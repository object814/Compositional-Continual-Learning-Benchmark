from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.expert_policy.action import Action
from metaworld.expert_policy.policy import Policy, assert_fully_parsed, move


class CompoPickPlaceBlockPolicy(Policy):
    """Expert policy for the compositional Pick-Place task.

    The task has two phases:
      Phase 1 — Pick up obj1 (red block) and place it at the goal position.
      Transition — After obj1 is placed, gently release the gripper and move
                straight up (+z) to avoid tilting the block.
      Phase 2 — Pick up obj2 (green block) and place it at the goal position.

    Transition detection: The environment switches observations from obj1 to obj2
    based on the `pickplace1_done` flag. We detect this by observing when the
    active object position changes significantly after we've completed placement.

    This policy is **stateful**: call ``reset()`` whenever the environment
    is reset so that the internal phase machine restarts correctly.
    """

    def __init__(self) -> None:
        super().__init__()
        self._phase: str = "pick_obj1"
        self._prev_obj_pos: npt.NDArray[np.float64] | None = None
        self._release_height: float | None = None
        self._obj1_placed: bool = False

    def reset(self) -> None:
        """Reset internal state. Must be called when the environment resets."""
        self._phase = "pick_obj1"
        self._prev_obj_pos = None
        self._release_height = None
        self._obj1_placed = False

    # ------------------------------------------------------------------
    # Observation parsing
    # ------------------------------------------------------------------
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(
        obs: npt.NDArray[np.float64],
    ) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "gripper_distance_apart": obs[3],
            "obj_pos": obs[4:7],  # obj1 in phase 1, obj2 in phase 2
            "obj_rot": obs[7:11],
            "goal_pos": obs[-3:],
            "unused_info_curr_obs": obs[11:18],
            "_prev_obs": obs[18:36],
        }

    # ------------------------------------------------------------------
    # Main action selection
    # ------------------------------------------------------------------
    def get_action(
        self, obs: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)
        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        # --- Detect phase transition from obj1 to obj2 ----------------
        # When obj1 is successfully placed, the environment switches
        # the observation to obj2, causing a significant position jump.
        if self._prev_obj_pos is not None and self._phase in ["place_obj1", "release_obj1", "lift_up"]:
            obj_pos_delta = float(np.linalg.norm(o_d["obj_pos"] - self._prev_obj_pos))
            # Large jump indicates the environment switched to obj2
            if obj_pos_delta > 0.1:
                self._phase = "pick_obj2"
                self._obj1_placed = True
        
        self._prev_obj_pos = o_d["obj_pos"].copy()

        # --- Execute phase-specific behavior --------------------------
        if self._phase == "pick_obj1":
            action["delta_pos"] = move(
                o_d["hand_pos"],
                self._pick_desired_pos(o_d),
                p=10.0,
            )
            action["grab_effort"] = self._grab_effort(o_d)
            
            # Transition to placement when object is grasped and lifted
            if (
                o_d["gripper_distance_apart"] < 0.75
                and o_d["obj_pos"][2] > 0.05
            ):
                self._phase = "place_obj1"

        elif self._phase == "place_obj1":
            action["delta_pos"] = move(
                o_d["hand_pos"],
                self._place_desired_pos(o_d),
                p=10.0,
            )
            action["grab_effort"] = 1.0  # Keep grasping
            
            # Transition to release when near goal position
            goal_dist = np.linalg.norm(o_d["hand_pos"][:2] - o_d["goal_pos"][:2])
            height_above_goal = o_d["hand_pos"][2] - o_d["goal_pos"][2]
            if goal_dist < 0.03 and height_above_goal < 0.06:
                self._phase = "release_obj1"

        elif self._phase == "release_obj1":
            # Hold position at goal and release gripper
            release_pos = o_d["goal_pos"].copy()
            release_pos[2] = o_d["goal_pos"][2] + 0.03

            action["delta_pos"] = move(
                o_d["hand_pos"],
                release_pos,
                p=5.0,  # Slower movement
            )
            action["grab_effort"] = -1.0  # Fully open gripper
            
            # Transition to lift up after releasing
            if o_d["gripper_distance_apart"] > 0.9:
                self._phase = "lift_up"

        elif self._phase == "lift_up":
            # Move straight up to avoid tilting the block
            lift_pos = o_d["hand_pos"].copy()
            lift_pos[2] = 0.15  # Lift up to safe height
            
            action["delta_pos"] = move(
                o_d["hand_pos"],
                lift_pos,
                p=8.0,
            )
            action["grab_effort"] = 0.0  # Keep gripper open
            
            # Stay in this phase until environment switches to obj2
            # (detected by observation change above)

        elif self._phase == "pick_obj2":
            action["delta_pos"] = move(
                o_d["hand_pos"],
                self._pick_desired_pos(o_d),
                p=10.0,
            )
            action["grab_effort"] = self._grab_effort(o_d)
            
            # Transition to final placement when object is grasped and lifted
            if (
                o_d["gripper_distance_apart"] < 0.75
                and o_d["obj_pos"][2] > 0.05
            ):
                self._phase = "place_obj2"

        elif self._phase == "place_obj2":
            action["delta_pos"] = move(
                o_d["hand_pos"],
                self._place_desired_pos(o_d),
                p=10.0,
            )
            action["grab_effort"] = 1.0  # Keep grasping
            
            # Final placement - release when near goal
            goal_dist = np.linalg.norm(o_d["hand_pos"][:2] - o_d["goal_pos"][:2])
            height_above_goal = o_d["hand_pos"][2] - o_d["goal_pos"][2]
            if goal_dist < 0.03 and height_above_goal < 0.06:
                # Release second object
                action["grab_effort"] = -1.0

        return action.array

    # ------------------------------------------------------------------
    # Desired position helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pick_desired_pos(o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        """Desired position for picking up an object."""
        pos_curr = o_d["hand_pos"]
        pos_obj = o_d["obj_pos"] + np.array([-0.005, 0, 0])
        gripper_separation = o_d["gripper_distance_apart"]
        
        # If XY error is large, move above the object
        if np.linalg.norm(pos_curr[:2] - pos_obj[:2]) > 0.02:
            return pos_obj + np.array([0.0, 0.0, 0.1])
        # Once XY aligned, drop down on top of object
        elif abs(pos_curr[2] - pos_obj[2]) > 0.05 and pos_obj[2] < 0.04:
            return pos_obj + np.array([0.0, 0.0, 0.03])
        # Wait for gripper to close before lifting
        elif gripper_separation > 0.73:
            return pos_curr
        # Lift object
        else:
            return pos_obj + np.array([0.0, 0.0, 0.15])

    @staticmethod
    def _place_desired_pos(o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        """Desired position for placing an object at the goal."""
        pos_curr = o_d["hand_pos"]
        pos_goal = o_d["goal_pos"].copy()

        # If not above goal in XY yet, move there at a safe height
        if np.linalg.norm(pos_curr[:2] - pos_goal[:2]) > 0.03:
            pos_goal[2] = max(pos_goal[2] + 0.1, 0.15)
            return pos_goal

        # Above goal in XY — lower down to just above goal height
        pos_goal[2] = pos_goal[2] + 0.03
        return pos_goal

    @staticmethod
    def _grab_effort(o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        """Gripper effort based on proximity to object."""
        pos_curr = o_d["hand_pos"]
        pos_obj = o_d["obj_pos"]
        
        if np.linalg.norm(pos_curr - pos_obj) < 0.07:
            return 1.0
        else:
            return 0.0
