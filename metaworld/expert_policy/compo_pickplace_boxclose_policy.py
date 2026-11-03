from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.expert_policy.action import Action
from metaworld.expert_policy.policy import Policy, assert_fully_parsed, move


class CompoPickPlaceBoxClosePolicy(Policy):
    """Expert policy for the compositional PickPlace + BoxClose task.

    The task has two phases:
      Phase 1 — Pick up the red block and place it inside the open box
          (from SawyerPickPlaceBlockV3Policy).
      Phase 2 — Pick up the lid and close the box
          (from SawyerBoxCloseV3Policy).

    Transition is detected by a large shift in the goal position, which
    happens when the environment switches from the pick-place target
    (inside the box) to the box-close target (lid on top of box).

    This policy is **stateful**: call ``reset()`` whenever the environment
    is reset so that the internal phase machine restarts correctly.
    """

    def __init__(self) -> None:
        super().__init__()
        self._phase: str = "pick"
        self._prev_obj: npt.NDArray[np.float64] | None = None

    def reset(self) -> None:
        """Reset internal state. Must be called when the environment resets."""
        self._phase = "pick"
        self._prev_obj = None

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
            "obj_pos": obs[4:7],       # block (phase 1) or lid (phase 2)
            "obj_rot": obs[7:11],
            "goal_pos": obs[-3:],
            "unused_info": obs[11:-3],
        }

    # ------------------------------------------------------------------
    # Main action selection
    # ------------------------------------------------------------------
    def get_action(
        self, obs: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)
        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        # --- Detect the environment's phase transition ----------------
        # When the env sets pickplace_completed = True, the observed
        # object switches from the block (inside the box) to the lid
        # (on the table far away).  Both phase targets share the same
        # XY (box centre) so the goal barely moves — but the *object*
        # position jumps dramatically, making it a reliable signal.
        if self._prev_obj is not None and self._phase in ("pick", "place", "release"):
            obj_delta = float(np.linalg.norm(o_d["obj_pos"] - self._prev_obj))
            if obj_delta > 0.1:
                self._phase = "release_block"
        self._prev_obj = o_d["obj_pos"].copy()

        # --- Phase 1: Pick up the block -------------------------------
        if self._phase == "pick":
            action["delta_pos"] = move(
                o_d["hand_pos"], self._pick_desired_pos(o_d), p=10.0
            )
            action["grab_effort"] = self._pick_grab_effort(o_d)

            # Transition to place when grasped and lifted
            if (
                o_d["gripper_distance_apart"] < 0.75
                and o_d["obj_pos"][2] > 0.05
            ):
                self._phase = "place"

        # --- Phase 1: Place block into the box ------------------------
        elif self._phase == "place":
            action["delta_pos"] = move(
                o_d["hand_pos"], self._place_desired_pos(o_d), p=10.0
            )
            action["grab_effort"] = 1.0

            # Transition to release when near goal
            goal_dist = np.linalg.norm(
                o_d["hand_pos"][:2] - o_d["goal_pos"][:2]
            )
            height_above_goal = o_d["hand_pos"][2] - o_d["goal_pos"][2]
            if goal_dist < 0.03 and height_above_goal < 0.06:
                self._phase = "release"

        # --- Phase 1: Release the block inside the box ----------------
        elif self._phase == "release":
            release_pos = o_d["goal_pos"].copy()
            release_pos[2] = o_d["goal_pos"][2] + 0.03
            action["delta_pos"] = move(
                o_d["hand_pos"], release_pos, p=5.0
            )
            action["grab_effort"] = -1.0

        # --- Transition: release block and raise gripper --------------
        elif self._phase == "release_block":
            action["delta_pos"] = np.zeros(3)
            action["grab_effort"] = -1.0
            # After opening gripper, raise up to clear the box
            if o_d["gripper_distance_apart"] > 0.9:
                self._phase = "raise"

        elif self._phase == "raise":
            pos_curr = o_d["hand_pos"]
            target = np.array([pos_curr[0], pos_curr[1], 0.3])
            action["delta_pos"] = move(pos_curr, target, p=4.0)
            action["grab_effort"] = -1.0
            if pos_curr[2] > 0.25:
                self._phase = "box_close"

        # --- Phase 2: Pick up lid and close the box -------------------
        elif self._phase == "box_close":
            action["delta_pos"] = move(
                o_d["hand_pos"],
                self._box_close_desired_pos(o_d),
                p=25.0,
            )
            action["grab_effort"] = self._box_close_grab_effort(o_d)

        return action.array

    # ------------------------------------------------------------------
    # Phase 1 helpers — pick block
    # ------------------------------------------------------------------
    @staticmethod
    def _pick_desired_pos(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[Any]:
        pos_curr = o_d["hand_pos"]
        pos_obj = o_d["obj_pos"] + np.array([-0.005, 0, 0])
        gripper_sep = o_d["gripper_distance_apart"]

        if np.linalg.norm(pos_curr[:2] - pos_obj[:2]) > 0.02:
            return pos_obj + np.array([0.0, 0.0, 0.1])
        elif abs(pos_curr[2] - pos_obj[2]) > 0.05 and pos_obj[-1] < 0.04:
            return pos_obj + np.array([0.0, 0.0, 0.03])
        elif gripper_sep > 0.73:
            return pos_curr
        else:
            return pos_obj + np.array([0.0, 0.0, 0.15])

    @staticmethod
    def _pick_grab_effort(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> float:
        pos_curr = o_d["hand_pos"]
        pos_obj = o_d["obj_pos"]
        if np.linalg.norm(pos_curr - pos_obj) < 0.07:
            return 1.0
        return 0.0

    # ------------------------------------------------------------------
    # Phase 1 helpers — place block into box
    # ------------------------------------------------------------------
    @staticmethod
    def _place_desired_pos(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[Any]:
        pos_curr = o_d["hand_pos"]
        pos_goal = o_d["goal_pos"].copy()

        # Height that clears the box walls when transporting
        SAFE_HEIGHT = 0.22

        # Raise to safe height first
        if (
            pos_curr[2] < SAFE_HEIGHT
            and np.linalg.norm(pos_curr[:2] - pos_goal[:2]) > 0.02
        ):
            return np.array([pos_curr[0], pos_curr[1], SAFE_HEIGHT])

        # Move horizontally to above the goal (inside the box opening)
        if np.linalg.norm(pos_curr[:2] - pos_goal[:2]) > 0.03:
            pos_goal[2] = max(pos_goal[2] + 0.1, SAFE_HEIGHT)
            return pos_goal

        # Lower the block into the box
        pos_goal[2] = pos_goal[2] + 0.03
        return pos_goal

    # ------------------------------------------------------------------
    # Phase 2 helpers — box close (pick lid and place on box)
    # ------------------------------------------------------------------
    @staticmethod
    def _box_close_desired_pos(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[Any]:
        """Mirrors SawyerBoxCloseV3Policy with elevated transport path."""
        pos_curr = o_d["hand_pos"]
        pos_lid = o_d["obj_pos"] + np.array([0.0, 0.0, 0.02])
        pos_box = o_d["goal_pos"]
        transport_z = 0.24

        # Stage 1: align over the lid before descending to grasp
        if np.linalg.norm(pos_curr[:2] - pos_lid[:2]) > 0.01:
            return np.array([*pos_lid[:2], transport_z])
        # Stage 2: descend onto the lid handle
        elif abs(pos_curr[2] - pos_lid[2]) > 0.05:
            return pos_lid
        # Stage 3: lift in place first to clear the box during transport
        elif abs(pos_curr[2] - transport_z) > 0.02:
            return np.array([pos_curr[0], pos_curr[1], transport_z])
        # Stage 4: move in XY to above the box
        elif np.linalg.norm(pos_curr[:2] - pos_box[:2]) > 0.01:
            return np.array([pos_box[0], pos_box[1], transport_z])
        # Stage 5: descend to place the lid on the box
        else:
            return pos_box

    @staticmethod
    def _box_close_grab_effort(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> float:
        pos_curr = o_d["hand_pos"]
        pos_lid = o_d["obj_pos"] + np.array([0.0, 0.0, 0.02])

        if (
            np.linalg.norm(pos_curr[:2] - pos_lid[:2]) > 0.01
            or abs(pos_curr[2] - pos_lid[2]) > 0.13
        ):
            return 0.5
        else:
            return 1.0
