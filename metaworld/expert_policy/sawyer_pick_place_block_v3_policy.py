from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.expert_policy.action import Action
from metaworld.expert_policy.policy import Policy, assert_fully_parsed, move


class SawyerPickPlaceBlockV3Policy(Policy):
    """Expert policy for pick-place-block tasks.

    Stateful: tracks pick/place/release phases. Call ``reset()`` when the
    environment resets.
    """

    def __init__(self) -> None:
        super().__init__()
        self._phase: str = "pick"

    def reset(self) -> None:
        self._phase = "pick"

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "gripper_distance_apart": obs[3],
            "obj_pos": obs[4:7],
            "obj_rot": obs[7:11],
            "goal_pos": obs[-3:],
            "unused_info_curr_obs": obs[11:18],
            "_prev_obs": obs[18:36],
        }

    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        if self._phase == "pick":
            action["delta_pos"] = move(
                o_d["hand_pos"], self._pick_desired_pos(o_d), p=10.0
            )
            action["grab_effort"] = self._grab_effort(o_d)

            # Transition to place when grasped and lifted
            if (
                o_d["gripper_distance_apart"] < 0.75
                and o_d["obj_pos"][2] > 0.05
            ):
                self._phase = "place"

        elif self._phase == "place":
            action["delta_pos"] = move(
                o_d["hand_pos"], self._place_desired_pos(o_d), p=10.0
            )
            action["grab_effort"] = 1.0

            # Transition to release when near goal
            goal_dist = np.linalg.norm(o_d["hand_pos"][:2] - o_d["goal_pos"][:2])
            height_above_goal = o_d["hand_pos"][2] - o_d["goal_pos"][2]
            if goal_dist < 0.03 and height_above_goal < 0.06:
                self._phase = "release"

        elif self._phase == "release":
            # Hold position at goal and open gripper
            release_pos = o_d["goal_pos"].copy()
            release_pos[2] = o_d["goal_pos"][2] + 0.03
            action["delta_pos"] = move(
                o_d["hand_pos"], release_pos, p=5.0
            )
            action["grab_effort"] = -1.0

        return action.array

    @staticmethod
    def _pick_desired_pos(o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        pos_curr = o_d["hand_pos"]
        pos_obj = o_d["obj_pos"] + np.array([-0.005, 0, 0])
        gripper_separation = o_d["gripper_distance_apart"]

        # If XY error is large, move above the object
        if np.linalg.norm(pos_curr[:2] - pos_obj[:2]) > 0.02:
            return pos_obj + np.array([0.0, 0.0, 0.1])
        # Once XY aligned, drop down on top of object
        elif abs(pos_curr[2] - pos_obj[2]) > 0.05 and pos_obj[-1] < 0.04:
            return pos_obj + np.array([0.0, 0.0, 0.03])
        # Wait for gripper to close before lifting
        elif gripper_separation > 0.73:
            return pos_curr
        # Lift object
        else:
            return pos_obj + np.array([0.0, 0.0, 0.15])

    @staticmethod
    def _place_desired_pos(o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
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
        pos_curr = o_d["hand_pos"]
        pos_obj = o_d["obj_pos"]
        if np.linalg.norm(pos_curr - pos_obj) < 0.07:
            return 1.0
        else:
            return 0.0
