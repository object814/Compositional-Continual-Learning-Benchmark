from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.expert_policy.sawyer_pick_place_v3_policy import SawyerPickPlaceV3Policy


class SawyerGraspV3Policy(SawyerPickPlaceV3Policy):
    @staticmethod
    def _desired_pos(o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        pos_curr = o_d["hand_pos"]
        pos_puck = o_d["puck_pos"] + np.array([-0.005, 0.0, 0.0])
        lift_z = max(o_d["goal_pos"][2], pos_puck[2] + 0.08)

        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02:
            return pos_puck + np.array([0.0, 0.0, 0.1])
        if abs(pos_curr[2] - pos_puck[2]) > 0.04 and pos_puck[2] < 0.05:
            return pos_puck + np.array([0.0, 0.0, 0.02])
        if o_d["gripper_distance_apart"] > 0.7:
            return pos_curr
        return np.array([pos_curr[0], pos_curr[1], lift_z])

    @staticmethod
    def _grab_effort(o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        pos_curr = o_d["hand_pos"]
        pos_puck = o_d["puck_pos"]
        close_enough_to_grasp = (
            np.linalg.norm(pos_curr[:2] - pos_puck[:2]) < 0.03
            and abs(pos_curr[2] - pos_puck[2]) < 0.08
        )
        if close_enough_to_grasp or pos_puck[2] > 0.05:
            return 1.0
        return 0.0
