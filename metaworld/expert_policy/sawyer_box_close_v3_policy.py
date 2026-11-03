from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.expert_policy.action import Action
from metaworld.expert_policy.policy import Policy, assert_fully_parsed, move


class SawyerBoxCloseV3Policy(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "gripper": obs[3],
            "lid_pos": obs[4:7],
            "extra_info_1": obs[7:-3],
            "box_pos": obs[-3:-1],
            "extra_info_2": obs[-1],
        }

    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})
        action["delta_pos"] = move(
            o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=25.0
        )
        action["grab_effort"] = self._grab_effort(o_d)

        return action.array

    @staticmethod
    def _desired_pos(o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        pos_curr = o_d["hand_pos"]
        pos_lid = o_d["lid_pos"] + np.array([0.0, 0.0, +0.02])
        pos_box = np.array([*o_d["box_pos"], 0.15])
        transport_z = 0.24

        # Stage 1: align over the lid before descending to grasp.
        if np.linalg.norm(pos_curr[:2] - pos_lid[:2]) > 0.01:
            return np.array([*pos_lid[:2], transport_z])
        # Stage 2: descend onto the lid handle.
        elif abs(pos_curr[2] - pos_lid[2]) > 0.05:
            return pos_lid
        # Stage 3: lift in place first (hold XY fixed) to clear the box during transport.
        elif abs(pos_curr[2] - transport_z) > 0.02:
            return np.array([pos_curr[0], pos_curr[1], transport_z])
        # Stage 4: move in XY only after reaching the lifted transport height.
        elif np.linalg.norm(pos_curr[:2] - pos_box[:2]) > 0.01:
            return np.array([pos_box[0], pos_box[1], transport_z])
        # Stage 5: descend vertically to place the lid on the box.
        else:
            return pos_box

    @staticmethod
    def _grab_effort(o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        pos_curr = o_d["hand_pos"]
        pos_lid = o_d["lid_pos"] + np.array([0.0, 0.0, +0.02])

        if (
            np.linalg.norm(pos_curr[:2] - pos_lid[:2]) > 0.01
            or abs(pos_curr[2] - pos_lid[2]) > 0.13
        ):
            return 0.5
        # While end effector is moving down toward the puck, begin closing the grabber
        else:
            return 1.0
