from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class SawyerReachXYEnvV3(SawyerXYZEnv):
    """Reach to a target in the XY plane (Z is fixed to current EE height).

    The goal is sampled in the XY plane at the same Z as the end effector's
    initial position.  Only table, arm, and goal indicator are present.
    """

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
    ) -> None:
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1.0, 0.5)

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            height=height,
            width=width,
        )
        self.reward_function_version = reward_function_version

        # Lock the Z action: this is a 2D (XY-plane) reach task, so the
        # world model should never observe end-effector motion along Z.
        self._locked_action_idx = 2
        low = self.action_space.low.copy()
        high = self.action_space.high.copy()
        low[self._locked_action_idx] = 0.0
        high[self._locked_action_idx] = 0.0
        self.action_space = Box(low, high, dtype=self.action_space.dtype)

        self.init_config: InitConfigDict = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0, 0.6, 0.02]),
            "hand_init_pos": np.array([0, 0.6, 0.2]),
        }

        self.goal = np.array([0.1, 0.8, 0.2])
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        # Goal varies in X and Y; Z is fixed at hand_init_pos height
        goal_low = np.array([-0.25, 0.4, self.hand_init_pos[2]])
        goal_high = np.array([0.25, 0.8, self.hand_init_pos[2]])

        # obj part is unused but needed by _get_state_rand_vec
        obj_low = np.array([0.0, 0.5, 0.02])
        obj_high = np.array([0.25, 0.8, 0.02])

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(goal_low, goal_high, dtype=np.float64)

        self.num_resets = 0

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_reach_xy_v3.xml")

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        action[self._locked_action_idx] = 0.0
        return super().step(action)

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        reward, reach_dist, in_place = self.compute_reward(action, obs)
        success = float(reach_dist <= 0.05)

        info = {
            "success": success,
            "near_object": reach_dist,
            "grasp_success": 1.0,
            "grasp_reward": reach_dist,
            "in_place_reward": in_place,
            "obj_to_target": reach_dist,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_id_main_object(self) -> int:
        return -1

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return np.zeros(3)

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return np.zeros(4)

    def reset_model(self) -> npt.NDArray[np.float64]:
        # Randomize hand init position in XY, keep Z fixed
        hand_low = np.array(self.hand_low)
        hand_high = np.array(self.hand_high)
        self.hand_init_pos = self.np_random.uniform(hand_low, hand_high)
        self.hand_init_pos[2] = self.init_config["hand_init_pos"][2]

        self._reset_hand()
        self.obj_init_angle = self.init_config["obj_init_angle"]

        sampled_state = self._get_state_rand_vec()
        self._target_pos = sampled_state[3:].copy()
        # Force Z of target to match hand init Z
        self._target_pos[2] = self.hand_init_pos[2]

        self.obj_init_pos = sampled_state[:3].copy()
        self.init_tcp = self.tcp_center
        self.init_left_pad = self.get_body_com("leftpad")
        self.init_right_pad = self.get_body_com("rightpad")

        self.model.site("goal").pos = self._target_pos

        self.maxReachDist = np.linalg.norm(
            self.init_tcp - np.array(self._target_pos)
        )

        return self._get_obs()

    def compute_reward(
        self, actions: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float]:
        assert self._target_pos is not None
        _TARGET_RADIUS: float = 0.05
        tcp = self.tcp_center
        target = self._target_pos

        tcp_to_target = float(np.linalg.norm(tcp - target))

        in_place_margin = float(np.linalg.norm(self.hand_init_pos - target))
        in_place = reward_utils.tolerance(
            tcp_to_target,
            bounds=(0, _TARGET_RADIUS),
            margin=in_place_margin,
            sigmoid="long_tail",
        )

        reward = 10 * in_place
        reward = (reward - 5.0) / 5.0

        return (reward, tcp_to_target, in_place)
