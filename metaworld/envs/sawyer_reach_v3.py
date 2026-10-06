from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from metaworld.envs.sawyer_pick_place_v3 import SawyerPickPlaceEnvV3
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.utils import reward_utils


class SawyerReachEnvV3(SawyerPickPlaceEnvV3):
    """SawyerReachEnv.

    Motivation for V3:
        V1 was very difficult to solve because the observation didn't say where
        to move (where to reach).
    Changelog from V1 to V3:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/15/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
        - (6/15/20) Separated reach-push-pick-place into 3 separate envs.
    """

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
        initialise_region: str = "large",
    ) -> None:
        super().__init__(
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            reward_function_version=reward_function_version,
            height=height,
            width=width,
            initialise_region=initialise_region,
        )

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

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.obj_init_angle = self.init_config["obj_init_angle"]

        sampled_state = self._get_state_rand_vec()
        self._target_pos = sampled_state[3:].copy()
        while np.linalg.norm(sampled_state[:2] - self._target_pos[:2]) < 0.15:
            sampled_state = self._get_state_rand_vec()
            self._target_pos = sampled_state[3:].copy()

        self.obj_init_pos = sampled_state[:3].copy()
        self.init_tcp = self.tcp_center
        self.init_left_pad = self.get_body_com("leftpad")
        self.init_right_pad = self.get_body_com("rightpad")
        self._set_obj_xyz(self.obj_init_pos)

        self.model.site("goal").pos = self._target_pos
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "objGeom")
        self.model.geom_rgba[geom_id] = np.array([1.0, 0.0, 0.0, 1.0])

        self.maxReachDist = np.linalg.norm(self.init_tcp - np.array(self._target_pos))

        return self._get_obs()

    def compute_reward(
        self, actions: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float]:
        assert self._target_pos is not None
        if self.reward_function_version == "v2":
            _TARGET_RADIUS: float = 0.05
            tcp = self.tcp_center
            # obj = obs[4:7]
            # tcp_opened = obs[3]
            target = self._target_pos

            tcp_to_target = float(np.linalg.norm(tcp - target))
            # obj_to_target = float(np.linalg.norm(obj - target))

            in_place_margin = float(np.linalg.norm(self.hand_init_pos - target))
            in_place = reward_utils.tolerance(
                tcp_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=in_place_margin,
                sigmoid="long_tail",
            )

            reward = 10 * in_place # Original metaworld reward definition

            # Normalise to [-1, 1]
            reward = (reward - 5.0) / 5.0

            return (reward, tcp_to_target, in_place)
        else:
            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2
            goal = self._target_pos

            del actions
            del obs

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            reachDist = np.linalg.norm(fingerCOM - goal)
            reachRew = c1 * (self.maxReachDist - reachDist) + c1 * (
                np.exp(-(reachDist**2) / c2) + np.exp(-(reachDist**2) / c3)
            )
            reachRew = max(reachRew, 0)
            reward = reachRew
            return float(reward), float(reachDist), float(0.0)
