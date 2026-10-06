from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt

from metaworld.envs.sawyer_pick_place_v3 import SawyerPickPlaceEnvV3
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.utils import reward_utils


class SawyerGraspEnvV3(SawyerPickPlaceEnvV3):
    """Grasp-only variant of pick-place using the same XML asset."""

    _SUCCESS_LIFT_HEIGHT = 0.02
    _TARGET_LIFT_HEIGHT = 0.08
    _OBJ_RADIUS = 0.015
    _SUCCESS_GRASP_REWARD = 0.85
    _SUCCESS_TCP_TO_OBJ = 0.03
    _SUCCESS_GRIPPER_OPEN = 0.45

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
        obj = obs[4:7]
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            lift_reward,
        ) = self.compute_reward(action, obs)

        success = float(
            self._has_secure_grasp(
                tcp_to_obj=tcp_to_obj,
                tcp_open=tcp_open,
                grasp_reward=grasp_reward,
            )
        )
        info = {
            "success": success,
            "near_object": float(tcp_to_obj <= 0.03),
            "grasp_success": success,
            "grasp_reward": grasp_reward,
            "in_place_reward": lift_reward,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }

        return reward, info

    def _has_secure_grasp(
        self,
        tcp_to_obj: float,
        tcp_open: float,
        grasp_reward: float,
    ) -> bool:
        return bool(
            self.touching_main_object
            and (tcp_open <= self._SUCCESS_GRIPPER_OPEN)
            and (tcp_to_obj <= self._SUCCESS_TCP_TO_OBJ)
            and (grasp_reward >= self._SUCCESS_GRASP_REWARD)
        )

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

        self.objHeight = float(self.data.geom("objGeom").xpos[2])
        self.heightTarget = self.objHeight + self._TARGET_LIFT_HEIGHT
        self.model.site("goal").pos = self._target_pos

        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "objGeom")
        self.model.geom_rgba[geom_id] = np.array([1.0, 0.0, 0.0, 1.0])

        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert self.obj_init_pos is not None

        tcp = self.tcp_center
        obj = obs[4:7]
        tcp_opened = float(obs[3])

        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        tcp_to_obj_xy = float(np.linalg.norm(obj[:2] - tcp[:2]))
        xy_margin = float(np.linalg.norm(self.obj_init_pos[:2] - self.init_tcp[:2]))
        xy_alignment = reward_utils.tolerance(
            tcp_to_obj_xy,
            bounds=(0.0, 0.015),
            margin=max(xy_margin, 1e-6),
            sigmoid="long_tail",
        )

        desired_tcp_z = obj[2] + self._OBJ_RADIUS
        tcp_to_hover_z = float(abs(tcp[2] - desired_tcp_z))
        z_margin = float(abs(self.init_tcp[2] - desired_tcp_z))
        z_alignment = reward_utils.tolerance(
            tcp_to_hover_z,
            bounds=(0.0, 0.02),
            margin=max(z_margin, 1e-6),
            sigmoid="long_tail",
        )
        approach = reward_utils.hamacher_product(xy_alignment, z_alignment)

        grasp_reward = self._gripper_caging_reward(action, obj)

        lift_height = float(max(obj[2] - self.obj_init_pos[2], 0.0))
        lift_to_target = float(max(self._SUCCESS_LIFT_HEIGHT - lift_height, 0.0))
        lift_margin = max(self._SUCCESS_LIFT_HEIGHT, 1e-6)
        lift_reward = reward_utils.tolerance(
            lift_to_target,
            bounds=(0.0, 0.003),
            margin=lift_margin,
            sigmoid="long_tail",
        )
        grasp_and_lift = reward_utils.hamacher_product(grasp_reward, lift_reward)
        closing_reward = reward_utils.hamacher_product(
            xy_alignment, float(np.clip(action[-1], 0.0, 1.0))
        )

        reward = (
            2.0 * xy_alignment
            + 1.5 * approach
            + 1.5 * closing_reward
            + 2.0 * grasp_reward
            + 3.0 * grasp_and_lift
        )
        reward = min(reward, 9.95)

        secure_grasp = self._has_secure_grasp(
            tcp_to_obj=tcp_to_obj,
            tcp_open=tcp_opened,
            grasp_reward=grasp_reward,
        )
        lifted = obj[2] >= self.obj_init_pos[2] + self._SUCCESS_LIFT_HEIGHT
        if lifted:
            reward = max(reward, 9.0 + lift_reward)
        if secure_grasp:
            reward = 10.0

        reward = (reward - 5.0) / 5.0
        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            lift_to_target,
            grasp_reward,
            lift_reward,
        )
