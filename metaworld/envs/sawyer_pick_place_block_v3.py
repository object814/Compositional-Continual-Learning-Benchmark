from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils

import mujoco


class SawyerPickPlaceBlockEnvV3(SawyerXYZEnv):
    """Pick-and-place with a box block instead of a cylinder.

    Uses the large initialisation region with goal z fixed at table height
    (0.02). Success requires the block to be near the target AND the gripper
    to be open (i.e. the block must be released).
    """

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
        obj_rgba: tuple[float, float, float, float] | None = None,
    ) -> None:
        # Large initial region, goal z fixed at 0.02
        goal_low = (-0.25, 0.4, 0.02)
        goal_high = (0.0, 0.7, 0.02)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0.0, 0.5, 0.02)
        obj_high = (0.25, 0.8, 0.02)

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
        self._obj_rgba = obj_rgba

        self.init_config: InitConfigDict = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0, 0.6, 0.02]),
            "hand_init_pos": np.array([0, 0.6, 0.2]),
        }

        self.goal = np.array([0.1, 0.8, 0.02])

        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

        self.num_resets = 0
        self.obj_init_pos = None

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_pick_place_block_v3.xml")

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
            in_place_reward,
        ) = self.compute_reward(action, obs)

        # Success requires block near target AND gripper released
        success = float(obj_to_target <= 0.05 and tcp_open > 0.8)
        near_object = float(tcp_to_obj <= 0.03)
        assert self.obj_init_pos is not None
        grasp_success = float(
            self.touching_main_object
            and (tcp_open > 0)
            and (obj[2] - 0.02 > self.obj_init_pos[2])
        )
        info = {
            "success": success,
            "near_object": near_object,
            "grasp_success": grasp_success,
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place_reward,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_id_main_object(self) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "objA")
        return self.model.body_geomadr[body_id]

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("obj")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("obj").xquat

    def fix_extreme_obj_pos(self, orig_init_pos: npt.NDArray[Any]) -> npt.NDArray[Any]:
        diff = self.get_body_com("obj")[:2] - self.get_body_com("obj")[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        return np.array(
            [adjusted_pos[0], adjusted_pos[1], self.get_body_com("obj")[-1]]
        )

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.fix_extreme_obj_pos(self.init_config["obj_init_pos"])
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_pos = self._get_state_rand_vec()
        self._target_pos = goal_pos[3:]
        while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
        self._target_pos = goal_pos[-3:]
        self.obj_init_pos = goal_pos[:3]
        self.init_tcp = self.tcp_center
        self.init_left_pad = self.get_body_com("leftpad")
        self.init_right_pad = self.get_body_com("rightpad")

        self._set_obj_xyz(self.obj_init_pos)
        self.model.site("goal").pos = self._target_pos

        self.objHeight = self.data.geom("objGeom").xpos[2]
        self.heightTarget = self.objHeight + 0.04

        self.maxPlacingDist = (
            np.linalg.norm(
                np.array(
                    [self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]
                )
                - np.array(self._target_pos)
            )
            + self.heightTarget
        )

        self.maxPushDist = np.linalg.norm(
            self.obj_init_pos[:2] - np.array(self._target_pos)[:2]
        )

        # Apply custom object colour to all geoms on the objA body
        if self._obj_rgba is not None:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "objA")
            start = self.model.body_geomadr[body_id]
            count = self.model.body_geomnum[body_id]
            for gid in range(start, start + count):
                self.model.geom_rgba[gid] = np.array(self._obj_rgba)

        return self._get_obs()

    def _gripper_caging_reward(
        self,
        action: npt.NDArray[np.float32],
        obj_pos: npt.NDArray[Any],
        obj_radius: float = 0,
        pad_success_thresh: float = 0,
        object_reach_radius: float = 0,
        xz_thresh: float = 0,
        desired_gripper_effort: float = 1.0,
        high_density: bool = False,
        medium_density: bool = False,
    ) -> float:
        pad_success_margin = 0.05
        x_z_success_margin = 0.005
        obj_radius = 0.015
        tcp = self.tcp_center
        left_pad = self.get_body_com("leftpad")
        right_pad = self.get_body_com("rightpad")
        delta_object_y_left_pad = left_pad[1] - obj_pos[1]
        delta_object_y_right_pad = obj_pos[1] - right_pad[1]
        right_caging_margin = abs(
            abs(obj_pos[1] - self.init_right_pad[1]) - pad_success_margin
        )
        left_caging_margin = abs(
            abs(obj_pos[1] - self.init_left_pad[1]) - pad_success_margin
        )

        right_caging = reward_utils.tolerance(
            delta_object_y_right_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_caging = reward_utils.tolerance(
            delta_object_y_left_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )

        y_caging = reward_utils.hamacher_product(left_caging, right_caging)

        tcp_xz = tcp + np.array([0.0, -tcp[1], 0.0])
        obj_position_x_z = np.copy(obj_pos) + np.array([0.0, -obj_pos[1], 0.0])
        tcp_obj_norm_x_z = float(np.linalg.norm(tcp_xz - obj_position_x_z, ord=2))

        assert self.obj_init_pos is not None
        init_obj_x_z = self.obj_init_pos + np.array([0.0, -self.obj_init_pos[1], 0.0])
        init_tcp_x_z = self.init_tcp + np.array([0.0, -self.init_tcp[1], 0.0])
        tcp_obj_x_z_margin = (
            np.linalg.norm(init_obj_x_z - init_tcp_x_z, ord=2) - x_z_success_margin
        )

        x_z_caging = reward_utils.tolerance(
            tcp_obj_norm_x_z,
            bounds=(0, x_z_success_margin),
            margin=tcp_obj_x_z_margin,
            sigmoid="long_tail",
        )

        caging = reward_utils.hamacher_product(y_caging, x_z_caging)

        gripper_closed = np.clip(action[-1], 0.0, 1.0)
        grasping = reward_utils.hamacher_product(caging, gripper_closed)
        return 0.7 * caging + 0.3 * grasping

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        _TARGET_RADIUS: float = 0.05
        _OBJ_RADIUS: float = 0.015
        tcp = self.tcp_center
        obj = obs[4:7]
        tcp_opened = obs[3]
        target = self._target_pos

        obj_to_target = float(np.linalg.norm(obj - target))
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        in_place_margin = float(np.linalg.norm(self.obj_init_pos - target))

        in_place = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, _TARGET_RADIUS),
            margin=in_place_margin,
            sigmoid="long_tail",
        )

        tcp_to_obj_xy = float(np.linalg.norm(obj[:2] - tcp[:2]))
        xy_margin = float(np.linalg.norm(self.obj_init_pos[:2] - self.init_tcp[:2]))
        xy_alignment = reward_utils.tolerance(
            tcp_to_obj_xy,
            bounds=(0.0, 0.015),
            margin=max(xy_margin, 1e-6),
            sigmoid="long_tail",
        )

        desired_tcp_z = obj[2] + _OBJ_RADIUS
        tcp_to_hover_z = float(abs(tcp[2] - desired_tcp_z))
        z_margin = float(abs(self.init_tcp[2] - desired_tcp_z))
        z_alignment = reward_utils.tolerance(
            tcp_to_hover_z,
            bounds=(0.0, 0.02),
            margin=max(z_margin, 1e-6),
            sigmoid="long_tail",
        )
        approach = reward_utils.hamacher_product(xy_alignment, z_alignment)

        object_grasped = self._gripper_caging_reward(action, obj)
        closing_reward = reward_utils.hamacher_product(
            xy_alignment, float(np.clip(action[-1], 0.0, 1.0))
        )

        lift_to_pick = float(max(self.heightTarget - obj[2], 0.0))
        lift_margin = max(self.heightTarget - self.obj_init_pos[2], 1e-6)
        lift_reward = reward_utils.tolerance(
            lift_to_pick,
            bounds=(0.0, 0.01),
            margin=lift_margin,
            sigmoid="long_tail",
        )

        grasp_and_lift = reward_utils.hamacher_product(object_grasped, lift_reward)
        lift_or_placed = max(lift_reward, in_place)
        transport_reward = reward_utils.hamacher_product(lift_or_placed, in_place)

        reward = (
            2.0 * xy_alignment
            + 1.0 * approach
            + 1.0 * closing_reward
            + 1.5 * object_grasped
            + 1.5 * grasp_and_lift
            + 3.0 * transport_reward
        )
        reward = min(reward, 9.5)

        if obj_to_target < _TARGET_RADIUS:
            reward = 9.7
        if obj_to_target < 0.01 and tcp_opened > 0.8:
            reward = 10.0

        # Normalise from [0, 10] to [-1, 1]
        reward = (reward - 5.0) / 5.0

        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            obj_to_target,
            object_grasped,
            in_place,
        )


class SawyerPickPlaceRedBlockEnvV3(SawyerPickPlaceBlockEnvV3):
    def __init__(self, **kwargs):
        kwargs.setdefault("obj_rgba", (1.0, 0.0, 0.0, 1.0))
        super().__init__(**kwargs)


class SawyerPickPlaceGreenBlockEnvV3(SawyerPickPlaceBlockEnvV3):
    def __init__(self, **kwargs):
        kwargs.setdefault("obj_rgba", (0.0, 1.0, 0.0, 1.0))
        super().__init__(**kwargs)
