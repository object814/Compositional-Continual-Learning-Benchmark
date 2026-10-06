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


class SawyerPickPlaceEnvV3(SawyerXYZEnv):
    """SawyerPickPlaceEnv.

    Motivation for V3:
        V1 was very difficult to solve because the observation didn't say where
        to move after picking up the puck.
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
        initialise_region: str = 'large',
    ) -> None:
        if initialise_region == 'small':
            # Small initial region: Metaworld original
            print(">>> Using small initial region")
            goal_low = (-0.1, 0.8, 0.05)
            goal_high = (0.1, 0.9, 0.3)
            hand_low = (-0.5, 0.40, 0.05)
            hand_high = (0.5, 1, 0.5)
            obj_low = (-0.1, 0.6, 0.02)
            obj_high = (0.1, 0.7, 0.02)
        elif initialise_region == 'large':
            # Large initial region
            print(">>> Using large initial region")
            goal_low = (-0.25, 0.4, 0.1)
            goal_high = (0.0, 0.7, 0.3)
            hand_low = (-0.5, 0.40, 0.05)
            hand_high = (0.5, 1, 0.5)
            obj_low = (0.0, 0.5, 0.02)
            obj_high = (0.25, 0.8, 0.02)
        elif initialise_region == 'fixed':
            print(">>> Using fixed initial region")
            goal_low = (-0.11, 0.8, 0.2)
            goal_high = (-0.1, 0.8, 0.2)
            hand_low = (-0.5, 0.40, 0.05)
            hand_high = (0.5, 1, 0.5)
            obj_low = (0.09, 0.6, 0.02)
            obj_high = (0.1, 0.6, 0.02)
        else:
            raise ValueError(
                f"Invalid initialise_region: {initialise_region}. "
                "Must be one of ['small', 'large', 'fixed']"
            )

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

        self.init_config: InitConfigDict = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0, 0.6, 0.02]),
            "hand_init_pos": np.array([0, 0.6, 0.2]),
        }

        self.goal = np.array([0.1, 0.8, 0.2])

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
        return full_V3_path_for("sawyer_xyz/sawyer_pick_place_v3.xml")

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
        success = float(obj_to_target <= 0.07)
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
        return self.data.geom("objGeom").id

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("obj")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return Rotation.from_matrix(
            self.data.geom("objGeom").xmat.reshape(3, 3)
        ).as_quat()

    def fix_extreme_obj_pos(self, orig_init_pos: npt.NDArray[Any]) -> npt.NDArray[Any]:
        # This is to account for meshes for the geom and object are not
        # aligned. If this is not done, the object could be initialized in an
        # extreme position
        diff = self.get_body_com("obj")[:2] - self.get_body_com("obj")[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        # The convention we follow is that body_com[2] is always 0,
        # and geom_pos[2] is the object height
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

        # Custom color for target object
        # Find the geom id by name
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "objGeom")
        # Change RGBA (R, G, B, A)
        self.model.geom_rgba[geom_id] = np.array([1.0, 0.0, 0.0, 1.0])

        return self._get_obs()

    def _gripper_caging_reward(
        self,
        action: npt.NDArray[np.float32],
        obj_pos: npt.NDArray[Any],
        obj_radius: float = 0,  # All of these args are unused, just here to match
        pad_success_thresh: float = 0,  # the parent's type signature
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

        # compute the tcp_obj distance in the x_z plane
        tcp_xz = tcp + np.array([0.0, -tcp[1], 0.0])
        obj_position_x_z = np.copy(obj_pos) + np.array([0.0, -obj_pos[1], 0.0])
        tcp_obj_norm_x_z = float(np.linalg.norm(tcp_xz - obj_position_x_z, ord=2))

        # used for computing the tcp to object object margin in the x_z plane
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

        gripper_closed = min(max(0, action[-1]), 1)
        caging = reward_utils.hamacher_product(y_caging, x_z_caging)

        """Original metaworld code for reference"""
        # gripping = gripper_closed if caging > 0.97 else 0.0
        # caging_and_gripping = reward_utils.hamacher_product(caging, gripping)
        # caging_and_gripping = (caging_and_gripping + caging) / 2
        # return caging_and_gripping
    
        # smooth grasp signal (no threshold)
        gripper_closed = np.clip(action[-1], 0.0, 1.0)
        grasping = reward_utils.hamacher_product(caging, gripper_closed)
        # emphasize alignment slightly more than closure
        return 0.7 * caging + 0.3 * grasping

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        if self.reward_function_version == "v2":
            _TARGET_RADIUS: float = 0.05
            _OBJ_RADIUS: float = 0.015
            tcp = self.tcp_center
            obj = obs[4:7]
            tcp_opened = obs[3]
            target = self._target_pos

            obj_to_target = float(np.linalg.norm(obj - target))
            tcp_to_obj = float(np.linalg.norm(obj - tcp))
            in_place_margin = np.linalg.norm(self.obj_init_pos - target)

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
            transport_reward = reward_utils.hamacher_product(lift_reward, in_place)

            reward = (
                2.0 * xy_alignment
                + 1.0 * approach
                + 1.0 * closing_reward
                + 1.5 * object_grasped
                + 1.5 * grasp_and_lift
                + 3.0 * transport_reward
            )
            reward = min(reward, 9.95)
            if obj_to_target < _TARGET_RADIUS:
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
        else:
            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            heightTarget = self.heightTarget
            goal = self._target_pos
            del obs

            reachDist = np.linalg.norm(objPos - fingerCOM)
            placingDist = np.linalg.norm(objPos - goal)
            assert np.all(goal == self._get_site_pos("goal"))

            reachRew = -reachDist
            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_tcp[-1])

            if reachDistxy < 0.05:
                reachRew = -reachDist
            else:
                reachRew = -reachDistxy - 2 * zRew

            # incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(action[-1], 0) / 50
            tolerance = 0.01
            if objPos[2] >= (heightTarget - tolerance):
                self.pickCompleted = True
            else:
                self.pickCompleted = False

            objDropped = (
                (objPos[2] < (self.objHeight + 0.005))
                and (placingDist > 0.02)
                and (reachDist > 0.02)
            )
            # Object on the ground, far away from the goal, and from the gripper
            # Can tweak the margin limits

            hScale = 100
            if self.pickCompleted and not (objDropped):
                pickRew = hScale * heightTarget
            elif (reachDist < 0.1) and (objPos[2] > (self.objHeight + 0.005)):
                pickRew = hScale * min(heightTarget, objPos[2])
            else:
                pickRew = 0

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            objDropped = (
                (objPos[2] < (self.objHeight + 0.005))
                and (placingDist > 0.02)
                and (reachDist > 0.02)
            )

            cond = self.pickCompleted and (reachDist < 0.1) and not (objDropped)
            if cond:
                placeRew = 1000 * (self.maxPlacingDist - placingDist) + c1 * (
                    np.exp(-(placingDist**2) / c2) + np.exp(-(placingDist**2) / c3)
                )
                placeRew = max(placeRew, 0)
            else:
                placeRew = 0

            assert (placeRew >= 0) and (pickRew >= 0)
            reward = reachRew + pickRew + placeRew

            return float(reward), 0.0, 0.0, float(placingDist), 0.0, 0.0
