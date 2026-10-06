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


class SawyerBoxCloseEnvV3(SawyerXYZEnv):
    # Minimum XY center-to-center distance between lid and box.
    BOX_MIN_DIST = 0.35
    # XY bounds for randomizing both lid and box centers.
    BOX_X_RANGE = (-0.25, 0.25)
    BOX_Y_RANGE = (0.45, 0.80)
    # Lid body position z for resting on table at reset.
    LID_INIT_Z = -0.02

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
        hand_high = (0.5, 1, 0.5)
        obj_low = (self.BOX_X_RANGE[0], self.BOX_Y_RANGE[0], 0.02)
        obj_high = (self.BOX_X_RANGE[1], self.BOX_Y_RANGE[1], 0.02)
        goal_low = np.array([self.BOX_X_RANGE[0], self.BOX_Y_RANGE[0], 0.133])
        goal_high = np.array([self.BOX_X_RANGE[1], self.BOX_Y_RANGE[1], 0.133])

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
            "obj_init_pos": np.array([0, 0.55, self.LID_INIT_Z], dtype=np.float32),
            "hand_init_pos": np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0.0, 0.75, 0.133])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._target_to_obj_init = None

        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)
        # _random_reset_space: [lid_x, lid_y, box_x, box_y]
        self._random_reset_space = Box(
            np.array(
                [
                    self.BOX_X_RANGE[0],
                    self.BOX_Y_RANGE[0],
                    self.BOX_X_RANGE[0],
                    self.BOX_Y_RANGE[0],
                ]
            ),
            np.array(
                [
                    self.BOX_X_RANGE[1],
                    self.BOX_Y_RANGE[1],
                    self.BOX_X_RANGE[1],
                    self.BOX_Y_RANGE[1],
                ]
            ),
            dtype=np.float64,
        )

        self.init_obj_quat = None
        self.liftThresh = 0.12

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_box.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            reward_grab,
            reward_ready,
            reward_success,
            success,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(success),
            "near_object": reward_ready,
            "grasp_success": reward_grab >= 0.5,
            "grasp_reward": reward_grab,
            "in_place_reward": reward_success,
            "obj_to_target": 0,
            "unscaled_reward": reward,
        }

        return reward, info

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        return []

    def _get_id_main_object(self) -> int:
        return self.model.geom_name2id("BoxHandleGeom")

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("top_link")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("top_link").xquat

    def _sample_non_overlapping_box_lid(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Sample lid and box XY positions with a minimum separation."""
        assert self._random_reset_space is not None
        low = self._random_reset_space.low
        high = self._random_reset_space.high
        rng = self.np_random if self.seeded_rand_vec else np.random

        for _ in range(1000):
            rand_vec = rng.uniform(low, high, size=low.size).astype(np.float64)
            lid_xy = rand_vec[:2]
            box_xy = rand_vec[2:4]
            if np.linalg.norm(lid_xy - box_xy) >= self.BOX_MIN_DIST:
                self._last_rand_vec = rand_vec
                return lid_xy, box_xy

        fallback = np.array([-0.15, 0.7, 0.15, 0.7], dtype=np.float64)
        self._last_rand_vec = fallback
        return fallback[:2].copy(), fallback[2:4].copy()

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        box_height = self.get_body_com("boxbody")[2]

        lid_xy, box_xy = self._sample_non_overlapping_box_lid()
        self.obj_init_pos = np.array(
            [lid_xy[0], lid_xy[1], self.LID_INIT_Z], dtype=np.float64
        )
        self._target_pos = np.array([box_xy[0], box_xy[1], self.goal[-1]-0.06])

        self.model.body("boxbody").pos = np.concatenate(
            [self._target_pos[:2], [box_height]]
        )

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._set_obj_xyz(self.obj_init_pos)
        self.model.site("goal").pos = self._target_pos

        # Cache initial gripper state for smooth caging-based grasp shaping.
        self.init_tcp = self.tcp_center
        self.init_left_pad = self.get_body_com("leftpad")
        self.init_right_pad = self.get_body_com("rightpad")

        self.objHeight = self.data.geom("BoxHandleGeom").xpos[2]
        self.heightTarget = self.objHeight + self.liftThresh

        self.maxPlacingDist = (
            np.linalg.norm(
                np.array(
                    [self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]
                )
                - np.array(self._target_pos)
            )
            + self.heightTarget
        )
        self.pickCompleted = False

        return self._get_obs()

    @staticmethod
    def _reward_grab_effort(actions: npt.NDArray[Any]) -> float:
        return float(np.clip(((np.clip(actions[3], -1, 1) + 1.0) / 2.0), 0.0, 1.0))

    @staticmethod
    def _reward_quat(obs) -> float:
        # Ideal upright lid has quat [.707, 0, 0, .707]
        # Rather than deal with an angle between quaternions, just approximate:
        ideal = np.array([0.707, 0, 0, 0.707])
        error = float(np.linalg.norm(obs[7:11] - ideal))
        return max(1.0 - error / 0.2, 0.0)

    @staticmethod
    def _reward_pos(
        obs: npt.NDArray[np.float64], target_pos: npt.NDArray[Any]
    ) -> tuple[float, float]:
        hand = obs[:3]
        lid = obs[4:7] + np.array([0.0, 0.0, 0.02])

        threshold = 0.02
        # floor is a 3D funnel centered on the lid's handle
        radius = np.linalg.norm(hand[:2] - lid[:2])
        if radius <= threshold:
            floor = 0.0
        else:
            floor = 0.04 * np.log(radius - threshold) + 0.4
        # prevent the hand from running into the handle prematurely by keeping
        # it above the "floor"
        above_floor = (
            1.0
            if hand[2] >= floor
            else reward_utils.tolerance(
                floor - hand[2],
                bounds=(0.0, 0.01),
                margin=floor / 2.0,
                sigmoid="long_tail",
            )
        )
        # grab the lid's handle
        in_place = reward_utils.tolerance(
            float(np.linalg.norm(hand - lid)),
            bounds=(0, 0.02),
            margin=0.5,
            sigmoid="long_tail",
        )
        ready_to_lift = reward_utils.hamacher_product(above_floor, in_place)

        # now actually put the lid on the box
        pos_error = target_pos - lid
        error_scale = np.array([1.0, 1.0, 3.0])  # Emphasize Z error
        a = 0.2  # Relative importance of just *trying* to lift the lid at all
        b = 0.8  # Relative importance of placing the lid on the box
        lifted = a * float(lid[2] > 0.04) + b * reward_utils.tolerance(
            float(np.linalg.norm(pos_error * error_scale)),
            bounds=(0, 0.05),
            margin=0.25,
            sigmoid="long_tail",
        )

        return ready_to_lift, lifted

    def _gripper_caging_reward(
        self,
        action: npt.NDArray[np.float32],
        obj_pos: npt.NDArray[Any],
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
        self, actions: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, bool]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        if self.reward_function_version == "v2":
            _TARGET_RADIUS: float = 0.05
            _LID_RADIUS: float = 0.015

            tcp = self.tcp_center
            lid_base = obs[4:7]
            lid_handle = lid_base + np.array([0.0, 0.0, 0.02])
            target = self._target_pos

            reward_quat = SawyerBoxCloseEnvV3._reward_quat(obs)
            reward_grab_effort = SawyerBoxCloseEnvV3._reward_grab_effort(actions)

            pos_error_scale = np.array([1.0, 1.0, 3.0])
            lid_to_target = float(np.linalg.norm((lid_handle - target) * pos_error_scale))
            init_lid_handle = self.obj_init_pos + np.array([0.0, 0.0, 0.02])
            in_place_margin = float(
                np.linalg.norm((init_lid_handle - target) * pos_error_scale)
            )
            in_place = reward_utils.tolerance(
                lid_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=max(in_place_margin, 1e-6),
                sigmoid="long_tail",
            )

            # Explicitly shape XY alignment of lid-to-box during transport.
            lid_to_target_xy = float(np.linalg.norm(lid_handle[:2] - target[:2]))
            target_xy_margin = float(np.linalg.norm(init_lid_handle[:2] - target[:2]))
            xy_transport_alignment = reward_utils.tolerance(
                lid_to_target_xy,
                bounds=(0.0, 0.015),
                margin=max(target_xy_margin, 1e-6),
                sigmoid="long_tail",
            )

            tcp_to_lid_xy = float(np.linalg.norm(lid_handle[:2] - tcp[:2]))
            xy_margin = float(np.linalg.norm(init_lid_handle[:2] - self.init_tcp[:2]))
            xy_alignment = reward_utils.tolerance(
                tcp_to_lid_xy,
                bounds=(0.0, 0.015),
                margin=max(xy_margin, 1e-6),
                sigmoid="long_tail",
            )

            desired_tcp_z = lid_handle[2] + _LID_RADIUS
            tcp_to_hover_z = float(abs(tcp[2] - desired_tcp_z))
            z_margin = float(abs(self.init_tcp[2] - desired_tcp_z))
            z_alignment = reward_utils.tolerance(
                tcp_to_hover_z,
                bounds=(0.0, 0.02),
                margin=max(z_margin, 1e-6),
                sigmoid="long_tail",
            )
            approach = reward_utils.hamacher_product(xy_alignment, z_alignment)

            object_caging = self._gripper_caging_reward(actions, lid_base)
            closing_reward = reward_utils.hamacher_product(xy_alignment, reward_grab_effort)
            grasp_reward = reward_utils.hamacher_product(object_caging, closing_reward)

            lift_to_pick = float(max(self.heightTarget - lid_handle[2], 0.0))
            lift_margin = max(self.heightTarget - init_lid_handle[2], 1e-6)
            lift_reward = reward_utils.tolerance(
                lift_to_pick,
                bounds=(0.0, 0.01),
                margin=lift_margin,
                sigmoid="long_tail",
            )

            grasp_and_lift = reward_utils.hamacher_product(grasp_reward, lift_reward)
            lift_or_placed = max(lift_reward, in_place)
            orientation_scale = 0.5 + 0.5 * reward_quat
            transport_reward = orientation_scale * reward_utils.hamacher_product(
                lift_or_placed, in_place
            )
            xy_transport_reward = reward_utils.hamacher_product(
                lift_or_placed, xy_transport_alignment
            )

            reward = (
                2.0 * xy_alignment
                + 1.0 * approach
                + 1.0 * closing_reward
                + 1.0 * grasp_reward
                + 1.0 * grasp_and_lift
                + 2.0 * xy_transport_reward
                + 2.0 * transport_reward
            )
            reward = min(float(reward), 9.95)

            # Override reward on success
            xy_aligned = np.linalg.norm(obs[4:6] - self._target_pos[:-1]) < 0.03
            z_reached = np.abs(obs[6] - self._target_pos[2]) < 0.03
            success = bool(xy_aligned and z_reached)
            if success:
                reward = 10.0

            # Normalise to [-1, 1]
            reward = (reward - 5.0) / 5.0

            return (
                reward,
                grasp_reward,
                approach,
                in_place,
                success,
            )
        else:
            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            heightTarget = self.heightTarget
            placeGoal = self._target_pos

            placingDist = np.linalg.norm(objPos - placeGoal)
            reachDist = np.linalg.norm(objPos - fingerCOM)

            reachRew = -reachDist
            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_tcp[-1])

            if reachDistxy < 0.05:
                reachRew = -reachDist
            else:
                reachRew = -reachDistxy - 2 * zRew

            # incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(actions[-1], 0) / 50

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
            if self.pickCompleted and not objDropped:
                pickRew = hScale * heightTarget
            elif (reachDist < 0.1) and (objPos[2] > (self.objHeight + 0.005)):
                pickRew = hScale * min(heightTarget, objPos[2])
            else:
                pickRew = 0

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            cond = self.pickCompleted and (reachDist < 0.1) and not objDropped
            if cond:
                placeRew = 1000 * (self.maxPlacingDist - placingDist) + c1 * (
                    np.exp(-(placingDist**2) / c2) + np.exp(-(placingDist**2) / c3)
                )
                placeRew = max(placeRew, 0)
                placeRew, placingDist = [placeRew, placingDist]
            else:
                placeRew, placingDist = [0, placingDist]

            assert (placeRew >= 0) and (pickRew >= 0)
            reward = reachRew + pickRew + placeRew

            xy_aligned = np.linalg.norm(obs[4:6] - self._target_pos[:-1]) < 0.03
            z_reached = np.abs(obs[6] - self._target_pos[2]) < 0.03
            success = bool(xy_aligned and z_reached)

            return float(reward), 0.0, 0.0, 0.0, success
