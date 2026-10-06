from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

import mujoco

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class SawyerBinPickingEnvV3(SawyerXYZEnv):
    """SawyerBinPickingEnv.

    Motivation for V3:
        V1 was often unsolvable because the cube could be located outside of
        the starting bin. It could even be near the base of the Sawyer and out
        of reach of the gripper. V3 changes the `obj_low` and `obj_high` bounds
        to fix this.
    Changelog from V1 to V3:
        - (7/20/20) Changed object initialization space
        - (7/24/20) Added Byron's XML changes
        - (11/23/20) Updated reward function to new pick-place style
    """

    # Each bin is ~0.2m x 0.2m (walls at ±0.095).
    # Minimum center-to-center distance to avoid overlap.
    BIN_MIN_DIST = 0.25
    # XY bounds for randomising each bin centre.
    BIN_X_RANGE = (-0.25, 0.25)
    BIN_Y_RANGE = (0.45, 0.85)

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
        bin_1_rgba: tuple[float, float, float, float] | None = None,
        bin_2_rgba: tuple[float, float, float, float] | None = None,
    ) -> None:
        hand_low = (-0.5, 0.40, 0.07)
        hand_high = (0.5, 1, 0.5)

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
            "obj_init_pos": np.array([-0.12, 0.7, 0.02]),
            "hand_init_pos": np.array((0, 0.6, 0.2)),
        }
        self.goal = np.array([0.12, 0.7, 0.02])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._target_to_obj_init: float | None = None

        self._bin_1_rgba = bin_1_rgba
        self._bin_2_rgba = bin_2_rgba

        self.liftThresh = 0.1

        # Observation spaces use wide bounds to cover all possible bin placements
        obj_low = (self.BIN_X_RANGE[0], self.BIN_Y_RANGE[0], 0.02)
        obj_high = (self.BIN_X_RANGE[1], self.BIN_Y_RANGE[1], 0.02)
        goal_low = np.array([self.BIN_X_RANGE[0], self.BIN_Y_RANGE[0], -0.001])
        goal_high = np.array([self.BIN_X_RANGE[1], self.BIN_Y_RANGE[1], +0.001])

        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
            dtype=np.float64,
        )

        self.goal_and_obj_space = Box(
            np.hstack((goal_low[:2], obj_low[:2])),
            np.hstack((goal_high[:2], obj_high[:2])),
            dtype=np.float64,
        )

        self.goal_space = Box(goal_low, goal_high, dtype=np.float64)
        # _random_reset_space: [bin1_x, bin1_y, bin2_x, bin2_y]
        self._random_reset_space = Box(
            np.array([self.BIN_X_RANGE[0], self.BIN_Y_RANGE[0],
                       self.BIN_X_RANGE[0], self.BIN_Y_RANGE[0]]),
            np.array([self.BIN_X_RANGE[1], self.BIN_Y_RANGE[1],
                       self.BIN_X_RANGE[1], self.BIN_Y_RANGE[1]]),
            dtype=np.float64,
        )

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_bin_picking.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            near_object,
            grasp_success,
            obj_to_target,
            grasp_reward,
            in_place_reward,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(obj_to_target <= 0.05),
            "near_object": float(near_object),
            "grasp_success": float(grasp_success),
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place_reward,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }

        return reward, info

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        return []

    def _get_id_main_object(self) -> int:
        # The object geoms in objA.xml are unnamed, so look up by body.
        # "objA" is a child body of "obj"; return its first geom.
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "objA")
        return self.model.body_geomadr[body_id]

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.get_body_com("obj")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("obj").xquat

    def _sample_non_overlapping_bins(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Sample two bin XY positions that do not overlap.

        Uses rejection sampling: draw from _random_reset_space until
        the two bin centres are far enough apart.
        """
        assert self._random_reset_space is not None
        low = self._random_reset_space.low
        high = self._random_reset_space.high
        rng = self.np_random if self.seeded_rand_vec else np.random
        for _ in range(1000):
            rand_vec = rng.uniform(low, high, size=low.size).astype(np.float64)
            bin1_xy = rand_vec[:2]
            bin2_xy = rand_vec[2:4]
            if np.linalg.norm(bin1_xy - bin2_xy) >= self.BIN_MIN_DIST:
                self._last_rand_vec = rand_vec
                return bin1_xy, bin2_xy
        # Fallback: place them far apart
        fallback = np.array([-0.15, 0.7, 0.15, 0.7])
        self._last_rand_vec = fallback
        return fallback[:2].copy(), fallback[2:4].copy()

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.obj_init_angle = self.init_config["obj_init_angle"]

        # Sample non-overlapping bin positions
        bin1_xy, bin2_xy = self._sample_non_overlapping_bins()

        # Move bin bodies (z stays at 0, on the table)
        self.model.body("bin_start").pos[:] = [bin1_xy[0], bin1_xy[1], 0.0]
        self.model.body("bin_goal").pos[:] = [bin2_xy[0], bin2_xy[1], 0.0]
        mujoco.mj_forward(self.model, self.data)

        # Place the object in the centre of bin_start (z = bin floor + cube half-size)
        obj_z = self.get_body_com("bin_start")[2] + 0.02
        self.obj_init_pos = np.array([bin1_xy[0], bin1_xy[1], obj_z])
        self._set_obj_xyz(self.obj_init_pos)

        # Cache initial gripper state (needed by _gripper_caging_reward)
        self.init_tcp = self.tcp_center
        self.init_left_pad = self.get_body_com("leftpad")
        self.init_right_pad = self.get_body_com("rightpad")

        # Goal is the centre of bin_goal
        self._target_pos = self.get_body_com("bin_goal").copy()
        self._target_to_obj_init = None

        self.objHeight = self.data.body("obj").xpos[2]
        self.heightTarget = self.objHeight + self.liftThresh

        self.maxPlacingDist = (
            np.linalg.norm(
                np.array([self.obj_init_pos[0], self.obj_init_pos[1]])
                - np.array(self._target_pos)[:-1]
            )
            + self.heightTarget
        )

        self.placeCompleted = False
        self.pickCompleted = False

        # Apply custom material colours
        if self._bin_1_rgba is not None:
            mat_id = self.model.mat("bin_red").id
            self.model.mat_rgba[mat_id] = self._bin_1_rgba
        if self._bin_2_rgba is not None:
            mat_id = self.model.mat("bin_blue").id
            self.model.mat_rgba[mat_id] = self._bin_2_rgba

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
        self, action: npt.NDArray[Any], obs: npt.NDArray[Any]
    ) -> tuple[float, bool, bool, float, float, float]:
        assert (
            self.obj_init_pos is not None and self._target_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        if self.reward_function_version == "v2":
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

            # Stage 1: Approach — move hand XY over the object
            tcp_to_obj_xy = float(np.linalg.norm(obj[:2] - tcp[:2]))
            xy_margin = float(np.linalg.norm(self.obj_init_pos[:2] - self.init_tcp[:2]))
            xy_alignment = reward_utils.tolerance(
                tcp_to_obj_xy,
                bounds=(0.0, 0.015),
                margin=max(xy_margin, 1e-6),
                sigmoid="long_tail",
            )

            # Stage 2: Approach — lower hand Z toward object
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

            # Stage 3: Grasp — caging reward
            object_grasped = self._gripper_caging_reward(action, obj)
            closing_reward = reward_utils.hamacher_product(
                xy_alignment, float(np.clip(action[-1], 0.0, 1.0))
            )

            # Stage 4: Lift — reward for lifting object above bin height
            lift_to_pick = float(max(self.heightTarget - obj[2], 0.0))
            lift_margin = max(self.heightTarget - self.obj_init_pos[2], 1e-6)
            lift_reward = reward_utils.tolerance(
                lift_to_pick,
                bounds=(0.0, 0.01),
                margin=lift_margin,
                sigmoid="long_tail",
            )

            # Stage 5: Transport — move lifted object toward target.
            # Use max(lift, in_place) so that lowering into the target bin
            # is not penalised: being at the goal is as good as being lifted.
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
            reward = min(reward, 9.95)

            if obj_to_target < _TARGET_RADIUS:
                reward = 10.0

            near_object = bool(tcp_to_obj <= 0.04)
            grasp_success = bool(
                self.touching_main_object
                and (tcp_opened > 0)
                and (obj[2] - 0.02 > self.obj_init_pos[2])
            )

            # Normalise to [-1, 1]
            reward = (reward - 5.0) / 5.0

            return (
                reward,
                near_object,
                grasp_success,
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
            placingGoal = self._target_pos

            reachDist = np.linalg.norm(objPos - fingerCOM)

            placingDist = np.linalg.norm(objPos[:2] - placingGoal[:-1])

            reachRew = -reachDist
            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_tcp[-1])
            if reachDistxy < 0.06:
                reachRew = -reachDist
            else:
                reachRew = -reachDistxy - zRew

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

            if (
                abs(objPos[0] - placingGoal[0]) < 0.05
                and abs(objPos[1] - placingGoal[1]) < 0.05
                and objPos[2] < self.objHeight + 0.05
            ):
                self.placeCompleted = True
            else:
                self.placeCompleted = False

            hScale = 100
            if self.placeCompleted or (self.pickCompleted and not objDropped):
                pickRew = hScale * heightTarget
            elif (reachDist < 0.1) and (objPos[2] > (self.objHeight + 0.005)):
                pickRew = hScale * min(heightTarget, objPos[2])
            else:
                pickRew = 0

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            placeRew = 1000 * (self.maxPlacingDist - placingDist) + c1 * (
                np.exp(-(placingDist**2) / c2) + np.exp(-(placingDist**2) / c3)
            )
            placeRew = max(placeRew, 0)
            cond = self.pickCompleted and (reachDist < 0.1) and not objDropped

            if self.placeCompleted:
                return (
                    float(-200 * action[-1] + placeRew),
                    False,
                    False,
                    float(placingDist),
                    0.0,
                    0.0,
                )
            elif cond:
                if (
                    abs(objPos[0] - placingGoal[0]) < 0.05
                    and abs(objPos[1] - placingGoal[1]) < 0.05
                ):
                    placeRew, placingDist = [-200 * action[-1] + placeRew, placingDist]
                else:
                    placeRew, placingDist = [placeRew, placingDist]
            else:
                placeRew, placingDist = [0, placingDist]

            if self.placeCompleted:
                reachRew = 0
                reachDist = 0
            reward = reachRew + pickRew + placeRew

            return float(reward), False, False, float(placingDist), 0.0, 0.0


class SawyerBinPickingRedBlueEnvV3(SawyerBinPickingEnvV3):
    def __init__(self, **kwargs):
        kwargs.setdefault("bin_1_rgba", (0.8, 0.0, 0.0, 1.0))
        kwargs.setdefault("bin_2_rgba", (0.0, 0.0, 0.8, 1.0))
        super().__init__(**kwargs)


class SawyerBinPickingYellowBlueEnvV3(SawyerBinPickingEnvV3):
    def __init__(self, **kwargs):
        kwargs.setdefault("bin_1_rgba", (0.8, 0.8, 0.0, 1.0))
        kwargs.setdefault("bin_2_rgba", (0.0, 0.0, 0.8, 1.0))
        super().__init__(**kwargs)


class SawyerBinPickingRedPurpleEnvV3(SawyerBinPickingEnvV3):
    def __init__(self, **kwargs):
        kwargs.setdefault("bin_1_rgba", (0.8, 0.0, 0.0, 1.0))
        kwargs.setdefault("bin_2_rgba", (0.8, 0.0, 0.8, 1.0))
        super().__init__(**kwargs)


class SawyerBinPickingYellowPurpleEnvV3(SawyerBinPickingEnvV3):
    def __init__(self, **kwargs):
        kwargs.setdefault("bin_1_rgba", (0.8, 0.8, 0.0, 1.0))
        kwargs.setdefault("bin_2_rgba", (0.8, 0.0, 0.8, 1.0))
        super().__init__(**kwargs)
