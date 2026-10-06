from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import CompoPickPlaceDrawerInitConfigDict
from metaworld.utils import reward_utils
import mujoco

class CompoDrawerOpenPickPlaceEnv(SawyerXYZEnv):
    """
    Sawyer Compositional DrawerOpen and PickPlaceEnvironment.
    """
    def __init__(
            self,
            render_mode: RenderMode | None = None,
            camera_name: str | None = None,
            camera_id: int | None = None,
            height: int = 480,
            width: int = 480,
            reward_function_version: str = "placeholder",
        ) -> None:
        # Control bound for hand
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)

        # Initialisation bound for hand and objects
        # We need to stagger the drawer and object init positions to avoid collisions
        drawer_low = (-0.4, 0.8, 0.0)
        drawer_high = (-0.2, 0.9, 0.0)
        obj_low = (0.05, 0.5, 0.02)
        obj_high = (0.25, 0.8, 0.02)

        # Task specific flag
        self.drawer_opened = False
        self.pickplace_completed = False
        
        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            height=height,
            width=width,
        )

        self.init_config: CompoPickPlaceDrawerInitConfigDict = {
            "hand_init_pos": np.array([0, 0.6, 0.2]),
            "drawer_init_pos": np.array([0.0, 0.9, 0.0]),
            "obj_init_pos": np.array([0, 0.6, 0.02]),
        }

        self.hand_init_pos = self.init_config["hand_init_pos"]
        self.drawer_init_pos = self.init_config["drawer_init_pos"]
        self.obj_init_pos = self.init_config["obj_init_pos"]
        
        self._random_reset_space = Box(
            np.hstack((drawer_low, obj_low)),
            np.hstack((drawer_high, obj_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(np.array(self.hand_low), np.array(self.hand_high), dtype=np.float64)

                
        self.maxDist = 0.2 # For drawer opening
        self._target_pos = np.zeros(3) 

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/compo_draweropen_pickplace.xml")

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()

        # Task specific flag
        self.drawer_opened = False
        self.pickplace_completed = False
        
        # Task specific reset
        rand_vec = self._get_state_rand_vec()
        # Drawer
        drawer_pos = rand_vec[:3]
        self.model.body("drawer").pos = drawer_pos
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "goal_slidey")
        qpos_adr = self.model.jnt_qposadr[joint_id]
        self.data.qpos[qpos_adr] = 0.0
        self.drawer_init_pos = drawer_pos
        # Block
        block_pos = rand_vec[3:]
        # Get free joint id
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj"
        )
        qpos_adr = self.model.jnt_qposadr[joint_id]
        # Set position
        self.data.qpos[qpos_adr : qpos_adr + 3] = block_pos
        self.obj_init_pos = block_pos
        # Apply changes
        mujoco.mj_forward(self.model, self.data)

        # Initial tcp and pad positions
        self.init_tcp = self.tcp_center
        self.init_left_pad = self.get_body_com("leftpad")
        self.init_right_pad = self.get_body_com("rightpad")
                
        # Target pos at reset is the drawer open target
        self._target_pos = drawer_pos + np.array([0.0, -0.16 - self.maxDist, 0.09])
        
        # Update visualization sites
        self.model.site("goal").pos = self._target_pos
        
        return self._get_obs()
    
    def _get_pos_objects(self) -> npt.NDArray[Any]:
        """
        Switches the 'Object' observation based on the active task.
        Phase 1: Returns Drawer Handle position.
        Phase 2: Returns Red Block position.
        """
        if not self.drawer_opened:
            # Return Handle Position (similar to DrawerOpenEnv)
            return self.get_body_com("drawer_link") + np.array([0.0, -0.16, 0.0])
        else:
            # Return Block Position (similar to PickPlaceEnv)
            return self.get_body_com("obj")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        if not self.drawer_opened:
            return self.data.body("drawer_link").xquat
        else:
            return Rotation.from_matrix(
                self.data.geom("objGeom").xmat.reshape(3, 3)
            ).as_quat()

    def _get_drawer_handle_pos(self):
        return self.get_body_com("drawer_link") + np.array([0.0, -0.16, 0.0])
    
    def _gripper_caging_reward(
        self,
        action: npt.NDArray[np.float32],
        obj_pos: npt.NDArray[Any],
    ) -> float:
        """
        Cleaned up version of SawyerPickPlaceEnvV3's caging reward.
        """
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
    ) -> tuple[float, dict[str, Any]]:
        assert self._target_pos is not None and self.obj_init_pos is not None and self.drawer_init_pos is not None
        
        gripper = obs[:3]
        tcp_opened = obs[3]
        obj_pos = obs[4:7] # This is Handle in Phase 1, Block in Phase 2
        
        # ---------------------------------------------------------
        # PHASE 1: OPEN DRAWER
        # ---------------------------------------------------------
        if not self.drawer_opened:
            handle_error = float(np.linalg.norm(obj_pos - self._target_pos))
            
            # Logic from SawyerDrawerOpenEnvV3
            reward_for_opening = reward_utils.tolerance(
                handle_error, bounds=(0, 0.02), margin=self.maxDist, sigmoid="long_tail"
            )
            
            handle_pos_init = self._target_pos + np.array([0.0, self.maxDist, 0.0])
            scale = np.array([3.0, 3.0, 1.0])
            gripper_error = (obj_pos - gripper) * scale
            gripper_error_init = (handle_pos_init - self.init_tcp) * scale
            
            reward_for_caging = reward_utils.tolerance(
                float(np.linalg.norm(gripper_error)),
                bounds=(0, 0.01),
                margin=np.linalg.norm(gripper_error_init),
                sigmoid="long_tail",
            )
            
            reward = reward_for_caging + reward_for_opening
            reward *= 5.0
            
            if handle_error <= 0.03:
                self.drawer_opened = True

            # Whole task reward has a range of [0, 20], normalise to [-1, 1]
            reward = (reward - 10.0) / 10.0
                
            return (
                reward, 
                float(np.linalg.norm(obj_pos - gripper)), 
                tcp_opened,
                handle_error,
                reward_for_caging,
                reward_for_opening,
            )

        # ---------------------------------------------------------
        # PHASE 2: PICK AND PLACE
        # ---------------------------------------------------------
        else:
            # Penalty that ensures the drawer stays open
            drawer_handle_pos = self._get_drawer_handle_pos()
            drawer_open_error = float(np.linalg.norm(drawer_handle_pos - (self.drawer_init_pos + np.array([0.0, -0.16 - self.maxDist, 0.09]))))
            penalty_for_opening = reward_utils.tolerance(
                drawer_open_error, bounds=(0, 0.02), margin=self.maxDist, sigmoid="long_tail"
            ) - 1.0

            current_handle_pos = self._get_drawer_handle_pos()
            
            # Offset: 0.15 back, -0.02 down
            place_offset = np.array([0.0, 0.15, -0.02]) 
            real_target_pos = current_handle_pos + place_offset
            
            # Update self._target_pos to place target for object
            self._target_pos = real_target_pos
            
            # Update visualization sites
            self.model.site("goal").pos = self._target_pos
            # Make visualisation site white
            self.model.site("goal").rgba = np.array([1.0, 1.0, 1.0, 1.0])
            
            # Logic from SawyerPickPlaceEnvV3
            _TARGET_RADIUS: float = 0.05
            target = self._target_pos
            
            obj_to_target = float(np.linalg.norm(obj_pos - target))
            tcp_to_obj = float(np.linalg.norm(obj_pos - gripper))
            in_place_margin = np.linalg.norm(self.obj_init_pos - target)

            in_place = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=in_place_margin,
                sigmoid="long_tail",
            )

            approach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, 0.04),
                margin=np.linalg.norm(self.obj_init_pos - self.init_tcp),
                sigmoid="long_tail",
            )

            object_grasped = self._gripper_caging_reward(action, obj_pos)

            """Original metaworld code for reference"""
            in_place_and_object_grasped = reward_utils.hamacher_product(
                object_grasped, in_place
            )
            # reward = in_place_and_object_grasped

            # if (
            #     tcp_to_obj < 0.02
            #     and (tcp_opened > 0)
            #     and (obj_pos[2] - 0.01 > self.obj_init_pos[2])
            # ):
            #     reward += 1.0 + 5.0 * in_place
            # if obj_to_target < _TARGET_RADIUS:
            #     reward = 10.0

            reward = (
                0.25 * penalty_for_opening
                + 1.25 * approach
                + 2.5 * object_grasped
                + 6.0 * in_place_and_object_grasped
            )
            if obj_to_target < _TARGET_RADIUS:
                reward = 10.0

            # Phase-2 baseline bonus to preserve stage-1 progress when opening the drawer.
            # Dropped from +11 to +10 so that "idle gripper still camped on the handle"
            # no longer produces a *positive* per-step reward (previously ~+0.1 after
            # normalisation). With +10 the idle-at-handle return is ~0 while a successful
            # place is ~+1, removing the local attractor that was causing the policy
            # to camp on the handle instead of releasing to grab the block.
            reward += 10.0

            # Whole task reward has a range of [0, 20], normalise to [-1, 1]
            reward = (reward - 10.0) / 10.0

            # Mark pickplace as completed
            if obj_to_target < _TARGET_RADIUS:
                self.pickplace_completed = True

            return (
                reward,
                tcp_to_obj,
                tcp_opened,
                obj_to_target,
                object_grasped,
                in_place,
            )

    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            tcp_to_obj,
            tcp_opened,
            dist_to_target,
            grasp_reward,
            in_place_reward,
        ) = self.compute_reward(action, obs)

        # Final Success if both tasks are done
        final_success = 0.0
        if self.drawer_opened and self.pickplace_completed:
            final_success = 1.0

        info = {
            "success": final_success,
            "drawer_opened": float(self.drawer_opened),
            "pickplace_completed": float(self.pickplace_completed),
            "near_object": float(tcp_to_obj <= 0.03),
            "obj_to_target": dist_to_target,
            "unscaled_reward": reward,
        }

        return reward, info