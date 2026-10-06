from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import CompoPickPlaceInitConfigDict
from metaworld.utils import reward_utils
import mujoco

class CompoPickPlaceBlockEnv(SawyerXYZEnv):
    """
    Sawyer Compositional Pick and Place Environment.
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

        # Initialisation bound for objects
        obj1_low = (0.0, 0.5, 0.02)
        obj1_high = (0.25, 0.8, 0.02)
        obj2_low = (0.0, 0.5, 0.02)
        obj2_high = (0.25, 0.8, 0.02)

        # Initialisation bound for goal (z fixed at table height for stacking)
        goal_low = (-0.25, 0.4, 0.02)
        goal_high = (0.0, 0.7, 0.02)

        # Task specific flag
        self.pickplace1_done = False
        self.pickplace2_done = False
        self.obj1_pos = None
        self.obj2_target = None
        
        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            height=height,
            width=width,
        )

        self.init_config: CompoPickPlaceInitConfigDict = {
            "hand_init_pos": np.array([0, 0.6, 0.2]),
            "obj1_init_pos": np.array([0, 0.6, 0.02]),
            "obj2_init_pos": np.array([0, 0.6, 0.02]),
        }

        self.hand_init_pos = self.init_config["hand_init_pos"]
        self.obj1_init_pos = self.init_config["obj1_init_pos"]
        self.obj2_init_pos = self.init_config["obj2_init_pos"]
        
        self._random_reset_space = Box(
            np.hstack((obj1_low, obj2_low, goal_low)),
            np.hstack((obj1_high, obj2_high, goal_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

        self._target_pos = np.zeros(3)

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/compo_pickplace_block.xml")

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()

        # Task specific flag
        self.pickplace1_done = False
        self.pickplace2_done = False
        self.obj1_pos = None
        self.obj2_target = None
        
        # Task specific reset
        rand_vec = self._get_state_rand_vec()

        # See if block 1 and 2 are too close, if so, re-sample
        while np.linalg.norm(rand_vec[:2] - rand_vec[3:5]) < 0.15:
            rand_vec = self._get_state_rand_vec()

        # Block 1
        block1_pos = rand_vec[:3]
        # Get free joint id
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "objjoint1"
        )
        qpos_adr = self.model.jnt_qposadr[joint_id]
        # Set position
        self.data.qpos[qpos_adr : qpos_adr + 3] = block1_pos
        self.obj1_init_pos = block1_pos

        # Block 2
        block2_pos = rand_vec[3:6]
        # Get free joint id
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "objjoint2"
        )
        qpos_adr = self.model.jnt_qposadr[joint_id]
        # Set position
        self.data.qpos[qpos_adr : qpos_adr + 3] = block2_pos
        self.obj2_init_pos = block2_pos

        # Initial tcp and pad positions
        self.init_tcp = self.tcp_center
        self.init_left_pad = self.get_body_com("leftpad")
        self.init_right_pad = self.get_body_com("rightpad")

        # Set target position        
        goal_pos = rand_vec[6:9]
        # # Ensure target and objects are not initialized too close to each other
        # while np.linalg.norm(goal_pos[:2] - self.obj1_init_pos[:2]) < 0.15:
        #     goal_pos = self._get_state_rand_vec()[6:9]
        self._target_pos = goal_pos
        
        # Update visualization sites
        self.model.site("goal").pos = self._target_pos

        # Apply changes
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs()
    
    def _get_pos_objects(self) -> npt.NDArray[Any]:
        """
        Switches the 'Object' observation based on the active task.
        Phase 1: Returns obj1 position.
        Phase 2: Returns obj2 position.
        """
        if not self.pickplace1_done:
            # Return obj1 position
            return self.get_body_com("obj1")
        else:
            # Return obj2 position
            return self.get_body_com("obj2")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        if not self.pickplace1_done:
            return self.data.body("obj1").xquat
        else:
            return self.data.body("obj2").xquat
    
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
        assert self.obj1_init_pos is not None
        init_obj_x_z = self.obj1_init_pos + np.array([0.0, -self.obj1_init_pos[1], 0.0])
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
        assert self._target_pos is not None and self.obj1_init_pos is not None and self.obj2_init_pos is not None
        
        gripper = obs[:3]
        tcp_opened = obs[3]
        obj_pos = obs[4:7] # This is obj1 in phase 1, obj2 in phase 2
        
        # ---------------------------------------------------------
        # PHASE 1: PICK AND PLACE OBJ1
        # ---------------------------------------------------------
        if not self.pickplace1_done:
            # Logic from SawyerPickPlaceEnvV3
            _TARGET_RADIUS: float = 0.03
            target = self._target_pos
            
            obj_to_target = float(np.linalg.norm(obj_pos - target))
            tcp_to_obj = float(np.linalg.norm(obj_pos - gripper))
            in_place_margin = np.linalg.norm(self.obj1_init_pos - target)

            in_place = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=in_place_margin,
                sigmoid="long_tail",
            )

            approach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, 0.04),
                margin=np.linalg.norm(self.obj1_init_pos - self.init_tcp),
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
            #     and (obj_pos[2] - 0.01 > self.obj1_init_pos[2])
            # ):
            #     reward += 1.0 + 5.0 * in_place
            # if obj_to_target < _TARGET_RADIUS:
            #     reward = 10.0

            # Phase 1 reward in [0, 5].
            #
            # Shape goal: create a smooth monotone path from
            #   far -> approach -> grasp -> carry -> place -> RELEASE.
            # The key change vs. the old formulation is the explicit release
            # term (`in_place * tcp_release`), which gives a positive gradient
            # toward *opening* the gripper once the block is at the target.
            # Without this, the old reward only rewarded holding the block at
            # the target (via in_place_and_grasped), so the policy had no
            # incentive to actually let go — which is exactly the failure
            # mode that was leaking into Phase 2.
            #
            # Component budget (max values shown):
            #   0.25 * approach                             -> 0.25
            #   1.00 * grasped * (1 - in_place)             -> 1.00  (only while carrying)
            #   2.00 * in_place                             -> 2.00  (doesn't require grasped)
            #   1.75 * in_place * tcp_release               -> 1.75  (release bonus at target)
            # Sum theoretical max ~5.00; snaps to exactly 5.0 on terminal.
            tcp_release = float(np.clip(tcp_opened, 0.0, 1.0))

            reward = (
                0.25 * approach
                + 1.0 * object_grasped * (1.0 - in_place)
                + 2.0 * in_place
                + 1.75 * in_place * tcp_release
            )
            # Terminal snap: block placed AND gripper released -> hard cap at 5.0
            if obj_to_target < _TARGET_RADIUS and tcp_opened > 0.8:
                reward = 5.0

            # Keep track of obj1_pos — this is FROZEN at phase-1 end and used
            # in Phase 2 both as the penalty reference and the stacking target.
            self.obj1_pos = self.get_body_com("obj1").copy()

            # Task end condition: obj1 placed near target and gripper opened
            self.pickplace1_done = obj_to_target < 0.05 and tcp_opened > 0.8

            # Whole task reward has a range of [0, 10], normalise to [-1, 1]
            reward = (reward - 5.0) / 5.0

            return (
                reward,
                tcp_to_obj,
                tcp_opened,
                obj_to_target,
                object_grasped,
                in_place,
            )     

        # ---------------------------------------------------------
        # PHASE 2: PICK AND PLACE OBJ2 ONTO OBJ1
        # ---------------------------------------------------------
        else:
            # Penalty that ensures obj1 stays where it was placed.
            # Uses self.obj1_pos (FROZEN at phase-1 end), so moving obj1 after
            # phase 1 is strictly punished regardless of where it ends up.
            current_obj1_pos = self.get_body_com("obj1")
            obj1_error = float(np.linalg.norm(current_obj1_pos - self.obj1_pos))
            penalty_for_obj1 = reward_utils.tolerance(
                obj1_error, bounds=(0, 0.03), margin=0.2, sigmoid="long_tail",
            ) - 1.0  # in [-1, 0]

            # Stacking target tracks the CURRENT obj1 position so that a
            # physically valid stack is always reachable — if obj1 has
            # settled/rolled slightly, the target follows it. The
            # "don't-drag-obj1" behaviour is enforced by penalty_for_obj1
            # above, which is computed against the FROZEN self.obj1_pos,
            # so the policy still gets punished for deliberately moving
            # obj1 after phase 1. Decoupling the two lets us keep the
            # anti-cheat pressure without making stacking impossible when
            # obj1 nudges a little under contact.
            obj2_target_pos = current_obj1_pos + np.array([0.0, 0.0, 0.03])
            
            # Update self._target_pos to place target for object
            self._target_pos = obj2_target_pos
            target = self._target_pos
            
            # Update visualization sites
            self.model.site("goal").pos = self._target_pos
            
            # Logic from SawyerPickPlaceEnvV3
            _TARGET_RADIUS: float = 0.03
            
            obj_to_target = float(np.linalg.norm(obj_pos - target))
            tcp_to_obj = float(np.linalg.norm(obj_pos - gripper))
            in_place_margin = np.linalg.norm(self.obj1_init_pos - target)

            in_place = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=in_place_margin,
                sigmoid="long_tail",
            )

            approach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, 0.04),
                margin=np.linalg.norm(self.obj1_init_pos - self.init_tcp),
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
            #     and (obj_pos[2] - 0.01 > self.obj1_init_pos[2])
            # ):
            #     reward += 1.0 + 5.0 * in_place
            # if obj_to_target < _TARGET_RADIUS:
            #     reward = 10.0

            # Phase 2 reward in roughly [5, 10].
            #
            # Structure: 5.0 baseline (so finishing Phase 1 is strictly
            # better than any Phase-1 state) + progress on obj2 (same shape
            # as Phase 1, budget 5) + penalty for moving obj1 (in [-1, 0]).
            # Worst case (obj1 fully dislodged, no obj2 progress) ~= 4.0,
            # success snaps to exactly 10.0. This keeps the overall env
            # reward within [~0, 10] with a clean monotone across phases:
            #   far -> approach obj2 -> grasp -> carry -> stack -> release.
            #
            # The penalty_for_obj1 term is what prevents the "grab red,
            # carry it next to green, drop target on top" shortcut. Combined
            # with the fixed obj2_target_pos above, the only high-return
            # policy is: leave obj1 alone, go pick up obj2, stack, release.
            tcp_release = float(np.clip(tcp_opened, 0.0, 1.0))

            progress = (
                1.0 * approach
                + 1.0 * object_grasped * (1.0 - in_place)
                + 2.0 * in_place
                + 1.0 * in_place * tcp_release
            )
            reward = 5.0 + progress + 1.0 * penalty_for_obj1
            # Terminal snap: obj2 stacked AND gripper released -> hard cap at 10.0
            if obj_to_target < _TARGET_RADIUS and tcp_opened > 0.8:
                reward = 10.0

            # Task end condition: obj2 stacked on target and gripper opened.
            # Added the tcp_opened requirement so "stacking" actually means
            # the policy committed to a release, matching Phase 1's style.
            self.pickplace2_done = obj_to_target < _TARGET_RADIUS and tcp_opened > 0.8


            # Whole task reward has a range of [0, 10], normalise to [-1, 1]
            reward = (reward - 5.0) / 5.0


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
        if self.pickplace1_done and self.pickplace2_done:
            final_success = 1.0

        info = {
            "success": final_success,
            "pickplace1_done": float(self.pickplace1_done),
            "pickplace2_done": float(self.pickplace2_done),
            "near_object": float(tcp_to_obj <= 0.03),
            "obj_to_target": dist_to_target,
            "unscaled_reward": reward,
        }

        return reward, info