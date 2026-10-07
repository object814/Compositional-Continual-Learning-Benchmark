"""The benchmark tasks as batched MuJoCo-Warp environments.

Each class mirrors one CPU environment in `metaworld/envs/` term for term:
the same layout sampling, object placement, observed objects, reward and
success test. Comments name the CPU method each part comes from.
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch

from . import reward_utils as ru
from .batched_env import BatchedSawyerEnv


def _norm(x):
    return x.norm(dim=-1)


def _full(like, value):
    return torch.full_like(like, float(value))


# =============================================================================
#  Reach
# =============================================================================

class _ReachPlaneEnv(BatchedSawyerEnv):
    """Base of `reach-xy/xz/yz/xyz-v3`: table, arm and goal only.

    The hand starts at a new random position every episode, the goal is one
    of the 50 MT1 layouts (the object part of the vector is unused), and the
    2-D variants lock one action dimension to 0.
    """

    GOAL_MARKER = "always"
    OBJ_LOW = (0.0, 0.5, 0.02)
    OBJ_HIGH = (0.25, 0.8, 0.02)
    FIXED_AXIS: int | None = None      # coordinate kept at its init value (hand start and goal)

    def _objects(self):
        z = torch.zeros(self.num_envs, 4, device=self.device)
        return z[:, :3], z

    def _task_reset(self, idx, vec):
        target = vec[:, 3:].clone()
        if self.FIXED_AXIS is not None:
            target[:, self.FIXED_AXIS] = self.HAND_INIT[self.FIXED_AXIS]
        self.target[idx] = target
        self.obj_init[idx] = vec[:, :3]

    def _evaluate(self, obs45, action):
        tcp_to_target = _norm(self.tcp_center() - self.target)
        in_place_margin = _norm(self.hand_init - self.target)
        in_place = ru.tolerance(tcp_to_target, bounds=(0.0, 0.05), margin=in_place_margin)
        reward = (10.0 * in_place - 5.0) / 5.0
        success = tcp_to_target <= 0.05
        info = {"success": success.float(), "near_object": tcp_to_target, "grasp_reward": tcp_to_target,
                "in_place_reward": in_place, "obj_to_target": tcp_to_target, "unscaled_reward": reward}
        return reward, success, info


def _reach_plane(name, xml, fixed_axis, goal_low, goal_high):
    lo, hi = list(BatchedSawyerEnv.HAND_LOW), list(BatchedSawyerEnv.HAND_HIGH)
    lo[fixed_axis] = hi[fixed_axis] = BatchedSawyerEnv.HAND_INIT[fixed_axis]
    attrs = dict(
        __doc__=f"metaworld `{name}`: reach in a plane; action {'xyz'[fixed_axis]} locked.",
        XML=xml, FIXED_AXIS=fixed_axis, LOCKED_ACTION=fixed_axis,
        RANDOM_HAND_INIT=(tuple(lo), tuple(hi)), GOAL_LOW=goal_low, GOAL_HIGH=goal_high,
        TASK_VEC_LOW=_ReachPlaneEnv.OBJ_LOW + goal_low, TASK_VEC_HIGH=_ReachPlaneEnv.OBJ_HIGH + goal_high,
    )
    return type(name.replace("-", "_"), (_ReachPlaneEnv,), attrs)


ReachXYEnv = _reach_plane("reach-xy-v3", "sawyer_reach_xy_v3.xml", 2, (-0.25, 0.4, 0.2), (0.25, 0.8, 0.2))
ReachXZEnv = _reach_plane("reach-xz-v3", "sawyer_reach_xz_v3.xml", 1, (-0.25, 0.6, 0.02), (0.25, 0.6, 0.4))
ReachYZEnv = _reach_plane("reach-yz-v3", "sawyer_reach_yz_v3.xml", 0, (0.0, 0.4, 0.02), (0.0, 0.8, 0.4))


class ReachXYZEnv(_ReachPlaneEnv):
    """metaworld `reach-xyz-v3`: goal and hand start vary in all of XYZ."""

    XML = "sawyer_reach_xyz_v3.xml"
    _MARGIN = np.array([0.08, 0.05, 0.04])     # keeps hand starts off the workspace edges
    RANDOM_HAND_INIT = (tuple(np.array(BatchedSawyerEnv.HAND_LOW) + _MARGIN),
                        tuple(np.array(BatchedSawyerEnv.HAND_HIGH) - _MARGIN))
    GOAL_LOW = (-0.25, 0.4, 0.02)
    GOAL_HIGH = (0.25, 0.8, 0.4)
    TASK_VEC_LOW = _ReachPlaneEnv.OBJ_LOW + GOAL_LOW
    TASK_VEC_HIGH = _ReachPlaneEnv.OBJ_HIGH + GOAL_HIGH


# =============================================================================
#  Pick-place family (sawyer_pick_place_v3.xml): pick-place, reach, grasp
# =============================================================================

class PickPlaceEnv(BatchedSawyerEnv):
    """metaworld `pick-place-v3` (SawyerPickPlaceEnvV3, 'large' init region, reward v2)."""

    XML = "sawyer_pick_place_v3.xml"
    GOAL_MARKER = "always"             # white goal site, alpha 1
    GOAL_LOW = (-0.25, 0.4, 0.1)
    GOAL_HIGH = (0.0, 0.7, 0.3)
    TASK_VEC_LOW = (0.0, 0.5, 0.02, -0.25, 0.4, 0.1)      # (obj, goal)
    TASK_VEC_HIGH = (0.25, 0.8, 0.02, 0.0, 0.7, 0.3)
    TARGET_RADIUS = 0.05
    OBJ_RADIUS = 0.015
    LIFT = 0.04                        # heightTarget = object height + LIFT

    def _edit_model(self, mjm):
        # reset_model: model.geom_rgba[objGeom] = red
        mjm.geom_rgba[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "objGeom")] = [1.0, 0.0, 0.0, 1.0]

    def _lookup_task_ids(self, mjm):
        self.bid["obj"] = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "obj")
        self.obj_geom = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "objGeom")
        self.obj_qadr, self.obj_vadr = self._joint_adr(mjm, "objjoint")

    def _alloc_extra_buffers(self):
        self.height_target = torch.zeros(self.num_envs, device=self.device)

    def _valid_task_vec(self, v):
        # reset_model's while-loop: object and goal at least 0.15 apart in XY.
        return np.linalg.norm(v[:2] - v[3:5]) >= 0.15

    def _objects(self):
        return self.body_xpos("obj"), self.geom_quat_xyzw(self.obj_geom)

    def _task_reset(self, idx, vec):
        obj, goal = vec[:, :3], vec[:, 3:]
        self._t["qpos"][idx, self.obj_qadr:self.obj_qadr + 3] = obj     # _set_obj_xyz
        self._t["qvel"][idx, self.obj_vadr:self.obj_vadr + 6] = 0.0
        self.obj_init[idx] = obj
        self.target[idx] = goal

    def _task_post_forward(self, idx):
        self.height_target[idx] = self.geom_xpos(self.obj_geom)[idx, 2] + self.LIFT

    def _evaluate(self, obs45, action):
        tcp = self.tcp_center()
        obj = obs45[:, 4:7]
        target = self.target
        obj_to_target = _norm(obj - target)
        tcp_to_obj = _norm(obj - tcp)
        in_place_margin = _norm(self.obj_init - target)
        in_place = ru.tolerance(obj_to_target, bounds=(0.0, self.TARGET_RADIUS), margin=in_place_margin)

        tcp_to_obj_xy = _norm(obj[:, :2] - tcp[:, :2])
        xy_margin = _norm(self.obj_init[:, :2] - self.init_tcp[:, :2]).clamp_min(1e-6)
        xy_alignment = ru.tolerance(tcp_to_obj_xy, bounds=(0.0, 0.015), margin=xy_margin)
        desired_tcp_z = obj[:, 2] + self.OBJ_RADIUS
        tcp_to_hover_z = (tcp[:, 2] - desired_tcp_z).abs()
        z_margin = (self.init_tcp[:, 2] - desired_tcp_z).abs().clamp_min(1e-6)
        z_alignment = ru.tolerance(tcp_to_hover_z, bounds=(0.0, 0.02), margin=z_margin)
        approach = ru.hamacher_product(xy_alignment, z_alignment)

        object_grasped = self._caging(action, obj, self.obj_init)
        closing = ru.hamacher_product(xy_alignment, action[:, 3].clamp(0.0, 1.0))
        lift_to_pick = (self.height_target - obj[:, 2]).clamp_min(0.0)
        lift_margin = (self.height_target - self.obj_init[:, 2]).clamp_min(1e-6)
        lift = ru.tolerance(lift_to_pick, bounds=(0.0, 0.01), margin=lift_margin)
        grasp_and_lift = ru.hamacher_product(object_grasped, lift)
        transport = ru.hamacher_product(lift, in_place)

        reward = (2.0 * xy_alignment + 1.0 * approach + 1.0 * closing
                  + 1.5 * object_grasped + 1.5 * grasp_and_lift + 3.0 * transport)
        reward = reward.clamp(max=9.95)
        reward = torch.where(obj_to_target < self.TARGET_RADIUS, _full(reward, 10.0), reward)
        reward = (reward - 5.0) / 5.0
        success = obj_to_target <= 0.07
        info = {"success": success.float(), "near_object": (tcp_to_obj <= 0.03).float(),
                "grasp_reward": object_grasped, "in_place_reward": in_place,
                "obj_to_target": obj_to_target, "unscaled_reward": reward}
        return reward, success, info


class ReachEnv(PickPlaceEnv):
    """metaworld `reach-v3`: move the gripper to the goal; the object is a distractor."""

    def _evaluate(self, obs45, action):
        tcp_to_target = _norm(self.tcp_center() - self.target)
        in_place_margin = _norm(self.hand_init - self.target)
        in_place = ru.tolerance(tcp_to_target, bounds=(0.0, 0.05), margin=in_place_margin)
        reward = (10.0 * in_place - 5.0) / 5.0
        success = tcp_to_target <= 0.05
        info = {"success": success.float(), "near_object": tcp_to_target, "grasp_reward": tcp_to_target,
                "in_place_reward": in_place, "obj_to_target": tcp_to_target, "unscaled_reward": reward}
        return reward, success, info


class GraspEnv(PickPlaceEnv):
    """metaworld `grasp-v3`: grasp the object securely and lift it 2 cm."""

    LIFT = 0.08
    SUCCESS_LIFT_HEIGHT = 0.02
    SUCCESS_GRASP_REWARD = 0.85
    SUCCESS_TCP_TO_OBJ = 0.03
    SUCCESS_GRIPPER_OPEN = 0.45

    def _evaluate(self, obs45, action):
        tcp = self.tcp_center()
        obj = obs45[:, 4:7]
        tcp_opened = obs45[:, 3]
        tcp_to_obj = _norm(obj - tcp)
        tcp_to_obj_xy = _norm(obj[:, :2] - tcp[:, :2])
        xy_margin = _norm(self.obj_init[:, :2] - self.init_tcp[:, :2]).clamp_min(1e-6)
        xy_alignment = ru.tolerance(tcp_to_obj_xy, bounds=(0.0, 0.015), margin=xy_margin)
        desired_tcp_z = obj[:, 2] + self.OBJ_RADIUS
        tcp_to_hover_z = (tcp[:, 2] - desired_tcp_z).abs()
        z_margin = (self.init_tcp[:, 2] - desired_tcp_z).abs().clamp_min(1e-6)
        z_alignment = ru.tolerance(tcp_to_hover_z, bounds=(0.0, 0.02), margin=z_margin)
        approach = ru.hamacher_product(xy_alignment, z_alignment)

        grasp_reward = self._caging(action, obj, self.obj_init)
        lift_height = (obj[:, 2] - self.obj_init[:, 2]).clamp_min(0.0)
        lift_to_target = (self.SUCCESS_LIFT_HEIGHT - lift_height).clamp_min(0.0)
        lift_reward = ru.tolerance(lift_to_target, bounds=(0.0, 0.003), margin=self.SUCCESS_LIFT_HEIGHT)
        grasp_and_lift = ru.hamacher_product(grasp_reward, lift_reward)
        closing = ru.hamacher_product(xy_alignment, action[:, 3].clamp(0.0, 1.0))
        reward = (2.0 * xy_alignment + 1.5 * approach + 1.5 * closing
                  + 2.0 * grasp_reward + 3.0 * grasp_and_lift)
        reward = reward.clamp(max=9.95)

        # _has_secure_grasp
        secure = (self._touching([self.obj_geom])
                  & (tcp_opened <= self.SUCCESS_GRIPPER_OPEN)
                  & (tcp_to_obj <= self.SUCCESS_TCP_TO_OBJ)
                  & (grasp_reward >= self.SUCCESS_GRASP_REWARD))
        lifted = obj[:, 2] >= self.obj_init[:, 2] + self.SUCCESS_LIFT_HEIGHT
        reward = torch.where(lifted, torch.maximum(reward, 9.0 + lift_reward), reward)
        reward = torch.where(secure, _full(reward, 10.0), reward)
        reward = (reward - 5.0) / 5.0
        info = {"success": secure.float(), "near_object": (tcp_to_obj <= 0.03).float(),
                "grasp_success": secure.float(), "grasp_reward": grasp_reward,
                "in_place_reward": lift_reward, "obj_to_target": lift_to_target, "unscaled_reward": reward}
        return reward, secure, info


# =============================================================================
#  Pick-place with a block (sawyer_pick_place_block_v3.xml)
# =============================================================================

class _PickPlaceBlockEnv(PickPlaceEnv):
    """metaworld `pick-place-<colour>block-v3`: place the block on the goal and release it."""

    XML = "sawyer_pick_place_block_v3.xml"
    GOAL_LOW = (-0.25, 0.4, 0.02)
    GOAL_HIGH = (0.0, 0.7, 0.02)
    TASK_VEC_LOW = (0.0, 0.5, 0.02, -0.25, 0.4, 0.02)
    TASK_VEC_HIGH = (0.25, 0.8, 0.02, 0.0, 0.7, 0.02)
    OBJ_RGBA = (1.0, 0.0, 0.0, 1.0)

    def _edit_model(self, mjm):
        # reset_model: all geoms of body objA take the block colour
        body = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "objA")
        start, count = mjm.body_geomadr[body], mjm.body_geomnum[body]
        mjm.geom_rgba[start:start + count] = self.OBJ_RGBA

    def _objects(self):
        return self.body_xpos("obj"), self.body_xquat("obj")

    def _evaluate(self, obs45, action):
        tcp = self.tcp_center()
        obj = obs45[:, 4:7]
        tcp_opened = obs45[:, 3]
        target = self.target
        obj_to_target = _norm(obj - target)
        tcp_to_obj = _norm(obj - tcp)
        in_place_margin = _norm(self.obj_init - target)
        in_place = ru.tolerance(obj_to_target, bounds=(0.0, self.TARGET_RADIUS), margin=in_place_margin)

        tcp_to_obj_xy = _norm(obj[:, :2] - tcp[:, :2])
        xy_margin = _norm(self.obj_init[:, :2] - self.init_tcp[:, :2]).clamp_min(1e-6)
        xy_alignment = ru.tolerance(tcp_to_obj_xy, bounds=(0.0, 0.015), margin=xy_margin)
        desired_tcp_z = obj[:, 2] + self.OBJ_RADIUS
        tcp_to_hover_z = (tcp[:, 2] - desired_tcp_z).abs()
        z_margin = (self.init_tcp[:, 2] - desired_tcp_z).abs().clamp_min(1e-6)
        z_alignment = ru.tolerance(tcp_to_hover_z, bounds=(0.0, 0.02), margin=z_margin)
        approach = ru.hamacher_product(xy_alignment, z_alignment)

        object_grasped = self._caging(action, obj, self.obj_init)
        closing = ru.hamacher_product(xy_alignment, action[:, 3].clamp(0.0, 1.0))
        lift_to_pick = (self.height_target - obj[:, 2]).clamp_min(0.0)
        lift_margin = (self.height_target - self.obj_init[:, 2]).clamp_min(1e-6)
        lift = ru.tolerance(lift_to_pick, bounds=(0.0, 0.01), margin=lift_margin)
        grasp_and_lift = ru.hamacher_product(object_grasped, lift)
        transport = ru.hamacher_product(torch.maximum(lift, in_place), in_place)

        reward = (2.0 * xy_alignment + 1.0 * approach + 1.0 * closing
                  + 1.5 * object_grasped + 1.5 * grasp_and_lift + 3.0 * transport)
        reward = reward.clamp(max=9.5)
        reward = torch.where(obj_to_target < self.TARGET_RADIUS, _full(reward, 9.7), reward)
        reward = torch.where((obj_to_target < 0.01) & (tcp_opened > 0.8), _full(reward, 10.0), reward)
        reward = (reward - 5.0) / 5.0
        success = (obj_to_target <= 0.05) & (tcp_opened > 0.8)
        info = {"success": success.float(), "near_object": (tcp_to_obj <= 0.03).float(),
                "grasp_reward": object_grasped, "in_place_reward": in_place,
                "obj_to_target": obj_to_target, "unscaled_reward": reward}
        return reward, success, info


class PickPlaceRedBlockEnv(_PickPlaceBlockEnv):
    OBJ_RGBA = (1.0, 0.0, 0.0, 1.0)


class PickPlaceGreenBlockEnv(_PickPlaceBlockEnv):
    OBJ_RGBA = (0.0, 1.0, 0.0, 1.0)


# =============================================================================
#  Bin picking
# =============================================================================

class _BinPickingEnv(BatchedSawyerEnv):
    """metaworld `bin-picking-<colours>-v3`: move the cube from one bin to the other.

    Both bins are placed anew at every reset (`_sample_non_overlapping_bins`
    ignores the MT1 task vector); the cube starts in the centre of bin 1 and
    the goal is the centre of bin 2. The cube keeps the orientation it settled
    into during the hand settle (see `BatchedSawyerEnv._settle`).
    """

    XML = "sawyer_bin_picking.xml"
    HAND_LOW = (-0.5, 0.40, 0.07)
    FRESH_LAYOUT = True
    SETTLE_LAYOUT = True
    BIN_MIN_DIST = 0.25
    BIN_LOW = (-0.25, 0.45)
    BIN_HIGH = (0.25, 0.85)
    GOAL_LOW = (-0.25, 0.45, -0.001)
    GOAL_HIGH = (0.25, 0.85, 0.001)
    BIN_1_RGBA = (0.8, 0.0, 0.0, 1.0)
    BIN_2_RGBA = (0.0, 0.0, 0.8, 1.0)
    TARGET_RADIUS = 0.05
    OBJ_RADIUS = 0.015
    LIFT = 0.1                         # liftThresh

    def _edit_model(self, mjm):
        mjm.mat_rgba[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_MATERIAL, "bin_red")] = self.BIN_1_RGBA
        mjm.mat_rgba[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_MATERIAL, "bin_blue")] = self.BIN_2_RGBA

    def _lookup_task_ids(self, mjm):
        for name in ("obj", "bin_start", "bin_goal"):
            self.bid[name] = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, name)
            assert name == "obj" or mjm.body_parentid[self.bid[name]] == 0, name
        self.obj_qadr, self.obj_vadr = 9, 9          # _set_obj_xyz: qpos[9:12], qvel[9:15]

    def _alloc_extra_buffers(self):
        self.height_target = torch.zeros(self.num_envs, device=self.device)

    def _sample_fresh_layout(self):
        # _sample_non_overlapping_bins: (bin1_xy, bin2_xy)
        lo, hi = np.array(self.BIN_LOW * 2), np.array(self.BIN_HIGH * 2)
        for _ in range(1000):
            v = self._rng.uniform(lo, hi)
            if np.linalg.norm(v[:2] - v[2:]) >= self.BIN_MIN_DIST:
                return v
        return np.array([-0.15, 0.7, 0.15, 0.7])

    def _objects(self):
        return self.body_xpos("obj"), self.body_xquat("obj")

    def _place_layout_bodies(self, idx, vec):
        zeros = torch.zeros(len(idx), 1, device=self.device)
        self._t["body_pos"][idx, self.bid["bin_start"]] = torch.cat([vec[:, :2], zeros], dim=-1)
        self._t["body_pos"][idx, self.bid["bin_goal"]] = torch.cat([vec[:, 2:4], zeros], dim=-1)

    def _task_reset(self, idx, vec):
        self._place_layout_bodies(idx, vec)
        zeros = torch.zeros(len(idx), 1, device=self.device)
        bin1 = torch.cat([vec[:, :2], zeros], dim=-1)
        bin2 = torch.cat([vec[:, 2:4], zeros], dim=-1)
        obj = bin1 + torch.tensor([0.0, 0.0, 0.02], device=self.device)
        self._t["qpos"][idx, self.obj_qadr:self.obj_qadr + 3] = obj
        self._t["qvel"][idx, self.obj_vadr:self.obj_vadr + 6] = 0.0
        self.obj_init[idx] = obj
        self.target[idx] = bin2

    def _task_post_forward(self, idx):
        self.height_target[idx] = self.body_xpos("obj")[idx, 2] + self.LIFT

    def _evaluate(self, obs45, action):
        tcp = self.tcp_center()
        obj = obs45[:, 4:7]
        target = self.target
        obj_to_target = _norm(obj - target)
        tcp_to_obj = _norm(obj - tcp)
        in_place_margin = _norm(self.obj_init - target)
        in_place = ru.tolerance(obj_to_target, bounds=(0.0, self.TARGET_RADIUS), margin=in_place_margin)

        tcp_to_obj_xy = _norm(obj[:, :2] - tcp[:, :2])
        xy_margin = _norm(self.obj_init[:, :2] - self.init_tcp[:, :2]).clamp_min(1e-6)
        xy_alignment = ru.tolerance(tcp_to_obj_xy, bounds=(0.0, 0.015), margin=xy_margin)
        desired_tcp_z = obj[:, 2] + self.OBJ_RADIUS
        tcp_to_hover_z = (tcp[:, 2] - desired_tcp_z).abs()
        z_margin = (self.init_tcp[:, 2] - desired_tcp_z).abs().clamp_min(1e-6)
        z_alignment = ru.tolerance(tcp_to_hover_z, bounds=(0.0, 0.02), margin=z_margin)
        approach = ru.hamacher_product(xy_alignment, z_alignment)

        object_grasped = self._caging(action, obj, self.obj_init)
        closing = ru.hamacher_product(xy_alignment, action[:, 3].clamp(0.0, 1.0))
        lift_to_pick = (self.height_target - obj[:, 2]).clamp_min(0.0)
        lift_margin = (self.height_target - self.obj_init[:, 2]).clamp_min(1e-6)
        lift = ru.tolerance(lift_to_pick, bounds=(0.0, 0.01), margin=lift_margin)
        grasp_and_lift = ru.hamacher_product(object_grasped, lift)
        transport = ru.hamacher_product(torch.maximum(lift, in_place), in_place)

        reward = (2.0 * xy_alignment + 1.0 * approach + 1.0 * closing
                  + 1.5 * object_grasped + 1.5 * grasp_and_lift + 3.0 * transport)
        reward = reward.clamp(max=9.95)
        reward = torch.where(obj_to_target < self.TARGET_RADIUS, _full(reward, 10.0), reward)
        reward = (reward - 5.0) / 5.0
        success = obj_to_target <= 0.05
        info = {"success": success.float(), "near_object": (tcp_to_obj <= 0.04).float(),
                "grasp_reward": object_grasped, "in_place_reward": in_place,
                "obj_to_target": obj_to_target, "unscaled_reward": reward}
        return reward, success, info


class BinPickingRedBlueEnv(_BinPickingEnv):
    BIN_1_RGBA, BIN_2_RGBA = (0.8, 0.0, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0)


class BinPickingYellowBlueEnv(_BinPickingEnv):
    BIN_1_RGBA, BIN_2_RGBA = (0.8, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0)


class BinPickingRedPurpleEnv(_BinPickingEnv):
    BIN_1_RGBA, BIN_2_RGBA = (0.8, 0.0, 0.0, 1.0), (0.8, 0.0, 0.8, 1.0)


class BinPickingYellowPurpleEnv(_BinPickingEnv):
    BIN_1_RGBA, BIN_2_RGBA = (0.8, 0.8, 0.0, 1.0), (0.8, 0.0, 0.8, 1.0)


# =============================================================================
#  Box close
# =============================================================================

def _lid_quat_reward(quat_wxyz):
    """SawyerBoxCloseEnvV3._reward_quat: closeness to the upright lid [.707, 0, 0, .707]."""
    ideal = torch.tensor([0.707, 0.0, 0.0, 0.707], device=quat_wxyz.device)
    return (1.0 - _norm(quat_wxyz - ideal) / 0.2).clamp_min(0.0)


class BoxCloseEnv(BatchedSawyerEnv):
    """metaworld `box-close-v3`: put the lid on the box.

    Lid and box are placed anew at every reset. The CPU reset runs 5 more
    physics steps after the hand settle, with the box already moved and
    before the lid is placed; the lid keeps the orientation it settled into.
    """

    XML = "sawyer_box.xml"
    FRESH_LAYOUT = True
    SETTLE_LAYOUT = True
    EXTRA_SETTLE_STEPS = 5
    BOX_MIN_DIST = 0.35
    XY_LOW = (-0.25, 0.45)
    XY_HIGH = (0.25, 0.80)
    LID_INIT_Z = -0.02
    GOAL_Z = 0.133                     # goal_space z; the target itself is 0.06 lower
    GOAL_LOW = (-0.25, 0.45, 0.133)
    GOAL_HIGH = (0.25, 0.80, 0.133)
    LIFT = 0.12                        # liftThresh
    TARGET_RADIUS = 0.05
    LID_RADIUS = 0.015

    def _lookup_task_ids(self, mjm):
        for name in ("boxbody", "top_link"):
            self.bid[name] = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, name)
        assert mjm.body_parentid[self.bid["boxbody"]] == 0
        self.box_height = float(mjm.body_pos[self.bid["boxbody"], 2])   # get_body_com("boxbody")[2]
        self.handle_geom = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "BoxHandleGeom")
        self.obj_qadr, self.obj_vadr = 9, 9          # _set_obj_xyz: qpos[9:12], qvel[9:15]

    def _alloc_extra_buffers(self):
        self.height_target = torch.zeros(self.num_envs, device=self.device)

    def _sample_fresh_layout(self):
        # _sample_non_overlapping_box_lid: (lid_xy, box_xy)
        lo, hi = np.array(self.XY_LOW * 2), np.array(self.XY_HIGH * 2)
        for _ in range(1000):
            v = self._rng.uniform(lo, hi)
            if np.linalg.norm(v[:2] - v[2:]) >= self.BOX_MIN_DIST:
                return v
        return np.array([-0.15, 0.7, 0.15, 0.7])

    def _objects(self):
        return self.body_xpos("top_link"), self.body_xquat("top_link")

    def _place_layout_bodies(self, idx, vec):
        box = torch.cat([vec[:, 2:4], torch.full((len(idx), 1), self.box_height, device=self.device)], dim=-1)
        self._t["body_pos"][idx, self.bid["boxbody"]] = box

    def _task_reset(self, idx, vec):
        k = len(idx)
        lid = torch.cat([vec[:, :2], torch.full((k, 1), self.LID_INIT_Z, device=self.device)], dim=-1)
        self._place_layout_bodies(idx, vec)
        self._t["qpos"][idx, self.obj_qadr:self.obj_qadr + 3] = lid
        self._t["qvel"][idx, self.obj_vadr:self.obj_vadr + 6] = 0.0
        self.obj_init[idx] = lid
        self.target[idx] = torch.cat([vec[:, 2:4], torch.full((k, 1), self.GOAL_Z - 0.06, device=self.device)], dim=-1)

    def _task_post_forward(self, idx):
        self.height_target[idx] = self.geom_xpos(self.handle_geom)[idx, 2] + self.LIFT

    def _evaluate(self, obs45, action):
        tcp = self.tcp_center()
        lid_base = obs45[:, 4:7]
        up = torch.tensor([0.0, 0.0, 0.02], device=self.device)
        lid_handle = lid_base + up
        target = self.target
        reward_quat = _lid_quat_reward(obs45[:, 7:11])
        grab_effort = ((action[:, 3].clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)

        scale = torch.tensor([1.0, 1.0, 3.0], device=self.device)
        lid_to_target = _norm((lid_handle - target) * scale)
        init_lid_handle = self.obj_init + up
        in_place_margin = _norm((init_lid_handle - target) * scale).clamp_min(1e-6)
        in_place = ru.tolerance(lid_to_target, bounds=(0.0, self.TARGET_RADIUS), margin=in_place_margin)

        lid_to_target_xy = _norm(lid_handle[:, :2] - target[:, :2])
        target_xy_margin = _norm(init_lid_handle[:, :2] - target[:, :2]).clamp_min(1e-6)
        xy_transport_alignment = ru.tolerance(lid_to_target_xy, bounds=(0.0, 0.015), margin=target_xy_margin)

        tcp_to_lid_xy = _norm(lid_handle[:, :2] - tcp[:, :2])
        xy_margin = _norm(init_lid_handle[:, :2] - self.init_tcp[:, :2]).clamp_min(1e-6)
        xy_alignment = ru.tolerance(tcp_to_lid_xy, bounds=(0.0, 0.015), margin=xy_margin)
        desired_tcp_z = lid_handle[:, 2] + self.LID_RADIUS
        tcp_to_hover_z = (tcp[:, 2] - desired_tcp_z).abs()
        z_margin = (self.init_tcp[:, 2] - desired_tcp_z).abs().clamp_min(1e-6)
        z_alignment = ru.tolerance(tcp_to_hover_z, bounds=(0.0, 0.02), margin=z_margin)
        approach = ru.hamacher_product(xy_alignment, z_alignment)

        object_caging = self._caging(action, lid_base, self.obj_init)
        closing = ru.hamacher_product(xy_alignment, grab_effort)
        grasp_reward = ru.hamacher_product(object_caging, closing)

        lift_to_pick = (self.height_target - lid_handle[:, 2]).clamp_min(0.0)
        lift_margin = (self.height_target - init_lid_handle[:, 2]).clamp_min(1e-6)
        lift = ru.tolerance(lift_to_pick, bounds=(0.0, 0.01), margin=lift_margin)
        grasp_and_lift = ru.hamacher_product(grasp_reward, lift)
        lift_or_placed = torch.maximum(lift, in_place)
        transport = (0.5 + 0.5 * reward_quat) * ru.hamacher_product(lift_or_placed, in_place)
        xy_transport = ru.hamacher_product(lift_or_placed, xy_transport_alignment)

        reward = (2.0 * xy_alignment + 1.0 * approach + 1.0 * closing + 1.0 * grasp_reward
                  + 1.0 * grasp_and_lift + 2.0 * xy_transport + 2.0 * transport)
        reward = reward.clamp(max=9.95)
        success = (_norm(obs45[:, 4:6] - target[:, :2]) < 0.03) & ((obs45[:, 6] - target[:, 2]).abs() < 0.03)
        reward = torch.where(success, _full(reward, 10.0), reward)
        reward = (reward - 5.0) / 5.0
        info = {"success": success.float(), "near_object": approach, "grasp_reward": grasp_reward,
                "in_place_reward": in_place, "unscaled_reward": reward}
        return reward, success, info


# =============================================================================
#  Compositional tasks
# =============================================================================

class CompoDrawerOpenPickPlaceEnv(BatchedSawyerEnv):
    """metaworld `compo-draweropen-pickplace` (open the drawer, then place the block in it)."""

    XML = "compo_draweropen_pickplace.xml"
    GOAL_MARKER = "phase2"             # goal site alpha 0 until the drawer is open, then white
    TASK_VEC_LOW = (-0.4, 0.8, 0.0, 0.05, 0.5, 0.02)       # (drawer, block)
    TASK_VEC_HIGH = (-0.2, 0.9, 0.0, 0.25, 0.8, 0.02)
    MAX_DIST = 0.2
    TARGET_RADIUS = 0.05

    def _lookup_task_ids(self, mjm):
        name2id = lambda t, n: mujoco.mj_name2id(mjm, t, n)
        self.bid["drawer"] = name2id(mujoco.mjtObj.mjOBJ_BODY, "drawer")
        self.bid["drawer_link"] = name2id(mujoco.mjtObj.mjOBJ_BODY, "drawer_link")
        self.bid["obj"] = name2id(mujoco.mjtObj.mjOBJ_BODY, "obj")
        self.obj_geom = name2id(mujoco.mjtObj.mjOBJ_GEOM, "objGeom")
        self.obj_qadr, _ = self._joint_adr(mjm, "objjoint")
        self.slide_qadr, _ = self._joint_adr(mjm, "goal_slidey")

    def _alloc_extra_buffers(self):
        n = self.num_envs
        self.drawer_opened = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.pickplace_completed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.drawer_init = torch.zeros(n, 3, device=self.device)
        self.open_offset = torch.tensor([0.0, -0.16 - self.MAX_DIST, 0.09], device=self.device)

    def _handle(self):
        h = self.body_xpos("drawer_link").clone()
        h[:, 1] -= 0.16
        return h

    def _objects(self):
        p2 = self.drawer_opened[:, None]
        pos = torch.where(p2, self.body_xpos("obj"), self._handle())
        quat = torch.where(p2, self.geom_quat_xyzw(self.obj_geom), self.body_xquat("drawer_link"))
        return pos, quat

    def _task_reset(self, idx, vec):
        drawer, block = vec[:, :3], vec[:, 3:]
        self._t["body_pos"][idx, self.bid["drawer"]] = drawer
        q = self._t["qpos"]
        q[idx, self.slide_qadr] = 0.0
        q[idx, self.obj_qadr:self.obj_qadr + 3] = block
        self.drawer_init[idx] = drawer
        self.obj_init[idx] = block
        self.target[idx] = drawer + self.open_offset
        self.drawer_opened[idx] = False
        self.pickplace_completed[idx] = False

    def _goal_marker_pos(self):
        return torch.where(self.drawer_opened[:, None], self.target, self.hidden_pos.expand_as(self.target))

    def _evaluate(self, obs45, action):
        gripper = obs45[:, :3]
        obj_pos = obs45[:, 4:7]               # handle in phase 1, block in phase 2
        phase2 = self.drawer_opened.clone()   # phase as seen by this observation

        # ---- phase 1: open drawer
        handle_error = _norm(obj_pos - self.target)
        opening = ru.tolerance(handle_error, bounds=(0.0, 0.02), margin=self.MAX_DIST)
        handle_pos_init = self.target + torch.tensor([0.0, self.MAX_DIST, 0.0], device=self.device)
        scale = torch.tensor([3.0, 3.0, 1.0], device=self.device)
        g_err = _norm((obj_pos - gripper) * scale)
        g_err_init = _norm((handle_pos_init - self.init_tcp) * scale)
        caging_p1 = ru.tolerance(g_err, bounds=(0.0, 0.01), margin=g_err_init)
        r1 = (caging_p1 + opening) * 5.0
        r1 = (r1 - 10.0) / 10.0

        # ---- phase 2: pick and place into the open drawer
        cur_handle = self._handle()
        drawer_open_error = _norm(cur_handle - (self.drawer_init + self.open_offset))
        penalty = ru.tolerance(drawer_open_error, bounds=(0.0, 0.02), margin=self.MAX_DIST) - 1.0
        place_target = cur_handle + torch.tensor([0.0, 0.15, -0.02], device=self.device)
        obj_to_target = _norm(obj_pos - place_target)
        tcp_to_obj = _norm(obj_pos - gripper)
        in_place_margin = _norm(self.obj_init - place_target)
        in_place = ru.tolerance(obj_to_target, bounds=(0.0, self.TARGET_RADIUS), margin=in_place_margin)
        approach = ru.tolerance(tcp_to_obj, bounds=(0.0, 0.04), margin=_norm(self.obj_init - self.init_tcp))
        grasped = self._caging(action, obj_pos, self.obj_init)
        r2 = 0.25 * penalty + 1.25 * approach + 2.5 * grasped + 6.0 * ru.hamacher_product(grasped, in_place)
        r2 = torch.where(obj_to_target < self.TARGET_RADIUS, _full(r2, 10.0), r2)
        r2 = r2 + 10.0
        r2 = (r2 - 10.0) / 10.0

        # ---- state updates (in compute_reward order)
        self.drawer_opened = self.drawer_opened | (~phase2 & (handle_error <= 0.03))
        self.target = torch.where(phase2[:, None], place_target, self.target)
        self.pickplace_completed = self.pickplace_completed | (phase2 & (obj_to_target < self.TARGET_RADIUS))

        reward = torch.where(phase2, r2, r1)
        success = self.drawer_opened & self.pickplace_completed
        info = {
            "success": success.float(),
            "drawer_opened": self.drawer_opened.float(),
            "pickplace_completed": self.pickplace_completed.float(),
            "near_object": torch.where(phase2, tcp_to_obj, _norm(obj_pos - gripper)).le(0.03).float(),
            "obj_to_target": torch.where(phase2, obj_to_target, handle_error),
            "unscaled_reward": reward,
        }
        return reward, success, info


class DrawerOpenEnv(BatchedSawyerEnv):
    """metaworld `drawer-open-v3` (SawyerDrawerOpenEnvV3, reward v2)."""

    XML = "sawyer_drawer.xml"
    GOAL_MARKER = "never"              # goal site has alpha 0
    TASK_VEC_LOW = (-0.4, 0.8, 0.0)
    TASK_VEC_HIGH = (0.4, 0.9, 0.0)
    MAX_DIST = 0.2

    def _lookup_task_ids(self, mjm):
        self.bid["drawer"] = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "drawer")
        self.bid["drawer_link"] = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "drawer_link")

    def _handle(self):
        h = self.body_xpos("drawer_link").clone()
        h[:, 1] -= 0.16
        return h

    def _objects(self):
        return self._handle(), self.body_xquat("drawer_link")

    def _task_reset(self, idx, vec):
        self._t["body_pos"][idx, self.bid["drawer"]] = vec
        self.obj_init[idx] = vec
        off = torch.tensor([0.0, -0.16 - self.MAX_DIST, 0.09], device=self.device)
        self.target[idx] = vec + off

    def _evaluate(self, obs45, action):
        gripper = obs45[:, :3]
        handle = obs45[:, 4:7]
        handle_error = _norm(handle - self.target)
        opening = ru.tolerance(handle_error, bounds=(0.0, 0.02), margin=self.MAX_DIST)
        handle_pos_init = self.target + torch.tensor([0.0, self.MAX_DIST, 0.0], device=self.device)
        scale = torch.tensor([3.0, 3.0, 1.0], device=self.device)
        gripper_error = _norm((handle - gripper) * scale)
        gripper_error_init = _norm((handle_pos_init - self.init_tcp) * scale)
        caging = ru.tolerance(gripper_error, bounds=(0.0, 0.01), margin=gripper_error_init)
        reward = (caging + opening) * 5.0
        reward = (reward - 5.0) / 5.0
        success = handle_error <= 0.03
        info = {"success": success.float(), "near_object": (_norm(handle - gripper) <= 0.03).float(),
                "grasp_reward": caging, "in_place_reward": opening,
                "obj_to_target": handle_error, "unscaled_reward": reward}
        return reward, success, info


class CompoPickPlaceBlockEnv(BatchedSawyerEnv):
    """metaworld `compo-pickplace-block`: place block 1 on the goal, then stack block 2 on it."""

    XML = "compo_pickplace_block.xml"
    GOAL_MARKER = "always"             # follows the stacking target in phase 2
    GOAL_LOW = (-0.25, 0.4, 0.02)
    GOAL_HIGH = (0.0, 0.7, 0.02)
    TASK_VEC_LOW = (0.0, 0.5, 0.02, 0.0, 0.5, 0.02) + GOAL_LOW       # (block1, block2, goal)
    TASK_VEC_HIGH = (0.25, 0.8, 0.02, 0.25, 0.8, 0.02) + GOAL_HIGH
    TARGET_RADIUS = 0.03

    def _lookup_task_ids(self, mjm):
        for name in ("obj1", "obj2"):
            self.bid[name] = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, name)
        self.obj1_qadr, _ = self._joint_adr(mjm, "objjoint1")
        self.obj2_qadr, _ = self._joint_adr(mjm, "objjoint2")

    def _alloc_extra_buffers(self):
        n = self.num_envs
        self.pickplace1_done = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.pickplace2_done = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.obj1_pos = torch.zeros(n, 3, device=self.device)   # frozen at the end of phase 1

    def _valid_task_vec(self, v):
        # reset_model's while-loop: the blocks at least 0.15 apart in XY.
        return np.linalg.norm(v[:2] - v[3:5]) >= 0.15

    def _objects(self):
        p2 = self.pickplace1_done[:, None]
        pos = torch.where(p2, self.body_xpos("obj2"), self.body_xpos("obj1"))
        quat = torch.where(p2, self.body_xquat("obj2"), self.body_xquat("obj1"))
        return pos, quat

    def _task_reset(self, idx, vec):
        q = self._t["qpos"]
        q[idx, self.obj1_qadr:self.obj1_qadr + 3] = vec[:, 0:3]
        q[idx, self.obj2_qadr:self.obj2_qadr + 3] = vec[:, 3:6]
        self.obj_init[idx] = vec[:, 0:3]                         # obj1_init_pos
        self.obj1_pos[idx] = vec[:, 0:3]
        self.target[idx] = vec[:, 6:9]
        self.pickplace1_done[idx] = False
        self.pickplace2_done[idx] = False

    def _evaluate(self, obs45, action):
        gripper = obs45[:, :3]
        tcp_opened = obs45[:, 3]
        obj_pos = obs45[:, 4:7]               # block 1 in phase 1, block 2 in phase 2
        phase2 = self.pickplace1_done.clone()
        release = tcp_opened.clamp(0.0, 1.0)
        approach_margin = _norm(self.obj_init - self.init_tcp)
        grasped = self._caging(action, obj_pos, self.obj_init)
        tcp_to_obj = _norm(obj_pos - gripper)
        approach = ru.tolerance(tcp_to_obj, bounds=(0.0, 0.04), margin=approach_margin)
        cur_obj1 = self.body_xpos("obj1")

        # ---- phase 1: place block 1 on the goal and release it
        obj_to_goal = _norm(obj_pos - self.target)
        in_place1 = ru.tolerance(obj_to_goal, bounds=(0.0, self.TARGET_RADIUS),
                                 margin=_norm(self.obj_init - self.target))
        r1 = 0.25 * approach + grasped * (1.0 - in_place1) + 2.0 * in_place1 + 1.75 * in_place1 * release
        r1 = torch.where((obj_to_goal < self.TARGET_RADIUS) & (tcp_opened > 0.8), _full(r1, 5.0), r1)
        r1 = (r1 - 5.0) / 5.0
        done1 = (obj_to_goal < 0.05) & (tcp_opened > 0.8)

        # ---- phase 2: stack block 2 on block 1, without moving block 1
        obj1_error = _norm(cur_obj1 - self.obj1_pos)
        penalty = ru.tolerance(obj1_error, bounds=(0.0, 0.03), margin=0.2) - 1.0
        stack_target = cur_obj1 + torch.tensor([0.0, 0.0, 0.03], device=self.device)
        obj_to_stack = _norm(obj_pos - stack_target)
        in_place2 = ru.tolerance(obj_to_stack, bounds=(0.0, self.TARGET_RADIUS),
                                 margin=_norm(self.obj_init - stack_target))
        progress = approach + grasped * (1.0 - in_place2) + 2.0 * in_place2 + in_place2 * release
        r2 = 5.0 + progress + penalty
        done2 = (obj_to_stack < self.TARGET_RADIUS) & (tcp_opened > 0.8)
        r2 = torch.where(done2, _full(r2, 10.0), r2)
        r2 = (r2 - 5.0) / 5.0

        # ---- state updates
        self.obj1_pos = torch.where(phase2[:, None], self.obj1_pos, cur_obj1)
        self.pickplace1_done = self.pickplace1_done | (~phase2 & done1)
        self.pickplace2_done = torch.where(phase2, done2, self.pickplace2_done)
        self.target = torch.where(phase2[:, None], stack_target, self.target)

        reward = torch.where(phase2, r2, r1)
        success = self.pickplace1_done & self.pickplace2_done
        info = {
            "success": success.float(),
            "pickplace1_done": self.pickplace1_done.float(),
            "pickplace2_done": self.pickplace2_done.float(),
            "near_object": (tcp_to_obj <= 0.03).float(),
            "obj_to_target": torch.where(phase2, obj_to_stack, obj_to_goal),
            "unscaled_reward": reward,
        }
        return reward, success, info


class CompoPickPlaceBoxCloseEnv(BatchedSawyerEnv):
    """metaworld `compo-pickplace-boxclose`: put the block in the box, then close the lid.

    Box, block and lid are placed anew at every reset.
    """

    XML = "compo_pickplace_boxclose.xml"
    GOAL_MARKER = "phase2"             # goal site alpha 0 until the block is in, then white
    FRESH_LAYOUT = True
    XY_LOW = (-0.25, 0.45)
    XY_HIGH = (0.25, 0.80)
    BOX_LID_MIN_DIST = 0.35
    BOX_BLOCK_MIN_DIST = 0.20
    BLOCK_LID_MIN_DIST = 0.20
    LID_INIT_Z = -0.02
    TARGET_RADIUS = 0.05

    def _lookup_task_ids(self, mjm):
        for name in ("boxbody", "top_link", "obj"):
            self.bid[name] = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, name)
        assert mjm.body_parentid[self.bid["boxbody"]] == 0
        self.box_height = float(mjm.body_pos[self.bid["boxbody"], 2])
        self.obj_geom = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "objGeom")
        self.block_qadr, self.block_vadr = self._joint_adr(mjm, "obj")
        self.lid_qadr, self.lid_vadr = self._joint_adr(mjm, "lid_joint")

    def _alloc_extra_buffers(self):
        n, dev = self.num_envs, self.device
        self.pickplace_completed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.boxclose_completed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.box_init = torch.zeros(n, 3, device=dev)
        self.lid_init = torch.zeros(n, 3, device=dev)
        self.lid_target = torch.zeros(n, 3, device=dev)
        self.obj_height = torch.zeros(n, device=dev)

    def _sample_fresh_layout(self):
        # _sample_non_overlapping_positions: (box_xy, block_xy, lid_xy)
        lo, hi = np.array(self.XY_LOW * 3), np.array(self.XY_HIGH * 3)
        for _ in range(1000):
            v = self._rng.uniform(lo, hi)
            box, block, lid = v[:2], v[2:4], v[4:6]
            if (np.linalg.norm(lid - box) >= self.BOX_LID_MIN_DIST
                    and np.linalg.norm(block - box) >= self.BOX_BLOCK_MIN_DIST
                    and np.linalg.norm(block - lid) >= self.BLOCK_LID_MIN_DIST):
                return v
        return np.array([0.1, 0.85, 0.25, 0.5, -0.25, 0.5])

    def _objects(self):
        p2 = self.pickplace_completed[:, None]
        pos = torch.where(p2, self.body_xpos("top_link"), self.body_xpos("obj"))
        quat = torch.where(p2, self.body_xquat("top_link"), self.geom_quat_xyzw(self.obj_geom))
        return pos, quat

    def _task_reset(self, idx, vec):
        k, dev = len(idx), self.device
        col = lambda v: torch.full((k, 1), v, device=dev)
        box = torch.cat([vec[:, 0:2], col(self.box_height)], dim=-1)
        block = torch.cat([vec[:, 2:4], col(0.02)], dim=-1)
        lid = torch.cat([vec[:, 4:6], col(self.LID_INIT_Z)], dim=-1)
        t = self._t
        t["body_pos"][idx, self.bid["boxbody"]] = box
        b, l = self.block_qadr, self.lid_qadr
        t["qpos"][idx, b:b + 3] = block
        t["qpos"][idx, b + 3:b + 7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
        t["qvel"][idx, self.block_vadr:self.block_vadr + 6] = 0.0
        t["qpos"][idx, l:l + 3] = lid
        t["qpos"][idx, l + 3:l + 7] = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=dev)
        t["qvel"][idx, self.lid_vadr:self.lid_vadr + 6] = 0.0
        self.box_init[idx] = box
        self.obj_init[idx] = block
        self.lid_init[idx] = lid
        self.target[idx] = box + torch.tensor([0.0, 0.0, 0.04], device=dev)
        self.lid_target[idx] = box + torch.tensor([0.0, 0.0, 0.073], device=dev)
        self.pickplace_completed[idx] = False
        self.boxclose_completed[idx] = False

    def _task_post_forward(self, idx):
        self.obj_height[idx] = self.geom_xpos(self.obj_geom)[idx, 2]

    def _goal_marker_pos(self):
        return torch.where(self.pickplace_completed[:, None], self.target,
                           self.hidden_pos.expand_as(self.target))

    def _evaluate(self, obs45, action):
        gripper = obs45[:, :3]
        tcp_opened = obs45[:, 3]
        obj_pos = obs45[:, 4:7]               # block in phase 1, lid in phase 2
        phase2 = self.pickplace_completed.clone()
        target = self.target                  # box inside in phase 1, box top in phase 2
        obj_to_target = _norm(obj_pos - target)
        tcp_to_obj = _norm(obj_pos - gripper)
        in_place = ru.tolerance(obj_to_target, bounds=(0.0, self.TARGET_RADIUS),
                                margin=_norm(self.obj_init - target).clamp_min(1e-6))
        approach = ru.tolerance(tcp_to_obj, bounds=(0.0, 0.04), margin=_norm(self.obj_init - self.init_tcp))
        grasped = self._caging(action, obj_pos, self.obj_init)

        # ---- phase 1: put the block in the box and release it
        lift_target = torch.maximum(target[:, 2] + 0.04, self.obj_height + 0.04)
        lift_to_pick = (lift_target - obj_pos[:, 2]).clamp_min(0.0)
        lift_margin = (lift_target - self.obj_init[:, 2]).clamp_min(1e-6)
        lift = ru.tolerance(lift_to_pick, bounds=(0.0, 0.01), margin=lift_margin)
        release = tcp_opened.clamp(0.0, 1.0)
        r1 = (0.5 * approach + 1.5 * grasped + 1.0 * ru.hamacher_product(grasped, lift)
              + 4.0 * in_place + 3.0 * in_place * release)
        r1 = r1.clamp(max=9.95)
        done1 = (obj_to_target < self.TARGET_RADIUS) & (tcp_opened > 0.8)
        r1 = torch.where(done1, _full(r1, 10.0), r1)
        r1 = (r1 - 10.0) / 10.0

        # ---- phase 2: put the lid on the box, keeping the block in it
        block_to_box_xy = _norm(self.body_xpos("obj")[:, :2] - self.box_init[:, :2])
        penalty = ru.tolerance(block_to_box_xy, bounds=(0.0, 0.05), margin=0.2) - 1.0
        reach_margin = (_norm(self.lid_init - self.init_tcp) + 0.1).clamp_min(1e-6)
        reach = (1.0 - tcp_to_obj / reach_margin).clamp_min(0.0)
        r2 = (0.5 * penalty + 2.0 * reach + 0.5 * approach + 1.5 * grasped
              + 5.5 * ru.hamacher_product(grasped, in_place))
        r2 = torch.where(obj_to_target < self.TARGET_RADIUS, _full(r2, 10.0), r2)
        r2 = (r2 + 11.0 - 10.0) / 10.0
        closed = (_norm(obj_pos[:, :2] - target[:, :2]) < 0.03) & ((obj_pos[:, 2] - target[:, 2]).abs() < 0.03)

        # ---- state updates (phase-1 completion switches to the lid)
        switch = ~phase2 & done1
        self.pickplace_completed = self.pickplace_completed | switch
        self.obj_init = torch.where(switch[:, None], self.lid_init, self.obj_init)
        self.target = torch.where(switch[:, None], self.lid_target, self.target)
        self.boxclose_completed = self.boxclose_completed | (phase2 & closed)

        reward = torch.where(phase2, r2, r1)
        success = self.pickplace_completed & self.boxclose_completed
        info = {
            "success": success.float(),
            "pickplace_completed": self.pickplace_completed.float(),
            "boxclose_completed": self.boxclose_completed.float(),
            "near_object": (tcp_to_obj <= 0.03).float(),
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }
        return reward, success, info


TASKS = {
    "reach-v3": ReachEnv,
    "reach-xy-v3": ReachXYEnv,
    "reach-xz-v3": ReachXZEnv,
    "reach-yz-v3": ReachYZEnv,
    "reach-xyz-v3": ReachXYZEnv,
    "grasp-v3": GraspEnv,
    "pick-place-v3": PickPlaceEnv,
    "pick-place-redblock-v3": PickPlaceRedBlockEnv,
    "pick-place-greenblock-v3": PickPlaceGreenBlockEnv,
    "drawer-open-v3": DrawerOpenEnv,
    "box-close-v3": BoxCloseEnv,
    "bin-picking-redblue-v3": BinPickingRedBlueEnv,
    "bin-picking-yellowblue-v3": BinPickingYellowBlueEnv,
    "bin-picking-redpurple-v3": BinPickingRedPurpleEnv,
    "bin-picking-yellowpurple-v3": BinPickingYellowPurpleEnv,
    "compo-draweropen-pickplace": CompoDrawerOpenPickPlaceEnv,
    "compo-pickplace-block": CompoPickPlaceBlockEnv,
    "compo-pickplace-boxclose": CompoPickPlaceBoxCloseEnv,
}


def make_batched_env(task: str, num_envs: int, **kwargs) -> BatchedSawyerEnv:
    """task: 'metaworld_<name>' or '<name>' for one of TASKS."""
    name = task.split("_", 1)[1] if task.startswith("metaworld_") else task
    if name not in TASKS:
        raise ValueError(f"Unknown task {task!r}; available: {sorted(TASKS)}")
    return TASKS[name](num_envs=num_envs, **kwargs)
