"""Batched MuJoCo-Warp base environment for the benchmark's Sawyer tasks.

One `BatchedSawyerEnv` simulates `num_envs` independent worlds: physics with
`mujoco_warp` (all worlds in one batch, CUDA-graph captured on a GPU) and
camera images with MJWarp's batched ray tracer. Observations, rewards and
success tests are computed with torch on tensors that alias the MJWarp state,
so only the final observation batch leaves the device.

It mirrors the CPU pipeline of the main branch,
  gymnasium.make("Meta-World/MT1", env_name=...)   (task reset / step / reward)
  -> ProprioMultiImageObsWrapper(topview, front, gripperPOV),
with these conventions:
  * same XML models (metaworld/assets), frame_skip=5, mocap XYZ control
    (action_scale=1/80, clipped to the task's hand bounds), gripper position
    actuators, and Meta-World's reset procedure: the hand is settled for
    50 x 5 physics steps from mj_resetData, then objects are placed;
  * layouts drawn as in the CPU env: one of 50 pre-sampled task vectors per
    world (MT1 + RandomTaskSelectWrapper), or a fresh layout at every reset
    for the tasks whose CPU `reset_model` samples its own (bins, box);
  * the 45-d Meta-World observation (current, previous, goal), clipped to
    the observation space, from which `proprio` (7) and `original_obs` (39)
    are taken exactly as in ProprioMultiImageObsWrapper;
  * episodes terminate on success and truncate after `max_episode_steps`.

The env does not auto-reset: the caller resets finished worlds by index.

Differences from the CPU env that cannot be avoided: float32 physics on
MuJoCo 3.14 with the solver tolerance raised to 1e-6, and ray-traced images
(no reflections, fog or anti-aliasing). Collision flags reproduce MuJoCo
3.1.2. Sites are not drawn by MJWarp, so visible sites become visual geoms.
Small ones kept for simplicity: the first observation of an episode reports
the hand velocity after the reset's final `mj_forward`, where some CPU tasks
report it one physics step earlier.

`scripts/check_warp_parity.py` compares every task with its CPU version.
"""
from __future__ import annotations

import contextlib
import datetime
import math
import uuid
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp

import mujoco_warp as mjw

from . import reward_utils as ru

ASSET_DIR = Path(__file__).resolve().parents[1] / "metaworld" / "assets"

FRAME_SKIP = 5
ACTION_SCALE = 1.0 / 80.0
N_TASK_VECS = 50                     # Meta-World MT1 pre-samples 50 layouts per env
HAND_SETTLE_ITERS = 50               # SawyerXYZEnv._reset_hand: 50 x frame_skip steps
HAND_POS_LOW = (-0.525, 0.348, -0.0525)   # SawyerXYZEnv._HAND_POS_SPACE
HAND_POS_HIGH = (0.525, 1.025, 0.7)
HIDDEN_POS = (0.0, 0.0, -5.0)        # goal marker parked under the floor
_HAND_QUAT = (1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0), 0.0)  # [1,0,1,0] normalised
_STATE_KEYS = ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "qacc_warmstart")


@wp.kernel
def _unpack_rgb_u8(
    packed: wp.array2d(dtype=wp.uint32),
    rgb_adr: wp.array(dtype=int),
    out: wp.array3d(dtype=wp.uint8),
):
    """Packed ARGB uint32 (per world, per camera) -> uint8 (world, pixel, 3*cam)."""
    worldid, camid, pixelid = wp.tid()
    val = packed[worldid, rgb_adr[camid] + pixelid]
    out[worldid, pixelid, camid * 3 + 0] = wp.uint8((val >> wp.uint32(16)) & wp.uint32(255))
    out[worldid, pixelid, camid * 3 + 1] = wp.uint8((val >> wp.uint32(8)) & wp.uint32(255))
    out[worldid, pixelid, camid * 3 + 2] = wp.uint8(val & wp.uint32(255))


def _quat_mul(a, b):
    """Hamilton product of (w, x, y, z) quaternions, batched over leading dims."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _quat_rotate(q, v):
    """Rotate vectors v (..., 3) by (w, x, y, z) quaternions q (..., 4)."""
    w, u = q[..., :1], q[..., 1:]
    return (2.0 * (u * v).sum(-1, keepdim=True) * u + (w * w - (u * u).sum(-1, keepdim=True)) * v
            + 2.0 * w * torch.cross(u, v, dim=-1))


def _quat_to_mat(q):
    """(w, x, y, z) quaternions (..., 4) -> rotation matrices (..., 3, 3)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def _find(items, name):
    for it in items:
        if it.name == name:
            return it
    raise KeyError(name)


def default_device() -> str:
    """The current CUDA device if there is one, else the CPU (slow; for tests)."""
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"


class BatchedSawyerEnv:
    """Base class; subclasses define the task (XML, layout, objects, reward)."""

    XML: str = ""
    HAND_LOW = (-0.5, 0.40, 0.05)      # SawyerXYZEnv hand_low / hand_high: the mocap clip
    HAND_HIGH = (0.5, 1.0, 0.5)
    HAND_INIT = (0.0, 0.6, 0.2)        # init_config["hand_init_pos"]
    # (low, high) to draw the hand start at every reset (reach-xy/xz/yz/xyz);
    # None: always HAND_INIT.
    RANDOM_HAND_INIT: tuple | None = None
    # True: the CPU reset settles the hand while the bodies of a throwaway
    # layout are in place (see `_settle`); objects resting at their spawn
    # point can be knocked by them (bins, box).
    SETTLE_LAYOUT = False
    GOAL_LOW: tuple | None = None      # goal_space (observation clip); None: hand bounds
    GOAL_HIGH: tuple | None = None
    GOAL_MARKER = "never"              # 'never' | 'always' | 'phase2'
    TASK_VEC_LOW: tuple = ()           # _random_reset_space
    TASK_VEC_HIGH: tuple = ()
    # True: draw a new layout at every reset, as the CPU `reset_model` of the
    # bin and box tasks does, instead of one of the 50 MT1 task vectors.
    FRESH_LAYOUT = False
    LOCKED_ACTION: int | None = None   # action dimension fixed to 0 (2-D reach tasks)
    EXTRA_SETTLE_STEPS = 0             # physics steps after the hand settle, final layout in place
    # Constraint rows per world (MJWarp njmax). When a step needs more, MJWarp
    # drops the extra contacts, so tasks with many contacts at once raise it.
    NJMAX = 512

    def __init__(
        self,
        num_envs: int,
        image_size: int = 128,
        cameras=("topview", "front", "gripperPOV"),
        action_repeat: int = 1,
        max_episode_steps: int = 250,
        seed: int = 0,
        device: str | None = None,
        solver_tolerance: float | None = 1e-6,
        nconmax: int = 96,
        njmax: int | None = None,      # default: the task's NJMAX
        use_cuda_graph: bool = True,
        render: bool = True,
        render_shadows: bool = True,
        reward_tuning: bool = False,
        fix_static_geoms: bool = True,
    ):
        self.num_envs = int(num_envs)
        njmax = int(njmax or self.NJMAX)
        self.image_size = int(image_size)
        self.cameras = list(cameras)
        self.action_repeat = int(action_repeat)
        self.max_episode_steps = int(max_episode_steps)
        self.reward_tuning = bool(reward_tuning)
        self.fix_static_geoms = bool(fix_static_geoms)
        self.render_enabled = bool(render)
        self.device = torch.device(device or default_device())
        on_gpu = self.device.type == "cuda"
        self._use_graph = bool(use_cuda_graph) and on_gpu
        self._rng = np.random.default_rng(seed)
        self.ids = [self._new_id() for _ in range(self.num_envs)]

        wp.init()
        self._wp_device = wp.get_device(str(self.device) if on_gpu else "cpu")
        self._stream = torch.cuda.Stream(device=self.device) if on_gpu else None
        self._wp_stream = wp.stream_from_torch(self._stream) if on_gpu else None

        mjm = self._build_mjmodel()
        self.mjm = mjm
        self._lookup_ids(mjm)
        self._reset0 = self._reset_data_state(mjm)
        # Tasks whose settled state is the same every episode settle once on the
        # CPU; the others settle on the device at every reset.
        self._settle_per_reset = bool(self.RANDOM_HAND_INIT) or self.SETTLE_LAYOUT
        canon = None if self._settle_per_reset else self._settled_state(mjm, self.HAND_INIT)

        # Float32 cannot reach Meta-World's tolerance=1e-10, which would make the
        # Newton solver run all its iterations every step; 1e-6 is the float32
        # equivalent of "fully converged".
        if solver_tolerance is not None:
            mjm.opt.tolerance = max(float(mjm.opt.tolerance), float(solver_tolerance))

        with self._on_stream():
            self.m = mjw.put_model(mjm, batch_sizes={"body_pos": self.num_envs})
            self.d = mjw.make_data(mjm, nworld=self.num_envs, nconmax=nconmax, njmax=njmax)
            self.rc = None
            if self.render_enabled:
                self.rc = mjw.create_render_context(
                    mjm,
                    nworld=self.num_envs,
                    cam_res=(self.image_size, self.image_size),
                    render_rgb=True,
                    render_depth=False,
                    render_seg=False,
                    use_textures=True,
                    use_shadows=bool(render_shadows),
                    enabled_geom_groups=[0, 1, 2],
                    cam_active=self.cameras,
                    render_skybox=True,
                )
            npix = self.image_size * self.image_size
            self._img_wp = wp.zeros((self.num_envs, npix, 3 * len(self.cameras)), dtype=wp.uint8)
            self._t = self._torch_views(self.d)
            self._t["body_pos"] = wp.to_torch(self.m.body_pos)
            self._t["img"] = wp.to_torch(self._img_wp)
            dev = self.device
            self._reset0 = {k: torch.as_tensor(v, dtype=torch.float32, device=dev)
                            for k, v in self._reset0.items()}
            self._canon = None if canon is None else {
                k: torch.as_tensor(v, dtype=torch.float32, device=dev) for k, v in canon.items()}
            self._alloc_task_buffers()
            self._init_static_geoms(mjm)
            # Compile all kernels and capture the per-control-step physics
            # (5 x mj_step + mj_forward) and the render pass as CUDA graphs.
            # This touches the initial state only; every world is reset
            # before its first step.
            init = self._canon or self._reset0
            for k, v in init.items():
                self._t[k][:] = v
            mjw.forward(self.m, self.d)
            self._graph_step = self._capture(self._physics_eager)
            self._graph_render = self._capture(self._render_eager) if self.render_enabled else None
            self._graph_settle = None
            if self._settle_per_reset:
                # Separate data for settling the hand at a new start position
                # without disturbing the worlds that are not being reset.
                self.d_settle = mjw.make_data(mjm, nworld=self.num_envs, nconmax=nconmax, njmax=njmax)
                self._ts = self._torch_views(self.d_settle)
                for k, v in self._reset0.items():
                    self._ts[k][:] = v
                self._graph_settle = self._capture(self._settle_eager)
        if on_gpu:
            torch.cuda.synchronize(self.device)
        self._last = None

    # ------------------------------------------------------------------ model
    def _build_mjmodel(self) -> mujoco.MjModel:
        path = str(ASSET_DIR / "sawyer_xyz" / self.XML)
        ref = mujoco.MjModel.from_xml_path(path)  # compiled reference for site data
        spec = mujoco.MjSpec.from_file(path)

        # Visible sites -> visual geoms (MJWarp's renderer only draws geoms).
        for sid in range(ref.nsite):
            name = mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_SITE, sid)
            rgba = ref.site_rgba[sid]
            if rgba[3] <= 0 or ref.site_group[sid] > 2 or name == "goal":
                continue
            body_name = mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, ref.site_bodyid[sid])
            body = _find(spec.bodies, body_name)
            body.add_geom(
                name=f"vis_site_{name}",
                type=mujoco.mjtGeom(ref.site_type[sid]),
                size=[float(x) for x in ref.site_size[sid]],
                pos=[float(x) for x in ref.site_pos[sid]],
                quat=[float(x) for x in ref.site_quat[sid]],
                rgba=[float(x) for x in rgba],
                contype=0, conaffinity=0, group=0, density=0.0,
            )

        # Goal marker: a mocap body carrying a white sphere (the 'goal' site).
        if self.GOAL_MARKER != "never":
            gsid = mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_SITE, "goal")
            goal = spec.worldbody.add_body(name="goal_marker", mocap=True, pos=list(HIDDEN_POS))
            goal.add_geom(
                name="vis_goal", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[float(ref.site_size[gsid][0]), 0.0, 0.0], rgba=[1.0, 1.0, 1.0, 1.0],
                contype=0, conaffinity=0, group=0, density=0.0,
            )

        mjm = spec.compile()
        # Match MuJoCo 3.1.2 (the CPU version) collision defaults: multi-contact
        # CCD was opt-in there and box-box used the primitive collider. In 3.14
        # both are on by default, and MJWarp's multi-contact CCD rejects the
        # non-zero geom margins (0.001) of the Sawyer gripper geoms.
        mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        # SawyerMocapBase.reset_mocap_welds
        for i in range(mjm.neq):
            if mjm.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                mjm.eq_data[i] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 5.0])
        # Colour changes the CPU env makes to the model at reset (same in every world).
        self._edit_model(mjm)
        # Hide geoms the OpenGL renderer would not show (alpha == 0).
        hidden = mjm.geom_rgba[:, 3] <= 0.0
        mjm.geom_group[hidden] = 5
        return mjm

    def _edit_model(self, mjm):
        pass

    def _init_static_geoms(self, mjm):
        """Geoms on bodies welded to the world, which MJWarp treats as static.

        MJWarp computes their world pose once in `make_data` and skips them in
        every later kinematics pass. Tasks that move such a body per world at
        reset (the drawer case, bins, box) must update these geoms themselves,
        or they stay at their XML position for collision and rendering.
        """
        welded = (mjm.body_weldid[mjm.geom_bodyid] == 0) & (mjm.geom_bodyid > 0)
        root_mocap = mjm.body_mocapid[mjm.body_rootid[mjm.geom_bodyid]] >= 0
        gids = np.nonzero(welded & ~root_mocap)[0]
        dev = self.device
        self._static_gid = torch.as_tensor(gids, dtype=torch.long, device=dev)
        self._static_gbody = torch.as_tensor(mjm.geom_bodyid[gids], dtype=torch.long, device=dev)
        self._static_gpos = torch.as_tensor(mjm.geom_pos[gids], dtype=torch.float32, device=dev)
        self._static_gquat = torch.as_tensor(mjm.geom_quat[gids], dtype=torch.float32, device=dev)

    def _update_static_geoms(self, idx):
        """Recompute static-geom poses for worlds `idx` from the current body poses."""
        if not self.fix_static_geoms or self._static_gid.numel() == 0:
            return
        t = self._t
        bpos = t["xpos"][idx][:, self._static_gbody]          # (k, g, 3)
        bquat = t["xquat"][idx][:, self._static_gbody]        # (k, g, 4) wxyz
        gpos = bpos + _quat_rotate(bquat, self._static_gpos.expand_as(bpos))
        gquat = _quat_mul(bquat, self._static_gquat.expand_as(bquat))
        rows = idx[:, None]
        t["geom_xpos"][rows, self._static_gid] = gpos
        t["geom_xmat"][rows, self._static_gid] = _quat_to_mat(gquat)

    def _lookup_ids(self, mjm):
        name2id = lambda t, n: mujoco.mj_name2id(mjm, t, n)
        bid = lambda n: name2id(mujoco.mjtObj.mjOBJ_BODY, n)
        sid = lambda n: name2id(mujoco.mjtObj.mjOBJ_SITE, n)
        self.bid = {n: bid(n) for n in ("hand", "rightclaw", "leftclaw", "leftpad", "rightpad", "mocap")}
        self.site_right = sid("rightEndEffector")
        self.site_left = sid("leftEndEffector")
        self.leftpad_geom = name2id(mujoco.mjtObj.mjOBJ_GEOM, "leftpad_geom")
        self.rightpad_geom = name2id(mujoco.mjtObj.mjOBJ_GEOM, "rightpad_geom")
        self.hand_mocap = int(mjm.body_mocapid[self.bid["mocap"]])
        self.goal_mocap = None
        if self.GOAL_MARKER != "never":
            self.goal_mocap = int(mjm.body_mocapid[bid("goal_marker")])
        self._lookup_task_ids(mjm)

    def _lookup_task_ids(self, mjm):
        pass

    def _joint_adr(self, mjm, name):
        jid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"joint {name!r} not in {self.XML}")
        return int(mjm.jnt_qposadr[jid]), int(mjm.jnt_dofadr[jid])

    @staticmethod
    def _reset_data_state(mjm):
        """State right after mj_resetData (the start of every CPU reset)."""
        d = mujoco.MjData(mjm)
        mujoco.mj_resetData(mjm, d)
        return {k: getattr(d, k).copy() for k in _STATE_KEYS}

    def _settled_state(self, mjm, hand_init):
        """State after Meta-World's reset: mj_resetData + `_reset_hand` (50 x 5 steps).

        The CPU env runs exactly this from mj_resetData on every reset, so for
        a fixed hand start the result is identical for every episode; objects
        are placed afterwards.
        """
        d = mujoco.MjData(mjm)
        mujoco.mj_resetData(mjm, d)
        for _ in range(HAND_SETTLE_ITERS):
            d.mocap_pos[self.hand_mocap] = hand_init
            d.mocap_quat[self.hand_mocap] = [1.0, 0.0, 1.0, 0.0]
            d.ctrl[:] = [-1.0, 1.0]
            mujoco.mj_step(mjm, d, nstep=FRAME_SKIP)
        for _ in range(self.EXTRA_SETTLE_STEPS):
            mujoco.mj_step(mjm, d)
        state = {k: getattr(d, k).copy() for k in _STATE_KEYS}
        state["mocap_quat"][self.hand_mocap] = _HAND_QUAT
        return state

    @staticmethod
    def _torch_views(d):
        t = {}
        names = ["qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "xpos", "xquat",
                 "cvel", "site_xpos", "geom_xpos", "geom_xmat", "qacc_warmstart"]
        for n in names:
            arr = getattr(d, n, None)
            if arr is not None:
                t[n] = wp.to_torch(arr)
        return t

    # ---------------------------------------------------------- task buffers
    def _alloc_task_buffers(self):
        n, dev = self.num_envs, self.device
        z3 = lambda: torch.zeros(n, 3, device=dev)
        self.target = z3()
        self.obj_init = z3()
        self.init_tcp = z3()
        self.hand_init = torch.tensor(self.HAND_INIT, dtype=torch.float32, device=dev).repeat(n, 1)
        self.prev21 = torch.zeros(n, 21, device=dev)
        self.t_agent = torch.zeros(n, dtype=torch.long, device=dev)
        self.task_vecs = None
        if not self.FRESH_LAYOUT:
            lo = np.asarray(self.TASK_VEC_LOW, dtype=np.float64)
            hi = np.asarray(self.TASK_VEC_HIGH, dtype=np.float64)
            self.task_vecs = np.stack([self._sample_task_vecs(lo, hi) for _ in range(n)])  # (n, 50, D)
        goal_low = self.GOAL_LOW if self.GOAL_LOW is not None else self.HAND_LOW
        goal_high = self.GOAL_HIGH if self.GOAL_HIGH is not None else self.HAND_HIGH
        obs_low = torch.tensor(HAND_POS_LOW + (-1.0,) + (-math.inf,) * 17, device=dev)
        obs_high = torch.tensor(HAND_POS_HIGH + (1.0,) + (math.inf,) * 17, device=dev)
        goal_low = torch.tensor(goal_low, dtype=torch.float32, device=dev)
        goal_high = torch.tensor(goal_high, dtype=torch.float32, device=dev)
        self.obs45_low = torch.cat([obs_low, obs_low, goal_low])
        self.obs45_high = torch.cat([obs_high, obs_high, goal_high])
        self.mocap_low = torch.tensor(self.HAND_LOW, dtype=torch.float32, device=dev)
        self.mocap_high = torch.tensor(self.HAND_HIGH, dtype=torch.float32, device=dev)
        self.hand_quat = torch.tensor(_HAND_QUAT, device=dev)
        self.hidden_pos = torch.tensor(HIDDEN_POS, device=dev)
        self._last = None
        self._alloc_extra_buffers()

    def _alloc_extra_buffers(self):
        pass

    def _sample_task_vecs(self, lo, hi):
        """The 50 MT1 layouts of one world, with the task's own rejection test."""
        out = np.empty((N_TASK_VECS, lo.size))
        k = 0
        while k < N_TASK_VECS:
            v = self._rng.uniform(lo, hi)
            if self._valid_task_vec(v):
                out[k] = v
                k += 1
        return out

    def _valid_task_vec(self, v):
        return True

    def _sample_fresh_layout(self):
        raise NotImplementedError

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _new_id():
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"{timestamp}-{uuid.uuid4().hex}"

    @contextlib.contextmanager
    def _on_stream(self):
        """Run warp and torch work on this env's device (and its own CUDA stream)."""
        with wp.ScopedDevice(self._wp_device):
            if self._stream is None:
                yield
            else:
                with torch.cuda.stream(self._stream), wp.ScopedStream(self._wp_stream):
                    self._stream.wait_stream(torch.cuda.default_stream(self.device))
                    yield

    def tcp_center(self):
        s = self._t["site_xpos"]
        return 0.5 * (s[:, self.site_right] + s[:, self.site_left])

    def body_xpos(self, name):
        return self._t["xpos"][:, self.bid[name]]

    def body_xquat(self, name):
        return self._t["xquat"][:, self.bid[name]]

    def geom_xpos(self, gid):
        return self._t["geom_xpos"][:, gid]

    def geom_quat_xyzw(self, gid):
        """scipy `Rotation.from_matrix(geom_xmat).as_quat()`, as the CPU env observes it."""
        return ru.quat_xyzw_from_matrix(self._t["geom_xmat"][:, gid])

    def _cur21(self):
        """Hand pos (3), gripper opening (1), object pos+quat padded (14), hand vel (3)."""
        t = self._t
        hand = t["xpos"][:, self.bid["hand"]]
        vel = t["cvel"][:, self.bid["hand"], 3:6]
        gap = (t["xpos"][:, self.bid["rightclaw"]] - t["xpos"][:, self.bid["leftclaw"]]).norm(dim=-1)
        gap = (gap / 0.1).clamp(0.0, 1.0).unsqueeze(-1)
        obj_pos, obj_quat = self._objects()
        pad = torch.zeros(self.num_envs, 7, device=self.device)
        return torch.cat([hand, gap, obj_pos, obj_quat, pad, vel], dim=-1)

    def _objects(self):
        """(pos (n, 3), quat (n, 4)) of the observed object: `_get_pos/quat_objects`."""
        raise NotImplementedError

    def _obs45(self, cur21, clip):
        obs = torch.cat([cur21, self.prev21, self.target], dim=-1)
        if clip:
            obs = torch.maximum(torch.minimum(obs, self.obs45_high), self.obs45_low)
        return obs

    def _caging(self, action, obj_pos, obj_init):
        """Meta-World's smooth gripper-caging reward (pick-place v3 and derived tasks)."""
        pad_success_margin, x_z_success_margin, obj_radius = 0.05, 0.005, 0.015
        tcp = self.tcp_center()
        left_pad, right_pad = self.body_xpos("leftpad"), self.body_xpos("rightpad")
        d_left = left_pad[:, 1] - obj_pos[:, 1]
        d_right = obj_pos[:, 1] - right_pad[:, 1]
        # The CPU env's `init_left_pad = get_body_com("leftpad")` is a view of
        # data.xpos, not a copy, so its "initial" pads are the current pads.
        right_margin = ((obj_pos[:, 1] - right_pad[:, 1]).abs() - pad_success_margin).abs()
        left_margin = ((obj_pos[:, 1] - left_pad[:, 1]).abs() - pad_success_margin).abs()
        right_caging = ru.tolerance(d_right, bounds=(obj_radius, pad_success_margin), margin=right_margin)
        left_caging = ru.tolerance(d_left, bounds=(obj_radius, pad_success_margin), margin=left_margin)
        y_caging = ru.hamacher_product(left_caging, right_caging)
        xz = [0, 2]
        tcp_obj_xz = (tcp[:, xz] - obj_pos[:, xz]).norm(dim=-1)
        xz_margin = ((obj_init[:, xz] - self.init_tcp[:, xz]).norm(dim=-1) - x_z_success_margin).clamp_min(0.0)
        x_z_caging = ru.tolerance(tcp_obj_xz, bounds=(0.0, x_z_success_margin), margin=xz_margin)
        caging = ru.hamacher_product(y_caging, x_z_caging)
        grasping = ru.hamacher_product(caging, action[:, 3].clamp(0.0, 1.0))
        return 0.7 * caging + 0.3 * grasping

    def _touching(self, obj_geoms):
        """`SawyerXYZEnv.touching_object`: both finger pads push on one of `obj_geoms`."""
        if not hasattr(self, "_contact"):
            c, efc = self.d.contact, self.d.efc
            self._contact = dict(
                geom=wp.to_torch(c.geom), worldid=wp.to_torch(c.worldid),
                efc_address=wp.to_torch(c.efc_address), nacon=wp.to_torch(self.d.nacon),
                force=wp.to_torch(efc.force),
            )
            self._obj_geoms = {}
        c = self._contact
        key = tuple(obj_geoms)
        if key not in self._obj_geoms:
            self._obj_geoms[key] = torch.as_tensor(key, dtype=torch.long, device=self.device)
        objs = self._obj_geoms[key]
        n_slots = c["geom"].shape[0]
        valid = torch.arange(n_slots, device=self.device) < c["nacon"][0]
        g = c["geom"].long()
        adr = c["efc_address"].reshape(n_slots, -1)[:, 0].long()
        world = c["worldid"].long().clamp(0, self.num_envs - 1)
        valid = valid & (adr >= 0)
        force = torch.where(valid, c["force"][world, adr.clamp_min(0)], torch.zeros((), device=self.device))
        g0_obj, g1_obj = torch.isin(g[:, 0], objs), torch.isin(g[:, 1], objs)

        def pad_force(pad):
            on_pair = ((g[:, 0] == pad) & g1_obj) | ((g[:, 1] == pad) & g0_obj)
            per_world = torch.zeros(self.num_envs, device=self.device)
            return per_world.index_add_(0, world, torch.where(on_pair, force, torch.zeros_like(force)))

        return (pad_force(self.leftpad_geom) > 0) & (pad_force(self.rightpad_geom) > 0)

    # ------------------------------------------------------- sim + render
    def _physics_eager(self):
        for _ in range(FRAME_SKIP):
            mjw.step(self.m, self.d)
        mjw.forward(self.m, self.d)

    def _settle_eager(self):
        for _ in range(FRAME_SKIP):
            mjw.step(self.m, self.d_settle)

    def _render_eager(self):
        mjw.refit_bvh(self.m, self.d, self.rc)
        mjw.render(self.m, self.d, self.rc)
        wp.launch(
            _unpack_rgb_u8,
            dim=(self.num_envs, len(self.cameras), self.image_size * self.image_size),
            inputs=[self.rc.rgb_data, self.rc.rgb_adr],
            outputs=[self._img_wp],
        )

    def _capture(self, fn):
        fn()  # warm-up: compile / load every kernel (also needed when eager)
        if not self._use_graph:
            return None
        # The warm-up loaded every module `fn` uses, so skip warp's default
        # "load all modules" step (it would compile all of MJWarp's kernels).
        for force_load in (False, None):
            try:
                with wp.ScopedCapture(force_module_load=force_load) as cap:
                    fn()
                return cap.graph
            except Exception as e:  # pragma: no cover - depends on driver
                err = e
        print(f"[mjwenv] CUDA graph capture failed ({err!r}); running eagerly.")
        return None

    def _physics(self):
        if self._graph_step is not None:
            wp.capture_launch(self._graph_step)
        else:
            self._physics_eager()

    def _render(self):
        if not self.render_enabled:
            return self._t["img"]
        if self._graph_render is not None:
            wp.capture_launch(self._graph_render)
        else:
            self._render_eager()
        return self._t["img"]

    def _settle(self, idx, hand_init, vec, settle_vec):
        """The CPU reset up to object placement, for worlds `idx`, in `d_settle`.

        `SawyerXYZEnv.reset` calls `reset_model` twice: once from the current
        state, then again after `mj_resetData`. A task that moves bodies
        (bins, box) leaves the first call's layout in the model, so the second
        call settles the hand (50 x 5 steps from mj_resetData) with that
        throwaway layout `settle_vec` in place, then moves the bodies to the
        final layout `vec`, runs EXTRA_SETTLE_STEPS and places the objects.
        """
        ts, t = self._ts, self._t
        for k, v in self._reset0.items():
            ts[k][idx] = v
        ts["mocap_pos"][idx, self.hand_mocap] = hand_init
        ts["mocap_quat"][idx, self.hand_mocap] = self.hand_quat
        ts["ctrl"][idx] = torch.tensor([-1.0, 1.0], device=self.device)
        if self.SETTLE_LAYOUT:
            self._place_layout_bodies(idx, settle_vec)
        launches = [HAND_SETTLE_ITERS]
        if self.EXTRA_SETTLE_STEPS:
            assert self.EXTRA_SETTLE_STEPS % FRAME_SKIP == 0
            launches.append(self.EXTRA_SETTLE_STEPS // FRAME_SKIP)
        for i, n in enumerate(launches):
            if i == 1:
                self._place_layout_bodies(idx, vec)
            for _ in range(n):
                if self._graph_settle is not None:
                    wp.capture_launch(self._graph_settle)
                else:
                    self._settle_eager()
        for k in _STATE_KEYS:
            t[k][idx] = ts[k][idx]

    def _place_layout_bodies(self, idx, vec):
        """Move the bodies a layout vector positions (model.body(...).pos edits)."""

    # ----------------------------------------------------------- task API
    def _task_reset(self, idx, vec):
        """Place objects / goals for worlds `idx` with layout vectors `vec` (len(idx), D)."""
        raise NotImplementedError

    def _task_post_forward(self, idx):
        pass

    def _evaluate(self, obs45, action):
        """Returns (reward in [-1, 1], success bool, info dict of tensors). May update task state."""
        raise NotImplementedError

    def _update_goal_marker(self):
        if self.goal_mocap is None:
            return
        self._t["mocap_pos"][:, self.goal_mocap] = self._goal_marker_pos()

    def _goal_marker_pos(self):
        return self.target

    # ---------------------------------------------------------- public API
    @property
    def obs_keys(self):
        return ("image", "proprio", "original_obs")

    @property
    def action_bounds(self):
        """(low, high) of the CPU env's action space (a locked dimension is [0, 0])."""
        low, high = -np.ones(4, np.float32), np.ones(4, np.float32)
        if self.LOCKED_ACTION is not None:
            low[self.LOCKED_ACTION] = high[self.LOCKED_ACTION] = 0.0
        return low, high

    def reseed(self, seed):
        """New random stream and new MT1 layouts, as for a freshly made env."""
        self._rng = np.random.default_rng(seed)
        if self.task_vecs is not None:
            lo = np.asarray(self.TASK_VEC_LOW, dtype=np.float64)
            hi = np.asarray(self.TASK_VEC_HIGH, dtype=np.float64)
            self.task_vecs = np.stack([self._sample_task_vecs(lo, hi) for _ in range(self.num_envs)])

    def sample_hand_init(self, k):
        lo, hi = (np.asarray(b, dtype=np.float64) for b in self.RANDOM_HAND_INIT)
        return self._rng.uniform(lo, hi, size=(k, 3))

    def reset(self, indices=None, task_vecs=None, hand_init=None, settle_vecs=None):
        """Reset worlds `indices` (all if None). Returns the obs of ALL worlds (numpy).

        `task_vecs` (len(indices), D) forces the layout of each reset world, in
        the format of the CPU env's `_last_rand_vec`; by default one is drawn
        as the CPU env would. `hand_init` (len(indices), 3) forces the hand
        start of the tasks with a random one, and `settle_vecs` the throwaway
        layout of the SETTLE_LAYOUT tasks (see `_settle`).
        """
        if indices is None:
            idx_np = np.arange(self.num_envs)
        else:
            idx_np = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx_np.size == 0:
            return self._last
        k = idx_np.size
        if task_vecs is None:
            if self.FRESH_LAYOUT:
                task_vecs = np.stack([self._sample_fresh_layout() for _ in range(k)])
            else:
                task_vecs = self.task_vecs[idx_np, self._rng.integers(0, N_TASK_VECS, size=k)]
        if hand_init is None:
            hand_init = self.sample_hand_init(k) if self.RANDOM_HAND_INIT else np.tile(self.HAND_INIT, (k, 1))
        if self.SETTLE_LAYOUT and settle_vecs is None:
            settle_vecs = np.stack([self._sample_fresh_layout() for _ in range(k)])
        with self._on_stream():
            dev = self.device
            idx = torch.as_tensor(idx_np, device=dev)
            vec = torch.as_tensor(np.asarray(task_vecs), dtype=torch.float32, device=dev).reshape(k, -1)
            t = self._t
            if self._settle_per_reset:
                hinit = torch.as_tensor(np.asarray(hand_init), dtype=torch.float32, device=dev).reshape(k, 3)
                svec = None if settle_vecs is None else torch.as_tensor(
                    np.asarray(settle_vecs), dtype=torch.float32, device=dev).reshape(k, -1)
                self._settle(idx, hinit, vec, svec)
                self.hand_init[idx] = hinit
            else:
                for key, v in self._canon.items():
                    t[key][idx] = v
            self._task_reset(idx, vec)
            self._update_goal_marker()
            mjw.forward(self.m, self.d)
            self._update_static_geoms(idx)
            self.init_tcp[idx] = self.tcp_center()[idx]
            self._task_post_forward(idx)
            cur21 = self._cur21()
            self.prev21[idx] = cur21[idx]
            obs45 = self._obs45(cur21, clip=False)
            self.t_agent[idx] = 0
            img = self._render()

            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
            mask[idx] = True
            if self._last is None:
                self._last_gpu = dict(obs45=obs45.clone(), img=img.clone())
            else:
                self._last_gpu["obs45"] = torch.where(mask[:, None], obs45, self._last_gpu["obs45"])
                self._last_gpu["img"] = torch.where(mask[:, None, None], img, self._last_gpu["img"])
            out = self._to_numpy(self._last_gpu["obs45"], self._last_gpu["img"])
        for i in idx_np.tolist():
            self.ids[i] = self._new_id()
        self._last = out
        return out

    def step(self, action):
        """Step all worlds with `action` (num_envs, 4).

        Returns obs (dict of numpy, all worlds), reward (n,), terminated (n,),
        truncated (n,), info (dict of numpy arrays).
        """
        with self._on_stream():
            a = torch.as_tensor(np.asarray(action) if not torch.is_tensor(action) else action,
                                dtype=torch.float32, device=self.device).reshape(self.num_envs, 4)
            if self.LOCKED_ACTION is not None:
                a = a.clone()
                a[:, self.LOCKED_ACTION] = 0.0
            a_clip = a.clamp(-1.0, 1.0)
            t = self._t
            n, R = self.num_envs, self.action_repeat
            ended = torch.zeros(n, dtype=torch.bool, device=self.device)
            term = torch.zeros(n, dtype=torch.bool, device=self.device)
            early = torch.zeros(n, dtype=torch.bool, device=self.device)   # ended before the last repeat
            total_r = torch.zeros(n, device=self.device)
            final_obs45 = early_img = None
            final_info = {}
            for r in range(R):
                # SawyerXYZEnv.set_xyz_action + do_simulation([a, -a], frame_skip)
                mp = t["mocap_pos"][:, self.hand_mocap]
                t["mocap_pos"][:, self.hand_mocap] = torch.maximum(
                    torch.minimum(mp + a_clip[:, :3] * ACTION_SCALE, self.mocap_high), self.mocap_low)
                t["mocap_quat"][:, self.hand_mocap] = self.hand_quat
                t["ctrl"][:, 0] = a[:, 3]
                t["ctrl"][:, 1] = -a[:, 3]
                self._physics()

                cur21 = self._cur21()
                obs45 = self._obs45(cur21, clip=True)
                self.prev21 = cur21
                reward, success, info = self._evaluate(obs45, a)
                self._update_goal_marker()
                if self.reward_tuning:
                    reward = (reward - 1.0) * 0.5          # RewardTuningWrapperV2
                total_r = total_r + torch.where(ended, torch.zeros_like(reward), reward)
                last = r == R - 1
                ended_now = (success | last) & ~ended
                term = term | (success & ~ended)
                if final_obs45 is None:
                    final_obs45 = obs45.clone()
                    final_info = {k: v.clone() for k, v in info.items()}
                else:
                    final_obs45 = torch.where(ended_now[:, None], obs45, final_obs45)
                    for k, v in info.items():
                        final_info[k] = torch.where(ended_now, v, final_info[k])
                if not last:
                    # A world that succeeds before the last repeat ends its episode
                    # here (ActionRepeat breaks); render its terminal frame now.
                    now = success & ~ended
                    if bool(now.any()):
                        img = self._render()
                        early_img = img.clone() if early_img is None else torch.where(now[:, None, None], img, early_img)
                        early = early | now
                ended = ended | ended_now
            img = self._render()
            final_img = img if early_img is None else torch.where(early[:, None, None], early_img, img)

            self.t_agent += 1
            trunc = (self.t_agent >= self.max_episode_steps) & ~term
            # Guard against simulation blow-ups: treat NaNs as a truncation.
            bad = ~torch.isfinite(final_obs45).all(dim=-1)
            final_obs45 = torch.nan_to_num(final_obs45, nan=0.0, posinf=0.0, neginf=0.0)
            trunc = trunc | (bad & ~term)
            total_r = torch.nan_to_num(total_r, nan=0.0)

            self._last_gpu = dict(obs45=final_obs45, img=final_img.clone())
            obs = self._to_numpy(final_obs45, final_img)
            out_r = total_r.float().cpu().numpy()
            out_term = term.cpu().numpy()
            out_trunc = trunc.cpu().numpy()
            info_np = {k: v.float().cpu().numpy() for k, v in final_info.items()}
            info_np["success"] = out_term.astype(np.float32)
            info_np["time_limit"] = (self.t_agent >= self.max_episode_steps).cpu().numpy()
        self._last = obs
        return obs, out_r, out_term, out_trunc, info_np

    def _to_numpy(self, obs45, img):
        h = w = self.image_size
        proprio = torch.cat([obs45[:, 0:3], obs45[:, 18:21], obs45[:, 3:4]], dim=-1)
        original = torch.cat([obs45[:, :18], obs45[:, 21:39], obs45[:, 42:45]], dim=-1)
        return {
            "image": img.reshape(self.num_envs, h, w, 3 * len(self.cameras)).cpu().numpy(),
            "proprio": proprio.float().cpu().numpy(),
            "original_obs": original.float().cpu().numpy(),
            "obs45": obs45.float().cpu().numpy(),
        }

    def close(self):
        self._graph_step = None
        self._graph_render = None
        self._graph_settle = None
