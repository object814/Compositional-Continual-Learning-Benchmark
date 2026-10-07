#!/usr/bin/env python3
"""Compare the MuJoCo-Warp tasks (mjwenv) with the CPU Meta-World tasks they port.

For each task, the CPU env (gymnasium "Meta-World/MT1") and a one-world warp
env start from the same layout and hand start, and receive the same actions
from a scripted controller that grasps, carries and places the objects (or
from smoothed random actions). Reported per task:

  reset   largest difference of the 45-d reset observation
  pos     largest difference of the position entries of the observation
  reward  largest difference of the reward (raw scale, [-1, 1])
  success steps where the success flags disagree / steps where CPU succeeded

Modes:
  logic (default)  the warp world takes the CPU state after every step in place
                   of its own physics, so observation, reward and success are
                   computed from identical states: differences beyond float32
                   rounding are porting errors.
  sync             the CPU state is copied in before every step and the warp
                   physics advances it: one-step physics differences.
  free             both run on their own: accumulated physics divergence.

The CPU env uses the collision flags of the warp env (MuJoCo 3.14 with
multi-contact and native CCD disabled, as in MuJoCo 3.1.2). Runs on a GPU, or
slowly on the CPU (Warp's CPU backend) when there is none.

    python scripts/check_warp_parity.py                       # all tasks
    python scripts/check_warp_parity.py --tasks grasp-v3 box-close-v3 --episodes 5
    python scripts/check_warp_parity.py --mode free --images parity_images
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium  # noqa: E402
import mujoco  # noqa: E402
import metaworld  # noqa: E402,F401  (registers Meta-World/MT1)
import mujoco_warp as mjw  # noqa: E402
from mjwenv import TASKS, make_batched_env  # noqa: E402

POS_DIMS = np.r_[0:18, 21:39, 42:45]          # observation entries that are positions / quats
STATE_KEYS = ("qpos", "qvel", "mocap_pos", "mocap_quat", "qacc_warmstart")


# ---------------------------------------------------------------- controller
def move(cur, to, gain=10.0):
    return np.clip(gain * (np.asarray(to) - np.asarray(cur)), -1.0, 1.0)


def pick_place(u, obj, goal, release=True, carry_z=0.15, max_steps=(80, 50, 15, 25, 100, 40)):
    """Generator of actions: above `obj()` -> descend -> close -> lift -> carry over
    `goal()` -> lower onto it -> (release and rise). `obj`/`goal` are callables."""
    up = np.array([0.0, 0.0, 1.0])
    k = 0
    while k < max_steps[0] and np.linalg.norm(u.tcp_center[:2] - obj()[:2]) > 0.01:
        yield np.r_[move(u.tcp_center, obj() + 0.1 * up), -1.0]; k += 1
    k = 0
    while k < max_steps[1] and abs(u.tcp_center[2] - obj()[2]) > 0.008:
        yield np.r_[move(u.tcp_center, obj()), -1.0]; k += 1
    for _ in range(max_steps[2]):
        yield np.r_[move(u.tcp_center, obj()), 1.0]
    for _ in range(max_steps[3]):
        yield np.r_[move(u.tcp_center, obj() + carry_z * up), 1.0]
    k = 0
    while k < max_steps[4] and np.linalg.norm(obj()[:2] - goal()[:2]) > 0.01:
        g = goal()
        yield np.r_[move(u.tcp_center, [g[0], g[1], max(g[2], carry_z)]), 1.0]; k += 1
    k = 0
    while k < max_steps[5] and np.linalg.norm(obj() - goal()) > 0.01:
        yield np.r_[move(u.tcp_center, u.tcp_center + goal() - obj()), 1.0]; k += 1
    if release:
        for _ in range(15):
            yield np.r_[0.0, 0.0, 0.0, -1.0]
        for _ in range(15):
            yield np.r_[0.0, 0.0, 1.0, -1.0]


def open_drawer(u, info):
    handle = lambda: u.data.body("drawer_link").xpos - np.array([0.0, 0.15, 0.0])
    yield from pick_place(u, handle, handle, release=False, carry_z=0.0, max_steps=(80, 50, 12, 0, 0, 0))
    k = 0
    while k < 100 and not info.get("drawer_opened", info.get("success", 0.0)):
        yield np.r_[move(u.tcp_center, handle() - np.array([0.0, 0.3, 0.0])), 1.0]; k += 1
    for _ in range(20):
        yield np.r_[move(u.tcp_center, u.tcp_center + np.array([0.0, 0.05, 0.15])), -1.0]


def scripted(task, u, info):
    """Action generator for `task`, reading the CPU env `u` (privileged state)."""
    body = lambda name: (lambda: u.data.body(name).xpos.copy())
    target = lambda: np.asarray(u._target_pos, dtype=float).copy()
    if task.startswith("reach"):
        def gen():
            while True:
                yield np.r_[move(u.tcp_center, target()), -1.0]
        return gen()
    if task == "grasp-v3":
        return pick_place(u, body("obj"), lambda: body("obj")() + [0.0, 0.0, 0.1], release=False)
    if task == "pick-place-v3":
        return pick_place(u, body("obj"), target, release=False)
    if task.startswith("pick-place-"):
        return pick_place(u, body("obj"), target)
    if task.startswith("bin-picking"):
        return pick_place(u, body("obj"), lambda: target() + [0.0, 0.0, 0.03], carry_z=0.2)
    if task == "drawer-open-v3":
        return open_drawer(u, info)
    handle = lambda: u.data.geom("BoxHandleGeom").xpos.copy()
    if task == "box-close-v3":
        return pick_place(u, handle, lambda: target() + [0.0, 0.0, 0.02], carry_z=0.25)
    if task == "compo-pickplace-block":
        def gen():
            yield from pick_place(u, body("obj1"), target)
            yield from pick_place(u, body("obj2"), lambda: body("obj1")() + [0.0, 0.0, 0.03])
        return gen()
    if task == "compo-draweropen-pickplace":
        def gen():
            yield from open_drawer(u, info)
            yield from pick_place(u, body("obj"), target, carry_z=0.2)
        return gen()
    if task == "compo-pickplace-boxclose":
        def gen():
            yield from pick_place(u, body("obj"), target, carry_z=0.25)
            yield from pick_place(u, handle, lambda: np.asarray(u._lid_target_pos) + [0.0, 0.0, 0.02], carry_z=0.25)
        return gen()
    raise ValueError(task)


class SmoothRandom:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.a = np.zeros(4)

    def __next__(self):
        self.a = 0.8 * self.a + 0.2 * self.rng.normal(0.0, 1.5, 4)
        return np.clip(self.a, -1.0, 1.0)


# ------------------------------------------------------------------ parity
def make_cpu_env(task, steps):
    env = gymnasium.make("Meta-World/MT1", env_name=task, max_episode_steps=steps + 10)
    u = env.unwrapped
    u.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    u.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    return env, u


def sync_state(wenv, u, prev_obs=True):
    """Copy the CPU state into warp world 0."""
    t = wenv._t
    for k in STATE_KEYS:
        src = np.array(getattr(u.data, k), dtype=np.float32)
        if k == "mocap_quat":
            src[wenv.hand_mocap] = wenv.hand_quat.cpu().numpy()
        n = src.shape[0]
        t[k][0, :n] = torch.as_tensor(src, device=wenv.device)
    if prev_obs:
        wenv.prev21[0] = torch.as_tensor(np.asarray(u._prev_obs, dtype=np.float32), device=wenv.device)


def inject_physics(wenv, u):
    """Replace the warp env's physics step by the CPU's post-step state (logic mode)."""
    def physics():
        sync_state(wenv, u, prev_obs=False)
        mjw.forward(wenv.m, wenv.d)
    wenv._physics = physics


def save_reset_images(path, u, warp_obs, cameras, size):
    """Reset frame of every camera: CPU (OpenGL) on top, warp (ray tracer) below."""
    from PIL import Image
    renderer = mujoco.Renderer(u.model, height=size, width=size)
    cpu = []
    for cam in cameras:
        renderer.update_scene(u.data, camera=cam)
        cpu.append(renderer.render().copy())
    renderer.close()
    img = warp_obs["image"][0]
    warp = [img[..., 3 * i:3 * i + 3] for i in range(len(cameras))]
    Image.fromarray(np.concatenate([np.concatenate(cpu, 1), np.concatenate(warp, 1)], 0)).save(path)


def run_task(task, args):
    env, u = make_cpu_env(task, args.steps)
    wenv = make_batched_env(task, 1, device=args.device, render=bool(args.images), seed=args.seed)
    if args.mode == "logic":
        inject_physics(wenv, u)
    # Record the layouts the CPU env draws at each reset: for the bin and box
    # tasks the first is the throwaway layout the hand settles with.
    layouts = []
    for name in ("_sample_non_overlapping_bins", "_sample_non_overlapping_box_lid"):
        if hasattr(u, name):
            def record(sample=getattr(u, name)):
                out = sample()
                layouts.append(np.array(u._last_rand_vec, dtype=np.float64))
                return out
            setattr(u, name, record)
    stats = dict(reset=0.0, pos=0.0, vel=0.0, reward=0.0, success_mismatch=0, cpu_success=0,
                 warp_success=0, steps=0)
    for ep in range(args.episodes):
        layouts.clear()
        obs_cpu, _ = env.reset(seed=args.seed + ep)
        vec = np.asarray(u._last_rand_vec, dtype=np.float64)
        hand = np.asarray(u.hand_init_pos, dtype=np.float64)[None] if wenv.RANDOM_HAND_INIT else None
        settle = layouts[0][None] if wenv.SETTLE_LAYOUT else None
        reset_obs = wenv.reset(task_vecs=vec[None], hand_init=hand, settle_vecs=settle)
        if args.images and ep == 0:
            save_reset_images(Path(args.images) / f"{task}.png", u, reset_obs, wenv.cameras, wenv.image_size)
        obs_w = reset_obs["obs45"][0]
        stats["reset"] = max(stats["reset"], float(np.abs(obs_w - obs_cpu).max()))
        info = {}
        policy = scripted(task, u, info) if args.policy == "scripted" else SmoothRandom(args.seed + ep)
        for _ in range(args.steps):
            try:
                action = np.asarray(next(policy), dtype=np.float32)
            except StopIteration:
                action = np.zeros(4, dtype=np.float32)
            if args.mode in ("sync", "logic"):
                sync_state(wenv, u)
            obs_cpu, r_cpu, term_cpu, _, info_cpu = env.step(action)
            info.clear()
            info.update(info_cpu)
            obs_w, r_w, term_w, _, info_w = wenv.step(action[None])
            d = np.abs(obs_w["obs45"][0] - obs_cpu)
            stats["pos"] = max(stats["pos"], float(d[POS_DIMS].max()))
            stats["vel"] = max(stats["vel"], float(np.delete(d, POS_DIMS).max()))
            stats["reward"] = max(stats["reward"], abs(float(r_w[0]) - float(r_cpu)))
            s_cpu, s_w = bool(info_cpu["success"]), bool(info_w["success"][0])
            stats["success_mismatch"] += int(s_cpu != s_w)
            stats["cpu_success"] += int(s_cpu)
            stats["warp_success"] += int(s_w)
            stats["steps"] += 1
    env.close()
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--episodes", type=int, default=3, help="episodes (layouts) per task")
    p.add_argument("--steps", type=int, default=200, help="agent steps per episode")
    p.add_argument("--policy", choices=("scripted", "random"), default="scripted")
    p.add_argument("--mode", choices=("logic", "sync", "free"), default="logic")
    p.add_argument("--device", default=None, help="warp device (default: GPU if available, else cpu)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--images", metavar="DIR", help="save each task's reset frame, CPU above warp, to DIR/<task>.png")
    args = p.parse_args()
    if args.images:
        Path(args.images).mkdir(parents=True, exist_ok=True)

    print(f"{'task':28s} {'reset':>8s} {'pos':>8s} {'vel':>8s} {'reward':>8s}  success (mismatch / cpu / warp of steps)")
    for task in args.tasks:
        t0 = time.time()
        s = run_task(task, args)
        print(f"{task:28s} {s['reset']:8.1e} {s['pos']:8.1e} {s['vel']:8.1e} {s['reward']:8.1e}  "
              f"{s['success_mismatch']:3d} / {s['cpu_success']:3d} / {s['warp_success']:3d} of {s['steps']}"
              f"   ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
