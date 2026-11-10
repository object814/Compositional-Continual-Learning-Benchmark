"""Roll out the scripted experts: success rates, videos and demonstrations.

Run from the repository root:

    python -m metaworld.expert_policy.rollout                     # every task, 10 episodes each
    python -m metaworld.expert_policy.rollout --tasks reach-xy-v3 compo-pickplace-block --episodes 20
    python -m metaworld.expert_policy.rollout --videos 3 --npz    # 3 videos per task, every episode as .npz

The environment is the one the baselines train on: gymnasium "Meta-World/MT1"
with ProprioMultiImageObsWrapper (topview, front and gripperPOV cameras), each
action repeated twice and at most 250 actions per episode, ending early on
success. `--sim warp` uses the MuJoCo-Warp version instead (main branch
only). Rendering needs MUJOCO_GL (osmesa or egl), which the conda environment
and the Docker image set.

Writes to --out (default outputs/expert_rollouts/<sim>/):
  summary.json               per task: success rate, episode length, return
  <task>/episode_<k>.mp4     the first --videos episodes: the observation
                             cameras side by side, one frame per action
  <task>/episode_<k>.npz     with --npz, every episode:
                               proprio (T+1, 7), image (T+1, H, W, 3 * cameras),
                               original_obs (T+1, 39): observations, the first
                                 at reset and one after each action
                               action (T, 4)
                               reward (T,): raw reward in [-1, 1] per simulator
                                 step, summed over the repeats of each action
                               success (T,): the task was solved during the action
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from metaworld.expert_policy import EXPERT_TASKS, make_expert

ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ["topview", "front", "gripperPOV"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=EXPERT_TASKS, metavar="TASK",
                   help="Tasks to roll out (default: all): " + ", ".join(EXPERT_TASKS))
    p.add_argument("--episodes", type=int, default=10, help="Episodes per task.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-episode-steps", type=int, default=250, help="Actions per episode at most.")
    p.add_argument("--action-repeat", type=int, default=2, help="Simulator steps per action.")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--cameras", nargs="+", default=CAMERAS)
    p.add_argument("--videos", type=int, default=1, help="Episodes per task saved as mp4 (0: none).")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--npz", action="store_true", help="Save every episode as .npz.")
    p.add_argument("--sim", choices=("cpu", "warp"), default="cpu",
                   help="cpu: MuJoCo (metaworld/); warp: MuJoCo-Warp (mjwenv/, main branch).")
    p.add_argument("--out", type=Path, help="Output directory (default: outputs/expert_rollouts/<sim>).")
    args = p.parse_args(argv)
    args.tasks = [t.removeprefix("metaworld_") for t in args.tasks]
    unknown = [t for t in args.tasks if t not in EXPERT_TASKS]
    if unknown:
        p.error(f"no expert for {', '.join(unknown)}; available: {', '.join(EXPERT_TASKS)}")
    args.out = args.out or ROOT / "outputs" / "expert_rollouts" / args.sim
    return args


def make_env(task, args):
    sim_steps = args.max_episode_steps * args.action_repeat
    if args.sim == "warp":
        try:
            from mjwenv import MetaWorldWarpEnv
        except ModuleNotFoundError as e:
            if e.name != "mjwenv":
                raise
            raise SystemExit("--sim warp needs the main branch (mjwenv/)") from None
        return MetaWorldWarpEnv(task, image_size=args.image_size, cameras=args.cameras,
                                max_episode_steps=sim_steps, seed=args.seed)
    import gymnasium as gym
    from metaworld.wrappers import ProprioMultiImageObsWrapper
    np.random.seed(args.seed)   # Meta-World draws a task's 50 layouts from numpy's global state
    env = gym.make("Meta-World/MT1", env_name=task, render_mode="rgb_array", max_episode_steps=sim_steps)
    return ProprioMultiImageObsWrapper(env, image_height=args.image_size, image_width=args.image_size,
                                       camera_names=args.cameras)


def run_episode(env, task, seed, args):
    """One expert episode; returns its record (observations, actions, ...) and summary."""
    expert = make_expert(task)
    obs, _ = env.reset(seed=seed)
    observations, actions, rewards, successes = [obs], [], [], []
    tuned_return, solved = 0.0, False
    for _ in range(args.max_episode_steps):
        action = np.asarray(expert.get_action(obs["original_obs"]), dtype=np.float32)
        reward, success, done = 0.0, False, False
        for _ in range(args.action_repeat):
            obs, r, terminated, truncated, info = env.step(action)
            reward += float(r)
            tuned_return += (float(r) - 1.0) / 2.0   # the training reward (RewardTuningWrapperV2)
            success = success or bool(info.get("success", 0.0))
            done = bool(terminated or truncated)
            if done:
                break
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        successes.append(success)
        solved = solved or success
        if done:
            break
    record = {
        "proprio": np.stack([o["proprio"] for o in observations]).astype(np.float32),
        "image": np.stack([o["image"] for o in observations]).astype(np.uint8),
        "original_obs": np.stack([o["original_obs"] for o in observations]).astype(np.float32),
        "action": np.stack(actions),
        "reward": np.asarray(rewards, dtype=np.float32),
        "success": np.asarray(successes, dtype=bool),
    }
    return record, {"success": solved, "length": len(actions), "return": tuned_return}


def save_video(path, images, n_cameras, fps):
    import imageio.v2 as imageio
    frames = [np.concatenate([img[..., 3 * i:3 * i + 3] for i in range(n_cameras)], axis=1) for img in images]
    imageio.mimwrite(path, frames, fps=fps, quality=8, macro_block_size=1)


def main(argv=None):
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"# {len(args.tasks)} tasks x {args.episodes} episodes, sim={args.sim}, "
          f"action_repeat={args.action_repeat}, max {args.max_episode_steps} actions -> {args.out}", flush=True)
    results = {}
    for task in args.tasks:
        start = time.time()
        env = make_env(task, args)
        task_dir = args.out / task
        task_dir.mkdir(exist_ok=True)
        episodes = []
        for k in range(args.episodes):
            record, summary = run_episode(env, task, args.seed * 1_000_000 + k, args)
            episodes.append(summary)
            if k < args.videos:
                save_video(task_dir / f"episode_{k:03d}.mp4", record["image"], len(args.cameras), args.fps)
            if args.npz:
                np.savez_compressed(task_dir / f"episode_{k:03d}.npz", **record)
        env.close()
        results[task] = {
            "episodes": len(episodes),
            "success_rate": float(np.mean([e["success"] for e in episodes])),
            "mean_length": float(np.mean([e["length"] for e in episodes])),
            "mean_return": float(np.mean([e["return"] for e in episodes])),
            "best_return": float(max(e["return"] for e in episodes)),
        }
        r = results[task]
        print(f"{task:<30} success {r['success_rate']:6.1%}   length {r['mean_length']:5.1f}   "
              f"return {r['mean_return']:8.2f} (best {r['best_return']:8.2f})   {time.time() - start:5.1f}s",
              flush=True)
    rates = [r["success_rate"] for r in results.values()]
    print(f"{'mean over tasks':<30} success {np.mean(rates):6.1%}")
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (args.out / "summary.json").write_text(json.dumps({"config": config, "tasks": results}, indent=2) + "\n")
    print(f"# summary: {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
