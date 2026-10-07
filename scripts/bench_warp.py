#!/usr/bin/env python3
"""Time the MuJoCo-Warp tasks: env construction, reset and step (physics + 3 cameras).

    python scripts/bench_warp.py                         # a few tasks, 1 world
    python scripts/bench_warp.py --tasks all --worlds 1 4 16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjwenv import TASKS, make_batched_env  # noqa: E402

DEFAULT_TASKS = ["reach-xy-v3", "pick-place-v3", "drawer-open-v3", "bin-picking-redblue-v3",
                 "compo-pickplace-boxclose"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS, help="task names, or 'all'")
    p.add_argument("--worlds", nargs="*", type=int, default=[1])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--image-size", type=int, default=128)
    args = p.parse_args()
    tasks = list(TASKS) if args.tasks == ["all"] else args.tasks
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    print(f"device {device} ({torch.cuda.get_device_name() if device != 'cpu' else 'Warp CPU backend'})")
    print(f"{'task':28s} {'worlds':>6s} {'build s':>8s} {'reset ms':>9s} {'step ms':>8s} {'steps/s':>9s}")
    for task in tasks:
        for n in args.worlds:
            t0 = time.perf_counter()
            env = make_batched_env(task, n, image_size=args.image_size, seed=0)
            build = time.perf_counter() - t0
            t0 = time.perf_counter()
            env.reset()
            reset = time.perf_counter() - t0
            rng = np.random.default_rng(0)
            for _ in range(10):                       # warm-up
                env.step(rng.uniform(-1, 1, (n, 4)))
            t0 = time.perf_counter()
            for _ in range(args.steps):
                _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, (n, 4)))
                done = np.nonzero(term | trunc)[0]
                if done.size:
                    env.reset(done)
            step = (time.perf_counter() - t0) / args.steps
            print(f"{task:28s} {n:6d} {build:8.1f} {reset * 1e3:9.1f} {step * 1e3:8.2f} {n / step:9.0f}",
                  flush=True)
            env.close()


if __name__ == "__main__":
    main()
