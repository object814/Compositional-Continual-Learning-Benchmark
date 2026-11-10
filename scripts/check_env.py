#!/usr/bin/env python3
"""Check that the current Python environment can run every baseline.

Reports package versions, tests the GPU, imports every training entry point
(by running it with --help, exactly as the launcher would invoke it), and
builds, resets, renders and steps each benchmark task with MuJoCo-Warp. The
first run compiles the GPU kernels into .cache/warp, which takes a few minutes.

    python scripts/check_env.py               # everything
    python scripts/check_env.py --quick       # three tasks only
    python scripts/check_env.py --skip-tasks  # packages, GPU and entry points
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

PACKAGES = ['torch', 'tensordict', 'torchrl', 'numpy', 'mujoco', 'mujoco_warp', 'warp',
            'gymnasium', 'gym',
            'OpenGL', 'cv2', 'scipy', 'hydra',
            'omegaconf', 'ruamel.yaml', 'yaml', 'termcolor', 'wandb', 'tensorboard',
            'moviepy', 'imageio', 'cloudpickle', 'psutil', 'PIL', 'pandas', 'matplotlib', 'tqdm']

ENTRY_POINTS = {
    'tdmpc2': ['tdmpc2/train.py', 'sequential_train.py', 'er_training/sequential_train_er.py',
               'ewc_training/sequential_train_ewc.py', 'packnet_training/sequential_train_packnet.py'],
    'pwm': ['tdmpc2/train.py', 'sequential/sequential_train.py', 'sequential/sequential_train_progressive.py'],
    'dreamerv3': ['dreamer_sequential.py', 'er_training/dreamer_sequential_er.py',
                  'ewc_training/dreamer_sequential_ewc.py', 'packnet_training/dreamer_sequential_packnet.py'],
}

TASKS = ['bin-picking-redblue-v3', 'bin-picking-yellowblue-v3', 'bin-picking-redpurple-v3',
         'bin-picking-yellowpurple-v3', 'box-close-v3', 'drawer-open-v3', 'grasp-v3',
         'pick-place-v3', 'pick-place-redblock-v3', 'pick-place-greenblock-v3', 'reach-v3',
         'reach-xy-v3', 'reach-xz-v3', 'reach-yz-v3', 'reach-xyz-v3', 'compo-draweropen-pickplace',
         'compo-pickplace-block', 'compo-pickplace-boxclose']
QUICK_TASKS = ['reach-v3', 'pick-place-redblock-v3', 'compo-pickplace-boxclose']


def report(ok, name, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f'  {detail}' if detail else ''), flush=True)
    return ok


def check_packages():
    print('Packages', flush=True)
    ok = True
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            ok &= report(True, name, getattr(module, '__version__', ''))
        except Exception as exc:  # noqa: BLE001 - report any import failure
            ok &= report(False, name, f'{type(exc).__name__}: {exc}')
    return ok


def check_gpu():
    print('GPU', flush=True)
    try:
        import torch
        if not torch.cuda.is_available():
            return report(False, 'CUDA', 'torch.cuda.is_available() is False')
        x = torch.randn(512, 512, device='cuda')
        float((x @ x).sum())
        return report(True, 'CUDA', f'{torch.cuda.get_device_name(0)}, torch CUDA {torch.version.cuda}')
    except Exception as exc:  # noqa: BLE001
        return report(False, 'CUDA', f'{type(exc).__name__}: {exc}')


def baseline_pythonpath(baseline):
    package = Path('baselines') / baseline
    if baseline != 'dreamerv3':
        package /= 'tdmpc2'
    return package


def check_entry_points(timeout):
    print('Training entry points (--help)', flush=True)
    ok = True
    for baseline, entries in ENTRY_POINTS.items():
        package = baseline_pythonpath(baseline)
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join(['.', str(package)])}
        for entry in entries:
            path = package / (entry[len('tdmpc2/'):] if baseline != 'dreamerv3' and entry.startswith('tdmpc2/') else entry)
            start = time.time()
            try:
                proc = subprocess.run([sys.executable, str(path), '--help'], cwd=ROOT, env=env,
                                      capture_output=True, text=True, timeout=timeout)
                passed = proc.returncode == 0
                lines = [line.strip() for line in (proc.stderr or proc.stdout).splitlines() if line.strip()]
                errors = [line for line in lines if 'Error' in line or 'error' in line]
                shown = ([errors[-1]] if errors else []) + (lines[-1:] if lines and lines[-1] not in errors else [])
                detail = f'{time.time() - start:.0f}s' if passed else (' | '.join(shown) or f'exit {proc.returncode}')
            except subprocess.TimeoutExpired:
                passed, detail = False, f'timed out after {timeout}s'
            ok &= report(passed, f'{baseline}: {path}', detail)
    return ok


def task_worker(tasks):
    """Runs in a subprocess, so a crash in one task is reported instead of ending the check."""
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from mjwenv import MetaWorldWarpEnv

    results = []
    for task in tasks:
        start = time.time()
        try:
            env = MetaWorldWarpEnv(task, image_size=128, max_episode_steps=500)
            obs, _ = env.reset(seed=0)
            for _ in range(5):
                obs, reward, *_ = env.step(env.action_space.sample())
            image = obs['image']
            assert image.shape == (128, 128, 9), f'image shape {image.shape}'
            assert image.std() > 0, 'blank image'
            assert np.isfinite(reward), 'non-finite reward'
            env.close()
            results.append((task, True, f'{time.time() - start:.1f}s'))
        except Exception as exc:  # noqa: BLE001
            results.append((task, False, f'{type(exc).__name__}: {exc}'))
    print('RESULTS ' + json.dumps(results))


def check_tasks(tasks, timeout):
    print('Benchmark tasks (MuJoCo-Warp)', flush=True)
    try:
        proc = subprocess.run([sys.executable, __file__, '--worker', ','.join(tasks)], cwd=ROOT,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return report(False, 'tasks', f'timed out after {timeout}s')
    lines = [line for line in proc.stdout.splitlines() if line.startswith('RESULTS ')]
    if not lines:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        return report(False, 'tasks', tail[-1] if tail else f'exit {proc.returncode}')
    ok = True
    for task, passed, detail in json.loads(lines[-1][len('RESULTS '):]):
        ok &= report(passed, task, detail)
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--quick', action='store_true', help='Check three tasks instead of all 18.')
    parser.add_argument('--skip-tasks', action='store_true', help='Skip building the benchmark tasks.')
    parser.add_argument('--skip-gpu', action='store_true', help='Skip the CUDA check (e.g. on a login node).')
    parser.add_argument('--timeout', type=int, default=1800, help='Seconds allowed per subprocess.')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        task_worker(args.worker.split(','))
        return 0

    print(f'Python {sys.version.split()[0]} at {sys.executable}', flush=True)
    results = {'packages': check_packages()}
    if not args.skip_gpu:
        results['gpu'] = check_gpu()
    results['entry points'] = check_entry_points(args.timeout)
    if not args.skip_tasks:
        results['tasks'] = check_tasks(QUICK_TASKS if args.quick else TASKS, args.timeout)

    print('\nSummary', flush=True)
    for name, passed in results.items():
        report(passed, name)
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
