#!/usr/bin/env python3
"""Run a benchmark configuration directly with the current Python environment."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile

import yaml

import wandb_restart
from sweep_config import (
    _is_sequential_schema, apply_filters, apply_pretrained_encoder,
    expand_dreamer_jobs, expand_jobs, expand_sequential_jobs, parse_filters,
)

ROOT = Path(__file__).resolve().parents[1]
ENCODER_VARIANTS = ('scratch', 'pretrained')
SMOKE_LOGDIR = Path(tempfile.gettempdir()) / 'cclb-smoke'  # outside the repository; removed per run
SEED_STEPS = 1250  # TD-MPC2/PWM warm-up: max(1000, 5 * 250-step episodes)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', choices=('tdmpc2', 'pwm', 'dreamerv3'))
    parser.add_argument('config', help='YAML filename in the baseline sweep directory, or a project-relative path.')
    parser.add_argument('--dry-run', action='store_true', help='Print runnable Python commands without writing files.')
    parser.add_argument('--filter', action='append', default=[], help='Select matrix jobs, e.g. method=regular,taskset=reach.')
    parser.add_argument('--seeds', help='Comma-separated seeds overriding the configuration.')
    parser.add_argument('--taskset', help='Comma-separated tasksets to run, e.g. reach,grasp (default: all).')
    parser.add_argument('--method', help='Comma-separated methods to run, e.g. er,ewc (default: all).')
    parser.add_argument('--max-jobs', type=int, help='Run at most this many jobs, sequentially.')
    parser.add_argument('--rerun-failed', action='store_true', help='Skip runs whose log directory contains done.flag.')
    parser.add_argument('--wandb-entity', default=None, help='Optional W&B account/team; otherwise use the W&B environment or login default.')
    parser.add_argument('--encoder', help='Comma-separated encoder variants: scratch, pretrained '
                        '(default: the configuration\'s encoder.variants, else scratch).')
    parser.add_argument('--pretrained-encoder', help='Encoder checkpoint, relative to the project root; overrides '
                        'encoder.checkpoint. Without --encoder it selects the pretrained variant only.')
    parser.add_argument('--freeze-encoder', dest='freeze_encoder', action='store_true',
                        help='Freeze the pretrained encoder (default).')
    parser.add_argument('--no-freeze-encoder', dest='freeze_encoder', action='store_false',
                        help='Fine-tune the pretrained encoder.')
    parser.set_defaults(freeze_encoder=True)
    parser.add_argument('--smoke', action='store_true',
                        help='Short end-to-end runs for testing an installation: tiny budgets, seed 0 unless '
                        '--seeds is given, W&B offline unless WANDB_MODE is set, no replay episodes, videos or '
                        f'periodic checkpoints. Remaining outputs go to {SMOKE_LOGDIR}/ and are deleted after '
                        'each successful run.')
    parser.add_argument('--smoke-steps', type=int, default=3000, help='Environment steps per task with --smoke.')
    args = parser.parse_args(argv)
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error('--max-jobs must be positive')
    if args.baseline == 'dreamerv3' and (args.pretrained_encoder or args.encoder):
        parser.error('--encoder and --pretrained-encoder are supported by TD-MPC2 and PWM only')
    if args.smoke and args.baseline != 'dreamerv3' and args.smoke_steps <= SEED_STEPS:
        parser.error(f'--smoke-steps must exceed the {SEED_STEPS}-step warm-up so training updates run')
    return args


def split_list(text):
    return [item.strip() for item in text.split(',') if item.strip()] if text else []


def resolve_config(baseline, config):
    path = Path(config).expanduser()
    if path.is_absolute():
        return path
    if len(path.parts) == 1:
        return ROOT / 'baselines' / baseline / 'sweep' / path
    return ROOT / path


def select_jobs(args):
    with resolve_config(args.baseline, args.config).open() as stream:
        cfg = yaml.safe_load(stream)
    if args.method or args.taskset:
        methods = split_list(args.method) or list(dict.fromkeys(c['method'] for c in cfg['matrix']))
        tasksets = split_list(args.taskset)
        for method in methods:
            if method not in cfg['methods']:
                raise ValueError(f'Unknown method {method!r}; choose from: ' + ', '.join(cfg['methods']))
        for taskset in tasksets:
            if taskset not in cfg['tasksets']:
                raise ValueError(f'Unknown taskset {taskset!r}; choose from: ' + ', '.join(cfg['tasksets']))
        matrix = []
        for method in methods:
            defined = cfg['methods'][method].get('tasksets')
            selected = (tasksets
                        or [t for c in cfg['matrix'] if c['method'] == method for t in c['tasksets']]
                        or list(defined or cfg['tasksets']))
            missing = [t for t in selected if defined is not None and t not in defined]
            if missing:
                raise ValueError(f'Method {method!r} has no settings for: ' + ', '.join(missing))
            matrix.append({'method': method, 'tasksets': selected})
        cfg['matrix'] = matrix
    if args.smoke:
        cfg['logdir_root'] = f'{SMOKE_LOGDIR}/{args.baseline}'
    if args.baseline == 'dreamerv3':
        expand = expand_dreamer_jobs
    else:
        expand = expand_sequential_jobs if _is_sequential_schema(cfg) else expand_jobs
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else ([0] if args.smoke else None)
    jobs = expand_encoder_variants(args, cfg, expand(cfg, seeds_override=seeds))
    if args.smoke:
        apply_smoke(args.baseline, jobs, args.smoke_steps)
    jobs = apply_filters(jobs, parse_filters(args.filter))
    if args.rerun_failed:
        jobs = [j for j in jobs if not (ROOT / j['logdir'] / 'done.flag').exists()]
    if args.max_jobs is not None:
        jobs = jobs[:args.max_jobs]
    if args.wandb_entity:
        for job in jobs:
            job['wandb_entity'] = args.wandb_entity
    return jobs


def expand_encoder_variants(args, cfg, jobs):
    """Repeat each job per encoder variant; pretrained runs get a `_preEncFr` (frozen) or `_preEnc` suffix."""
    encoder_cfg = cfg.get('encoder') or {}
    variants = (split_list(args.encoder)
                or (['pretrained'] if args.pretrained_encoder else [])
                or list(encoder_cfg.get('variants', ['scratch'])))
    for variant in variants:
        if variant not in ENCODER_VARIANTS:
            raise ValueError(f'Unknown encoder variant {variant!r}; choose from: ' + ', '.join(ENCODER_VARIANTS))
    checkpoint = args.pretrained_encoder or encoder_cfg.get('checkpoint')
    expanded = []
    for variant in dict.fromkeys(variants):
        batch = copy.deepcopy(jobs)
        for job in batch:
            job['encoder'] = variant
        if variant == 'pretrained':
            if not checkpoint:
                raise ValueError('No encoder checkpoint: set encoder.checkpoint in the configuration '
                                 'or pass --pretrained-encoder')
            batch = apply_pretrained_encoder(batch, str(Path(checkpoint).expanduser()), args.freeze_encoder)
            for job in batch:
                job['encoder_checkpoint'] = checkpoint
        expanded += batch
    return expanded


def apply_smoke(baseline, jobs, steps):
    """Shrink budgets so warm-up, updates, evaluation and task switches each run briefly, and save as
    little as possible: no replay episodes, videos or periodic checkpoints. Files a method needs while
    running, such as end-of-task models, are still written."""
    every = max(steps // 3, 1)
    for job in jobs:
        n = len(job.get('tasks', [None]))
        extra = list(job.get('extra_args', []))
        if baseline == 'dreamerv3':
            job['steps'] = [steps] * n
            job['dataset_sizes'] = [steps] * n
            job['extra_args'] = extra + [
                '--eval_every', str(every), '--log_every', str(every), '--prefill', str(min(500, steps // 4)),
                '--no-eval-prev-video']
        elif job['mode'] == 'single':
            job.update(steps=steps, buffer_size=steps, eval_freq=every, eval_episodes=1, save_freq=0,
                       save_episodes=False)
            job['extra_args'] = extra + ['save_video=false', 'save_agent=false']
        else:
            job.update(task_steps=[steps] * n, task_buffer_sizes=[steps] * n, eval_freq=every, eval_episodes=1)
            if job.get('save_freq') is not None:
                job['save_freq'] = 0
            job['extra_args'] = extra + ['--no-save-video']


def is_sequential(baseline, job):
    """Sequential training scripts (all DreamerV3 runs use one, single-task ones included)."""
    return baseline == 'dreamerv3' or job['mode'] == 'sequential'


def check_checkpoints(jobs):
    missing = sorted({j['encoder_checkpoint'] for j in jobs if j.get('encoder_checkpoint')
                      and not (ROOT / Path(j['encoder_checkpoint']).expanduser()).exists()})
    return missing


def value(item):
    if isinstance(item, (list, dict, bool)) or item is None:
        return json.dumps(item, separators=(',', ':'))
    return str(item)


def build_command(baseline, job, python='python', run_id=None):
    """Build an argv list; all relative paths are interpreted from ROOT."""
    base = Path('baselines') / baseline
    package = base if baseline == 'dreamerv3' else base / 'tdmpc2'
    entry = Path(job['entry'])
    if baseline != 'dreamerv3' and entry.parts[0] == 'tdmpc2':
        entry = Path(*entry.parts[1:])
    cmd = [python, str(package / entry)]
    if baseline != 'dreamerv3' and job['mode'] == 'single':
        keys = ('task', 'seed', 'steps', 'episodic', 'model_size', 'batch_size',
                'buffer_size', 'horizon', 'mpc', 'eval_freq', 'eval_episodes',
                'save_freq', 'image_size', 'max_episode_steps', 'action_repeat',
                'cameras', 'save_episodes', 'prune_every_n_episodes', 'buffer_storage_device')
        overrides = {key: job[key] for key in keys}
        overrides.update(work_dir=job['logdir'], exp_name=job['run_name'],
                         wandb_project=job['wandb_project'], wandb_run_name=job['run_name'],
                         wandb_group=job['wandb_group'], wandb_tags=job['wandb_tags'],
                         enable_wandb=True)
        if job.get('wandb_entity'):
            overrides['wandb_entity'] = job['wandb_entity']
        if run_id:
            # Quoted so Hydra keeps IDs such as 25640038 or 123e4567 as strings.
            overrides['wandb_run_id'] = f"'{run_id}'"
        cmd += [key + '=' + value(val) for key, val in overrides.items()]
        cmd += ['hydra.run.dir=' + job['logdir'], 'hydra.job.chdir=false']
    else:
        flags = {'tasks': job['tasks'], 'logdir': job['logdir'], 'seed': job['seed'],
                 'wandb-project': job['wandb_project'], 'wandb-run-name': job['run_name'],
                 'wandb-group': job['wandb_group'], 'wandb-tags': job['wandb_tags']}
        if job.get('wandb_entity'):
            flags['wandb-entity'] = job['wandb_entity']
        if run_id:
            flags['wandb-run-id'] = run_id
        if baseline == 'dreamerv3':
            flags.update(configs=job['configs'], **{
                'task-steps': job['steps'], 'dataset-sizes': job['dataset_sizes']})
        else:
            flags.update({'task-steps': job['task_steps'], 'task-buffer-sizes': job['task_buffer_sizes'],
                          'logger': 'wandb', 'exp-name': job['run_name']})
            for key in ('model_size', 'batch_size', 'horizon', 'eval_freq', 'eval_episodes',
                        'save_freq', 'image_size', 'max_episode_steps', 'action_repeat', 'cameras'):
                if job.get(key) is not None:                     # save_freq: PWM only
                    flags[key.replace('_', '-')] = job[key]
        for key, val in flags.items():
            cmd.append('--' + key)
            if isinstance(val, list):
                cmd.extend(value(v) for v in val)
            else:
                cmd.append(value(val))
        if baseline == 'dreamerv3':
            cmd += ['--eval-prev-video', '--skip-config-check']
    cmd += [value(arg) for arg in job.get('extra_args', [])]
    return cmd


def pythonpath(baseline):
    package = Path('baselines') / baseline
    if baseline != 'dreamerv3':
        package /= 'tdmpc2'
    return os.pathsep.join(('.', str(package)))


def main(argv=None):
    args = parse_args(argv)
    jobs = select_jobs(args)
    if not jobs:
        print('No jobs match the selection.', file=sys.stderr)
        return 1
    missing = check_checkpoints(jobs)
    if missing:
        message = 'Encoder checkpoint not found: ' + ', '.join(missing)
        if not args.dry_run:
            print(message + ' (see README: pretrained encoder).', file=sys.stderr)
            return 2
        print('# Warning: ' + message, flush=True)
    extra_env = {}
    if args.smoke and not os.environ.get('WANDB_MODE'):
        extra_env['WANDB_MODE'] = 'offline'
    print('# Run these commands from the project root.', flush=True)
    for job in jobs:
        run_id = None
        logdir = ROOT / job['logdir']
        if not args.dry_run:
            if args.smoke:
                shutil.rmtree(logdir, ignore_errors=True)  # smoke runs always start fresh
            logdir.mkdir(parents=True, exist_ok=True)
            id_file = logdir / 'wandb_run_id.txt'
            run_id = id_file.read_text().strip() if id_file.exists() else ''
            if (run_id and is_sequential(args.baseline, job) and not args.smoke
                    and not (logdir / 'done.flag').exists()):
                # Started again after an interruption: training resumes at the first unfinished
                # task, and W&B continues in a new run holding the completed tasks' history.
                run_id = wandb_restart.restart(job, run_id, logdir)
            run_id = run_id or secrets.token_hex(4)
            id_file.write_text(run_id + '\n')
        cmd = build_command(args.baseline, job, run_id=run_id)
        prefix = pythonpath(args.baseline)
        env_text = ''.join(f'{k}={shlex.quote(v)} ' for k, v in extra_env.items())
        print(env_text + 'PYTHONPATH=' + shlex.quote(prefix) + '${PYTHONPATH:+:$PYTHONPATH} ' + shlex.join(cmd),
              flush=True)
        if args.dry_run:
            continue
        env = {**os.environ, **extra_env}
        env['PYTHONPATH'] = prefix + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        cmd[0] = sys.executable
        proc = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
        if proc.returncode:
            if args.smoke:
                print(f'# Smoke run failed; its outputs are kept in {logdir}', file=sys.stderr, flush=True)
            return proc.returncode
        if args.smoke:
            shutil.rmtree(logdir, ignore_errors=True)
            print(f'# Smoke run passed: {job["run_name"]}', flush=True)
        else:
            (logdir / 'done.flag').touch()
    return 0


if __name__ == '__main__':
    sys.exit(main())
