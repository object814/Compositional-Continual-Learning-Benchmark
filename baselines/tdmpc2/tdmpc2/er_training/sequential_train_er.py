"""Sequential TD-MPC2 training with Experience Replay (ER) on a shared backbone.

Architecture (matches `sequential_train.py`):
  - **Backbone** = encoder + dynamics — shared across tasks. At each task
    boundary the new agent loads the previous task's `backbone.pt` (weights
    + Adam moments), so the shared representation carries over.
  - **Per-task heads** = reward / Qs / termination / pi + RunningScale + MPC
    `_prev_mean` — reset to a fresh random init at the start of every task
    and saved to `task_modules.pt` at task end. Heads from prior tasks
    live on disk only — they are never loaded into the training agent.

ER additions over the naive script:
  1. During each task's training, every completed episode is appended to a
     Python-side log of episode TensorDicts.
  2. At the end of a task, the log is reservoir-sampled down to a budget
     (in transitions, computed as `er_buffer_ratio * cfg.steps`) and saved
     to disk at `<task_logdir>/er_episodes.pt`.
  3. Before training a subsequent task, episodes from every previous task's
     dump are loaded and inserted into the new (otherwise empty) replay
     buffer via `Buffer.add` BEFORE training begins. The buffer therefore
     contains a mix of old-task and (eventually) new-task transitions.

Notes:
  - Reward / Q heads are fresh per task, so ER transitions from previous
    tasks still teach the encoder/dynamics a richer representation
    (consistency / value losses operate through the backbone) without
    asking a single reward head to fit multiple tasks' reward functions —
    the heads themselves are per-task by construction.

Checkpoint layout per task:
    <logdir>/task{N}_<task>/
        backbone.pt            -- encoder + dynamics weights + Adam moments
        task_modules.pt        -- reward / Qs / termination / pi / scale +
                                  head-slice Adam moments + pi_optim state
        er_episodes.pt         -- reservoir-sampled episodes for ER prefill
        models/{step}.pt       -- periodic whole-agent snapshots (if save_freq > 0)
        eval_video/            -- TD-MPC2 video recorder output

Example:
    python sequential_train_er.py \\
        --tasks metaworld_drawer-open-v3 metaworld_pick-place-v3 \\
        --task-steps 200000 500000 \\
        --logdir ../logdir/seq_tdmpc2_er/run0 \\
        --er-buffer-ratio 0.025 \\
        --wandb-project Metaworld_Tdmpc2_Sequential_ER
"""

import os

os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"
os.environ["EGL_LOG_LEVEL"] = "fatal"

import warnings
warnings.filterwarnings("ignore", message="Constant.*may be too high")
warnings.filterwarnings("ignore", message=".*Please upgrade to Gymnasium.*")
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
warnings.filterwarnings('ignore')

import argparse
import gc
import random
import shutil
import sys
from pathlib import Path
from time import time

import torch
from termcolor import colored
from tqdm.auto import tqdm

# Make `tdmpc2/` importable so we can reuse logger / progress helpers.
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))

from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2

# Reuse helpers from the naive sequential script.
from sequential_train import (
    build_task_cfg,
    save_progress,
    load_progress,
    save_wandb_run_id,
    SequentialLogger,
    eval_on_env,
    _to_td,
)


CONFIG_PATH = _PARENT / 'config.yaml'

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


# ==========================================================================
# ER episode dump / load
# ==========================================================================

def _episode_transitions(td):
    """Number of real transitions in a single-episode TensorDict.

    `_to_td` produces one td per env step including the reset row; the
    first row is the reset (no real transition), so transition count is
    `len(td) - 1`.
    """
    return max(int(td.batch_size[0]) - 1, 0)


class OnlineReservoir:
    """Bounded online reservoir over completed episode TensorDicts.

    Implements Vitter's Algorithm R: maintains at most `k` items in memory;
    when the `(i+1)`-th item arrives with `i >= k`, it replaces a random
    slot with probability `k/(i+1)`. The result is a uniform random sample
    over every episode ever added.

    Replaces the previous unbounded `episode_log: list` capture, which
    grew linearly with step count and triggered OOM (SIGKILL, rc=137) on
    long tasks at ~360-400k internal steps.

    Args:
        k: maximum number of episodes to keep in memory. Sized to fit the
           eventual ER transition budget with modest headroom; see
           `_auto_reservoir_size` for the heuristic.
        seed: RNG seed for reproducible replacement.
    """

    def __init__(self, k: int, seed: int = 0):
        self.k = max(int(k), 0)
        self._rng = random.Random(seed)
        self._items = []
        self._seen = 0

    def add(self, item):
        self._seen += 1
        if self.k <= 0:
            return
        if len(self._items) < self.k:
            self._items.append(item)
            return
        # Reservoir replace: index in [0, _seen-1], keep if it falls in [0, k).
        j = self._rng.randrange(self._seen)
        if j < self.k:
            self._items[j] = item

    def items(self):
        return list(self._items)

    def __len__(self):
        return len(self._items)

    @property
    def seen(self):
        return self._seen


def _auto_reservoir_size(cfg, er_buffer_ratio: float) -> int:
    """Auto-size the per-task reservoir.

    Target transitions = `er_buffer_ratio * cfg.steps`.
    Estimate internal episode length conservatively as
    `max_episode_steps / action_repeat` (TD-MPC2 internal steps), then add
    ~50% headroom so the at-task-end transition-based truncation still has
    a few episodes to choose from.
    """
    ep_len = max(
        int(getattr(cfg, 'max_episode_steps', 250))
        // max(int(getattr(cfg, 'action_repeat', 1)), 1),
        1,
    )
    target_eps = int(er_buffer_ratio * int(cfg.steps) / ep_len)
    return max(16, int(1.5 * target_eps) + 8)


def reservoir_sample(episodes, budget, seed=0):
    if budget <= 0 or not episodes:
        return []
    rng = random.Random(seed)
    order = list(range(len(episodes)))
    rng.shuffle(order)
    out, total = [], 0
    for idx in order:
        td = episodes[idx]
        n = _episode_transitions(td)
        if n <= 0:
            continue
        if total + n > budget and out:
            break
        out.append(td)
        total += n
        if total >= budget:
            break
    return out


def save_er_episodes(episodes, path, budget, seed=0):
    sampled = reservoir_sample(episodes, budget, seed=seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_eps = [td.detach().cpu() for td in sampled]
    torch.save(cpu_eps, str(path))
    n_trans = sum(_episode_transitions(td) for td in cpu_eps)
    print(colored(
        f'>>> ER: saved {len(cpu_eps)} episodes ({n_trans} transitions) -> {path}',
        'cyan',
    ))
    return len(cpu_eps), n_trans


def load_er_episodes(path):
    path = Path(path)
    if not path.exists():
        return []
    return torch.load(str(path), map_location='cpu', weights_only=False)


# ==========================================================================
# Per-task online training (mirrors sequential_train.train_one_task with
# an additional episode_log capture for cross-task ER dumping).
# ==========================================================================

def train_one_task(cfg, agent, env, buffer, logger,
                   global_step_start, eval_fn, skip_pretrain,
                   episode_reservoir,
                   task_idx=0, num_tasks=1, task_name='',
                   episode_dir=None, save_episodes=True,
                   buffer_cap=None, prune_every_n_eps=5,
                   initial_local_step=0):
    """Online training that records completed episodes into a bounded
    online reservoir for cross-task ER dumping.

    `episode_reservoir` is an `OnlineReservoir` instance; the function
    calls `.add(ep_td.detach().cpu())` after every completed episode.
    Memory usage is bounded at `O(k * episode_size)` rather than growing
    linearly with step count.

    `initial_local_step` lets the caller continue a task mid-flight from
    a `models/{step}.pt` snapshot: training loop start, eval cadence,
    pretrain gating, and `global_step_fn` all see a non-zero starting
    step. The wandb axis therefore picks up exactly where the killed
    run left off (any overlap in `[initial_local_step, max_old_step]` is
    silently overwritten by wandb).
    """
    logger.global_step_fn = lambda s: global_step_start + s
    if save_episodes and episode_dir is not None:
        Path(episode_dir).mkdir(parents=True, exist_ok=True)
    if buffer_cap is None:
        buffer_cap = int(min(cfg.buffer_size, cfg.steps))

    local_step = max(int(initial_local_step), 0)

    pbar = tqdm(
        total=cfg.steps,
        initial=min(local_step, cfg.steps),
        desc=f'T{task_idx + 1}/{num_tasks} {task_name}',
        unit='step',
        dynamic_ncols=True,
    )

    ep_idx = 0
    start_time = time()
    train_metrics = {}
    done = True
    eval_next = False
    tds = []
    obs = None
    info = None

    def _common():
        elapsed = time() - start_time
        return dict(
            step=local_step,
            global_step=global_step_start + local_step,
            episode=ep_idx,
            elapsed_time=elapsed,
            steps_per_second=local_step / max(elapsed, 1e-9),
        )

    try:
        while local_step <= cfg.steps:
            if local_step % cfg.eval_freq == 0:
                eval_next = True

            if (cfg.save_freq > 0 and local_step > 0
                    and local_step % cfg.save_freq == 0):
                logger.save_agent(agent, identifier=local_step)

            if done:
                if eval_next:
                    eval_fn(agent, local_step, global_step_start + local_step)
                    eval_next = False

                # `info is not None` (not `local_step > 0`) gates the
                # "log the just-finished episode" block. With mid-task
                # resume the loop can begin at local_step > 0 without
                # any env.step having been called yet, so `info` is None.
                if info is not None:
                    if info['terminated'] and not cfg.episodic:
                        raise ValueError(
                            'Termination detected but cfg.episodic=false. '
                            'Set episodic=true to support episodic tasks.'
                        )
                    train_metrics.update(
                        episode_reward=torch.tensor(
                            [td['reward'] for td in tds[1:]]).sum(),
                        episode_success=info['success'],
                        episode_length=len(tds),
                        episode_terminated=info['terminated'],
                    )
                    train_metrics.update(_common())
                    logger.log_train(train_metrics)
                    ep_td = torch.cat(tds)
                    ep_idx = buffer.add(ep_td)
                    # ER capture: bounded online reservoir over completed
                    # episodes. Keeps a uniform random sample of size <= k
                    # in CPU memory regardless of how many episodes a task
                    # actually runs.
                    episode_reservoir.add(ep_td.detach().cpu())
                    if save_episodes and episode_dir is not None:
                        try:
                            Buffer.save_episode(ep_td, episode_dir, ep_idx)
                            if ep_idx % prune_every_n_eps == 0:
                                Buffer.prune_episode_dir_to_cap(episode_dir, buffer_cap)
                        except Exception as e:
                            print(colored(f'[seq_train_er] save_episode failed: {e}', 'red'))

                obs = env.reset()
                tds = [_to_td(env, obs)]

            if local_step > cfg.seed_steps:
                action = agent.act(obs, t0=(len(tds) == 1))
            else:
                action = env.rand_act()
            obs, reward, done, info = env.step(action)
            tds.append(_to_td(env, obs, action, reward, info['terminated']))

            if local_step >= cfg.seed_steps:
                if local_step == cfg.seed_steps and not skip_pretrain:
                    num_updates = cfg.seed_steps
                    pbar.write(colored(
                        'Pretraining agent on seed data...', 'yellow'))
                else:
                    num_updates = 1
                for _ in range(num_updates):
                    _m = agent.update(buffer)
                train_metrics.update(_m)

            local_step += 1
            if local_step <= cfg.steps:
                pbar.update(1)
    finally:
        pbar.close()

    return local_step


# ==========================================================================
# Main sequential ER loop
# ==========================================================================

def main(args):
    assert torch.cuda.is_available(), 'TD-MPC2 requires CUDA.'

    num_tasks = len(args.tasks)
    if len(args.task_steps) != num_tasks:
        raise ValueError('--task-steps must have one value per task.')
    if args.task_buffer_sizes is not None:
        if len(args.task_buffer_sizes) != num_tasks:
            raise ValueError('--task-buffer-sizes must have one value per task.')
        task_buffer_sizes = list(args.task_buffer_sizes)
    else:
        task_buffer_sizes = [args.buffer_size] * num_tasks

    base_logdir = Path(args.logdir).expanduser().resolve()
    base_logdir.mkdir(parents=True, exist_ok=True)

    # ---- shared overrides --------------------------------------------------
    shared = dict(
        model_size=args.model_size,
        batch_size=args.batch_size,
        horizon=args.horizon,
        mpc=args.mpc,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
        image_size=args.image_size,
        max_episode_steps=args.max_episode_steps,
        action_repeat=args.action_repeat,
        cameras=list(args.cameras),
        episodic=args.episodic,
        enable_wandb=False,
        save_video=args.save_video,
        save_agent=True,
        exp_name=args.exp_name,
        compile=args.compile,
        save_episodes=args.save_episodes,
        episode_dir=None,
        buffer_storage_device=args.buffer_storage_device,
        prune_every_n_episodes=args.prune_every_n_episodes,
        # Offline-pretrained encoder (loaded + optionally frozen inside
        # WorldModel.__init__). None is skipped by build_task_cfg.
        pretrained_encoder=args.pretrained_encoder,
        freeze_encoder=bool(args.freeze_encoder),
    )

    task_cfgs = []
    for i in range(num_tasks):
        task_logdir = base_logdir / f'task{i + 1}_{args.tasks[i]}'
        per_task = {**shared, 'buffer_size': int(task_buffer_sizes[i])}
        cfg = build_task_cfg(
            args.tasks[i], args.task_steps[i], task_logdir,
            per_task, args.seed,
        )
        task_cfgs.append(cfg)

    set_seed(args.seed)

    # ---- Resume detection --------------------------------------------------
    resume_from = 0
    global_step = 0
    prev_backbone_path = None
    prev_legacy_ckpt = None  # back-compat: pre-split `final.pt`
    progress = load_progress(base_logdir)
    if progress is not None:
        for i in range(num_tasks):
            entry = progress.get('tasks', {}).get(str(i))
            if entry and entry.get('completed'):
                resume_from = i + 1
                global_step = int(entry['global_step_at_end'])
                task_dir = base_logdir / f'task{i + 1}_{args.tasks[i]}'
                bb = task_dir / 'backbone.pt'
                if bb.exists():
                    prev_backbone_path = bb
                    prev_legacy_ckpt = None
                else:
                    legacy = task_dir / 'models' / 'final.pt'
                    if legacy.exists():
                        prev_backbone_path = None
                        prev_legacy_ckpt = legacy
                print(colored(
                    f'>>> RESUME: task {i + 1} ({args.tasks[i]}) '
                    f'already done (@g={global_step}).', 'cyan'))
            else:
                break
    if resume_from >= num_tasks:
        print(colored('>>> All tasks already completed. Nothing to do.',
                      'green', attrs=['bold']))
        return

    # ---- Single wandb run spanning every task ------------------------------
    stored_run_id = (progress or {}).get('wandb_run_id') if progress else None
    effective_run_id = args.wandb_run_id or stored_run_id

    wandb_run = None
    if args.logger == 'wandb':
        import wandb
        run_name = args.wandb_run_name or (
            f'seq_tdmpc2_er_{"-".join(args.tasks)}_s{args.seed}'
        )
        if effective_run_id is not None:
            print(colored(
                f'>>> RESUME: attaching to existing wandb run id='
                f'{effective_run_id}', 'cyan',
            ))
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            id=effective_run_id,
            resume='allow' if effective_run_id else None,
            dir=str(base_logdir),
            config={
                'tasks': args.tasks,
                'task_steps': args.task_steps,
                'seed': args.seed,
                'er_buffer_ratio': args.er_buffer_ratio,
                'er_seed': args.er_seed,
                **shared,
            },
            tags=['sequential', 'tdmpc2', 'er'] + list(args.tasks),
        )
        try:
            save_wandb_run_id(base_logdir, wandb_run.id)
        except Exception as e:
            print(colored(f'[warn] could not persist wandb run id: {e}', 'red'))

    # ---- Wandb step floor (for mid-task resume) ----------------------------
    # When resuming an existing wandb run, wandb enforces strict step
    # monotonicity from the run's last logged step. Re-logging at an older
    # step is *silently dropped* (it does NOT overwrite — that was the
    # earlier wrong assumption). To get the curve to pick up cleanly, we
    # advance the in-task step counter forward to this floor so every new
    # log lands at step >= last_logged_step.
    wandb_floor = 0
    if wandb_run is not None:
        try:
            wandb_floor = int(getattr(wandb_run, 'step', 0) or 0)
        except Exception:
            wandb_floor = 0
        if wandb_floor > 0:
            print(colored(
                f'>>> Wandb step floor (resume): logs must start at step '
                f'>= {wandb_floor}. Per-task initial_local_step will be '
                f'advanced to this floor when applicable.',
                'cyan',
            ))

    # ---- Plan summary ------------------------------------------------------
    print('=' * 64)
    print(colored('>>> SEQUENTIAL TD-MPC2 ER TRAINING', 'cyan', attrs=['bold']))
    print('=' * 64)
    for i, (t, s) in enumerate(zip(args.tasks, args.task_steps)):
        mark = '  <-- resume here' if i == resume_from else ''
        print(f'  Task {i + 1}: {t}  steps={s}{mark}')
    print(f'  Logdir:           {base_logdir}')
    print(f'  Seed:             {args.seed}')
    print(f'  ER buffer ratio:  {args.er_buffer_ratio} '
          f'(budget per prev task = ratio * that task\'s cfg.steps)')
    print(f'  ER seed:          {args.er_seed}')
    print(f'  Wandb project:    {args.wandb_project}')
    print('=' * 64)

    # ---- Task loop ---------------------------------------------------------
    for task_idx in range(num_tasks):
        if task_idx < resume_from:
            continue

        cfg = task_cfgs[task_idx]
        task_name = args.tasks[task_idx]
        task_logdir = Path(cfg.work_dir)
        task_logdir.mkdir(parents=True, exist_ok=True)

        print('=' * 64)
        print(colored(
            f'>>> TASK {task_idx + 1}/{num_tasks}: {task_name}',
            'cyan', attrs=['bold'],
        ))
        print(f'    internal steps: {cfg.steps}  '
              f'(action_repeat={cfg.action_repeat} -> '
              f'{cfg.steps * cfg.action_repeat} env frames)')
        print(f'    global step start: {global_step}')
        print('=' * 64)

        save_progress(
            base_logdir, task_idx, task_name=task_name,
            global_step_at_start=global_step,
        )

        # Buffer isolation for in-task `train_eps/`: cross-task replay in ER
        # comes from each previous task's reservoir-sampled `er_episodes.pt`,
        # not from the in-task `train_eps/` dir. Once a previous task is fully
        # completed its `train_eps/` is no longer needed for resume.
        for j in range(task_idx):
            prev_done = (
                progress is not None
                and progress.get('tasks', {}).get(str(j), {}).get('completed')
            )
            if not prev_done:
                continue
            prev_eps = (
                base_logdir / f'task{j + 1}_{args.tasks[j]}' / 'train_eps'
            )
            if prev_eps.is_dir():
                print(colored(
                    f'>>> Removing previous-task replay dir: {prev_eps}',
                    'yellow'))
                shutil.rmtree(prev_eps, ignore_errors=True)

        # Fresh env + buffer for this task --------------------------------
        train_env = make_env(cfg)
        buffer = Buffer(cfg)

        task_episode_dir = task_logdir / 'train_eps'
        task_buffer_cap = int(min(cfg.buffer_size, cfg.steps))
        if args.save_episodes and task_episode_dir.is_dir():
            stats = buffer.load_from_directory(
                task_episode_dir, max_total_steps=task_buffer_cap)
            if stats['episodes_restored'] > 0:
                print(colored(
                    f'>>> Replayed {stats["transitions_restored"]:,} transitions '
                    f'from {stats["episodes_restored"]} episodes '
                    f'(cap {task_buffer_cap:,}).',
                    'green'))
                if stats['kept_files']:
                    dropped = Buffer.erase_over_episode_files(
                        task_episode_dir, stats['kept_files'])
                    if dropped:
                        print(colored(f'>>> Pruned {dropped} stale episode files.', 'yellow'))

        logger = SequentialLogger(
            cfg, task_idx, num_tasks, task_name, wandb_run,
            save_video=args.save_video,
        )

        # Build agent. For task > 0, inherit the shared backbone from the
        # previous task (weights + Adam moments). Per-task heads stay at
        # their fresh init from `TDMPC2(cfg)`.
        agent = TDMPC2(cfg)
        if task_idx > 0 and prev_backbone_path is not None and prev_backbone_path.exists():
            print(colored(
                f'>>> Loading previous-task backbone: {prev_backbone_path}',
                'cyan',
            ))
            agent.load_backbone(prev_backbone_path)
        elif task_idx > 0 and prev_legacy_ckpt is not None and prev_legacy_ckpt.exists():
            print(colored(
                f'>>> [legacy] Loading whole-agent ckpt: {prev_legacy_ckpt}',
                'yellow',
            ))
            agent.load(prev_legacy_ckpt)
        elif task_idx == 0 and args.from_checkpoint is not None:
            ck = Path(args.from_checkpoint)
            if ck.exists():
                print(colored(f'>>> Loading external checkpoint: {ck}',
                              'cyan'))
                agent.load(ck)
            else:
                raise FileNotFoundError(ck)

        # ----- Mid-task resume from latest models/{step}.pt snapshot ------
        # The naive resume logic only re-enters a task that finished cleanly
        # (`completed: true` in progress.json). If a task died mid-flight
        # (OOM, SIGKILL, wall-clock), the periodic snapshots written by
        # `logger.save_agent(agent, identifier=local_step)` at every
        # `cfg.save_freq` steps are still on disk under `<task_logdir>/models/`.
        # We pick up the latest one here so the agent doesn't restart from a
        # random init.
        #
        # This deliberately runs AFTER the previous-task-backbone load so
        # that, for task > 0 mid-flight resumes, the snapshot (which is a
        # whole-agent state-dict) overrides the encoder/dynamics with their
        # latest in-task values rather than the previous-task hand-off.
        resumed_from_snapshot_step = None
        if not args.disable_mid_task_resume:
            snap_dir = task_logdir / 'models'
            if snap_dir.is_dir():
                numeric_snaps = []
                for p in snap_dir.glob('*.pt'):
                    try:
                        numeric_snaps.append((int(p.stem), p))
                    except ValueError:
                        pass
                if numeric_snaps:
                    step_n, latest = max(numeric_snaps, key=lambda x: x[0])
                    if step_n < cfg.steps:
                        print(colored(
                            f'>>> MID-TASK RESUME: loading snapshot '
                            f'{latest.name} (local step {step_n}) on top of '
                            f'the current agent.',
                            'cyan', attrs=['bold'],
                        ))
                        agent.load(latest)
                        resumed_from_snapshot_step = step_n
                    else:
                        print(colored(
                            f'>>> Snapshot {latest.name} at step {step_n} '
                            f'is at/past cfg.steps={cfg.steps}; ignoring.',
                            'yellow',
                        ))

        # ============================================================
        # [ER] Prefill replay buffer with episodes from previous tasks
        # ============================================================
        if task_idx > 0 and args.er_buffer_ratio > 0:
            print(colored(
                f'>>> ER: loading episodes from {task_idx} previous task(s)...',
                'cyan',
            ))
            total_eps_loaded = 0
            total_trans_loaded = 0
            for j in range(task_idx):
                er_path = (base_logdir / f'task{j + 1}_{args.tasks[j]}'
                           / 'er_episodes.pt')
                eps = load_er_episodes(er_path)
                if not eps:
                    print(colored(
                        f'    WARNING: no ER dump for task {j + 1} '
                        f'({args.tasks[j]}) at {er_path} — skipping',
                        'yellow',
                    ))
                    continue
                for td in eps:
                    buffer.add(td)
                n_trans = sum(_episode_transitions(td) for td in eps)
                total_eps_loaded += len(eps)
                total_trans_loaded += n_trans
                print(f'    task {j + 1} ({args.tasks[j]}): '
                      f'{len(eps)} eps, {n_trans} transitions')
            print(colored(
                f'>>> ER: prefilled buffer with {total_eps_loaded} eps '
                f'({total_trans_loaded} transitions, '
                f'buffer.num_eps={buffer.num_eps})',
                'cyan',
            ))

        # Cross-task evaluation closure -----------------------------------
        # For the current task: live agent. For each prior task `j`: build
        # a throw-away TDMPC2 shell whose encoder + dynamics are rebound to
        # the live backbone (by reference, no copy), then load that task's
        # heads from `task_modules.pt`.
        def eval_fn(agent, local_step, gstep,
                    _task_idx=task_idx, _logger=logger, _cfg=cfg):
            for j in range(_task_idx + 1):
                is_current = (j == _task_idx)
                # Build the eval env fresh for every evaluation and close it
                # immediately afterwards. A cached/long-lived eval env shares
                # a MuJoCo rendering context with the other (different-task)
                # envs alive in this process and returns corrupted camera
                # observations, which silently collapses prior-task eval.
                env_j = make_env(task_cfgs[j])
                try:
                    video_rec = _logger.video if (is_current and _logger.video
                                                  ) else None
                    if video_rec is not None:
                        video_rec.set_prefix(
                            f'eval/task{j + 1}_{args.tasks[j]}/videos/eval_video'
                        )
                    if is_current:
                        eval_agent = agent
                    else:
                        tm_path = (base_logdir / f'task{j + 1}_{args.tasks[j]}'
                                   / 'task_modules.pt')
                        eval_agent = TDMPC2(task_cfgs[j])
                        eval_agent.model._encoder = agent.model._encoder
                        eval_agent.model._dynamics = agent.model._dynamics
                        eval_agent.load_task_modules(tm_path)
                        eval_agent.model.eval()
                    metrics = eval_on_env(
                        eval_agent, env_j, _cfg.eval_episodes,
                        video_recorder=video_rec,
                        video_step=gstep if video_rec else None,
                    )
                    metrics['step'] = local_step
                    _logger.log_eval(j, args.tasks[j], metrics, gstep)
                    if not is_current:
                        del eval_agent
                finally:
                    try:
                        env_j.close()
                    except Exception:
                        pass

        # Train -----------------------------------------------------------
        # Bounded online reservoir: prevents the OOM caused by the previous
        # unbounded `episode_log: list` (rc=137 SIGKILL at ~360-400k steps
        # on long tasks). Size is auto-derived from the ER transition
        # budget unless overridden via --er-reservoir-size.
        reservoir_k = (
            args.er_reservoir_size
            if args.er_reservoir_size is not None
            else _auto_reservoir_size(cfg, args.er_buffer_ratio)
        )
        # Make the per-task RNG deterministic but distinct per task.
        episode_reservoir = OnlineReservoir(
            k=reservoir_k, seed=args.er_seed + task_idx,
        )
        print(colored(
            f'>>> ER reservoir: cap={reservoir_k} episodes '
            f'(er_buffer_ratio={args.er_buffer_ratio}, '
            f'cfg.steps={cfg.steps}, '
            f'auto={args.er_reservoir_size is None})',
            'cyan',
        ))
        # Skip the pretraining burst whenever we're not strictly starting
        # task 1 from a random-init agent: explicit --skip-pretrain, a
        # later task (which inherits the prior backbone), or a mid-task
        # snapshot resume (the agent is already well past seed_steps).
        skip_pre = (
            args.skip_pretrain
            or (task_idx > 0)
            or (resumed_from_snapshot_step is not None)
        )

        # Local-step start: max of (a) the snapshot we loaded weights from,
        # and (b) the wandb step floor minus this task's global offset, so
        # the first wandb.log call lands at step >= wandb_floor and isn't
        # dropped. Clamp at cfg.steps so a stale wandb floor doesn't try to
        # force us past the task budget.
        snap_start = resumed_from_snapshot_step or 0
        wandb_local_floor = max(0, wandb_floor - global_step)
        init_local = min(max(snap_start, wandb_local_floor), int(cfg.steps))
        if init_local > snap_start:
            print(colored(
                f'>>> initial_local_step advanced past snapshot '
                f'({snap_start}) to {init_local} to clear the wandb floor '
                f'({wandb_floor}). The model weights are from step '
                f'{snap_start}; the wandb axis just resumes at '
                f'{global_step + init_local}.',
                'yellow',
            ))

        train_one_task(
            cfg, agent, train_env, buffer, logger,
            global_step_start=global_step,
            eval_fn=eval_fn,
            skip_pretrain=skip_pre,
            episode_reservoir=episode_reservoir,
            task_idx=task_idx,
            num_tasks=num_tasks,
            task_name=task_name,
            episode_dir=task_episode_dir,
            save_episodes=args.save_episodes,
            buffer_cap=task_buffer_cap,
            prune_every_n_eps=args.prune_every_n_episodes,
            initial_local_step=init_local,
        )

        # Save split checkpoint BEFORE the final cross-task eval so that
        # eval shells for j < t see this task's heads.
        backbone_path = task_logdir / 'backbone.pt'
        task_modules_path = task_logdir / 'task_modules.pt'
        agent.save_backbone(backbone_path)
        agent.save_task_modules(task_modules_path)

        # Final cross-task eval ------------------------------------------
        eval_fn(agent, cfg.steps, global_step + cfg.steps)

        # ============================================================
        # [ER] Dump reservoir-sampled episodes for this task
        # ============================================================
        # The reservoir already holds a uniform sample of size <= k. The
        # subsequent transition-based `save_er_episodes` truncates to the
        # configured ER transition budget — it's a pass-through when the
        # reservoir is already under budget.
        if args.er_buffer_ratio > 0:
            er_budget = int(args.er_buffer_ratio * cfg.steps)
            er_path = task_logdir / 'er_episodes.pt'
            print(colored(
                f'>>> ER reservoir at task end: {len(episode_reservoir)} '
                f'episodes kept out of {episode_reservoir.seen} seen.',
                'cyan',
            ))
            save_er_episodes(
                episode_reservoir.items(), er_path, er_budget,
                seed=args.er_seed + task_idx,
            )
        del episode_reservoir

        prev_backbone_path = backbone_path
        prev_legacy_ckpt = None

        global_step += cfg.steps
        save_progress(
            base_logdir, task_idx, task_name=task_name,
            completed=True, global_step_at_end=global_step,
        )

        try:
            train_env.close()
        except Exception:
            pass
        del train_env, buffer, agent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(colored(
            f'>>> TASK {task_idx + 1} ({task_name}) done @g={global_step}.',
            'green', attrs=['bold'],
        ))
        print()

    if wandb_run is not None:
        wandb_run.finish()

    print('=' * 64)
    print(colored('>>> ALL TASKS COMPLETED.', 'green', attrs=['bold']))
    print(f'>>> Final global step: {global_step}')
    print('=' * 64)


# ==========================================================================
# Entry point
# ==========================================================================

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Sequential TD-MPC2 training with Experience Replay '
                    'on a shared backbone.',
    )

    p.add_argument('--tasks', nargs='+', required=True)
    p.add_argument('--task-steps', nargs='+', type=int, required=True)

    p.add_argument('--logdir', type=str, required=True)
    p.add_argument('--logger', type=str, default='wandb',
                   choices=['wandb', 'none'])
    p.add_argument('--wandb-entity', type=str, default=None)
    p.add_argument('--wandb-project', type=str,
                   default='Metaworld_Tdmpc2_Sequential_ER')
    p.add_argument('--wandb-run-name', type=str, default=None)
    p.add_argument('--wandb-run-id', type=str, default=None)
    p.add_argument('--exp-name', type=str, default='seq_tdmpc2_er')
    p.add_argument('--from-checkpoint', type=str, default=None,
                   help='Optional external checkpoint for task 1.')

    p.add_argument('--seed', type=int, default=1)

    p.add_argument('--model-size', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--buffer-size', type=int, default=50000)
    p.add_argument('--task-buffer-sizes', nargs='+', type=int, default=None)
    p.add_argument('--horizon', type=int, default=3)
    p.add_argument('--mpc', dest='mpc', action='store_true')
    p.add_argument('--no-mpc', dest='mpc', action='store_false')
    p.set_defaults(mpc=True)
    p.add_argument('--eval-freq', type=int, default=10000)
    p.add_argument('--eval-episodes', type=int, default=5)
    p.add_argument('--save-freq', type=int, default=0)
    p.add_argument('--save-video', dest='save_video', action='store_true')
    p.add_argument('--no-save-video', dest='save_video', action='store_false')
    p.set_defaults(save_video=True)
    p.add_argument('--compile', dest='compile', action='store_true')
    p.add_argument('--no-compile', dest='compile', action='store_false')
    p.set_defaults(compile=False)
    p.add_argument('--skip-pretrain', action='store_true')

    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--max-episode-steps', type=int, default=250)
    p.add_argument('--action-repeat', type=int, default=2)
    p.add_argument('--cameras', nargs='+',
                   default=['topview', 'front', 'gripperPOV'])
    p.add_argument('--episodic', dest='episodic', action='store_true')
    p.add_argument('--no-episodic', dest='episodic', action='store_false')
    p.set_defaults(episodic=True)

    p.add_argument('--save-episodes', dest='save_episodes', action='store_true')
    p.add_argument('--no-save-episodes', dest='save_episodes', action='store_false')
    p.set_defaults(save_episodes=True)
    p.add_argument('--prune-every-n-episodes', type=int, default=5)
    p.add_argument('--buffer-storage-device', type=str, default='auto',
                   choices=['auto', 'cuda', 'cpu'])

    # Offline-pretrained encoder (see tdmpc2/offline encoder pretraining).
    p.add_argument('--pretrained-encoder', type=str, default=None,
                   help='Path to an encoder checkpoint from '
                        'offline encoder pretraining. Loaded into the world '
                        'model at init; default None = fresh random encoder.')
    p.add_argument('--freeze-encoder', dest='freeze_encoder',
                   action='store_true',
                   help='Freeze the loaded pretrained encoder for the whole '
                        'run (requires_grad=False, excluded from the optimizer).')
    p.add_argument('--no-freeze-encoder', dest='freeze_encoder',
                   action='store_false',
                   help='Fine-tune the loaded pretrained encoder instead.')
    p.set_defaults(freeze_encoder=False)

    # ER knobs
    p.add_argument(
        '--er-buffer-ratio', type=float, default=0.025,
        help='Fraction of each previous task\'s cfg.steps to keep as ER '
             'transitions. 0 disables ER (default: 0.025).',
    )
    p.add_argument(
        '--er-seed', type=int, default=42,
        help='Random seed for reservoir sampling of ER episodes.',
    )
    p.add_argument(
        '--er-reservoir-size', type=int, default=None,
        help='Max #episodes the in-memory online reservoir keeps per task. '
             'When None (default), auto-derived from `er_buffer_ratio * '
             'cfg.steps / (max_episode_steps / action_repeat)` plus ~50%% '
             'headroom (min 16). Set this directly to cap RAM use if the '
             'auto value is too large for your node.',
    )
    p.add_argument(
        '--disable-mid-task-resume', action='store_true',
        help='Disable auto-loading of the latest models/{step}.pt snapshot '
             'when re-entering a task that did not finish. Default behaviour '
             'is to resume from the highest-numbered snapshot and continue '
             'training; pass this flag to force restart-from-scratch instead.',
    )

    main(p.parse_args())
