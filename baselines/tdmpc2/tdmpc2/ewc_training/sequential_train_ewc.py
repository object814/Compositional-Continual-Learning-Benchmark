"""Sequential TD-MPC2 training with EWC on the shared backbone only.

Architecture (matches `sequential_train.py`):
  - **Backbone** = encoder + dynamics — shared across tasks, persisted as
    `<task_logdir>/backbone.pt`.
  - **Per-task heads** = reward / Qs / termination / pi + RunningScale +
    MPC `_prev_mean` — fresh per task, persisted as
    `<task_logdir>/task_modules.pt`.

EWC adaptation (mirrors the Dreamer EWC backbone-only protocol):
  Since per-task heads are re-initialised every task, catastrophic
  forgetting can only happen inside the shared backbone. EWC therefore
  protects ONLY encoder + dynamics weights:
    1. After each task except the last, compute a diagonal Fisher matrix
       over backbone parameters using the world-model loss.
    2. Add λ · Σ F · (θ - θ*)² to the world-model `total_loss` on every
       subsequent training step.
    3. Persist EWC state to `<task_logdir>/ewc_state_task{N}.pt` and
       reload it on resume.
  Policy gradients flow only through `zs.detach()` and never reach the
  backbone, so we do NOT add the penalty to the policy backward.

Compile must be off — the Python-side EWC penalty is not compatible with
`torch.compile(_update, "reduce-overhead")`'s cudagraph capture.

Example:
    python sequential_train_ewc.py \\
        --tasks metaworld_drawer-open-v3 metaworld_pick-place-v3 \\
        --task-steps 200000 500000 \\
        --logdir ../logdir/seq_tdmpc2_ewc/run0 \\
        --ewc-lambda 1000.0 \\
        --ewc-fisher-batches 50 \\
        --wandb-project Metaworld_Tdmpc2_Sequential_EWC
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
import shutil
import sys
import time as _time
from pathlib import Path

import torch
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from termcolor import colored
from tqdm.auto import tqdm

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))

from common.seed import set_seed
from common.buffer import Buffer
from common import math as tdmpc_math
from envs import make_env
from tdmpc2 import TDMPC2

from sequential_train import (
    build_task_cfg,
    save_progress,
    load_progress,
    save_wandb_run_id,
    SequentialLogger,
    eval_on_env,
    _to_td,
)

from ewc_training.ewc import EWCManager


CONFIG_PATH = _PARENT / 'config.yaml'

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


# ==========================================================================
# EWC-aware TD-MPC2 subclass
# ==========================================================================
# Override `_update` to add `λ · Σ F · (θ - θ*)²` to the world-model loss
# before backward. The penalty's gradient flows only into backbone params
# (Fisher / θ* are tracked over encoder + dynamics only), but adding it to
# `total_loss` is the cleanest place: total_loss.backward() already
# distributes gradients across all params in `self.optim`. The policy
# backward operates on `zs.detach()` so no penalty is needed there.

class EWCTDMPC2(TDMPC2):
    """TD-MPC2 + backbone-only EWC penalty injected into the world-model backward."""

    def __init__(self, cfg, ewc_manager: EWCManager = None):
        if getattr(cfg, 'compile', False):
            print(colored(
                '>>> EWC: forcing cfg.compile=False (penalty is incompatible '
                'with cudagraph capture)', 'yellow',
            ))
            cfg.compile = False
        super().__init__(cfg)
        self.ewc_manager = ewc_manager

    def _update(self, obs, action, reward, terminated, task=None):
        # ---- world-model loss (mirrors parent _update body) --------------
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, terminated, task)

        self.model.train()

        zs = torch.empty(
            self.cfg.horizon + 1, self.cfg.batch_size, self.cfg.latent_dim,
            device=self.device,
        )
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
            z = self.model.next(z, _action, task)
            consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho ** t
            zs[t + 1] = z

        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type='all')
        reward_preds = self.model.reward(_zs, action, task)
        if self.cfg.episodic:
            termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

        reward_loss, value_loss = 0, 0
        for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(
            zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))
        ):
            reward_loss = reward_loss + tdmpc_math.soft_ce(
                rew_pred_unbind, rew_unbind, self.cfg
            ).mean() * self.cfg.rho ** t
            for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
                value_loss = value_loss + tdmpc_math.soft_ce(
                    qs_unbind_unbind, td_targets_unbind, self.cfg
                ).mean() * self.cfg.rho ** t

        consistency_loss = consistency_loss / self.cfg.horizon
        reward_loss = reward_loss / self.cfg.horizon
        if self.cfg.episodic:
            termination_loss = F.binary_cross_entropy_with_logits(
                termination_pred, terminated
            )
        else:
            termination_loss = 0.
        value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)

        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.termination_coef * termination_loss
            + self.cfg.value_coef * value_loss
        )

        # ---- EWC penalty on backbone only --------------------------------
        ewc_penalty = torch.tensor(0.0, device=self.device)
        if self.ewc_manager is not None and self.ewc_manager.num_tasks_consolidated > 0:
            ewc_penalty = self.ewc_manager.penalty(self)
            total_loss = total_loss + ewc_penalty

        # ---- world-model backward + step ---------------------------------
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm,
        )
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        # ---- policy update (parent's update_pi — no EWC penalty needed) --
        pi_info = self.update_pi(zs.detach(), task)

        # ---- target-Q soft update (unchanged) ----------------------------
        self.model.soft_update_target_Q()

        self.model.eval()
        info = TensorDict({
            "consistency_loss": consistency_loss,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
            "termination_loss": termination_loss,
            "total_loss": total_loss,
            "grad_norm": grad_norm,
            "ewc_penalty": ewc_penalty.detach(),
        })
        if self.cfg.episodic:
            info.update(tdmpc_math.termination_statistics(
                torch.sigmoid(termination_pred[-1]), terminated[-1],
            ))
        info.update(pi_info)
        return info.detach().mean()


# ==========================================================================
# Per-task online training (mirrors sequential_train.train_one_task)
# ==========================================================================

def train_one_task(cfg, agent, env, buffer, logger,
                   global_step_start, eval_fn, skip_pretrain,
                   task_idx=0, num_tasks=1, task_name='',
                   episode_dir=None, save_episodes=True,
                   buffer_cap=None, prune_every_n_eps=5):
    logger.global_step_fn = lambda s: global_step_start + s
    if save_episodes and episode_dir is not None:
        Path(episode_dir).mkdir(parents=True, exist_ok=True)
    if buffer_cap is None:
        buffer_cap = int(min(cfg.buffer_size, cfg.steps))

    pbar = tqdm(
        total=cfg.steps,
        desc=f'T{task_idx + 1}/{num_tasks} {task_name}',
        unit='step',
        dynamic_ncols=True,
    )

    local_step = 0
    ep_idx = 0
    start_time = _time.time()
    train_metrics = {}
    done = True
    eval_next = False
    tds = []
    obs = None
    info = None

    def _common():
        elapsed = _time.time() - start_time
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

                if local_step > 0:
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
                    if save_episodes and episode_dir is not None:
                        try:
                            Buffer.save_episode(ep_td, episode_dir, ep_idx)
                            if ep_idx % prune_every_n_eps == 0:
                                Buffer.prune_episode_dir_to_cap(episode_dir, buffer_cap)
                        except Exception as e:
                            print(colored(f'[seq_train_ewc] save_episode failed: {e}', 'red'))

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
# Main sequential EWC loop
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

    # ---- shared overrides (compile forced off) -----------------------------
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
        compile=False,
        save_episodes=args.save_episodes,
        episode_dir=None,
        buffer_storage_device=args.buffer_storage_device,
        prune_every_n_episodes=args.prune_every_n_episodes,
        # Offline-pretrained encoder (loaded + optionally frozen inside
        # WorldModel.__init__). None is skipped by build_task_cfg. A frozen
        # encoder is automatically excluded from the EWC penalty (EWCManager
        # skips params with requires_grad=False).
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

    # ---- EWC manager (single instance for entire run) ----------------------
    ewc_manager = EWCManager(lambda_ewc=args.ewc_lambda)

    # ---- Resume detection --------------------------------------------------
    resume_from = 0
    global_step = 0
    prev_backbone_path = None
    prev_legacy_ckpt = None
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

    # [EWC] Resume: reload latest consolidated EWC state, if any.
    if resume_from > 0:
        for j in range(resume_from - 1, -1, -1):
            ewc_file = (base_logdir / f'task{j + 1}_{args.tasks[j]}'
                        / f'ewc_state_task{j + 1}.pt')
            if ewc_file.exists():
                print(colored(f'>>> EWC: loading state from {ewc_file}', 'cyan'))
                ewc_state = torch.load(str(ewc_file), map_location='cpu',
                                       weights_only=False)
                ewc_manager.load_state_dict(ewc_state)
                print(colored(
                    f'>>> EWC: restored {ewc_manager.num_tasks_consolidated} '
                    f'consolidated task(s)', 'cyan',
                ))
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
            f'seq_tdmpc2_ewc_{"-".join(args.tasks)}_s{args.seed}'
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
                'ewc_lambda': args.ewc_lambda,
                'ewc_fisher_batches': args.ewc_fisher_batches,
                **shared,
            },
            tags=['sequential', 'tdmpc2', 'ewc'] + list(args.tasks),
        )
        try:
            save_wandb_run_id(base_logdir, wandb_run.id)
        except Exception as e:
            print(colored(f'[warn] could not persist wandb run id: {e}', 'red'))

    # ---- Plan summary ------------------------------------------------------
    print('=' * 64)
    print(colored('>>> SEQUENTIAL TD-MPC2 EWC TRAINING (backbone-only)',
                  'cyan', attrs=['bold']))
    print('=' * 64)
    for i, (t, s) in enumerate(zip(args.tasks, args.task_steps)):
        mark = '  <-- resume here' if i == resume_from else ''
        print(f'  Task {i + 1}: {t}  steps={s}{mark}')
    print(f'  Logdir:               {base_logdir}')
    print(f'  Seed:                 {args.seed}')
    print(f'  EWC lambda:           {args.ewc_lambda}')
    print(f'  EWC Fisher batches:   {args.ewc_fisher_batches}')
    print(f'  Compile:              FORCED OFF (EWC requirement)')
    print(f'  Wandb project:        {args.wandb_project}')
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
        print(f'    internal steps:   {cfg.steps}  '
              f'(action_repeat={cfg.action_repeat} -> '
              f'{cfg.steps * cfg.action_repeat} env frames)')
        print(f'    global step start:{global_step}')
        print(f'    EWC consolidated: {ewc_manager.num_tasks_consolidated} '
              f'task(s)')
        print('=' * 64)

        save_progress(
            base_logdir, task_idx, task_name=task_name,
            global_step_at_start=global_step,
        )

        # Buffer isolation: EWC keeps replay in-task only.
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

        # Build the EWC-aware agent. For task > 0, inherit the shared
        # backbone from the previous task. Per-task heads stay fresh.
        agent = EWCTDMPC2(cfg, ewc_manager=ewc_manager)
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

        # [EWC] Move stored Fishers/θ* to device + rebuild penalty cache.
        if ewc_manager.num_tasks_consolidated > 0:
            ewc_manager.to_device(agent.device)
            ewc_manager._rebuild_penalty_cache(agent)
            n_tracked = len(ewc_manager._tracked_param_names)
            print(colored(
                f'>>> EWC: rebuilt penalty cache on {agent.device} '
                f'({n_tracked} tracked backbone param groups, '
                f'{ewc_manager.num_tasks_consolidated} consolidated tasks)',
                'cyan',
            ))

        # Cross-task evaluation closure (same scheme as naive sequential).
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
        train_one_task(
            cfg, agent, train_env, buffer, logger,
            global_step_start=global_step,
            eval_fn=eval_fn,
            skip_pretrain=args.skip_pretrain or (task_idx > 0),
            task_idx=task_idx,
            num_tasks=num_tasks,
            task_name=task_name,
            episode_dir=task_episode_dir,
            save_episodes=args.save_episodes,
            buffer_cap=task_buffer_cap,
            prune_every_n_eps=args.prune_every_n_episodes,
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
        # [EWC] Compute Fisher + consolidate (skip on the last task)
        # ============================================================
        if task_idx < num_tasks - 1:
            print(colored(
                f'>>> EWC: computing Fisher (backbone only) for task '
                f'{task_idx + 1} ({task_name})...', 'cyan',
            ))
            t0 = _time.time()
            importance, task_param = ewc_manager.compute_fisher(
                agent, buffer,
                num_batches=args.ewc_fisher_batches,
                device=agent.device,
            )
            ewc_manager.consolidate(agent, importance, task_param, task_idx)

            elapsed = _time.time() - t0
            n_params = sum(v.numel() for v in importance.values())
            fisher_mean = (
                sum(v.mean().item() for v in importance.values())
                / max(len(importance), 1)
            )
            fisher_max = max(v.max().item() for v in importance.values())
            print(colored(
                f'>>> EWC: consolidated task {task_idx + 1} in {elapsed:.1f}s '
                f'({len(importance)} backbone param groups, {n_params:,} params, '
                f'mean F={fisher_mean:.2e}, max F={fisher_max:.2e})',
                'cyan',
            ))

            if wandb_run is not None:
                wandb_run.log({
                    'ewc/fisher_mean': fisher_mean,
                    'ewc/fisher_max': fisher_max,
                    'ewc/fisher_compute_time_s': elapsed,
                    'ewc/num_tasks_consolidated': ewc_manager.num_tasks_consolidated,
                    'global_step': global_step + cfg.steps,
                }, step=global_step + cfg.steps)

            ewc_path = task_logdir / f'ewc_state_task{task_idx + 1}.pt'
            torch.save(ewc_manager.state_dict(), str(ewc_path))
            print(colored(f'>>> EWC: saved state -> {ewc_path}', 'cyan'))

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
        description='Sequential TD-MPC2 training with backbone-only EWC.',
    )

    p.add_argument('--tasks', nargs='+', required=True)
    p.add_argument('--task-steps', nargs='+', type=int, required=True)

    p.add_argument('--logdir', type=str, required=True)
    p.add_argument('--logger', type=str, default='wandb',
                   choices=['wandb', 'none'])
    p.add_argument('--wandb-entity', type=str, default=None)
    p.add_argument('--wandb-project', type=str,
                   default='Metaworld_Tdmpc2_Sequential_EWC')
    p.add_argument('--wandb-run-name', type=str, default=None)
    p.add_argument('--wandb-run-id', type=str, default=None)
    p.add_argument('--exp-name', type=str, default='seq_tdmpc2_ewc')
    p.add_argument('--from-checkpoint', type=str, default=None)

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

    # EWC knobs
    p.add_argument(
        '--ewc-lambda', type=float, default=1000.0,
        help='EWC regularisation strength (default: 1000.0). Backbone-only '
             'EWC is typically less constraining than full-model EWC; sweep '
             '[100, 500, 1000, 5000, 10000] to calibrate.',
    )
    p.add_argument(
        '--ewc-fisher-batches', type=int, default=50,
        help='Number of buffer mini-batches used to estimate the diagonal '
             'Fisher at each task boundary (default: 50).',
    )

    main(p.parse_args())
