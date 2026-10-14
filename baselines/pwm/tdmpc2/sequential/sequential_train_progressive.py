"""Sequential PWM training with progressively allocated and frozen experts.

The encoder is shared across tasks, while each task has its own prediction
heads. Training samples the current task's replay buffer. Earlier experts are
frozen at task boundaries and remain available to the task-conditioned router.
"""

import os
os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = '1'
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['XDG_RUNTIME_DIR'] = '/tmp'
os.environ['EGL_LOG_LEVEL'] = 'fatal'

import warnings
warnings.filterwarnings('ignore', message='Constant.*may be too high')
warnings.filterwarnings('ignore', message='.*Please upgrade to Gymnasium.*')
warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*')
warnings.filterwarnings('ignore')

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))

from sequential_train import run_continual_sequential


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def main(args):
    run_continual_sequential(
        args,
        progressive=True,
        er_buffer_ratio=0.0,
        er_seed=int(getattr(args, 'er_seed', 42)),
        per_task_tau=args.per_task_tau,
        merged=True,
        sdp_freeze=args.sdp_freeze,
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Sequential TD-MPC2 + PRISM-WM continual learning, '
                    'progressive masked MoE.',
    )

    p.add_argument('--tasks', nargs='+', required=True)
    p.add_argument('--task-steps', nargs='+', type=int, required=True)

    p.add_argument('--logdir', type=str, required=True)
    p.add_argument('--logger', type=str, default='wandb',
                   choices=['wandb', 'none'])
    p.add_argument('--wandb-entity', type=str, default=None)
    p.add_argument('--wandb-project', type=str,
                   default='prismatic_seq_progressive')
    p.add_argument('--wandb-run-name', type=str, default=None)
    p.add_argument('--wandb-run-id', type=str, default=None)
    p.add_argument('--wandb-group', type=str, default=None,
                   help='Wandb group; sweep launcher passes the taskset name.')
    p.add_argument('--wandb-tags', nargs='*', default=None,
                   help='Wandb tags; sweep launcher fills with method/taskset/etc.')
    p.add_argument('--exp-name', type=str, default='progmoe_merged_tdmpc2')
    p.add_argument('--from-checkpoint', type=str, default=None,
                   help='Directory containing backbone.pt (+ optional '
                        'task_modules.pt) to seed task 0.')

    p.add_argument('--seed', type=int, default=1)

    # TDMPC2 hyperparams
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

    # Env knobs
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--max-episode-steps', type=int, default=250)
    p.add_argument('--action-repeat', type=int, default=2)
    p.add_argument('--cameras', nargs='+',
                   default=['topview', 'front', 'gripperPOV'])
    p.add_argument('--episodic', dest='episodic', action='store_true')
    p.add_argument('--no-episodic', dest='episodic', action='store_false')
    p.set_defaults(episodic=True)

    # Replay-buffer storage
    p.add_argument('--buffer-storage-device', type=str, default='auto',
                   choices=['auto', 'cuda', 'cpu'])

    # Progressive-MoE knob
    p.add_argument('--K-per-task', dest='K_per_task',
                   type=int, default=3,
                   help='Number of expert slots opened per task. total_K = '
                        'num_tasks * K_per_task; at task t experts '
                        '[t·K, (t+1)·K) train and [0, t·K) are frozen.')

    # Gram-Schmidt orthogonalization
    p.add_argument('--use-orthogonal', dest='use_orthogonal',
                   action='store_true',
                   help='Enable Gram-Schmidt orthogonalization of the expert '
                        'feature stack inside the dynamics MoE.')
    p.add_argument('--no-use-orthogonal', dest='use_orthogonal',
                   action='store_false')
    p.set_defaults(use_orthogonal=True)

    # Per-task τ snapshot (eval-time; unchanged from progmoe).
    p.add_argument('--per-task-tau', dest='per_task_tau',
                   action='store_true',
                   help='At cross-task eval, restore each prior task\'s '
                        'converged τ from its backbone.pt snapshot.')
    p.add_argument('--no-per-task-tau', dest='per_task_tau',
                   action='store_false')
    p.set_defaults(per_task_tau=True)

    # SDP-faithful (Plan A) freeze. After task 0 the shared MoE head + gate
    # τ-column + τ schedule are frozen so prior tasks' dynamics path is fully
    # preserved (encoder already frozen via --freeze-encoder; old experts
    # progressively frozen; per-task heads isolated). Old experts' gate rows
    # are left trainable so a new task can still route to — reuse — them.
    p.add_argument('--sdp-freeze', dest='sdp_freeze', action='store_true',
                   help='Freeze shared MoE head + gate τ-column + τ schedule '
                        'after task 0 (SDP-style; prevents forgetting on prior '
                        'tasks while keeping new-task transfer).')
    p.add_argument('--no-sdp-freeze', dest='sdp_freeze', action='store_false')
    p.set_defaults(sdp_freeze=False)

    # Kept (unused) so resumed runs that pass --er-seed don't crash.
    p.add_argument('--er-seed', type=int, default=42, help=argparse.SUPPRESS)

    # Offline-pretrained encoder (weights supplied separately). Default: off.
    p.add_argument('--pretrained-encoder', type=str, default=None,
                   help='Path to an encoder checkpoint from '
                        'offline encoder pretraining. Loaded into the '
                        'backbone encoder; None -> fresh random encoder.')
    p.add_argument('--freeze-encoder', dest='freeze_encoder',
                   action='store_true',
                   help='Freeze the (pretrained) encoder for ALL tasks '
                        '(requires_grad=False; excluded from the optimizer).')
    p.add_argument('--no-freeze-encoder', dest='freeze_encoder',
                   action='store_false')
    p.set_defaults(freeze_encoder=False)

    main(p.parse_args())
