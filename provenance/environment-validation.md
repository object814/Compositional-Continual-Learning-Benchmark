# Environment validation

The original experiments used two stacks: TD-MPC2 and PWM ran in a conda
environment (Python 3.9, MuJoCo 3.1.2, torch 2.6.0 nightly of 2024-11-12,
torchrl/tensordict nightly 2025.1.1), and DreamerV3 ran on Python 3.10 with
MuJoCo 2.3.5 and packages from the user's `~/.local`. The release uses one
environment for all three baselines (`environment.yml`, `requirements.txt`).

Candidates tried, each checked with `scripts/check_env.py` and one `--smoke`
run per method and encoder variant (TD-MPC2 and PWM `grasp` curriculum and
`reach_xy` single task; DreamerV3 `grasp`), seed 0, 3,000 steps per task:

| Candidate | Result |
| --- | --- |
| Python 3.10, torch 2.5.1, torchrl 0.6.0, MuJoCo 3.1.2 | All passed except PWM single-task: torch 2.5's `torch.compile` compares cached `TensorDictParams` with `!=` and fails. |
| Python 3.10, torch 2.6.0, torchrl/tensordict 0.7.2, MuJoCo 3.1.2 | All passed after two launcher-side fixes below. Adopted. |

Fixes made during validation, none in the training code:

- PWM single-task runs with `compile=false` (in its sweep configuration). The
  mixture-of-experts update fails under `torch.compile` with the stable
  torch/tensordict releases; the sequential PWM drivers already disable
  compilation for the same layer. This changes speed, not the algorithm.
- The launcher quotes the W&B run ID passed to Hydra, so IDs that look like
  numbers stay strings.
- Packages added that the original environments provided implicitly:
  `hydra-submitit-launcher` (selected by the TD-MPC2/PWM `config.yaml`) and
  `tensorboard` (imported by DreamerV3).

Results may differ slightly from the original runs because DreamerV3 moved
from MuJoCo 2.3.5 to 3.1.2, the torch/torchrl/tensordict nightlies became
stable releases, and PWM single-task runs uncompiled.

Smoke runs passed (2026-10-05): TD-MPC2 fine-tuning (scratch and pretrained),
ER, EWC, PackNet, single-task (scratch and pretrained); PWM progressive and
single-task (scratch and pretrained); DreamerV3 fine-tuning, ER, EWC, PackNet.
PWM static sequential was not run.
