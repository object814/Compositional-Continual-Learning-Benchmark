# Benchmarking World Models for Continual Learning on Compositional Tasks

[Project page](https://object814.github.io/Compositional-Continual-Learning/) ·
[Paper (arXiv)](https://arxiv.org/abs/2609.22055) ·
[Citation](#citation)

### MuJoCo CPU version

Haoyu Zhou, Joe Watson, Anson Lei, Ingmar Posner<br>
Applied Artificial Intelligence Lab (A2I), Oxford Robotics Institute, University of Oxford

![Overview of the benchmark](https://object814.github.io/Compositional-Continual-Learning/static/images/teaser_figure.png)

The benchmark arranges Meta-World manipulation tasks into curricula that end
with a composition task built from the primitives before it. Forward transfer
on that final task measures how well a world model reuses what it learnt;
evaluating the earlier tasks as training goes on measures forgetting. The
compositions vary what the robot does (action), what it sees (perception), or
both. Three model-based baselines are included:

| Baseline | Methods |
| --- | --- |
| TD-MPC2 | single-task; sequential fine-tuning, experience replay (ER), EWC, PackNet |
| PWM | single-task mixture of experts; progressive sequential mixture of experts |
| DreamerV3 | single-task; sequential fine-tuning, ER, EWC, PackNet |

TD-MPC2 and PWM train their CNN encoder from scratch or start from a
pretrained, frozen encoder.

**This is the `mujoco-cpu` branch**, which simulates every task with CPU
MuJoCo, as in the paper. The default [`main`](https://github.com/object814/Compositional-Continual-Learning-Benchmark/tree/main) branch runs
the same tasks, baselines and commands with
[MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp), simulating and
rendering every task on the GPU: training is about 2.7× faster for TD-MPC2 and
1.3× faster for DreamerV3. Its float32 physics and GPU renderer change training
dynamics slightly, so use this branch (`mujoco-cpu`) to reproduce the paper exactly.

## Installation

Requires Linux with an NVIDIA GPU (driver supporting CUDA 12.4).

```bash
git clone https://github.com/object814/Compositional-Continual-Learning-Benchmark.git
cd Compositional-Continual-Learning-Benchmark
git checkout mujoco-cpu
cp .env.example .env        # then set WANDB_API_KEY (or WANDB_MODE=offline)
```

Then set up **either** conda **or** Docker.

### Option A: conda

```bash
sudo apt-get install -y libosmesa6 libgl1 libegl1 libglib2.0-0   # or: sudo dnf install -y mesa-libOSMesa
PYTHONNOUSERSITE=1 conda env create -f environment.yml
conda activate cclb
set -a; source .env; set +a                                      # load W&B settings
```

### Option B: Docker

Needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).
The image contains only the environment; the repository is mounted at
`/workspace`, so code changes need no rebuild.

With Docker Compose:

```bash
printf 'HOST_UID=%s\nHOST_GID=%s\n' "$(id -u)" "$(id -g)" >> .env   # once: run as your user

docker compose up -d --build      # build the image and start the container in the background
docker compose exec cclb bash     # open a shell (`exit` leaves the container running)
docker compose stop               # stop it; `docker compose start` resumes it
docker compose down               # stop and remove it
```

Or with plain Docker:

```bash
docker build -t cclb .
docker run -d --init --name cclb --gpus all --shm-size 16g \
    --user "$(id -u):$(id -g)" --env-file .env -v "$PWD":/workspace cclb
docker exec -it cclb bash
docker stop cclb                  # `docker start cclb` resumes it
docker rm -f cclb
```

Rebuild only after changing `requirements.txt` or `Dockerfile`. To keep
training after you leave the shell, start it inside `tmux` (`tmux new -s
train`, detach with `Ctrl-b d`, reattach with `tmux attach -t train`).
`.env` is read when the container is created; after editing it, remove and
recreate the container.

### Pretrained encoder

The pretrained-encoder runs (TD-MPC2, PWM) load
[`checkpoints/encoder_ms19.pt`](checkpoints/encoder_ms19.pt), which comes with
the repository. This CNN encoder was pretrained on demonstrations from all
benchmark tasks and is kept frozen during training, so these runs have seen
the full task distribution (the frozen-encoder diagnostic in the
[paper](https://object814.github.io/Compositional-Continual-Learning/#experimental-setup)).

### Caches

Every cache the code writes (`torch.compile`, Mesa, W&B, ...) goes to
`.cache/` in the repository, not to your home directory.

## Verify the installation

```bash
python scripts/check_env.py           # every line should read [ok]
python scripts/launch.py tdmpc2 sweep_sequential_regular.yaml --smoke --taskset grasp --encoder scratch
```

`check_env.py` checks packages and the GPU, and builds and renders all
benchmark tasks. `--smoke` trains for 3,000 steps per task and keeps no
output. With several GPUs, select one first, for example
`export CUDA_VISIBLE_DEVICES=1`.

## The benchmark

Six task suites (`--taskset`), each trained in order. The last task of every
suite composes the ones before it.

| Suite | Composition | Primitive tasks, in order | Composition task |
| --- | --- | --- | --- |
| `reach` | action | `reach-xy`, `reach-xz`, `reach-yz`: move the gripper to a target in one plane | `reach-xyz`: reach a target anywhere in 3D |
| `grasp` | action | `reach`: move the gripper to a target, with the cube on the table; `grasp`: pick up the cube | `pick-place`: pick up the cube and place it at a target |
| `binpnp` | perception | `bin-picking-redblue`, `-yellowblue`, `-redpurple`: move the cube from the first bin to the second (bins coloured as named) | `bin-picking-yellowpurple`: the same, from the yellow bin to the purple bin |
| `pnpblock` | perception | `pick-place-redblock`, `pick-place-greenblock`: place one block at a target | `compo-pickplace-block`: place the red block, then stack the green block on it |
| `drawerpnp` | full | `drawer-open`; `pick-place`: place the cube at a target | `compo-draweropen-pickplace`: open the drawer, then place the cube inside |
| `pnpboxclose` | full | `pick-place-redblock`; `box-close`: put the lid on the box | `compo-pickplace-boxclose`: place the block inside the box, then close the lid |

Task names are Meta-World environment names; all but the composition tasks
end in `-v3` (for example `reach-xy-v3`).

**Observations.** A dictionary with

- `proprio` (7, float32): gripper position (3), gripper velocity (3) and
  gripper opening (1);
- `image` (128 × 128 × 9, uint8): RGB images from the `topview`, `front` and
  `gripperPOV` cameras, stacked along the channel axis;
- `original_obs` (39): Meta-World's state vector (gripper, object
  poses, previous frame, goal). It is privileged information, and the baselines
  do not use it.

**Actions.** 4 numbers in [-1, 1]: the end-effector displacement in x, y, z and
the gripper command.

**Rewards and episodes.** The environment returns a dense reward in [-1, 1] per
step; the training scripts rescale it to [-1, 0] as (r - 1) / 2. An episode
terminates when the task succeeds (`info['success']`) and is truncated after
`max_episode_steps`.

### Using the environments

From the repository root, after activating the environment (it sets
`MUJOCO_GL=osmesa` for offscreen rendering):

```python
import gymnasium as gym
import metaworld  # registers Meta-World/MT1
from metaworld.wrappers import ProprioMultiImageObsWrapper

env = gym.make("Meta-World/MT1", env_name="compo-pickplace-block",
               render_mode="rgb_array", max_episode_steps=500)
env = ProprioMultiImageObsWrapper(env, image_height=128, image_width=128,
                                  camera_names=["topview", "front", "gripperPOV"])
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

On the `main` branch, `mjwenv.MetaWorldWarpEnv` provides the same tasks
with the same interface on the GPU, and `mjwenv.make_batched_env` simulates
many worlds of one task in a single batch.

### Scripted experts

Every task has a privileged, state-based scripted policy in
`metaworld/expert_policy/`; the best of its returns is the 100 of the
normalised scores (see [Evaluation protocol](#evaluation-protocol)). To roll
them out and record success rates, videos or demonstrations:

```bash
python -m metaworld.expert_policy.rollout                    # every task, 10 episodes each
python -m metaworld.expert_policy.rollout --tasks reach-xy-v3 --episodes 20 --videos 3
python -m metaworld.expert_policy.rollout --npz              # also save every episode as .npz
```

Results go to `outputs/expert_rollouts/`.

## Repository layout

```
baselines/tdmpc2/      TD-MPC2: code (tdmpc2/) and sweep configurations (sweep/)
baselines/pwm/         PWM: code, sweep configurations, expert-reuse analysis
baselines/dreamerv3/   DreamerV3: code and sweep configurations
metaworld/             benchmark environments and assets
metaworld/expert_policy/   scripted expert policies and their rollout script
scripts/launch.py      runs a sweep configuration
scripts/check_env.py   installation check
```

Each experiment is a YAML file in `baselines/<baseline>/sweep/`.
`scripts/launch.py` expands it into runs (tasksets × seeds × encoder
variants) and runs them one after another.

## Training

Every experiment is a sweep file in `baselines/<baseline>/sweep/`, started
with `python scripts/launch.py <baseline> <sweep file>`. A sweep file runs all
its curricula (or single tasks) with seeds 0, 1 and 2, one run after another;
the options further down select part of it.

**Continual-learning runs**, one command per configuration:

```bash
# TD-MPC2: fine-tuning (FT), with the CNN encoder trained from scratch or pretrained and frozen
python scripts/launch.py tdmpc2 sweep_sequential_regular.yaml --encoder scratch
python scripts/launch.py tdmpc2 sweep_sequential_regular.yaml --encoder pretrained
# TD-MPC2: experience replay (ER), EWC and PackNet
python scripts/launch.py tdmpc2 sweep_sequential_er.yaml
python scripts/launch.py tdmpc2 sweep_sequential_ewc.yaml
python scripts/launch.py tdmpc2 sweep_sequential_packnet.yaml

# PWM: progressive mixture of experts, encoder from scratch or pretrained and frozen
python scripts/launch.py pwm sweep_prismatic_seq_progressive.yaml --encoder scratch
python scripts/launch.py pwm sweep_prismatic_seq_progressive.yaml --encoder pretrained

# DreamerV3: FT, ER, EWC and PackNet
python scripts/launch.py dreamerv3 sweep_sequential.yaml --method regular
python scripts/launch.py dreamerv3 sweep_sequential.yaml --method er
python scripts/launch.py dreamerv3 sweep_sequential.yaml --method ewc
python scripts/launch.py dreamerv3 sweep_sequential.yaml --method packnet
```

Without `--encoder` or `--method`, a command runs every variant in its file.

**Single-task runs**, which are also the from-scratch references for forward
transfer:

```bash
python scripts/launch.py tdmpc2 sweep_single.yaml --encoder scratch       # or pretrained
python scripts/launch.py pwm sweep_single_prism.yaml --encoder scratch    # or pretrained
python scripts/launch.py dreamerv3 sweep_single.yaml
```

Their tasksets are `reach_xy`, `reach_xyz`, `draweropen`, `graspcube`,
`pnpcube`, `pnpblock`, `boxclose`, `binpnp`, `compo_drawerpnp`,
`compo_pnpblock` and `compo_pnpboxclose`.

For example, one TD-MPC2 EWC run on the `grasp` suite with seed 0:

```bash
python scripts/launch.py tdmpc2 sweep_sequential_ewc.yaml --taskset grasp --seeds 0
```

Useful options:

| Option | Effect |
| --- | --- |
| `--dry-run` | print the commands without running them |
| `--taskset grasp,reach` | only these curricula (see [The benchmark](#the-benchmark)) or single-task tasksets |
| `--seeds 0` | only these seeds |
| `--method er,ewc` | only these methods (DreamerV3: `regular`, `er`, `ewc`, `packnet`) |
| `--encoder scratch` or `pretrained` | only this encoder variant (TD-MPC2, PWM) |
| `--rerun-failed` | skip runs that already finished |

Logs go to
`baselines/<baseline>/sweep/logdir/cpu/`, and metrics to Weights & Biases as
set in `.env`. W&B run names and groups start with `cpu_` (`gpu_warp_` on the
`main` branch), so runs of the two versions never share a log directory
or a W&B run.

**Resuming.** If a run stops, run the same command again (add
`--rerun-failed` to skip finished runs). Each task's models, and what the
method carries to later tasks (ER episodes, EWC Fisher, PackNet masks), are
saved when the task ends, and `sequential_progress.json` in the run's log
directory records which tasks are complete. A resumed run skips those and
starts the first unfinished task again from its beginning, seeded as in an
uninterrupted run; nothing is saved during a task, so training is not slowed
down. In Weights & Biases the resumed run continues in a new run holding the
history of the completed tasks; the interrupted one is moved to the group
`<group>_failed`. Single-task runs of every baseline restart from the
beginning in the same way; their model is saved once, when training ends.
A finished run is never overwritten: without `--rerun-failed`, running it
again trains a new copy in `<logdir>_<timestamp>`, logged to Weights & Biases
as `<run name>_<timestamp>`.

## Evaluation protocol

To compare a new method with the results below, keep the following fixed.

**Training.** Each suite is trained in order, with the per-task budgets below
and seeds 0, 1 and 2. Every task starts with a fresh replay buffer, so only
what the method itself carries over (weights, ER episodes, EWC Fisher
information, PackNet masks) crosses a task boundary. A shared backbone
(encoder, dynamics model and decoder, if any) learns across all tasks; each
task gets its own reward, actor and critic heads, which are frozen when the
task ends. Earlier tasks are evaluated with their stored heads and the current
backbone.

Budgets per task, in millions of steps, in curriculum order. They are the
same for every baseline:

| Suite | Budget per task |
| --- | --- |
| `reach` | 0.2, 0.2, 0.2, 0.2 |
| `grasp` | 0.2, 0.6, 0.6 |
| `binpnp` | 0.8, 0.8, 0.8, 0.8 |
| `pnpblock` | 0.8, 0.8, 1.8 |
| `drawerpnp` | 0.2, 0.5, 1.8 |
| `pnpboxclose` | 0.8, 0.6, 1.8 |

TD-MPC2 and PWM count agent steps: each action is repeated twice, and an
episode lasts 250 agent steps (500 simulator steps). DreamerV3 repeats no
actions, and its episodes last 250 steps. The sweep files in
`baselines/<baseline>/sweep/` hold these budgets and every other setting.

<details>
<summary>Budgets of the paper's runs</summary>

The [reference results](#reference-results) come from runs with per-baseline
budgets, in millions of steps:

| Suite | TD-MPC2, PWM | DreamerV3 |
| --- | --- | --- |
| `reach` | 0.2, 0.2, 0.2, 0.2 | 0.4, 0.4, 0.4, 0.5 |
| `grasp` | 0.2, 0.6, 0.5 | 0.2, 0.4, 0.6 |
| `binpnp` | 0.8, 0.8, 0.8, 0.8 | 0.7, 0.5, 0.5, 0.5 |
| `pnpblock` | 0.8, 0.8, 1.8 | 0.8, 0.5, 0.8 |
| `drawerpnp` | 0.1, 0.5, 1.8 | 0.2, 0.5, 1.0 |
| `pnpboxclose` | 0.8, 0.6, 1.8 | 0.8, 0.5, 0.8 |

The single-task budgets differed between baselines too. The sweep files at
commit `7cd687d` of the `mujoco-cpu` branch hold all of the paper's budgets.

</details>

**Evaluation.** During sequential training, every task seen so far is
evaluated at a fixed interval: TD-MPC2 and PWM every 5,000 steps over 5
episodes, DreamerV3 every 10,000 steps over 10 episodes. The episode returns
are logged to W&B as `eval/task<i>_<task>/episode_reward` (TD-MPC2, PWM) and
`eval_task<i>_<task>/eval_return` (DreamerV3).

**Metrics.** Forward transfer (FWT↑: how much the earlier tasks help on the
final composition task, compared with learning it from scratch) and backward
transfer (BWT↓: forgetting of the earlier tasks) are defined in
[Appendix A of the project page](https://object814.github.io/Compositional-Continual-Learning/#appendix-a). Both are computed on
expert-normalised returns, R = 100 · (return − f) / (R<sub>expert</sub> − f),
where the floor f is the return of an episode that gets the worst reward at
every simulator step (−500 for TD-MPC2 and PWM, −250 for DreamerV3), and
R<sub>expert</sub> is the best of 10 rollouts of a privileged, state-based
scripted policy, listed in
[`expert_return.txt`](baselines/pwm/tdmpc2/expert_reuse/expert_return.txt).
So 0 is total failure and 100 matches the scripted policy. Curves are averaged
over the three seeds before the metrics are computed. The from-scratch
reference for FWT is this single-task configuration:

| Suite | Final task | Single-task taskset (`sweep_single*.yaml`) | Budget (M steps) |
| --- | --- | --- | --- |
| `reach` | reach-xyz | `reach_xyz` | 0.2 |
| `grasp` | pick-place | `pnpcube` | 0.6 |
| `binpnp` | bin-picking-yellowpurple | `binpnp` | 0.8 |
| `pnpblock` | compo-pickplace-block | `compo_pnpblock` | 1.8 |
| `drawerpnp` | compo-draweropen-pickplace | `compo_drawerpnp` | 1.5 |
| `pnpboxclose` | compo-pickplace-boxclose | `compo_pnpboxclose` | 1.5 |

## Reference results

From the paper, run with this branch; means over three seeds. FT is
sequential fine-tuning (`regular` in the sweep files). These runs used the
paper's per-baseline budgets (see [Evaluation protocol](#evaluation-protocol)).

Averages over the six suites:

| | FT BWT↓ | FT FWT↑ | ER BWT↓ | ER FWT↑ | EWC BWT↓ | EWC FWT↑ | PackNet BWT↓ | PackNet FWT↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DreamerV3 | 34.14 | 26.18 | 15.69 | 33.55 | 12.95 | −1.42 | 1.04 | 23.86 |
| TD-MPC2 | 75.91 | 7.57 | 62.25 | 10.65 | 68.52 | −3.63 | 4.86 | 7.31 |

| | Scratch encoder BWT↓ | Scratch encoder FWT↑ | Pretrained frozen encoder BWT↓ | Pretrained frozen encoder FWT↑ |
| --- | --- | --- | --- | --- |
| PWM | 79.21 | 2.41 | 3.85 | 36.18 |
| TD-MPC2 (FT) | 75.91 | 7.57 | 34.97 | 38.11 |

Per-suite scores are in [Appendix B](https://object814.github.io/Compositional-Continual-Learning/#appendix-b) of the project page,
and the learning curves for every suite and method in
[Appendix C](https://object814.github.io/Compositional-Continual-Learning/#appendix-c).

## Compute requirements

Each run uses one GPU, and the simulation runs on the CPU. Our runs used one
NVIDIA L40S (48 GB) and 64 GB of RAM each. The hours below are estimates for
one curriculum with one seed. They are derived from the speeds of our
sequential runs on the `main` branch (MuJoCo-Warp) and its measured speed-up
over this branch (2.7× for TD-MPC2, 1.3× for DreamerV3), giving about 14,000
steps per hour for TD-MPC2 and 45,000 for DreamerV3. PWM's speed on this
branch was not measured; its estimate of about 12,000 steps per hour assumes
the same simulation cost per step as TD-MPC2. Allow ±30%.

| Suite | TD-MPC2 | PWM | DreamerV3 |
| --- | --- | --- | --- |
| `reach` | 57 h | 69 h | 18 h |
| `grasp` | 99 h | 121 h | 31 h |
| `binpnp` | 227 h | 276 h | 72 h |
| `pnpblock` | 242 h | 294 h | 76 h |
| `drawerpnp` | 178 h | 216 h | 56 h |
| `pnpboxclose` | 227 h | 276 h | 72 h |
| all six | 1,030 h | 1,252 h | 325 h |

Running every configuration in the sweep files (all methods, encoder variants
and seeds, plus the single-task runs) takes about 34,000 GPU-hours on this
branch, and about 16,000 on `main` (MuJoCo-Warp).

## Citation

If you use this benchmark or code, please cite:

```bibtex
@article{zhou2026benchmarking,
  title={Benchmarking World Models for Continual Learning on Compositional Tasks},
  author={Zhou, Haoyu and Watson, Joe and Lei, Anson and Posner, Ingmar},
  journal={arXiv preprint arXiv:2609.22055},
  year={2026}
}
```

## License

MIT (see `LICENSE`). Code adapted from other projects keeps its original
license.

## Acknowledgements

The environments in `metaworld/` are built upon
[Meta-World](https://github.com/Farama-Foundation/Metaworld) (MIT license in
`metaworld/LICENSE`). The baselines are adapted from TD-MPC2, PWM and
DreamerV3; their licenses are in `baselines/<baseline>/LICENSE`. If you use
the environments, please also cite Meta-World:

```bibtex
@inproceedings{mclean2025metaworld,
  title={Meta-World+: An Improved, Standardized, {RL} Benchmark},
  author={Reginald McLean and Evangelos Chatzaroulas and Luc McCutcheon and Frank R{\"o}der and Tianhe Yu and Zhanpeng He and K.R. Zentner and Ryan Julian and J K Terry and Isaac Woungang and Nariman Farsad and Pablo Samuel Castro},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2025},
  url={https://openreview.net/forum?id=1de3azE606}
}
```
