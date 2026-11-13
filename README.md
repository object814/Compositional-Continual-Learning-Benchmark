# Benchmarking World Models for Continual Learning on Compositional Tasks

### MuJoCo-Warp version

[Project page](https://object814.github.io/Compositional-Continual-Learning/) ·
[Paper (arXiv)](https://arxiv.org/abs/2609.22055) ·
[Citation](#citation)

Haoyu Zhou, Joe Watson, Anson Lei, Ingmar Posner<br>
Applied Artificial Intelligence Lab (A2I), Oxford Robotics Institute, University of Oxford

This branch (`main`, the default) runs the benchmark with
[MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp): every task is
simulated and rendered on the GPU instead of the CPU, so training is faster.
Tasks, baselines, hyperparameters and commands are the same as on the
[`mujoco-cpu`](https://github.com/object814/Compositional-Continual-Learning-Benchmark/tree/mujoco-cpu) branch, which uses CPU MuJoCo as in the
paper.

**Use `mujoco-cpu` to reproduce the paper exactly.** This version follows the
paper's tasks step for step, but GPU physics (float32) and the GPU renderer make training
dynamics differ slightly from the published results (see
[Speed and consistency](#speed-and-consistency)).

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

## Installation

Requires Linux with an NVIDIA GPU (driver supporting CUDA 12.4).

```bash
git clone https://github.com/object814/Compositional-Continual-Learning-Benchmark.git
cd Compositional-Continual-Learning-Benchmark
cp .env.example .env        # then set WANDB_API_KEY (or WANDB_MODE=offline)
```

Then set up **either** conda **or** Docker.

### Option A: conda

```bash
PYTHONNOUSERSITE=1 conda env create -f environment.yml      # creates cclb-mujoco-warp
conda activate cclb-mujoco-warp
set -a; source .env; set +a                                   # load W&B settings
```

### Option B: Docker

Needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).
The repository is mounted at `/workspace`, so code changes need no rebuild.

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
docker build -t cclb-warp .
docker run -d --init --name cclb-warp --gpus all --shm-size 16g \
    --user "$(id -u):$(id -g)" --env-file .env -v "$PWD":/workspace cclb-warp
docker exec -it cclb-warp bash
docker stop cclb-warp             # `docker start cclb-warp` resumes it
docker rm -f cclb-warp
```

### Pretrained encoder

Runs with the pretrained, frozen encoder (TD-MPC2, PWM) load
[`checkpoints/encoder_ms19.pt`](checkpoints/encoder_ms19.pt), included in the
repository. It was pretrained on demonstrations from all benchmark tasks (see
the [paper](https://object814.github.io/Compositional-Continual-Learning/#experimental-setup)).

## Verify the installation

```bash
python scripts/check_env.py           # every line should read [ok]
python scripts/launch.py tdmpc2 sweep_sequential_regular.yaml --smoke --taskset grasp --encoder scratch
```

`check_env.py` checks the packages and the GPU and builds every task;
`--smoke` trains for 3,000 steps per task and keeps no output.

Two optional checks of the GPU simulation itself:

```bash
python scripts/check_warp_parity.py   # compare every task with the CPU version, step by step
python scripts/bench_warp.py          # time construction, reset and step of the GPU tasks
```

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

From the repository root, after activating the environment:

```python
from mjwenv import MetaWorldWarpEnv

env = MetaWorldWarpEnv("compo-pickplace-block", image_size=128, max_episode_steps=500)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`mjwenv.make_batched_env(task, num_envs=...)` simulates many worlds of one
task in a single batch. The CPU version of the same tasks (as on `mujoco-cpu`) is in
`metaworld/`:

```python
import gymnasium as gym
import metaworld  # registers Meta-World/MT1
from metaworld.wrappers import ProprioMultiImageObsWrapper

env = gym.make("Meta-World/MT1", env_name="compo-pickplace-block",
               render_mode="rgb_array", max_episode_steps=500)
env = ProprioMultiImageObsWrapper(env, image_height=128, image_width=128,
                                  camera_names=["topview", "front", "gripperPOV"])
```

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
metaworld/             benchmark environments (CPU reference) and assets
mjwenv/                the same tasks on MuJoCo-Warp (GPU), used for training
metaworld/expert_policy/   scripted expert policies and their rollout script
scripts/launch.py      runs a sweep configuration
scripts/check_env.py   installation check
scripts/check_warp_parity.py, scripts/bench_warp.py   GPU-simulation checks
```

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

Logs go to `baselines/<baseline>/sweep/logdir/gpu_warp/` and metrics to
Weights & Biases, with run names starting with `gpu_warp_` (`cpu_` on
`mujoco-cpu`).

**Resuming.** Run the same command again. Sequential runs save their models
at the end of each task and restart from the beginning of the first
unfinished task; single-task runs start over. `--rerun-failed` skips runs that
already finished; without it, a finished run is trained again as a new copy.

## Evaluation protocol

Each suite is trained in order with seeds 0, 1 and 2. Every task starts with
a fresh replay buffer, so only what the method carries over (weights, ER
episodes, EWC Fisher information, PackNet masks) crosses a task boundary. A
shared backbone (encoder, dynamics model and decoder, if any) learns across
tasks; each task has its own reward, actor and critic heads, frozen when the
task ends, and earlier tasks are evaluated with their heads and the current
backbone.

**Budgets.** The sweep files use the budgets below, in millions of agent
steps per task. They are a reference rather than part of the benchmark: FWT
and BWT normalise out the number of steps, so other budgets work as long as
the composition task gets the same budget in the curriculum as in its
single-task reference run.

| Suite | Budget per task | Single-task reference |
| --- | --- | --- |
| `reach` | 0.2, 0.2, 0.2, 0.2 | `reach_xyz` |
| `grasp` | 0.2, 0.6, 0.6 | `pnpcube` |
| `binpnp` | 0.8, 0.8, 0.8, 0.8 | `binpnp` |
| `pnpblock` | 0.8, 0.8, 1.8 | `compo_pnpblock` |
| `drawerpnp` | 0.2, 0.5, 1.8 | `compo_drawerpnp` |
| `pnpboxclose` | 0.8, 0.6, 1.8 | `compo_pnpboxclose` |

TD-MPC2 and PWM repeat each action twice (episodes of 250 agent steps, 500
simulator steps); DreamerV3 repeats none (episodes of 250 steps).

**Metrics.** Every task seen so far is evaluated during training: TD-MPC2 and
PWM every 5,000 steps over 5 episodes (W&B key
`eval/task<i>_<task>/episode_reward`), DreamerV3 every 10,000 steps over 10
episodes (`eval_task<i>_<task>/eval_return`). Returns are expert-normalised,
R = 100 · (return − f) / (R<sub>expert</sub> − f), where f is the return of an
episode with the worst reward at every simulator step (−500 for TD-MPC2 and
PWM, −250 for DreamerV3) and R<sub>expert</sub> is the best of 10 scripted-expert
rollouts
([`expert_return.txt`](baselines/pwm/tdmpc2/expert_reuse/expert_return.txt)).
Forward transfer (FWT↑) and backward transfer (BWT↓) are computed on these
curves, averaged over seeds, as defined in
[Appendix A of the project page](https://object814.github.io/Compositional-Continual-Learning/#appendix-a). Our results are in the
paper and in [Appendix B](https://object814.github.io/Compositional-Continual-Learning/#appendix-b).

## Compute requirements

Each run uses one GPU. Our runs used one NVIDIA L40S (48 GB) and 64 GB of
RAM each. The hours below are for one curriculum with one seed. They come from
the training speed of our sequential runs on this branch: about 38,000 steps
per hour for TD-MPC2, 24,000 for PWM and 58,000 for DreamerV3. TD-MPC2 and PWM
slow down as a curriculum goes on, since they evaluate every earlier task and
PWM adds experts, so allow ±30%.

| Suite | TD-MPC2 | PWM | DreamerV3 |
| --- | --- | --- | --- |
| `reach` | 21 h | 33 h | 14 h |
| `grasp` | 37 h | 58 h | 24 h |
| `binpnp` | 84 h | 133 h | 55 h |
| `pnpblock` | 89 h | 142 h | 59 h |
| `drawerpnp` | 66 h | 104 h | 43 h |
| `pnpboxclose` | 84 h | 133 h | 55 h |
| all six | 381 h | 603 h | 250 h |

Running every configuration in the sweep files (all methods, encoder variants
and seeds, plus the single-task runs) takes about 16,000 GPU-hours on this
branch.

**Smaller models.** We did not sweep model sizes: every run uses TD-MPC2's
19M-parameter model (PWM builds on it) and DreamerV3's 200M configuration.
Smaller models train faster and are a good place to start. For TD-MPC2 and
PWM, set `model_size` in the sweep files to 1 or 5 (with `--encoder scratch`;
the pretrained encoder fits size 19 only). For DreamerV3, reduce `dyn_hidden`,
`dyn_deter`, `units` and the encoder and decoder `cnn_depth` and `mlp_units`
of `metaworld_visual_200M_heavy_long_speedup` in
`baselines/dreamerv3/configs.yaml`.

## Speed and consistency

Measured against `mujoco-cpu` with the same tasks, seeds and settings, the two
versions running side by side on the same GPU:

| | Training speed-up |
| --- | --- |
| TD-MPC2 | about 2.7× |
| DreamerV3 | about 1.3× (most of its time is spent on gradient updates) |

**What is the same as `mujoco-cpu`:** task layouts, resets, observations, rewards,
success tests, episode length, and every training script, wrapper and
hyperparameter. Given the same simulator state, the GPU tasks return the same
observations (to about 1e-6), the same success flags and the same rewards (to
about 0.002) as the CPU tasks. TD-MPC2 (reach-xy, drawer-open, grasp) and
DreamerV3 (drawer-open) learned at the same rate on both versions, within
seed-to-seed variation.

**What differs slightly:**

- Physics runs in float32 on the GPU (MuJoCo 3.14 with MuJoCo-Warp), with
  MuJoCo 3.1.2's collision settings. Contact-rich motion therefore drifts
  slowly away from the CPU version over long rollouts.
- Images come from a GPU ray tracer: the same scene and cameras, but slightly
  different lighting and shading, with no reflections. Policies trained on
  one version are not expected to transfer to the other without retraining.
- DreamerV3 steps its environments in its own process instead of in worker
  processes.

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
`metaworld/LICENSE`), and `mjwenv/` re-implements them on
[MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp). The baselines
are adapted from TD-MPC2, PWM and DreamerV3; their licenses are in
`baselines/<baseline>/LICENSE`. If you use the environments, please also cite
Meta-World:

```bibtex
@inproceedings{mclean2025metaworld,
  title={Meta-World+: An Improved, Standardized, {RL} Benchmark},
  author={Reginald McLean and Evangelos Chatzaroulas and Luc McCutcheon and Frank R{\"o}der and Tianhe Yu and Zhanpeng He and K.R. Zentner and Ryan Julian and J K Terry and Isaac Woungang and Nariman Farsad and Pablo Samuel Castro},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2025},
  url={https://openreview.net/forum?id=1de3azE606}
}
```
