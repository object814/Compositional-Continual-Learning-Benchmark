# Benchmarking World Models for Continual Learning on Compositional Tasks

### MuJoCo-Warp version

[Project page](https://object814.github.io/Compositional-Continual-Learning/) ·
[Paper (arXiv)](https://arxiv.org/abs/2609.22055) ·
[Citation](#citation)

Haoyu Zhou, Joe Watson, Anson Lei, Ingmar Posner<br>
Applied Artificial Intelligence Lab (A2I), Oxford Robotics Institute, University of Oxford

This branch (`mujoco-warp`) runs the benchmark with
[MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp): every task is
simulated and rendered on the GPU instead of the CPU, so training is faster.
Tasks, baselines, hyperparameters and commands are the same as on `main`.

**Use `main` to reproduce the paper.** This version follows the paper's tasks
step for step, but GPU physics (float32) and the GPU renderer make training
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
git checkout mujoco-warp
cp .env.example .env        # then set WANDB_API_KEY (or WANDB_MODE=offline)
```

Then set up **either** conda **or** Docker.

### Option A: conda

```bash
PYTHONNOUSERSITE=1 conda env create -f environment.yml      # creates cclb-mujoco-warp
conda activate cclb-mujoco-warp
set -a; source .env; set +a                                   # load W&B settings
```

To create the environment in a directory of your choice, use
`-p <dir>/cclb-mujoco-warp` in both commands. If `python` still points to
another installation after `conda activate` (`type -a python` shows it), check
that conda's own `bin` directory is not placed ahead of the environment on
your `PATH`.

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
docker build -t cclb-warp .
docker run -d --init --name cclb-warp --gpus all --shm-size 16g \
    --user "$(id -u):$(id -g)" --env-file .env -v "$PWD":/workspace cclb-warp
docker exec -it cclb-warp bash
docker stop cclb-warp             # `docker start cclb-warp` resumes it
docker rm -f cclb-warp
```

Rebuild only after changing `requirements.txt` or `Dockerfile`. To keep
training after you leave the shell, start it inside `tmux` (`tmux new -s
train`, detach with `Ctrl-b d`, reattach with `tmux attach -t train`).
`.env` is read when the container is created; after editing it, remove and
recreate the container.

### Pretrained encoder

Pretrained-encoder runs (TD-MPC2, PWM) load `checkpoints/encoder_ms19.pt`,
which is distributed separately:

```bash
mkdir -p checkpoints && cp /path/to/encoder_ms19.pt checkpoints/
```

### Caches

The first run compiles the GPU kernels, which takes a few minutes; they are
cached afterwards. This and every other cache the code writes (Warp,
`torch.compile`, W&B, ...) go to `.cache/` in the repository, not to your home
directory.

## Verify the installation

```bash
python scripts/check_env.py           # every line should read [ok]
python scripts/launch.py tdmpc2 sweep_sequential_regular.yaml --smoke --taskset grasp --encoder scratch
```

`check_env.py` checks packages and the GPU, and builds and renders all
benchmark tasks. `--smoke` trains for 3,000 steps per task and keeps no
output. With several GPUs, select one first, for example
`export CUDA_VISIBLE_DEVICES=1`.

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
task in a single batch. The CPU version of the same tasks (as on `main`) is in
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

## Repository layout

```
baselines/tdmpc2/      TD-MPC2: code (tdmpc2/) and sweep configurations (sweep/)
baselines/pwm/         PWM: code, sweep configurations, expert-reuse analysis
baselines/dreamerv3/   DreamerV3: code and sweep configurations
metaworld/             benchmark environments (CPU reference) and assets
mjwenv/                the same tasks on MuJoCo-Warp (GPU), used for training
scripts/launch.py      runs a sweep configuration
scripts/check_env.py   installation check
scripts/check_warp_parity.py, scripts/bench_warp.py   GPU-simulation checks
```

Each experiment is a YAML file in `baselines/<baseline>/sweep/`.
`scripts/launch.py` expands it into runs (tasksets × seeds × encoder
variants) and runs them one after another.

## Sequential training

By default each command runs every curriculum with seeds 0, 1 and 2 (and,
for TD-MPC2/PWM, each encoder variant listed in the configuration).

```bash
# TD-MPC2
python scripts/launch.py tdmpc2 sweep_sequential_regular.yaml   # fine-tuning
python scripts/launch.py tdmpc2 sweep_sequential_er.yaml
python scripts/launch.py tdmpc2 sweep_sequential_ewc.yaml
python scripts/launch.py tdmpc2 sweep_sequential_packnet.yaml

# PWM
python scripts/launch.py pwm sweep_prismatic_seq_progressive.yaml

# DreamerV3 (fine-tuning, ER, EWC and PackNet)
python scripts/launch.py dreamerv3 sweep_sequential.yaml
```

Useful options:

| Option | Effect |
| --- | --- |
| `--dry-run` | print the commands without running them |
| `--taskset grasp,reach` | only these curricula (see [The benchmark](#the-benchmark)) |
| `--seeds 0` | only these seeds |
| `--method er,ewc` | only these methods (DreamerV3 `sweep_sequential.yaml`) |
| `--encoder scratch` or `pretrained` | only this encoder variant (TD-MPC2, PWM) |
| `--rerun-failed` | skip runs that already finished |

Single-task baselines use `sweep_single.yaml` (TD-MPC2, DreamerV3) and
`sweep_single_prism.yaml` (PWM). Logs go to
`baselines/<baseline>/sweep/logdir/gpu_warp/`, and metrics to Weights & Biases
as set in `.env`. W&B run names and groups start with `gpu_warp_` (`cpu_` on
`main`), so runs of the two versions never share a log directory or a W&B run.

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

Budgets per task, in millions of steps, in curriculum order:

| Suite | TD-MPC2, PWM | DreamerV3 |
| --- | --- | --- |
| `reach` | 0.2, 0.2, 0.2, 0.2 | 0.4, 0.4, 0.4, 0.5 |
| `grasp` | 0.2, 0.6, 0.5 | 0.2, 0.4, 0.6 |
| `binpnp` | 0.8, 0.8, 0.8, 0.8 | 0.7, 0.5, 0.5, 0.5 |
| `pnpblock` | 0.8, 0.8, 1.8 | 0.8, 0.5, 0.8 |
| `drawerpnp` | 0.1, 0.5, 1.8 | 0.2, 0.5, 1.0 |
| `pnpboxclose` | 0.8, 0.6, 1.8 | 0.8, 0.5, 0.8 |

TD-MPC2 and PWM count agent steps: each action is repeated twice, and an
episode lasts 250 agent steps (500 simulator steps). DreamerV3 repeats no
actions, and its episodes last 250 steps. The sweep files in
`baselines/<baseline>/sweep/` hold these budgets and every other setting.

**Evaluation.** During sequential training, every task seen so far is
evaluated at a fixed interval: TD-MPC2 and PWM every 5,000 steps over 5
episodes, DreamerV3 every 10,000 steps over 10 episodes. The episode returns
are logged to W&B as `eval/task<i>_<task>/episode_reward` (TD-MPC2, PWM) and
`eval_task<i>_<task>/eval_return` (DreamerV3).

**Normalisation.** Returns are expert-normalised:
R = 100 · (return − f) / (R<sub>expert</sub> − f), where the floor f is the
return of an episode that gets the worst reward at every simulator step
(−500 for TD-MPC2 and PWM, −250 for DreamerV3), and R<sub>expert</sub> is the best of
10 rollouts of a privileged, state-based scripted policy,
listed in
[`expert_return.txt`](baselines/pwm/tdmpc2/expert_reuse/expert_return.txt).
So 0 is total failure and 100 matches the scripted policy; scores are not
capped above. Curves are averaged over the three seeds before the metrics are
computed, and integrals use the trapezoidal rule over the evaluation points.

**Backward transfer (BWT↓, forgetting).** Let b<sub>i</sub> be the step at
which training of task i ends, N the number of tasks, and
C<sub>i</sub> = R<sub>i</sub>(b<sub>i</sub>) the task's score at that point.
For every earlier task that was learnt (C<sub>i</sub> > 0), the drop below
C<sub>i</sub> over the rest of the curriculum is expressed as a fraction of
the largest possible drop:

```
D_i = 100 · ∫[b_i, b_N] max(0, C_i − R_i(s)) ds / ((b_N − b_i) · C_i)
BWT = mean of D_i over the earlier tasks with C_i > 0
```

BWT lies in [0, 100]: 0 if no task ever drops below its end-of-training score,
100 if every task collapses to the floor immediately. Tasks that were never
learnt are left out. This is a forgetting score, not the higher-is-better
backward transfer of Lopez-Paz and Ranzato (2017).

**Forward transfer (FWT↑, reuse).** Measured on the final task only, against
the same task learnt from scratch (the single-task sweeps):

```
FWT = (1/T) · ∫[0, T] (R_N^seq(t) − R_N^scratch(t)) dt
```

where R<sub>N</sub><sup>seq</sup> is the final task's curve within the
curriculum (restarted at t = 0), R<sub>N</sub><sup>scratch</sup> the
from-scratch curve, and T the training budget the two share. Positive values
mean the curriculum helped. The from-scratch reference for each suite is this
single-task configuration:

| Suite | Final task | Single-task taskset (`sweep_single*.yaml`) |
| --- | --- | --- |
| `reach` | reach-xyz | `reach_xyz` |
| `grasp` | pick-place | `pnpcube` |
| `binpnp` | bin-picking-yellowpurple | `binpnp` |
| `pnpblock` | compo-pickplace-block | `compo_pnpblock` |
| `drawerpnp` | compo-draweropen-pickplace | `compo_drawerpnp` |
| `pnpboxclose` | compo-pickplace-boxclose | `compo_pnpboxclose` |

## Reference results

From the paper, run on `main` (CPU MuJoCo); means over three seeds. FT is
sequential fine-tuning (`regular` in the sweep files). Runs on this branch
should come close but will not match exactly (see
[Speed and consistency](#speed-and-consistency)).

Averages over the six suites:

| | FT BWT↓ | FT FWT↑ | ER BWT↓ | ER FWT↑ | EWC BWT↓ | EWC FWT↑ | PackNet BWT↓ | PackNet FWT↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DreamerV3 | 34.14 | 26.18 | 15.69 | 33.55 | 12.95 | −1.42 | 1.04 | 23.86 |
| TD-MPC2 | 75.91 | 7.57 | 62.25 | 10.65 | 68.52 | −3.63 | 4.86 | 7.31 |

| | Scratch encoder BWT↓ | Scratch encoder FWT↑ | Pretrained frozen encoder BWT↓ | Pretrained frozen encoder FWT↑ |
| --- | --- | --- | --- | --- |
| PWM | 79.21 | 2.41 | 3.85 | 36.18 |
| TD-MPC2 (FT) | 75.91 | 7.57 | 34.97 | 38.11 |

<details>
<summary>Per-suite results</summary>

**DreamerV3**

| Suite | FT BWT↓ | FT FWT↑ | ER BWT↓ | ER FWT↑ | EWC BWT↓ | EWC FWT↑ | PackNet BWT↓ | PackNet FWT↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `reach` | 11.77 | 43.91 | 5.68 | 49.42 | 7.22 | −2.25 | 0.67 | 36.95 |
| `grasp` | 10.07 | 21.28 | 5.07 | 26.71 | 4.62 | −20.26 | 2.43 | 16.71 |
| `binpnp` | 29.27 | 42.09 | 6.46 | 49.26 | 8.36 | 0.90 | 1.57 | 38.00 |
| `pnpblock` | 31.80 | 11.64 | 11.80 | 36.37 | 4.36 | 8.74 | 0.71 | 9.20 |
| `drawerpnp` | 56.15 | 7.06 | 21.90 | 35.52 | 13.92 | −1.65 | 0.20 | 13.45 |
| `pnpboxclose` | 65.75 | 31.12 | 43.22 | 3.99 | 39.20 | 6.01 | 0.63 | 28.86 |

**TD-MPC2**

| Suite | FT BWT↓ | FT FWT↑ | ER BWT↓ | ER FWT↑ | EWC BWT↓ | EWC FWT↑ | PackNet BWT↓ | PackNet FWT↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `reach` | 38.14 | 10.47 | 25.38 | 12.07 | 23.09 | 10.35 | 0.77 | 12.94 |
| `grasp` | 62.62 | 27.17 | 46.70 | 24.74 | 51.54 | 17.27 | 8.98 | 19.04 |
| `binpnp` | 77.16 | 10.50 | 44.61 | 19.69 | 64.46 | 5.18 | 3.84 | 9.73 |
| `pnpblock` | 94.89 | 2.51 | 82.79 | 2.39 | 90.20 | −54.28 | 4.56 | −10.73 |
| `drawerpnp` | 89.59 | 1.07 | 88.51 | 1.81 | 91.65 | −3.53 | 4.10 | −0.17 |
| `pnpboxclose` | 93.07 | −6.27 | 85.48 | 3.19 | 90.16 | 3.21 | 6.90 | 13.04 |

**PWM and TD-MPC2 (FT) by encoder**

| Suite | PWM scratch BWT↓ | PWM scratch FWT↑ | TD-MPC2 scratch BWT↓ | TD-MPC2 scratch FWT↑ | PWM frozen BWT↓ | PWM frozen FWT↑ | TD-MPC2 frozen BWT↓ | TD-MPC2 frozen FWT↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `reach` | 44.45 | 9.35 | 38.14 | 10.47 | 2.47 | 5.84 | 35.07 | 10.64 |
| `grasp` | 90.73 | 14.49 | 62.62 | 27.17 | 5.69 | 48.29 | 13.80 | 49.44 |
| `binpnp` | 66.97 | 1.88 | 77.16 | 10.50 | 6.65 | 34.30 | 10.27 | 37.50 |
| `pnpblock` | 92.36 | −3.21 | 94.89 | 2.51 | 3.99 | 50.02 | 45.36 | 58.38 |
| `drawerpnp` | 90.81 | 2.01 | 89.59 | 1.07 | 1.64 | 41.02 | 59.06 | 40.22 |
| `pnpboxclose` | 89.95 | −10.09 | 93.07 | −6.27 | 2.68 | 37.62 | 46.24 | 32.51 |

Learning curves for every suite and method are on the
[project page](https://object814.github.io/Compositional-Continual-Learning/).

</details>

## Compute requirements

Each run uses one GPU. Our runs used one NVIDIA L40S (48 GB) and 64 GB of
RAM each. Training speed on this branch, measured on single-task runs:
TD-MPC2 about 52,000 steps per hour, PWM 28,500 and DreamerV3 44,000.
Sequential runs also evaluate every earlier task, so the hours below are
estimates (±30%) for one curriculum with one seed:

| Suite | TD-MPC2 | PWM | DreamerV3 |
| --- | --- | --- | --- |
| `reach` | 15 h | 28 h | 39 h |
| `grasp` | 25 h | 46 h | 27 h |
| `binpnp` | 62 h | 112 h | 50 h |
| `pnpblock` | 65 h | 119 h | 48 h |
| `drawerpnp` | 46 h | 84 h | 39 h |
| `pnpboxclose` | 62 h | 112 h | 48 h |
| all six | 275 h | 502 h | 250 h |

Running every configuration in the sweep files (all methods, encoder variants
and seeds, plus the single-task runs) takes about 13,500 GPU-hours on this
branch.

## Speed and consistency

Measured against `main` with the same tasks, seeds and settings, the two
versions running side by side on the same GPU:

| | Training speed-up |
| --- | --- |
| TD-MPC2 | about 2.7× |
| DreamerV3 | about 1.3× (most of its time is spent on gradient updates) |

**What is the same as `main`:** task layouts, resets, observations, rewards,
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
