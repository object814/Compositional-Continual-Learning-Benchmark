# Compositional Continual Learning Benchmark: MuJoCo-Warp version

This branch (`mujoco-warp`) runs the benchmark with
[MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp): every task is
simulated and rendered on the GPU instead of the CPU, so training is faster.
Tasks, baselines, hyperparameters and commands are the same as on `main`.

**Use `main` to reproduce the paper.** This version follows the paper's tasks
step for step, but GPU physics (float32) and the GPU renderer make training
dynamics differ slightly from the published results (see
[Speed and consistency](#speed-and-consistency)).

Meta-World manipulation tasks arranged into curricula, where later tasks
compose skills from earlier ones, with three model-based baselines:

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
| `--taskset grasp,reach` | only these curricula |
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

Curricula (`--taskset`), trained in order:

| Taskset | Tasks |
| --- | --- |
| `reach` | reach-xy → reach-xz → reach-yz → reach-xyz |
| `grasp` | reach → grasp → pick-place |
| `binpnp` | bin-picking-redblue → bin-picking-yellowblue → bin-picking-redpurple → bin-picking-yellowpurple |
| `drawerpnp` | drawer-open → pick-place → compo-draweropen-pickplace |
| `pnpblock` | pick-place-redblock → pick-place-greenblock → compo-pickplace-block |
| `pnpboxclose` | pick-place-redblock → box-close → compo-pickplace-boxclose |

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
