# Compositional Continual Learning Benchmark

Training code for the compositional continual learning benchmark: a set of
Meta-World manipulation tasks arranged into curricula, where later tasks
compose skills from earlier ones, together with three model-based baselines.

| Baseline | Methods |
| --- | --- |
| TD-MPC2 | Single-task; sequential fine-tuning, experience replay (ER), EWC, PackNet |
| PWM | Single-task mixture of experts; progressive sequential mixture of experts |
| DreamerV3 | Single-task; sequential fine-tuning, ER, EWC, PackNet |

TD-MPC2 and PWM can train their CNN encoder from scratch or start from a
pretrained, frozen encoder.

## Requirements

- Linux (x86_64) with an NVIDIA GPU and a driver that supports CUDA 12.4
- Either conda (Miniforge, Miniconda or Anaconda) or Docker (no conda
  needed) with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

All baselines render MuJoCo scenes offscreen with OSMesa, on the CPU.

## Installation

Clone the repository and work from its root directory:

```bash
git clone https://github.com/object814/Compositional-Continual-Learning-Benchmark.git
cd Compositional-Continual-Learning-Benchmark
```

### Option A: conda

1. Install the system OSMesa library:

   ```bash
   sudo apt-get install -y libosmesa6 libgl1 libegl1 libglib2.0-0   # Debian/Ubuntu
   sudo dnf install -y mesa-libOSMesa                               # Fedora/RHEL/Rocky
   ```

2. Create and activate the environment. It installs Python 3.10, PyTorch 2.6
   (CUDA 12.4), MuJoCo 3.1.2 and the remaining packages from
   `requirements.txt`:

   ```bash
   PYTHONNOUSERSITE=1 conda env create -f environment.yml
   conda activate cclb
   ```

   `PYTHONNOUSERSITE=1` keeps packages from `~/.local` from being used in
   place of the environment's own; the activated environment sets it
   automatically, together with `MUJOCO_GL=osmesa`. Reading the conda-forge
   package index takes several GB of memory; if conda reports that `python`
   or `pip` cannot be found, run `conda clean --index-cache -y` and retry on a
   machine with more free memory.

### Option B: Docker

`docker/run.sh` wraps the Docker commands. Your user must be able to run
`docker` without `sudo` (see Docker's
[post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/)).
Do not run the helper with `sudo`: you would be root inside the container and
the files you create would belong to root.

1. Build the image. It contains Python 3.10 and the packages in
   `requirements.txt` (the same versions as the conda environment), but not
   the code: the repository is mounted when the container starts, so code
   changes need no rebuild. Rebuild after changes to `requirements.txt`,
   `Dockerfile` or `docker/entrypoint.sh`. The first build downloads several
   GB.

   ```bash
   bash docker/run.sh build
   ```

2. Create your W&B settings file (see [Weights & Biases](#weights--biases)
   for what to put in it):

   ```bash
   cp .env.example .env
   ```

   The container reads `.env` when it starts. After editing the file, restart
   the container with `bash docker/run.sh stop` followed by
   `bash docker/run.sh`.

3. Open a shell in the container, or run a single command:

   ```bash
   bash docker/run.sh
   bash docker/run.sh python scripts/check_env.py
   ```

   The first call starts a container named `cclb-<your user name>` in the
   background; later calls open new shells in the same container. Inside it
   `python` already has every package installed, you are a user with your
   host name and IDs and passwordless `sudo`, the
   repository is at `/workspace`, and files you write there, such as logs and
   checkpoints, are the same files as in your checkout. Commands in the rest
   of this guide run unchanged inside this shell.

4. Keep long runs going after you leave. The container keeps running when
   you `exit` a shell, so start runs inside `tmux`:

   ```bash
   bash docker/run.sh          # shell in the container
   tmux new -s train           # start a tmux session, then launch training
   # detach with Ctrl-b d, then exit the shell; training keeps running
   bash docker/run.sh          # later: a new shell in the same container
   tmux attach -t train        # back to the training session
   ```

5. When you are finished, stop and remove the container. This ends
   everything running inside it, including tmux sessions:

   ```bash
   bash docker/run.sh stop
   ```

   After rebuilding the image, `docker/run.sh` reminds you that the running
   container still uses the old one; stop it and open a new shell to switch.

Optional settings for `docker/run.sh`, set in your host shell, for example
`CCLB_SHM_SIZE=32g bash docker/run.sh`. Container settings take effect when a
container starts:

| Setting | Meaning | Default |
| --- | --- | --- |
| `CCLB_IMAGE` | Image name used by `build` and when starting the container. | `cclb` |
| `CCLB_CONTAINER` | Name of the container. Set different names to run several containers at once. | `cclb-<your user name>` |
| `CCLB_SHM_SIZE` | Shared memory for the container. Increase it if DreamerV3 reports shared-memory errors. | `16g` |

## Pretrained encoder

The pretrained-encoder runs of TD-MPC2 and PWM load
`checkpoints/encoder_ms19.pt`. The checkpoint is distributed separately from
the repository; place it there:

```bash
mkdir -p checkpoints
cp /path/to/encoder_ms19.pt checkpoints/
```

Runs that need the checkpoint stop before starting if it is missing. Runs
with encoders trained from scratch do not need it.

## Weights & Biases

Training logs metrics to [Weights & Biases](https://wandb.ai) (W&B). Put your
settings in a `.env` file at the repository root; it is never committed:

```bash
cp .env.example .env
```

Then edit `.env`:

| Setting | Meaning | Options |
| --- | --- | --- |
| `WANDB_API_KEY` | Your W&B API key. Required for online logging. | Copy it from <https://wandb.ai/authorize>. |
| `WANDB_ENTITY` | The W&B account or team that runs are logged under. | Leave it commented out to use your personal account, or set a team name. Overridden by `--wandb-entity TEAM` on the launcher. |
| `WANDB_MODE` | Where metrics go. | `online` (default when unset): upload to wandb.ai. `offline`: save only in each run's log directory; upload later with `wandb sync <run directory>/wandb/offline-run-*`. `disabled`: no W&B logging. |

Only `WANDB_API_KEY` is needed for the usual setup; leave the other two
commented out unless you need them. Without an API key, set `WANDB_MODE` to
`offline` or `disabled`. Write each value without quotes or spaces, for
example `WANDB_API_KEY=0123abcd`, because Docker keeps quotes as part of the
value.

- **Docker:** `docker/run.sh` passes `.env` into the container automatically
  each time a container starts.
- **Conda:** load `.env` into your shell before training:

  ```bash
  set -a; source .env; set +a
  ```

  Alternatively, run `wandb login` once, which stores the API key on your
  machine.

## Verify the installation

1. Check the environment. This reports package versions, tests the GPU,
   imports every training entry point, and builds and renders all 18
   benchmark tasks. It writes no files.

   ```bash
   python scripts/check_env.py
   ```

   Every line should read `[ok]`. Add `--quick` to build three tasks instead
   of 18.

2. Optionally, run short end-to-end smoke tests. `--smoke` trains each
   selected run for 3,000 steps per task with seed 0, logs to `smoke_logs/`,
   and logs W&B offline unless `WANDB_MODE` is set. One command per method
   and encoder variant:

   ```bash
   python scripts/launch.py tdmpc2 sweep_sequential_regular.yaml --smoke --taskset grasp
   python scripts/launch.py tdmpc2 sweep_sequential_er.yaml --smoke --taskset grasp
   python scripts/launch.py tdmpc2 sweep_sequential_ewc.yaml --smoke --taskset grasp
   python scripts/launch.py tdmpc2 sweep_sequential_packnet.yaml --smoke --taskset grasp
   python scripts/launch.py tdmpc2 sweep_single.yaml --smoke --taskset reach_xy
   python scripts/launch.py pwm sweep_prismatic_seq_progressive.yaml --smoke --taskset grasp
   python scripts/launch.py pwm sweep_single_prism.yaml --smoke --taskset reach_xy
   python scripts/launch.py dreamerv3 sweep_sequential.yaml --smoke --taskset grasp
   ```

   A single-task smoke run takes about 15 minutes and a three-task curriculum
   about 2 hours on one GPU, mostly spent rendering. Add `--encoder scratch`
   to skip the pretrained-encoder runs.

## Running experiments

Experiments are defined in `baselines/<baseline>/sweep/*.yaml` and started
with `scripts/launch.py`, which runs the selected jobs one after another in
the current Python environment. By default a configuration runs all of its
tasksets with seeds 0, 1 and 2:

| Baseline | Configuration | Runs |
| --- | --- | --- |
| TD-MPC2 | `sweep_single.yaml`: scratch and pretrained encoder | 33 + 33 |
| TD-MPC2 | `sweep_sequential_regular.yaml` (fine-tuning): scratch and pretrained encoder | 18 + 18 |
| TD-MPC2 | `sweep_sequential_er.yaml`, `sweep_sequential_ewc.yaml`, `sweep_sequential_packnet.yaml` | 18 each |
| PWM | `sweep_single_prism.yaml` (3 experts): scratch and pretrained encoder | 33 + 33 |
| PWM | `sweep_prismatic_seq_progressive.yaml`: scratch (trained throughout) and pretrained (frozen) encoder | 18 + 18 |
| PWM | `sweep_prismatic_seq_static.yaml` (static mixture of experts) | 18 |
| DreamerV3 | `sweep_single.yaml` | 33 |
| DreamerV3 | `sweep_sequential_regular.yaml`, `_er`, `_ewc`, `_packnet` (or all four in `sweep_sequential.yaml`) | 18 each |

Preview the commands with `--dry-run`, which prints them and writes nothing:

```bash
python scripts/launch.py tdmpc2 sweep_sequential_ewc.yaml --dry-run
```

Remove `--dry-run` to run them. Select a subset with comma-separated options:

```bash
python scripts/launch.py tdmpc2 sweep_sequential_regular.yaml --taskset reach,grasp --seeds 0
python scripts/launch.py dreamerv3 sweep_sequential.yaml --method er,ewc --taskset pnpboxclose
python scripts/launch.py pwm sweep_single_prism.yaml --encoder scratch --max-jobs 1
```

The printed commands can also be copied and run individually, for example on
a job scheduler; run them from the repository root.

### Encoder variants (TD-MPC2 and PWM)

Each configuration lists its variants under `encoder:`. `scratch` trains the
CNN encoder from scratch; `pretrained` loads `encoder.checkpoint` and freezes
it. Choose with `--encoder scratch` or `--encoder pretrained`, point to another
checkpoint with `--pretrained-encoder PATH`, or fine-tune the pretrained
encoder with `--no-freeze-encoder`. Pretrained runs get the suffix `_preEncFr`
(frozen) or `_preEnc` (fine-tuned) in their names and log directories.

### Logs, resuming and W&B

Runs write to `baselines/<baseline>/sweep/logdir/<run type>/<taskset>/seed_<n>/`,
for example `baselines/tdmpc2/sweep/logdir/sequential_ewc/grasp/seed_0/`.
A successful run creates `done.flag` there; rerunning a command with
`--rerun-failed` skips finished runs. Each training script keeps its own
checkpointing and resume behaviour when an interrupted run is relaunched.

Metrics go to W&B as configured in [Weights & Biases](#weights--biases). The
launcher stores each run's W&B ID in its log directory, so a resumed run
continues the same W&B run.

## Tasks and curricula

Sequential tasksets (curricula), trained in order:

| Taskset | Tasks |
| --- | --- |
| `binpnp` | bin-picking-redblue → bin-picking-yellowblue → bin-picking-redpurple → bin-picking-yellowpurple |
| `drawerpnp` | drawer-open → pick-place → compo-draweropen-pickplace |
| `pnpblock` | pick-place-redblock → pick-place-greenblock → compo-pickplace-block |
| `reach` | reach-xy → reach-xz → reach-yz → reach-xyz |
| `grasp` | reach → grasp → pick-place |
| `pnpboxclose` | pick-place-redblock → box-close → compo-pickplace-boxclose |

Single-task tasksets: `binpnp`, `boxclose`, `draweropen`, `graspcube`,
`pnpblock`, `pnpcube`, `reach_xy`, `reach_xyz`, `compo_drawerpnp`,
`compo_pnpblock`, `compo_pnpboxclose`.

## Repository layout

```
baselines/tdmpc2/      TD-MPC2 code and sweep configurations
baselines/pwm/         PWM code, sweep configurations and expert-reuse analysis
baselines/dreamerv3/   DreamerV3 code and sweep configurations
metaworld/             benchmark environments and assets
scripts/launch.py      experiment launcher
scripts/check_env.py   installation check
environment.yml        conda environment
requirements.txt       Python packages
Dockerfile             container image
docker/                container entrypoint and run helper
provenance/            migration and environment-validation records
```

PWM expert-reuse analysis is in `baselines/pwm/tdmpc2/expert_reuse/`; see its
README.

## License

This project is released under the MIT License (see `LICENSE`). Code adapted
from other projects keeps its original license, as listed below.

## Acknowledgements

The benchmark environments and assets in `metaworld/` are built upon the
[Meta-World](https://github.com/Farama-Foundation/Metaworld) benchmark; only
the tasks used here are kept, and Meta-World's MIT license is in
`metaworld/LICENSE`. The baselines are adapted from TD-MPC2, PWM and DreamerV3;
their licenses are in `baselines/<baseline>/LICENSE`. If you use the
environments, please also cite Meta-World:

```bibtex
@inproceedings{mclean2025metaworld,
  title={Meta-World+: An Improved, Standardized, {RL} Benchmark},
  author={Reginald McLean and Evangelos Chatzaroulas and Luc McCutcheon and Frank R{\"o}der and Tianhe Yu and Zhanpeng He and K.R. Zentner and Ryan Julian and J K Terry and Isaac Woungang and Nariman Farsad and Pablo Samuel Castro},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2025},
  url={https://openreview.net/forum?id=1de3azE606}
}
```
