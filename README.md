# Compositional Continual Learning Benchmark

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

Pretrained-encoder runs (TD-MPC2, PWM) load `checkpoints/encoder_ms19.pt`,
which is distributed separately:

```bash
mkdir -p checkpoints && cp /path/to/encoder_ms19.pt checkpoints/
```

## Verify the installation

```bash
python scripts/check_env.py           # every line should read [ok]
python scripts/launch.py tdmpc2 sweep_sequential_regular.yaml --smoke --taskset grasp --encoder scratch
```

`check_env.py` checks packages and the GPU, and builds and renders all
benchmark tasks. `--smoke` trains for 3,000 steps per task and keeps no
output. With several GPUs, select one first, for example
`export CUDA_VISIBLE_DEVICES=1`.

## Repository layout

```
baselines/tdmpc2/      TD-MPC2: code (tdmpc2/) and sweep configurations (sweep/)
baselines/pwm/         PWM: code, sweep configurations, expert-reuse analysis
baselines/dreamerv3/   DreamerV3: code and sweep configurations
metaworld/             benchmark environments and assets
scripts/launch.py      runs a sweep configuration
scripts/check_env.py   installation check
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
`baselines/<baseline>/sweep/logdir/`, and metrics to Weights & Biases as set
in `.env`.

Curricula (`--taskset`), trained in order:

| Taskset | Tasks |
| --- | --- |
| `reach` | reach-xy → reach-xz → reach-yz → reach-xyz |
| `grasp` | reach → grasp → pick-place |
| `binpnp` | bin-picking-redblue → bin-picking-yellowblue → bin-picking-redpurple → bin-picking-yellowpurple |
| `drawerpnp` | drawer-open → pick-place → compo-draweropen-pickplace |
| `pnpblock` | pick-place-redblock → pick-place-greenblock → compo-pickplace-block |
| `pnpboxclose` | pick-place-redblock → box-close → compo-pickplace-boxclose |

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
