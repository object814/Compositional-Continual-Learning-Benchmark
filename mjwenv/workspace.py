"""Keep every cache the training stack writes inside the repository (.cache/).

By default Warp, Triton (torch.compile), the CUDA JIT, matplotlib, Mesa and
W&B write caches and settings to the home directory. Importing `mjwenv` (as
every training script does) points them at <repo>/.cache/ instead, so the
checkout is self-contained. A variable that is already set is left alone.
CUDA_CACHE_PATH takes effect only if this runs before CUDA is initialised,
which is the case when `mjwenv` is imported at the top of a script.
"""
import os
from pathlib import Path

CACHE_ROOT = Path(__file__).resolve().parents[1] / ".cache"

_DIRS = {
    "XDG_CACHE_HOME": "",                    # pip, matplotlib, Mesa, torch hub, ...
    "XDG_CONFIG_HOME": "config",
    "XDG_DATA_HOME": "data",
    "WARP_CACHE_PATH": "warp",               # compiled MuJoCo-Warp kernels
    "TRITON_HOME": "triton",                 # torch.compile
    "TRITON_CACHE_DIR": "triton/cache",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "CUDA_CACHE_PATH": "nv",                 # CUDA driver JIT
    "WANDB_CONFIG_DIR": "wandb/config",
    "WANDB_CACHE_DIR": "wandb/cache",
    "WANDB_DATA_DIR": "wandb/data",
}


def use_workspace_caches():
    for var, sub in _DIRS.items():
        os.environ.setdefault(var, str(CACHE_ROOT / sub))
