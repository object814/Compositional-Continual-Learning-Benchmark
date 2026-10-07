"""GPU (MuJoCo-Warp) versions of the benchmark tasks.

`MetaWorldWarpEnv` is what the training scripts use: one world behind the CPU
environment's gymnasium interface. `make_batched_env` gives the underlying
batched environment (many worlds per call).
"""
import os
from pathlib import Path

# Keep Warp's compiled-kernel cache in the repository (.cache/warp) rather than
# the home directory; set WARP_CACHE_PATH to put it elsewhere.
os.environ.setdefault("WARP_CACHE_PATH", str(Path(__file__).resolve().parents[1] / ".cache" / "warp"))

from .batched_env import BatchedSawyerEnv  # noqa: E402
from .gym_env import MetaWorldWarpEnv  # noqa: E402
from .tasks import TASKS, make_batched_env  # noqa: E402

__all__ = ["BatchedSawyerEnv", "MetaWorldWarpEnv", "TASKS", "make_batched_env"]
