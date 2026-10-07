"""GPU (MuJoCo-Warp) versions of the benchmark tasks.

`MetaWorldWarpEnv` is what the training scripts use: one world behind the CPU
environment's gymnasium interface. `make_batched_env` gives the underlying
batched environment (many worlds per call).
"""
from .workspace import use_workspace_caches

use_workspace_caches()   # caches go to <repo>/.cache, not the home directory

from .batched_env import BatchedSawyerEnv  # noqa: E402
from .gym_env import MetaWorldWarpEnv  # noqa: E402
from .tasks import TASKS, make_batched_env  # noqa: E402

__all__ = ["BatchedSawyerEnv", "MetaWorldWarpEnv", "TASKS", "make_batched_env"]
