"""One MuJoCo-Warp world behind the CPU environment's gymnasium interface.

`MetaWorldWarpEnv(task, image_size, cameras, max_episode_steps)` replaces

    env = gymnasium.make("Meta-World/MT1", env_name=task, render_mode="rgb_array",
                         max_episode_steps=max_episode_steps)
    env = ProprioMultiImageObsWrapper(env, image_size, image_size, cameras)

in the training scripts: the same observation dict ('proprio', 'image',
'original_obs'), action space, raw reward in [-1, 1], termination on success
and truncation at `max_episode_steps`. Every wrapper the scripts add on top
(reward scaling, action repeat, time limit, ...) is unchanged.

Building a warp env takes seconds (model upload, CUDA-graph capture), and the
scripts create a new env for every evaluation, so closed or garbage-collected
envs go back to a pool and are reused by the next env of the same task.
"""
from __future__ import annotations

import weakref

import gymnasium
import numpy as np

from .tasks import TASKS, make_batched_env

DEFAULT_CAMERAS = ("topview", "front", "gripperPOV")

_POOL: dict[tuple, list] = {}


def _release(key, env):
    _POOL.setdefault(key, []).append(env)


def task_name(task: str) -> str:
    """'metaworld_<name>', 'mw-<name>' or '<name>' -> '<name>'."""
    for prefix in ("metaworld_", "mw-"):
        if task.startswith(prefix):
            task = task[len(prefix):]
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; available: {sorted(TASKS)}")
    return task


class MetaWorldWarpEnv(gymnasium.Env):
    """A single benchmark task simulated and rendered with MuJoCo-Warp."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task: str,
        image_size: int = 128,
        cameras=DEFAULT_CAMERAS,
        max_episode_steps: int = 250,
        seed: int | None = None,
    ):
        self.task = task_name(task)
        self.render_mode = "rgb_array"
        self.cameras = tuple(cameras)
        if seed is None:
            # Follow the global numpy seed, as the CPU env's task sampling does.
            seed = int(np.random.randint(2**31 - 1))
        key = (self.task, int(image_size), self.cameras, int(max_episode_steps))
        pool = _POOL.get(key)
        if pool:
            self._env = pool.pop()
            self._env.reseed(seed)
        else:
            self._env = make_batched_env(
                self.task, num_envs=1, image_size=image_size, cameras=self.cameras,
                max_episode_steps=max_episode_steps, seed=seed)
        self._release = weakref.finalize(self, _release, key, self._env)

        low, high = self._env.action_bounds
        self.action_space = gymnasium.spaces.Box(low, high, dtype=np.float32)
        h = w = int(image_size)
        self.observation_space = gymnasium.spaces.Dict({
            "proprio": gymnasium.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            "image": gymnasium.spaces.Box(0, 255, shape=(h, w, 3 * len(self.cameras)), dtype=np.uint8),
        })
        self._last_image = None

    def _obs(self, obs):
        self._last_image = obs["image"][0]
        return {"proprio": obs["proprio"][0], "image": obs["image"][0], "original_obs": obs["original_obs"][0]}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs(self._env.reset()), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(1, 4)
        obs, reward, terminated, truncated, info = self._env.step(action)
        info = {k: float(v[0]) for k, v in info.items()}
        return self._obs(obs), float(reward[0]), bool(terminated[0]), bool(truncated[0]), info

    def render(self):
        """The observation cameras of the last frame, side by side (H, W * n_cams, 3)."""
        if self._last_image is None:
            return None
        img = self._last_image
        return np.concatenate([img[..., 3 * i:3 * i + 3] for i in range(len(self.cameras))], axis=1)

    def close(self):
        self._release()
