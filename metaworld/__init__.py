"""The public-facing Metaworld API."""

from __future__ import annotations

import abc
import pickle
from collections import OrderedDict
from typing import Any, Literal

import gymnasium as gym  # type: ignore
import numpy as np
import numpy.typing as npt

# noqa: D104
from gymnasium.envs.registration import register

import metaworld.env_dict as _env_dict
from metaworld.env_dict import ALL_V3_ENVIRONMENTS
from metaworld.sawyer_xyz_env import SawyerXYZEnv  # type: ignore
from metaworld.types import Task  # type: ignore
from metaworld.wrappers import (
    AutoTerminateOnSuccessWrapper,
    CheckpointWrapper,
    NormalizeRewardsExponential,
    OneHotWrapper,
    PseudoRandomTaskSelectWrapper,
    RandomTaskSelectWrapper,
    RNNBasedMetaRLWrapper,
)
from metaworld.workspace import use_workspace_caches

use_workspace_caches()   # caches go to <repo>/.cache, not the home directory


class Benchmark(abc.ABC):
    """A Benchmark.

    When used to evaluate an algorithm, only a single instance should be used.
    """

    _train_classes: _env_dict.EnvDict
    _test_classes: _env_dict.EnvDict
    _train_tasks: list[Task]
    _test_tasks: list[Task]

    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    def train_classes(self) -> _env_dict.EnvDict:
        """Returns all of the environment classes used for training."""
        return self._train_classes

    @property
    def test_classes(self) -> _env_dict.EnvDict:
        """Returns all of the environment classes used for testing."""
        return self._test_classes

    @property
    def train_tasks(self) -> list[Task]:
        """Returns all of the training tasks for this benchmark."""
        return self._train_tasks

    @property
    def test_tasks(self) -> list[Task]:
        """Returns all of the test tasks for this benchmark."""
        return self._test_tasks


_MT_OVERRIDE = dict(partially_observable=False)
"""The overrides for the Multi-Task benchmarks. Enables the inclusion of the goal position in the observation."""

_N_GOALS = 50
"""The number of goals to generate for each environment."""


def _encode_task(env_name, data) -> Task:
    """Instantiates a new `Task` object after pickling the data.

    Args:
        env_name: The name of the environment.
        data: The task data (will be pickled).

    Returns:
        A `Task` object.
    """
    return Task(env_name=env_name, data=pickle.dumps(data))


def _make_tasks(
    classes: _env_dict.EnvDict,
    args_kwargs: _env_dict.EnvArgsKwargsDict,
    kwargs_override: dict,
    seed: int | None = None,
    env_init_kwargs: dict[str, Any] | None = None,
) -> list[Task]:
    """Initialises goals for a given set of environments.

    Args:
        classes: The environment classes as an `EnvDict`.
        args_kwargs: The environment arguments and keyword arguments.
        kwargs_override: Any kwarg overrides.
        seed: The random seed to use.
        env_init_kwargs: Any custom kwargs to pass when initialising the environment.

    Returns:
        A flat list of `Task` objects, `_N_GOALS` for each environment in `classes`.
    """
    # Cache existing random state
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)

    tasks = []
    for env_name, args in args_kwargs.items():
        kwargs = args["kwargs"].copy()
        assert isinstance(kwargs, dict)
        assert len(args["args"]) == 0

        # Init env
        env = classes[env_name](**(env_init_kwargs or {}))
        env._freeze_rand_vec = False
        env._set_task_called = True
        rand_vecs: list[npt.NDArray[Any]] = []

        # Set task
        del kwargs["task_id"]
        env._set_task_inner(**kwargs)

        for _ in range(_N_GOALS):  # Generate random goals
            env.reset()
            assert env._last_rand_vec is not None
            rand_vecs.append(env._last_rand_vec)
        unique_task_rand_vecs = np.unique(np.array(rand_vecs), axis=0)
        assert (
            unique_task_rand_vecs.shape[0] == _N_GOALS
        ), f"Only generated {unique_task_rand_vecs.shape[0]} unique goals, not {_N_GOALS}"
        env.close()

        # Create a task for each random goal
        for rand_vec in rand_vecs:
            kwargs = args["kwargs"].copy()
            assert isinstance(kwargs, dict)
            del kwargs["task_id"]

            kwargs.update(dict(rand_vec=rand_vec, env_cls=classes[env_name]))
            kwargs.update(kwargs_override)

            tasks.append(_encode_task(env_name, kwargs))

        del env

    # Restore random state
    if seed is not None:
        np.random.set_state(st0)

    return tasks


# MT Benchmarks


class MT1(Benchmark):
    """
    The MT1 benchmark.
    A goal-conditioned RL environment for a single Metaworld task.
    """

    ENV_NAMES = list(_env_dict.ALL_V3_ENVIRONMENTS.keys())

    def __init__(self, env_name, seed=None, custom_kwargs: dict[str, Any] | None = None):
        super().__init__()
        if env_name not in _env_dict.ALL_V3_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a V3 environment")
        cls = _env_dict.ALL_V3_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = OrderedDict([(env_name, cls)])
        args_kwargs = _env_dict.ML1_args_kwargs[env_name]
        task_env_init_kwargs: dict[str, Any] = {}
        # Handle any custom kwargs for initialising the env
        if custom_kwargs is not None:
            task_env_init_kwargs.update(custom_kwargs)

        self._train_tasks = _make_tasks(
            self._train_classes, {env_name: args_kwargs}, _MT_OVERRIDE, seed=seed, env_init_kwargs=task_env_init_kwargs
        )

        self._test_tasks = []


def _init_each_env(
    env_cls: type[SawyerXYZEnv],
    tasks: list[Task],
    seed: int | None = None,
    max_episode_steps: int | None = None,
    terminate_on_success: bool = False,
    use_one_hot: bool = False,
    env_id: int | None = None,
    num_tasks: int | None = None,
    recurrent_info_in_obs: bool = False,
    normalize_reward_in_recurrent_info: bool = True,
    task_select: Literal["random", "pseudorandom"] = "random",
    reward_function_version: Literal["v1", "v2"] = "v2",
    reward_normalization_method: Literal["gymnasium", "exponential"] | None = None,
    normalize_observations: bool = False,
    reward_alpha: float = 0.001,
    render_mode: Literal["human", "rgb_array", "depth_array"] | None = None,
    camera_name: str | None = None,
    camera_id: int | None = None,
    width: int = 480,
    height: int = 480,
    custom_kwargs: dict[str, Any] | None = None,
) -> gym.Env:
    env_init_kwargs: dict[str, Any] = {
        "reward_function_version": reward_function_version,
        "render_mode": render_mode,
        "camera_name": camera_name,
        "camera_id": camera_id,
        "width": width,
        "height": height,
    }
    if custom_kwargs is not None:
        env_init_kwargs.update(custom_kwargs)

    env: gym.Env = env_cls(**env_init_kwargs)
    if seed is not None:
        env.seed(seed)  # type: ignore
    env = gym.wrappers.TimeLimit(env, max_episode_steps or env.max_path_length)  # type: ignore
    env = AutoTerminateOnSuccessWrapper(env)
    env.toggle_terminate_on_success(terminate_on_success)
    if use_one_hot:
        assert env_id is not None, "Need to pass env_id through constructor"
        assert num_tasks is not None, "Need to pass num_tasks through constructor"
        env = OneHotWrapper(env, env_id, num_tasks)
    if recurrent_info_in_obs:
        env = RNNBasedMetaRLWrapper(
            env, normalize_reward=normalize_reward_in_recurrent_info
        )
    if reward_normalization_method == "gymnasium":
        env = gym.wrappers.NormalizeReward(env)
    elif reward_normalization_method == "exponential":
        env = NormalizeRewardsExponential(reward_alpha=reward_alpha, env=env)
    if normalize_observations:
        env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if task_select != "random":
        env = PseudoRandomTaskSelectWrapper(env, tasks)
    else:
        env = RandomTaskSelectWrapper(env, tasks)

    env = CheckpointWrapper(env, f"{env_cls}_{env_id}")
    if seed is not None:
        env.action_space.seed(seed)
    return env


def make_mt_envs(
    name: str,
    seed: int | None = None,
    num_tasks: int | None = None,
    vector_strategy: Literal["sync", "async"] = "sync",
    autoreset_mode: gym.vector.AutoresetMode | str = gym.vector.AutoresetMode.SAME_STEP,
    **kwargs,
) -> gym.Env | gym.vector.VectorEnv:
    benchmark: Benchmark
    if name in ALL_V3_ENVIRONMENTS.keys():
        # Separate kwargs for _init_each_env from env-specific custom kwargs
        _INIT_EACH_ENV_PARAMS = {
            'max_episode_steps', 'terminate_on_success', 'use_one_hot', 'env_id',
            'num_tasks', 'recurrent_info_in_obs', 'normalize_reward_in_recurrent_info',
            'task_select', 'reward_function_version', 'reward_normalization_method',
            'normalize_observations', 'reward_alpha', 'render_mode', 'camera_name',
            'camera_id', 'width', 'height', 'custom_kwargs',
        }
        init_env_kwargs = {k: v for k, v in kwargs.items() if k in _INIT_EACH_ENV_PARAMS}
        env_custom_kwargs = {k: v for k, v in kwargs.items() if k not in _INIT_EACH_ENV_PARAMS}

        # Merge any explicit custom_kwargs with auto-detected env-specific kwargs
        custom_kwargs = init_env_kwargs.pop('custom_kwargs', None) or {}
        custom_kwargs.update(env_custom_kwargs)

        benchmark = MT1(name, seed=seed, custom_kwargs=custom_kwargs or None)
        tasks = [task for task in benchmark.train_tasks]
        return _init_each_env(  # type: ignore[misc]
            env_cls=benchmark.train_classes[name],
            tasks=tasks,
            seed=seed,
            num_tasks=num_tasks or 1,
            custom_kwargs=custom_kwargs or None,
            **init_env_kwargs,
        )
    else:
        raise ValueError(
            "Invalid env name. Must be a benchmark task name (e.g. 'reach-v3')."
        )


def register_mw_envs() -> None:
    def _mt_bench_vector_entry_point(
        mt_bench: str,
        vector_strategy: Literal["sync", "async"],
        autoreset_mode: gym.vector.AutoresetMode
        | str = gym.vector.AutoresetMode.SAME_STEP,
        seed=None,
        use_one_hot=False,
        num_envs=None,
        **lamb_kwargs,
    ):
        if "num_goals" in lamb_kwargs:
            global _N_GOALS
            _N_GOALS = lamb_kwargs["num_goals"]
            del lamb_kwargs["num_goals"]
        return make_mt_envs(  # type: ignore
            mt_bench,
            seed=seed,
            use_one_hot=use_one_hot,
            vector_strategy=vector_strategy,  # type: ignore
            autoreset_mode=autoreset_mode,
            **lamb_kwargs,
        )

    register(
        id="Meta-World/MT1",
        entry_point=lambda env_name, use_one_hot=False, vector_strategy="sync", autoreset_mode=gym.vector.AutoresetMode.SAME_STEP, seed=None, num_envs=None, **kwargs: _mt_bench_vector_entry_point(
            env_name,
            vector_strategy,
            autoreset_mode,
            seed,
            use_one_hot,
            num_envs,
            **kwargs,
        ),
        kwargs={},
    )


register_mw_envs()
__all__: list[str] = []
