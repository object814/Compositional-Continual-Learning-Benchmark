"""Dictionaries mapping benchmark environment name strings to environment classes."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Literal
from typing import OrderedDict as Typing_OrderedDict
from typing import Sequence, Union

from typing_extensions import TypeAlias

from metaworld import envs
from metaworld.sawyer_xyz_env import SawyerXYZEnv

# Utils

EnvDict: TypeAlias = "Typing_OrderedDict[str, type[SawyerXYZEnv]]"
EnvArgsKwargsDict: TypeAlias = (
    "Dict[str, Dict[Literal['args', 'kwargs'], Union[List, Dict]]]"
)

ENV_CLS_MAP = {
    # Compo tasks
    "compo-draweropen-pickplace": envs.CompoDrawerOpenPickPlaceEnv,
    "compo-pickplace-block": envs.CompoPickPlaceBlockEnv,
    "compo-pickplace-boxclose": envs.CompoPickPlaceBoxCloseEnv,

    "bin-picking-redblue-v3": envs.SawyerBinPickingRedBlueEnvV3,
    "bin-picking-yellowblue-v3": envs.SawyerBinPickingYellowBlueEnvV3,
    "bin-picking-redpurple-v3": envs.SawyerBinPickingRedPurpleEnvV3,
    "bin-picking-yellowpurple-v3": envs.SawyerBinPickingYellowPurpleEnvV3,
    "box-close-v3": envs.SawyerBoxCloseEnvV3,
    "drawer-open-v3": envs.SawyerDrawerOpenEnvV3,
    "grasp-v3": envs.SawyerGraspEnvV3,
    "reach-v3": envs.SawyerReachEnvV3,
    "reach-xy-v3": envs.SawyerReachXYEnvV3,
    "reach-xz-v3": envs.SawyerReachXZEnvV3,
    "reach-yz-v3": envs.SawyerReachYZEnvV3,
    "reach-xyz-v3": envs.SawyerReachXYZEnvV3,
    "pick-place-redblock-v3": envs.SawyerPickPlaceRedBlockEnvV3,
    "pick-place-greenblock-v3": envs.SawyerPickPlaceGreenBlockEnvV3,
    "pick-place-v3": envs.SawyerPickPlaceEnvV3,
}


def _get_env_dict(env_names: Sequence[str]) -> EnvDict:
    """Returns an `OrderedDict` containing `(env_name, env_cls)` tuples for the given env_names.

    Args:
        env_names: The environment names

    Returns:
        The appropriate `OrderedDict.
    """
    return OrderedDict([(env_name, ENV_CLS_MAP[env_name]) for env_name in env_names])


def _get_args_kwargs(all_envs: EnvDict, env_subset: EnvDict) -> EnvArgsKwargsDict:
    """Returns containing a `dict` of "args" and "kwargs" for each environment in a given list of environments.
    Specifically, sets an empty "args" array and a "kwargs" dictionary with a "task_id" key for each env.

    Args:
        all_envs: The full list of envs
        env_subset: The subset of envs to get args and kwargs for

    Returns:
        The args and kwargs dictionary.
    """
    return {
        key: dict(args=[], kwargs={"task_id": list(all_envs.keys()).index(key)})
        for key, _ in env_subset.items()
    }


# V3 DICTS

ALL_V3_ENVIRONMENTS = _get_env_dict(
    [
        # Compo tasks
        "compo-draweropen-pickplace",
        "compo-pickplace-block",
        "compo-pickplace-boxclose",

        "bin-picking-redblue-v3",
        "bin-picking-yellowblue-v3",
        "bin-picking-redpurple-v3",
        "bin-picking-yellowpurple-v3",
        "box-close-v3",
        "drawer-open-v3",
        "grasp-v3",
        "pick-place-redblock-v3",
        "pick-place-greenblock-v3",
        "pick-place-v3",
        "reach-v3",
        "reach-xy-v3",
        "reach-xz-v3",
        "reach-yz-v3",
        "reach-xyz-v3",
    ]
)

ML1_args_kwargs = _get_args_kwargs(ALL_V3_ENVIRONMENTS, ALL_V3_ENVIRONMENTS)
