# Compo tasks
from metaworld.envs.compo_draweropen_pickplace import CompoDrawerOpenPickPlaceEnv
from metaworld.envs.compo_pickplace_block import CompoPickPlaceBlockEnv
from metaworld.envs.compo_pickplace_boxclose import CompoPickPlaceBoxCloseEnv


from metaworld.envs.sawyer_bin_picking_v3 import (
    SawyerBinPickingRedBlueEnvV3,
    SawyerBinPickingYellowBlueEnvV3,
    SawyerBinPickingRedPurpleEnvV3,
    SawyerBinPickingYellowPurpleEnvV3,
)
from metaworld.envs.sawyer_box_close_v3 import SawyerBoxCloseEnvV3
from metaworld.envs.sawyer_drawer_open_v3 import SawyerDrawerOpenEnvV3
from metaworld.envs.sawyer_grasp import SawyerGraspEnvV3
from metaworld.envs.sawyer_pick_place_block_v3 import (
    SawyerPickPlaceRedBlockEnvV3,
    SawyerPickPlaceGreenBlockEnvV3,
)
from metaworld.envs.sawyer_pick_place_v3 import SawyerPickPlaceEnvV3
from metaworld.envs.sawyer_reach_v3 import SawyerReachEnvV3
from metaworld.envs.sawyer_reach_xy_v3 import SawyerReachXYEnvV3
from metaworld.envs.sawyer_reach_xz_v3 import SawyerReachXZEnvV3
from metaworld.envs.sawyer_reach_yz_v3 import SawyerReachYZEnvV3
from metaworld.envs.sawyer_reach_xyz_v3 import SawyerReachXYZEnvV3

__all__ = [
    # Compo tasks
    "CompoDrawerOpenPickPlaceEnv",
    "CompoPickPlaceBlockEnv",
    "CompoPickPlaceBoxCloseEnv",

    "SawyerBinPickingRedBlueEnvV3",
    "SawyerBinPickingYellowBlueEnvV3",
    "SawyerBinPickingRedPurpleEnvV3",
    "SawyerBinPickingYellowPurpleEnvV3",
    "SawyerBoxCloseEnvV3",
    "SawyerDrawerOpenEnvV3",
    "SawyerGraspEnvV3",
    "SawyerPickPlaceRedBlockEnvV3",
    "SawyerPickPlaceGreenBlockEnvV3",
    "SawyerPickPlaceEnvV3",
    "SawyerReachEnvV3",
    "SawyerReachXYEnvV3",
    "SawyerReachXZEnvV3",
    "SawyerReachYZEnvV3",
    "SawyerReachXYZEnvV3",
]
