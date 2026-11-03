"""Scripted expert policies for the benchmark tasks.

Each policy reads the simulator state (the 39-dim Meta-World observation,
`obs["original_obs"]` from `ProprioMultiImageObsWrapper`) and returns a 4-dim
action. Several keep track of the phase of their task, so build a fresh one
per episode with `make_expert`. They come from Meta-World's
`metaworld/policies/`, with policies added for the benchmark's own tasks.

`python -m metaworld.expert_policy.rollout` rolls them out, reports success
rates and saves videos and demonstrations (see rollout.py).
"""
import importlib

# task -> (module, class); the colour variants of a task share its policy
_POLICIES = {
    "reach-v3": ("sawyer_reach_v3_policy", "SawyerReachV3Policy"),
    "reach-xy-v3": ("sawyer_reach_xy_v3_policy", "SawyerReachXYV3Policy"),
    "reach-xz-v3": ("sawyer_reach_xz_v3_policy", "SawyerReachXZV3Policy"),
    "reach-yz-v3": ("sawyer_reach_yz_v3_policy", "SawyerReachYZV3Policy"),
    "reach-xyz-v3": ("sawyer_reach_xyz_v3_policy", "SawyerReachXYZV3Policy"),
    "grasp-v3": ("sawyer_grasp_policy", "SawyerGraspV3Policy"),
    "pick-place-v3": ("sawyer_pick_place_v3_policy", "SawyerPickPlaceV3Policy"),
    "bin-picking-redblue-v3": ("sawyer_bin_picking_v3_policy", "SawyerBinPickingV3Policy"),
    "bin-picking-yellowblue-v3": ("sawyer_bin_picking_v3_policy", "SawyerBinPickingV3Policy"),
    "bin-picking-redpurple-v3": ("sawyer_bin_picking_v3_policy", "SawyerBinPickingV3Policy"),
    "bin-picking-yellowpurple-v3": ("sawyer_bin_picking_v3_policy", "SawyerBinPickingV3Policy"),
    "pick-place-redblock-v3": ("sawyer_pick_place_block_v3_policy", "SawyerPickPlaceBlockV3Policy"),
    "pick-place-greenblock-v3": ("sawyer_pick_place_block_v3_policy", "SawyerPickPlaceBlockV3Policy"),
    "drawer-open-v3": ("sawyer_drawer_open_v3_policy", "SawyerDrawerOpenV3Policy"),
    "box-close-v3": ("sawyer_box_close_v3_policy", "SawyerBoxCloseV3Policy"),
    "compo-pickplace-block": ("compo_pickplace_block_policy", "CompoPickPlaceBlockPolicy"),
    "compo-draweropen-pickplace": ("compo_draweropen_pickplace_policy", "CompoDrawerOpenPickPlacePolicy"),
    "compo-pickplace-boxclose": ("compo_pickplace_boxclose_policy", "CompoPickPlaceBoxClosePolicy"),
}
EXPERT_TASKS = list(_POLICIES)


def make_expert(task: str):
    """A new expert for `task` ('reach-xy-v3', or 'metaworld_reach-xy-v3' as in the sweeps)."""
    task = task.removeprefix("metaworld_")
    if task not in _POLICIES:
        raise ValueError(f"No expert for {task!r}; available: {', '.join(EXPERT_TASKS)}")
    module, cls = _POLICIES[task]
    return getattr(importlib.import_module(f"{__name__}.{module}"), cls)()


__all__ = ["EXPERT_TASKS", "make_expert"]
