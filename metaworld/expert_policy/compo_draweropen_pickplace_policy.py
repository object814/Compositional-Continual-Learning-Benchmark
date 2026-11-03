from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.expert_policy.action import Action
from metaworld.expert_policy.policy import Policy, assert_fully_parsed, move


class CompoDrawerOpenPickPlacePolicy(Policy):
    """Expert policy for the compositional DrawerOpen + PickPlace task.

    The task has two phases:
      Phase 1 — Open the drawer (from SawyerDrawerOpenV3Policy).
      Transition — Carefully disengage from the drawer handle:
          1. Stop horizontal movement and let the drawer coast to a stop.
          2. Raise the gripper straight up to clear the handle.
      Phase 2 — Pick up the block and place it inside the open drawer
          (from SawyerPickPlaceV3Policy), with a modified transport path
          that raises the block above the drawer before lowering it in.

    This policy is **stateful**: call ``reset()`` whenever the environment
    is reset so that the internal phase machine restarts correctly.
    """

    def __init__(self) -> None:
        super().__init__()
        self._phase: str = "drawer_open"
        self._stabilize_counter: int = 0
        self._prev_goal: npt.NDArray[np.float64] | None = None

    def reset(self) -> None:
        """Reset internal state.  Must be called when the environment resets."""
        self._phase = "drawer_open"
        self._stabilize_counter = 0
        self._prev_goal = None

    # ------------------------------------------------------------------
    # Observation parsing
    # ------------------------------------------------------------------
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(
        obs: npt.NDArray[np.float64],
    ) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "gripper_distance_apart": obs[3],
            "obj_pos": obs[4:7],       # handle (phase 1) or block (phase 2)
            "obj_rot": obs[7:11],
            "goal_pos": obs[-3:],
            "unused_info": obs[11:-3],  # prev-frame + padding + velocity
        }

    # ------------------------------------------------------------------
    # Main action selection
    # ------------------------------------------------------------------
    def get_action(
        self, obs: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)
        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        # --- Detect the environment's phase transition ----------------
        # When the env sets ``drawer_opened = True`` inside
        # ``compute_reward()``, the goal position changes from the
        # drawer-open target to the pick-place target.  The magnitude of
        # this shift (≈0.19 m) is a reliable transition signal.
        if self._prev_goal is not None and self._phase == "drawer_open":
            goal_delta = float(np.linalg.norm(o_d["goal_pos"] - self._prev_goal))
            if goal_delta > 0.1:
                self._phase = "stabilize"
                self._stabilize_counter = 0
        self._prev_goal = o_d["goal_pos"].copy()

        # --- Phase: open the drawer -----------------------------------
        if self._phase == "drawer_open":
            action["delta_pos"], action["grab_effort"] = self._drawer_open_action(o_d)

        # --- Phase: stabilize (hold still, let drawer settle) ---------
        elif self._phase == "stabilize":
            action["delta_pos"] = np.zeros(3)
            action["grab_effort"] = -1.0
            self._stabilize_counter += 1
            if self._stabilize_counter >= 10:
                self._phase = "raise"

        # --- Phase: raise gripper to clear the drawer handle ----------
        elif self._phase == "raise":
            pos_curr = o_d["hand_pos"]
            target = np.array([pos_curr[0], pos_curr[1], 0.3])
            action["delta_pos"] = move(pos_curr, target, p=4.0)
            action["grab_effort"] = -1.0
            if pos_curr[2] > 0.25:
                self._phase = "pick_place"

        # --- Phase: pick up block & place it in the drawer ------------
        elif self._phase == "pick_place":
            action["delta_pos"] = move(
                o_d["hand_pos"],
                self._pick_place_desired_pos(o_d),
                p=10.0,
            )
            action["grab_effort"] = self._pick_place_grab_effort(o_d)

        return action.array

    # ------------------------------------------------------------------
    # Phase 1 helpers — drawer open
    # ------------------------------------------------------------------
    @staticmethod
    def _drawer_open_action(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> tuple[npt.NDArray[Any], float]:
        """Return ``(delta_pos, grab_effort)`` for the drawer-open phase.

        Logic mirrors ``SawyerDrawerOpenV3Policy``:
        1. Align XY above the handle.
        2. Drop down to the handle.
        3. Push behind the handle with high gain to pull the drawer open.
        """
        pos_curr = o_d["hand_pos"]
        pos_handle = o_d["obj_pos"] + np.array([0.0, 0.0, -0.02])

        if np.linalg.norm(pos_curr[:2] - pos_handle[:2]) > 0.06:
            # Align above handle
            to_pos = pos_handle + np.array([0.0, 0.0, 0.3])
            delta = move(pos_curr, to_pos, p=4.0)
        elif abs(pos_curr[2] - pos_handle[2]) > 0.04:
            # Drop down to handle level
            delta = move(pos_curr, pos_handle, p=4.0)
        else:
            # Pull drawer open (push behind handle)
            to_pos = pos_handle + np.array([0.0, -0.06, 0.0])
            delta = move(pos_curr, to_pos, p=50.0)

        return delta, -1.0  # keep gripper open

    # ------------------------------------------------------------------
    # Phase 2 helpers — pick & place (with elevated transport path)
    # ------------------------------------------------------------------
    @staticmethod
    def _pick_place_desired_pos(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[Any]:
        """Compute the target position for the pick-place phase.

        The key difference from :class:`SawyerPickPlaceV3Policy` is that,
        after grasping, the block is first raised to a safe height that
        clears the drawer walls, moved horizontally to above the goal
        (inside the open drawer), and only then lowered down.
        """
        pos_curr = o_d["hand_pos"]
        pos_block = o_d["obj_pos"] + np.array([-0.005, 0.0, 0.0])
        pos_goal = o_d["goal_pos"]
        gripper_sep = o_d["gripper_distance_apart"]

        # Height that clears the drawer walls when transporting the block
        SAFE_HEIGHT = 0.22

        # 1. Align XY above the block
        if np.linalg.norm(pos_curr[:2] - pos_block[:2]) > 0.02:
            return pos_block + np.array([0.0, 0.0, 0.1])

        # 2. Descend onto the block (only while it is still on the ground)
        elif abs(pos_curr[2] - pos_block[2]) > 0.05 and pos_block[-1] < 0.04:
            return pos_block + np.array([0.0, 0.0, 0.03])

        # 3. Wait for the gripper to close around the block
        elif gripper_sep > 0.73:
            return pos_curr

        # --- Block is now grasped — transport it over the drawer ------

        # 4. Raise to safe height (skip if already above goal in XY,
        #    which would mean we are descending into the drawer)
        elif (
            pos_curr[2] < SAFE_HEIGHT
            and np.linalg.norm(pos_curr[:2] - pos_goal[:2]) > 0.02
        ):
            return np.array([pos_curr[0], pos_curr[1], SAFE_HEIGHT])

        # 5. Move horizontally to above the goal (inside the drawer opening)
        elif np.linalg.norm(pos_curr[:2] - pos_goal[:2]) > 0.02:
            return np.array([pos_goal[0], pos_goal[1], SAFE_HEIGHT])

        # 6. Lower the block into the drawer
        else:
            return pos_goal

    @staticmethod
    def _pick_place_grab_effort(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> float:
        """Return the gripper effort for the pick-place phase."""
        pos_curr = o_d["hand_pos"]
        pos_block = o_d["obj_pos"]
        if np.linalg.norm(pos_curr - pos_block) < 0.07:
            return 1.0
        return 0.0
