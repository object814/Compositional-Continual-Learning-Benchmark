"""Expert-normalised return, matching the benchmark's reporting scale.

    normalised = 100 * (R - floor) / (R_expert - floor)

with `floor = -(max_episode_steps * action_repeat)` -- the return of an agent
that collects the worst per-step reward (-1) at every environment step -- and
`R_expert` read from `expert_return.txt` (best of 10 scripted-policy rollouts).
So 0 is total failure and 100 is the scripted expert.

Two properties of the scale worth knowing when reading the numbers:

  * The expert rollouts terminated on success while agent evaluation episodes
    always run to the timeout, so the expert stops accumulating negative
    reward earlier than the agent can. The scale is therefore somewhat
    generous to the expert, and an agent that solves the task still scores
    below 100. This is the benchmark's own convention and is kept so these
    numbers are comparable with the main results.
  * It does NOT distort the ablation. Normalisation is a monotone affine map
    and both arms of an ablation are full-length episodes, so the ratio
    between ablated and unablated return is unaffected by the choice of
    anchors.

Unlike success rate, this scale keeps resolving differences on tasks the
agent never solves: the composition tasks sit at 0% success but their raw
returns span roughly -250 to -430, which is a live signal an ablation can
move.
"""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve().parent
EXPERT_RETURN_FILE = _HERE / 'expert_return.txt'

# Tasks that share a scripted policy (and therefore an expert reference) with
# another task. Mirrors POLICY_REGISTRY in pretrain/collect_data.py, where the
# colour variants of bin-picking all use SawyerBinPickingV3Policy and the
# block pick-place variants all use SawyerPickPlaceBlockV3Policy.
TASK_ALIASES = {
    'bin-picking-redblue-v3': 'bin-picking-v3',
    'bin-picking-yellowblue-v3': 'bin-picking-v3',
    'bin-picking-redpurple-v3': 'bin-picking-v3',
    'bin-picking-yellowpurple-v3': 'bin-picking-v3',
    'pick-place-redblock-v3': 'pick-place-block-v3',
    'pick-place-greenblock-v3': 'pick-place-block-v3',
}


def load_expert_returns(path: Path = EXPERT_RETURN_FILE) -> dict[str, float]:
    """Parse `expert_return.txt` -> {env_name: best_return}."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        env_name, ret = parts[1], parts[2]
        try:
            out[env_name] = float(ret)
        except ValueError:
            continue          # the header row
    return out


def strip_prefix(task: str) -> str:
    for p in ('metaworld_', 'mw-'):
        if task.startswith(p):
            return task[len(p):]
    return task


def expert_return_for(task: str,
                      table: dict[str, float] | None = None) -> float | None:
    """Expert reference return for a task, or None if it has no entry."""
    table = load_expert_returns() if table is None else table
    name = strip_prefix(task)
    if name in table:
        return table[name]
    alias = TASK_ALIASES.get(name)
    if alias and alias in table:
        return table[alias]
    return None


def failure_floor(cfg) -> float:
    """Worst attainable return: -1 per environment step for a whole episode.

    `ActionRepeat` sums the reward over its repeats and the timeout is applied
    in agent steps, so an episode spans `max_episode_steps * action_repeat`
    environment steps.
    """
    steps = int(getattr(cfg, 'max_episode_steps', 250))
    repeat = int(getattr(cfg, 'action_repeat', 2))
    return -float(steps * repeat)


class Normaliser:
    """Maps raw episode return onto the 0 (failure) .. 100 (expert) scale."""

    def __init__(self, task: str, cfg, table: dict[str, float] | None = None):
        self.task = task
        self.floor = failure_floor(cfg)
        self.expert = expert_return_for(task, table)

    @property
    def available(self) -> bool:
        return self.expert is not None

    def __call__(self, raw_return: float) -> float:
        """Normalised score. Returns NaN when the task has no expert entry,
        so a missing reference shows up as missing rather than as a wrong
        number."""
        if self.expert is None:
            return float('nan')
        denom = self.expert - self.floor
        if abs(denom) < 1e-9:
            return float('nan')
        return 100.0 * (raw_return - self.floor) / denom

    def describe(self) -> str:
        if self.expert is None:
            return (f'{strip_prefix(self.task)}: no expert reference '
                    f'(raw return only)')
        return (f'{strip_prefix(self.task)}: floor={self.floor:.0f}, '
                f'expert={self.expert:.2f} -> 100')
