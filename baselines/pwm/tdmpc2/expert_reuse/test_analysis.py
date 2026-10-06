"""CPU self-test for the normalisation, checkpoint-pair discovery and the
trainer's mid-task snapshot finder.

Runs against the real logdir, so it checks the pieces the GPU tier depends on
without needing a GPU.

    python expert_reuse/test_analysis.py
"""

import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE.parent), str(_HERE.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from expert_reuse.normalise import (                       # noqa: E402
    Normaliser, load_expert_returns, expert_return_for, failure_floor)
from expert_reuse.runspec import discover_runs             # noqa: E402

FAILED = []


def check(name, cond, detail=''):
    print(f'  {"PASS" if cond else "FAIL"}  {name}' + (f'  {detail}' if detail else ''))
    if not cond:
        FAILED.append(name)


class _Cfg:
    max_episode_steps = 250
    action_repeat = 2


def main():
    print('normalisation scale')
    table = load_expert_returns()
    check('expert_return.txt parsed', len(table) >= 13, f'{len(table)} entries')
    check('failure floor is -(steps x repeat)', failure_floor(_Cfg()) == -500.0,
          f'{failure_floor(_Cfg())}')

    n = Normaliser('metaworld_compo-pickplace-boxclose', _Cfg(), table)
    check('expert maps to 100', abs(n(n.expert) - 100.0) < 1e-6)
    check('floor maps to 0', abs(n(-500.0)) < 1e-6)
    check('monotone increasing', n(-200.0) > n(-400.0))

    # The whole point of switching off success rate.
    zero_success = {'compo-pickplace-block': -414.6,
                    'compo-draweropen-pickplace': -257.8,
                    'compo-pickplace-boxclose': -429.5}
    vals = []
    for task, raw in zero_success.items():
        nn = Normaliser('metaworld_' + task, _Cfg(), table)
        vals.append(nn(raw))
    check('0%-success tasks still get a graded, non-zero score',
          all(2 < v < 90 for v in vals),
          ' '.join(f'{v:.1f}' for v in vals))

    print('\ntask -> expert reference resolution')
    curriculum = [
        'reach-xy-v3', 'reach-xz-v3', 'reach-yz-v3', 'reach-xyz-v3',
        'grasp-v3', 'pick-place-v3',
        'bin-picking-redblue-v3', 'bin-picking-yellowblue-v3',
        'bin-picking-redpurple-v3', 'bin-picking-yellowpurple-v3',
        'pick-place-redblock-v3', 'pick-place-greenblock-v3',
        'compo-pickplace-block', 'drawer-open-v3',
        'compo-draweropen-pickplace', 'box-close-v3',
        'compo-pickplace-boxclose',
    ]
    missing = [t for t in curriculum
               if expert_return_for('metaworld_' + t, table) is None]
    check('every FINAL task has a reference', not missing, str(missing))
    check('colour variants alias to their shared scripted policy',
          expert_return_for('metaworld_bin-picking-redpurple-v3', table)
          == table['bin-picking-v3'])
    check('missing reference yields NaN, not a wrong number',
          Normaliser('metaworld_reach-v3', _Cfg(), table)(-100.0) != Normaliser(
              'metaworld_reach-v3', _Cfg(), table)(-100.0))

    print('\ncheckpoint-pair discovery on the real logdir')
    runs = discover_runs(variant='preEncFr')
    reached = [r for r in runs if r.reached_final_task(200_000)]
    check('runs discovered', len(reached) >= 10, f'{len(reached)} runs')
    n_causal = 0
    for r in reached:
        bb, tm, step, at_end = r.latest_checkpoint_pair(r.final_task_idx)
        ok = bb is not None and tm is not None
        n_causal += int(ok)
        if ok:
            check(f'{r.name}: pair from one step, both files exist',
                  bb.parent == tm.parent and bb.exists() and tm.exists(),
                  f'step {step:,}{" (end)" if at_end else ""}')
    check('every reached run is causal-eligible', n_causal == len(reached),
          f'{n_causal}/{len(reached)}')

    print('\ntrainer mid-task snapshot finder')
    from sequential.sequential_train import find_task_snapshot
    r = next(x for x in reached if not x.is_finished())
    snap = find_task_snapshot(r.task_dirs[r.final_task_idx])
    check('finds a snapshot for an unfinished task', snap is not None)
    if snap:
        step, bb, tm = snap
        check('snapshot files exist and share a directory',
              bb.exists() and tm.exists() and bb.parent == tm.parent,
              f'{r.name} @ {step:,}')
        check('agrees with the analysis-side discovery',
              step == r.latest_checkpoint_pair(r.final_task_idx)[2])

    with tempfile.TemporaryDirectory() as td:
        check('returns None when there is no models/ dir',
              find_task_snapshot(td) is None)
        m = Path(td) / 'models' / '1000'
        m.mkdir(parents=True)
        (m / 'backbone.pt').write_bytes(b'x')
        check('skips a snapshot missing task_modules.pt',
              find_task_snapshot(td) is None)
        (m / 'task_modules.pt').write_bytes(b'x')
        check('accepts it once both files are present',
              find_task_snapshot(td)[0] == 1000)

    print()
    if FAILED:
        print(f'{len(FAILED)} check(s) FAILED: {FAILED}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
