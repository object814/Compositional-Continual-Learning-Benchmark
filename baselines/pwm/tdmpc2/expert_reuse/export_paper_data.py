"""Export the raw numbers behind the expert-reuse figures, as JSON.

One checkpoint per task suite (the seed chosen in `DEFAULT_RUNS`), evaluated
under several EVALUATION seeds. The evaluation seed changes the environment
reset sequence and the MPC sampling noise, so it gives error bars on the
rollout measurements while holding the model fixed -- these are not training
seeds, and the checkpoint is identical across them.

What gets exported per suite
----------------------------
router/          read once from the checkpoint; no seeds involved, because the
                 gate is conditioned on the task one-hot alone and its output
                 is a constant vector per task
    raw_weights                  the [num_tasks x total_K] routing matrix
    final_start_vs_end           the final task's router before and after its
                                 own training (the exact null)
    primitive_vs_composition     per prior block: that primitive's routing
                                 over its own experts vs the composition
                                 task's, in three scalings (see below)

evaluation/      repeated for every evaluation seed
    closedloop   expert-normalised return per ablation condition
    openloop     multi-step latent prediction MSE per ablation condition,
                 both raw and as a ratio to the unablated model

Conditions: unablated, prior-experts-only, new-experts-only, uniform routing,
shuffled routing (control), and leave-one-primitive-out for each primitive.
`drop_block{final}` is deliberately absent: removing the final task's own
block leaves exactly the prior blocks, so it is `old_only` recomputed --
verified bit-identical, and it doubles the cost of the most expensive tier.

Three scalings are exported for the routing weights because they answer
different questions and only one of them is comparable across tasks:
  raw                     as the model uses it. NOT comparable between tasks:
                          task 1 softmaxes over 3 experts, the composition
                          task over 9, so later weights are smaller by
                          construction.
  uniform_normalised      weight / (1/K_active). 1.0 = this expert's fair
                          share on any task. Use this to compare tasks.
  renormalised_in_block   renormalised within the block, which removes the
                          block's size and isolates the shape.

Usage
-----
    python expert_reuse/export_paper_data.py --eval-seeds 0 1 2

Writes `expert_reuse/paper_data/<suite>.json` incrementally (one file per
suite, as soon as it finishes) plus a combined `all_suites.json` and a flat
`summary.csv` at the end.
"""

from __future__ import annotations

import os

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('LAZY_LEGACY_OP', '0')
os.environ.setdefault('XDG_RUNTIME_DIR', '/tmp')
os.environ.setdefault('EGL_LOG_LEVEL', 'fatal')

import argparse
import csv
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
for p in (str(_PKG), str(_PKG.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from expert_reuse import gate as gate_mod                      # noqa: E402
from expert_reuse import rollout as ro                         # noqa: E402
from expert_reuse.interventions import (                       # noqa: E402
    build_conditions, make_weight_fns)
from expert_reuse.normalise import Normaliser                  # noqa: E402
from expert_reuse.runspec import (                             # noqa: E402
    discover_runs, load_sweep_defaults)

# One checkpoint per suite. Chosen on the strength of the old-only vs
# new-only asymmetry in the closed-loop ablation, i.e. the seed where the
# prior experts most clearly carry the task on their own. Override with
# --runs taskset=seed_dir.
DEFAULT_RUNS = {
    'reach':       'seed_1_preEncFr',
    'grasp':       'seed_1_preEncFr',
    'binpnp':      'seed_1_preEncFr',
    'pnpblock':    'seed_1_preEncFr',
    'drawerpnp':   'seed_2_preEncFr',
    'pnpboxclose': 'seed_1_preEncFr',
}

SUITE_AXIS = {
    'reach': 'action', 'grasp': 'action',
    'binpnp': 'perception', 'pnpblock': 'perception',
    'drawerpnp': 'full', 'pnpboxclose': 'full',
}


# =======================================================================
# helpers
# =======================================================================

def _f(x):
    """float or None (JSON has no NaN)."""
    if x is None:
        return None
    v = float(x)
    return None if not np.isfinite(v) else v


def _fl(xs):
    return [_f(x) for x in xs]


def _agg(values: list[float]) -> dict:
    """mean / std / sem over the evaluation seeds."""
    v = np.array([x for x in values if x is not None and np.isfinite(x)],
                 dtype=np.float64)
    if v.size == 0:
        return dict(mean=None, std=None, sem=None, n=0)
    return dict(mean=_f(v.mean()), std=_f(v.std(ddof=0)),
                sem=_f(v.std(ddof=0) / np.sqrt(v.size)), n=int(v.size))


def condition_names(task_idx: int) -> list[str]:
    """The conditions exported, in a fixed reporting order."""
    names = ['full', 'old_only', 'new_only', 'ctl_uniform', 'ctl_shuffle']
    names += [f'drop_block{j}' for j in range(task_idx)]
    return names


# =======================================================================
# router section (no seeds; the gate output is constant per task)
# =======================================================================

def router_section(run, tab: dict) -> dict:
    K, nT = tab['K_per_task'], tab['num_tasks']
    t = nT - 1
    aK_final = tab['active_Ks'][t]
    W = np.asarray(tab['weights'], dtype=np.float64)
    W_start = np.asarray(tab['weights_start'], dtype=np.float64)

    owner = [k // K for k in range(tab['total_K'])]

    start_vs_end = dict(
        task=tab['tasks'][t],
        active_K=int(aK_final),
        expert_index=list(range(aK_final)),
        owner_task=owner[:aK_final],
        is_prior_expert=[k < t * K for k in range(aK_final)],
        # Exact null: a one-hot gate zeroes the gradient on every column but
        # the active one, so the previous checkpoint holds this task's router
        # at its initial value.
        start=_fl(W_start[t, :aK_final]),
        end=_fl(W[t, :aK_final]),
        start_uniform_normalised=_fl(W_start[t, :aK_final] * aK_final),
        end_uniform_normalised=_fl(W[t, :aK_final] * aK_final),
        old_expert_mass=_f(W[t, :t * K].sum()) if t else None,
        old_expert_mass_at_start=_f(W_start[t, :t * K].sum()) if t else None,
        uniform_null_old_mass=_f(t / (t + 1)) if t else None,
    )

    sim = tab.get('copy_similarity', {}).get(f'task{t}', {})
    pvc = {}
    for j in range(t):
        sl = slice(j * K, (j + 1) * K)
        own = W[j, sl]
        comp = W[t, sl]
        aK_own = tab['active_Ks'][j]
        b = sim.get(f'block{j}', {})
        pvc[f'block{j}'] = dict(
            primitive_task=tab['tasks'][j],
            expert_index=list(range(j * K, (j + 1) * K)),
            own_raw=_fl(own),
            composition_raw=_fl(comp),
            own_uniform_normalised=_fl(own * aK_own),
            composition_uniform_normalised=_fl(comp * aK_final),
            own_renormalised_in_block=_fl(own / own.sum() if own.sum() else own),
            composition_renormalised_in_block=_fl(
                comp / comp.sum() if comp.sum() else comp),
            # block-level fair share = 1 / (number of blocks active)
            block_mass_own=_f(own.sum()),
            block_mass_composition=_f(comp.sum()),
            block_share_own=_f(own.sum() * (j + 1)),
            block_share_composition=_f(comp.sum() * (t + 1)),
            cosine=_f(b.get('cosine')),
            cosine_no_tau=_f(b.get('cosine_no_tau')),
            cosine_tau_only_baseline=_f(b.get('cosine_tau_only_vs_late')),
            argmax_match=b.get('argmax_match'),
            argmax_match_no_tau=b.get('argmax_match_no_tau'),
        )

    return dict(
        raw_weights=[[_f(v) for v in W[i]] for i in range(nT)],
        raw_weights_note=('row i = task i, column k = expert k; null where '
                          'expert k was outside task i active window'),
        active_K_per_task=[int(x) for x in tab['active_Ks']],
        tau_per_task=_fl(tab['taus']),
        owner_task_per_expert=owner,
        prior_gate_columns_frozen=tab['columns_frozen'],
        tau_column_frozen=tab['tau_col_frozen'],
        final_start_vs_end=start_vs_end,
        primitive_vs_composition=pvc,
    )


# =======================================================================
# rollout section (repeated per evaluation seed)
# =======================================================================

def evaluation_section(run, tab, defaults, geom, args) -> dict:
    t = run.final_task_idx
    aK = geom['active_K']
    K = geom['K_per_task']
    names = condition_names(t)

    agent, cfg, env, ck_info = ro.load_final_agent(run, defaults, geom)
    normaliser = Normaliser(run.tasks[t], cfg)
    print(f'  scale: {normaliser.describe()}')
    learned = np.asarray(tab['weights'], dtype=np.float64)[t, :aK]
    conds = build_conditions(aK, K, t, per_expert=False)

    closed = {n: {'per_seed': []} for n in names}
    openl = {n: {'per_seed': []} for n in names}
    collected = []

    try:
        for seed in args.eval_seeds:
            print(f'  --- evaluation seed {seed} ---')
            fns = make_weight_fns(conds, renormalize=True, keep_stack=True,
                                  learned_weights=learned, active_K=aK,
                                  seed=seed)

            # ---- open-loop: collect once, then every condition is cheap ----
            eps = ro.collect_episodes(agent, env, args.episodes, seed=seed,
                                      verbose=args.verbose)
            collected.append(dict(
                seed=int(seed),
                n_episodes=len(eps),
                n_transitions=int(sum(e['length'] for e in eps)),
                success=_f(np.mean([e['success'] for e in eps])),
                episode_reward=_f(np.mean([e['ep_reward'] for e in eps])),
                normalised_return=_f(np.mean(
                    [normaliser(e['ep_reward']) for e in eps])),
            ))
            base_mse = None
            for n in names:
                r = ro.openloop_error(agent, eps, int(cfg.horizon),
                                      weight_fn=fns.get(n),
                                      max_starts=args.max_starts)
                if n == 'full':
                    base_mse = r['mse']
                openl[n]['per_seed'].append(dict(
                    seed=int(seed), mse=_f(r['mse']),
                    mse_h1=_f(r['mse_h1']), mse_per_horizon=_fl(r['mse_per_h']),
                    cosine_to_truth=_f(r['cos']),
                    reward_mae=_f(r['reward_mae']),
                    # normalised: how many times the unablated error
                    mse_ratio_to_full=_f(r['mse'] / base_mse)
                    if base_mse else None,
                ))
            print(f'    open-loop done (baseline MSE {base_mse:.5f})')

            # ---- closed-loop: one MPC eval per condition ----
            for n in names:
                print(f'    closed-loop [{n}] '
                      f'({args.closedloop_episodes} episodes)...')
                r = ro.closedloop_eval(
                    agent, env, args.closedloop_episodes,
                    weight_fn=fns.get(n), seed=seed, tag=f'{n} s{seed}',
                    normaliser=normaliser)
                closed[n]['per_seed'].append(dict(
                    seed=int(seed),
                    normalised_return=_f(r.get('normalised_return')),
                    episode_reward=_f(r['episode_reward']),
                    success=_f(r['episode_success']),
                    episode_length=_f(r['episode_length']),
                    episode_normalised_returns=_fl(
                        r.get('episode_normalised_returns', [])),
                    episode_returns=_fl(r['episode_returns']),
                ))
    finally:
        try:
            env.close()
        except Exception:
            pass
        del agent, env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- aggregate across evaluation seeds ----
    for n in names:
        closed[n]['across_seeds'] = _agg(
            [s['normalised_return'] for s in closed[n]['per_seed']])
        closed[n]['across_seeds_success'] = _agg(
            [s['success'] for s in closed[n]['per_seed']])
        openl[n]['across_seeds_mse'] = _agg(
            [s['mse'] for s in openl[n]['per_seed']])
        openl[n]['across_seeds_mse_ratio_to_full'] = _agg(
            [s['mse_ratio_to_full'] for s in openl[n]['per_seed']])

    base = closed['full']['across_seeds']['mean']
    for n in names:
        m = closed[n]['across_seeds']['mean']
        closed[n]['drop_vs_unablated'] = (
            _f(base - m) if (base is not None and m is not None) else None)

    return dict(
        eval_seeds=[int(s) for s in args.eval_seeds],
        eval_seed_note=('evaluation seeds only: they change env resets and '
                        'MPC sampling noise; the checkpoint is identical '
                        'across them'),
        closedloop_episodes_per_seed=int(args.closedloop_episodes),
        openloop_collect_episodes_per_seed=int(args.episodes),
        openloop_horizon=int(cfg.horizon),
        conditions=names,
        condition_meaning={
            'full': 'unablated model',
            'old_only': 'prior tasks experts only (new block removed)',
            'new_only': 'new experts only (all prior blocks removed)',
            'ctl_uniform': 'control: learned routing replaced by uniform',
            'ctl_shuffle': 'control: learned coefficients permuted across '
                           'experts',
            **{f'drop_block{j}': f'leave-one-out: primitive task T{j + 1} '
                                 f'experts removed' for j in range(t)},
        },
        checkpoint=dict(path=ck_info['checkpoint'], step=int(ck_info['step']),
                        budget=int(ck_info['budget']),
                        at_end=bool(ck_info['at_end']),
                        frac=_f(ck_info['step'] / ck_info['budget'])
                        if ck_info['budget'] > 0 else None),
        scale=dict(task=run.tasks[t], failure_floor=_f(normaliser.floor),
                   expert_return=_f(normaliser.expert),
                   note='normalised = 100*(R-floor)/(expert-floor)'),
        collection=collected,
        closedloop=closed,
        openloop=openl,
    )


# =======================================================================
# main
# =======================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--eval-seeds', nargs='+', type=int, default=[0, 1, 2],
                   help='Evaluation seeds (env resets + MPC noise). The '
                        'checkpoint is the same for all of them.')
    p.add_argument('--closedloop-episodes', type=int, default=10,
                   help='MPC episodes per condition per seed.')
    p.add_argument('--episodes', type=int, default=20,
                   help='Episodes collected per seed for the open-loop tier.')
    p.add_argument('--max-starts', type=int, default=None)
    p.add_argument('--tasksets', nargs='*', default=None,
                   help='Restrict to these suites.')
    p.add_argument('--runs', nargs='*', default=None,
                   help='Override the chosen checkpoint, as '
                        'taskset=seed_dir (e.g. grasp=seed_2_preEncFr).')
    p.add_argument('--out', type=Path, default=_HERE / 'paper_data')
    p.add_argument('--skip-existing', action='store_true',
                   help='Skip suites whose json already exists.')
    p.add_argument('--verbose', action='store_true',
                   help='Print each collected episode.')
    args = p.parse_args()

    chosen = dict(DEFAULT_RUNS)
    for kv in (args.runs or []):
        k, _, v = kv.partition('=')
        chosen[k] = v

    runs = {r.name: r for r in discover_runs(variant=None)}
    defaults = load_sweep_defaults()
    args.out.mkdir(parents=True, exist_ok=True)

    wanted = [ts for ts in chosen
              if args.tasksets is None or ts in args.tasksets]

    print('=' * 72)
    print('expert-reuse paper data export')
    print(f'  evaluation seeds : {args.eval_seeds}  '
          f'({args.closedloop_episodes} closed-loop episodes each)')
    print(f'  output           : {args.out}')
    print('=' * 72)

    all_out, failures = {}, []
    for ts in wanted:
        name = f'{ts}__{chosen[ts]}'
        dest = args.out / f'{ts}.json'
        if args.skip_existing and dest.exists():
            print(f'--- {ts}: already exported, skipping')
            all_out[ts] = json.loads(dest.read_text())
            continue
        run = runs.get(name)
        if run is None:
            print(f'--- {ts}: run {name} NOT FOUND, skipping')
            failures.append((ts, 'run not found'))
            continue
        if not run.has_causal_checkpoint():
            print(f'--- {ts}: {name} has no backbone+heads checkpoint, '
                  f'skipping')
            failures.append((ts, 'no usable checkpoint'))
            continue

        print(f'\n--- {ts}  ({name}) ---')
        t0 = time.time()
        try:
            tab = gate_mod.analyze_run(run)
            geom = run.moe_geometry()
            rec = dict(
                suite=ts,
                composition_axis=SUITE_AXIS.get(ts),
                run=name,
                seed_dir=chosen[ts],
                tasks=run.short_tasks(),
                primitive_tasks=run.short_tasks()[:-1],
                composition_task=run.short_tasks()[-1],
                num_tasks=run.num_tasks,
                K_per_task=geom['K_per_task'],
                total_K=geom['total_K'],
                final_task_complete=bool(tab['final_complete']),
                final_task_step=int(tab['final_step']),
                final_task_budget=int(tab['final_budget']),
                router=router_section(run, tab),
            )
            rec['evaluation'] = evaluation_section(
                run, tab, defaults, geom, args)
            rec['seconds'] = _f(time.time() - t0)
            # Written as soon as the suite finishes, so an interrupted batch
            # keeps everything already computed.
            dest.write_text(json.dumps(rec, indent=2))
            all_out[ts] = rec
            print(f'--- {ts}: wrote {dest} ({time.time() - t0:.0f}s)')
        except Exception:
            traceback.print_exc()
            failures.append((ts, 'exception'))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Rebuild the combined files from EVERY per-suite json on disk, not just
    # the suites this invocation happened to process. Running one suite per
    # GPU is the recommended way to do this, and each of those invocations
    # would otherwise overwrite all_suites.json / summary.csv with its own
    # single suite.
    combined = {}
    for f in sorted(args.out.glob('*.json')):
        if f.name == 'all_suites.json':
            continue
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        if isinstance(rec, dict) and 'suite' in rec:
            combined[rec['suite']] = rec
    if combined:
        (args.out / 'all_suites.json').write_text(
            json.dumps(combined, indent=2))
        _write_csv(combined, args.out / 'summary.csv')
        print(f'\nwrote {args.out}/all_suites.json and summary.csv '
              f'({len(combined)} suites on disk)')
    all_out = combined or all_out

    print('\n' + '=' * 72)
    print(f'{"suite":<14}{"run":<22}{"cond":<14}{"norm return":>12}'
          f'{"±sem":>7}{"MSE ratio":>11}')
    print('-' * 72)
    for ts, rec in all_out.items():
        ev = rec.get('evaluation', {})
        for n in ev.get('conditions', []):
            a = ev['closedloop'][n]['across_seeds']
            o = ev['openloop'][n]['across_seeds_mse_ratio_to_full']
            print(f'{ts:<14}{rec["seed_dir"]:<22}{n:<14}'
                  f'{(a["mean"] if a["mean"] is not None else float("nan")):>12.1f}'
                  f'{(a["sem"] if a["sem"] is not None else float("nan")):>7.1f}'
                  f'{(o["mean"] if o["mean"] is not None else float("nan")):>11.2f}')
    if failures:
        print('\nnot exported:')
        for ts, why in failures:
            print(f'  {ts}: {why}')
    print('=' * 72)


def _write_csv(all_out: dict, path: Path):
    """Flat long-format table: one row per suite x condition."""
    cols = ['suite', 'composition_axis', 'run', 'checkpoint_step',
            'checkpoint_frac', 'condition',
            'closedloop_norm_mean', 'closedloop_norm_std',
            'closedloop_norm_sem', 'closedloop_drop_vs_unablated',
            'closedloop_success_mean',
            'openloop_mse_mean', 'openloop_mse_ratio_to_full_mean',
            'n_eval_seeds']
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for ts, rec in all_out.items():
            ev = rec.get('evaluation', {})
            ck = ev.get('checkpoint', {})
            for n in ev.get('conditions', []):
                c = ev['closedloop'][n]
                o = ev['openloop'][n]
                w.writerow({
                    'suite': ts,
                    'composition_axis': rec.get('composition_axis'),
                    'run': rec['run'],
                    'checkpoint_step': ck.get('step'),
                    'checkpoint_frac': ck.get('frac'),
                    'condition': n,
                    'closedloop_norm_mean': c['across_seeds']['mean'],
                    'closedloop_norm_std': c['across_seeds']['std'],
                    'closedloop_norm_sem': c['across_seeds']['sem'],
                    'closedloop_drop_vs_unablated': c['drop_vs_unablated'],
                    'closedloop_success_mean':
                        c['across_seeds_success']['mean'],
                    'openloop_mse_mean': o['across_seeds_mse']['mean'],
                    'openloop_mse_ratio_to_full_mean':
                        o['across_seeds_mse_ratio_to_full']['mean'],
                    'n_eval_seeds': c['across_seeds']['n'],
                })


if __name__ == '__main__':
    main()
