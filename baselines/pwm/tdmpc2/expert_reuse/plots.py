"""Figures for the expert-reuse analysis.

Colour follows the data's job, not taste:

  routing mass, leave-one-out damage magnitude   sequential, one blue hue,
                                                 light -> dark
  change against a reference (start vs end,
  ablation vs unablated)                         diverging blue<->red with a
                                                 neutral gray midpoint
  expert-block identity (which task owns an
  expert)                                        categorical, fixed slot order

Old vs new experts are additionally marked with hatching and an axis strip, so
that distinction never rests on colour alone. Figures are rendered for a light
paper surface.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ---- palette ----------------------------------------------------------
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
GRID = '#d8d7d2'
NEUTRAL = '#f0efec'

SEQ_BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
            '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281',
            '#0d366b']
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4',
               '#008300', '#4a3aa7', '#e34948']
DIV_RED = '#d03b3b'
DIV_BLUE = '#2a78d6'

CMAP_SEQ = LinearSegmentedColormap.from_list('seq_blue', SEQ_BLUE)
CMAP_DIV = LinearSegmentedColormap.from_list(
    'div_br', [DIV_BLUE, '#9ec5f4', NEUTRAL, '#eda49f', DIV_RED])

plt.rcParams.update({
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID,
    'axes.labelcolor': INK,
    'text.color': INK,
    'xtick.color': INK_2,
    'ytick.color': INK_2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 9,
    'axes.titlesize': 10,
    'figure.dpi': 140,
})


def _save(fig, outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f'{name}.png'
    fig.savefig(png, bbox_inches='tight')
    fig.savefig(outdir / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    return png


def _block_colors(num_blocks: int) -> list[str]:
    return [CATEGORICAL[i % len(CATEGORICAL)] for i in range(num_blocks)]


def _expert_labels(total_K: int, K: int) -> list[str]:
    return [f'{k}' for k in range(total_K)]


def _draw_block_strip(ax, total_K: int, K: int, nT: int, y=-0.9, h=0.35):
    """Thin strip under the x-axis showing which task owns each expert."""
    cols = _block_colors(nT)
    for j in range(nT):
        ax.add_patch(plt.Rectangle(
            (j * K - 0.5, y), K, h, facecolor=cols[j], edgecolor=SURFACE,
            linewidth=1.5, clip_on=False))
        ax.text(j * K + K / 2 - 0.5, y + h / 2, f'T{j + 1}', ha='center',
                va='center', fontsize=7.5, color='white', fontweight='bold',
                clip_on=False)


# =======================================================================
# 1. The router heatmap  (the headline figure)
# =======================================================================

def router_heatmap(tab: dict, outdir: Path, name='router_heatmap') -> Path:
    """Rows = tasks, columns = experts, cell = softmax routing weight.

    A cell is blank where the expert was outside that task's active window
    (progressive masking means task t only ever routes over experts
    [0, (t+1)*K)).
    """
    W = tab['weights']
    nT, total_K, K = tab['num_tasks'], tab['total_K'], tab['K_per_task']

    fig, ax = plt.subplots(figsize=(0.62 * total_K + 3.2, 0.62 * nT + 2.4))
    masked = np.ma.masked_invalid(W)
    cmap = CMAP_SEQ.copy()
    cmap.set_bad(NEUTRAL)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=float(np.nanmax(W)),
                   aspect='auto')

    # Cell values. The grid is small enough that the annotated heatmap
    # doubles as the table view.
    vmax = float(np.nanmax(W))
    for t in range(nT):
        for k in range(total_K):
            v = W[t, k]
            if not np.isfinite(v):
                continue
            ax.text(k, t, f'{v:.2f}'.lstrip('0') if v < 1 else f'{v:.2f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if v > 0.55 * vmax else INK)

    # Hatch the frozen prior-task experts of each row so "old vs new" is not
    # carried by position alone.
    for t in range(1, nT):
        ax.add_patch(plt.Rectangle(
            (-0.5, t - 0.5), t * K, 1, fill=False, hatch='///',
            edgecolor='#ffffff', linewidth=0, alpha=0.35, zorder=3))

    for j in range(1, nT):
        ax.axvline(j * K - 0.5, color=SURFACE, linewidth=2.5)

    ax.set_xticks(range(total_K))
    ax.set_xticklabels(_expert_labels(total_K, K), fontsize=7.5)
    ax.set_yticks(range(nT))
    ax.set_yticklabels([f'T{t + 1}  {tab["tasks"][t]}' for t in range(nT)],
                       fontsize=8)
    ax.set_xlabel('dynamics-MoE expert index', labelpad=16)
    ax.set_title(
        f'{tab["taskset"]} / {tab["seed_dir"]} — router weight per expert\n'
        f'gate input is the task one-hot only, so each row is constant '
        f'across states', loc='left', color=INK_2,
        fontsize=9)
    _draw_block_strip(ax, total_K, K, nT, y=nT - 0.5 + 0.35, h=0.32)
    ax.set_ylim(nT - 0.5 + 0.95, -0.5)

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label('routing weight', fontsize=8)
    cb.outline.set_visible(False)

    # Hatched-cell meaning, stated rather than left to the reader.
    ax.annotate('hatched = experts frozen from earlier tasks   ·   '
                'gray = outside that task’s active window',
                xy=(0, 0), xycoords='axes fraction',
                xytext=(0, -46), textcoords='offset points',
                fontsize=7.5, color=INK_2, annotation_clip=False)
    return _save(fig, outdir, name)


# =======================================================================
# 2. Final task: router at the start vs end of training
# =======================================================================

def start_vs_end(tab: dict, outdir: Path, name='final_start_vs_end') -> Path:
    """The exact null. A one-hot gate zeroes the gradient on every column but
    the active one, so the final task's gate column is untouched until that
    task begins: the previous checkpoint's row IS the initial router.
    """
    t = tab['num_tasks'] - 1
    aK = tab['active_Ks'][t]
    K, nT = tab['K_per_task'], tab['num_tasks']
    w_end = tab['weights'][t, :aK]
    w_start = tab['weights_start'][t, :aK]

    x = np.arange(aK)
    fig, ax = plt.subplots(figsize=(0.62 * aK + 3.0, 3.4))
    bw = 0.38
    ax.bar(x - bw / 2 - 0.01, w_start, bw, label='start of final task (null)',
           color='#9ec5f4', edgecolor=SURFACE, linewidth=1.2)
    ax.bar(x + bw / 2 + 0.01, w_end, bw, label='end of final task (learned)',
           color='#1c5cab', edgecolor=SURFACE, linewidth=1.2)

    n_old = t * K
    ax.axvspan(-0.5, n_old - 0.5, color=NEUTRAL, zorder=0)
    ax.text((n_old - 1) / 2, ax.get_ylim()[1] * 0.96,
            'experts from earlier tasks (frozen)', ha='center', va='top',
            fontsize=8, color=INK_2)
    ax.text(n_old + (aK - n_old - 1) / 2, ax.get_ylim()[1] * 0.96,
            'new experts', ha='center', va='top', fontsize=8, color=INK_2)

    ax.axhline(1.0 / aK, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(aK - 0.4, 1.0 / aK, '  uniform', va='center', fontsize=7.5,
            color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in range(aK)], fontsize=7.5)
    ax.set_xlabel('expert index', labelpad=14)
    ax.set_ylabel('routing weight')
    ax.set_title(
        f'{tab["taskset"]} / {tab["seed_dir"]} — final task '
        f'({tab["tasks"][t]}): the router is learned, not inherited',
        loc='left', fontsize=9, color=INK_2)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.grid(axis='y', color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    _draw_block_strip(ax, aK, K, t + 1, y=-0.052 * ax.get_ylim()[1],
                      h=0.036 * ax.get_ylim()[1])
    return _save(fig, outdir, name)


# =======================================================================
# 3. Reuse mass across runs
# =======================================================================

def reuse_summary(tables: list[dict], outdir: Path,
                  name='reuse_summary') -> Path:
    """Old-expert routing mass on each run's final task, against both nulls."""
    labels, old, uni, start = [], [], [], []
    for tab in tables:
        f = tab['final']
        labels.append(f'{tab["taskset"]}\n'
                      f'{tab["seed_dir"].replace("seed_", "s")}')
        old.append(f['old_mass'])
        uni.append(f['uniform_null'])
        start.append(f['start_null'])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(1.35 * len(labels) + 2.6, 3.9))
    bars = ax.bar(x, old, 0.5, color='#1c5cab', edgecolor=SURFACE,
                  linewidth=1.2)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor='#1c5cab', edgecolor=SURFACE,
                     label='learned mass on prior experts')]
    # The two nulls land almost on top of each other (that agreement is
    # itself the result), so they are drawn at different widths to stay
    # separately legible.
    for i, (u, sv) in enumerate(zip(uni, start)):
        ax.plot([x[i] - 0.36, x[i] + 0.36], [u, u], color=INK_2, linewidth=1.6,
                linestyle=(0, (4, 3)), zorder=4)
        if np.isfinite(sv):
            ax.plot([x[i] - 0.2, x[i] + 0.2], [sv, sv], color=DIV_RED,
                    linewidth=2.4, zorder=5, solid_capstyle='butt')
    from matplotlib.lines import Line2D
    handles += [
        Line2D([], [], color=INK_2, linewidth=1.6, linestyle=(0, (4, 3)),
               label='uniform null  t/(t+1)'),
        Line2D([], [], color=DIV_RED, linewidth=2.4,
               label='start-of-task null (exact)'),
    ]
    for b, v in zip(bars, old):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f'{v:.2f}',
                ha='center', fontsize=8, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('routing mass on prior-task experts')
    ax.set_ylim(0, 1)
    ax.set_title('Final-task reuse mass vs. what the router started from',
                 loc='left', fontsize=9, color=INK_2)
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=2,
              loc='upper right')
    ax.grid(axis='y', color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    return _save(fig, outdir, name)


# =======================================================================
# 3b. Primitive task vs composition task, over the SAME experts
# =======================================================================

def primitive_vs_composition(tab: dict, outdir: Path,
                             name='primitive_vs_composition') -> Path | None:
    """For each primitive task's block of experts, how that task routed over
    its own experts versus how the composition task routes over them.

    One panel per prior block. Both rows answer a different question and are
    both needed:

      top     routing weight as a MULTIPLE OF ITS UNIFORM SHARE. Raw weights
              are not comparable across tasks here: each task softmaxes over
              a different number of active experts (3 after task 1, 6 after
              task 2, 9 after task 3), so every weight is drawn from a
              distribution summing to 1 over a different support and the
              later task's bars would sit lower purely by construction.
              Dividing by 1/K_active removes that, so 1.0 means "exactly this
              expert's fair share" on every task and the two series can be
              read against each other. Above 1 = favoured, below 1 = starved.
      bottom  the same weights renormalised WITHIN the block, which removes
              the block's overall size and isolates the shape: does the
              composition task prefer the same expert the primitive did?

    The bottom row answers "did it arrive at a similar weighting". The top row
    is what stops that being over-read: a perfect shape match on a block the
    composition task has starved to a third of its fair share is a much weaker
    statement than the same match on a block held at twice its share.
    """
    K, nT = tab['K_per_task'], tab['num_tasks']
    t = nT - 1
    n_blocks = t
    if n_blocks < 1:
        return None
    sim = tab.get('copy_similarity', {}).get(f'task{t}', {})

    fig, axes = plt.subplots(
        2, n_blocks, figsize=(2.9 * n_blocks + 1.6, 5.4),
        squeeze=False, sharey='row')

    C_OWN, C_COMP = CATEGORICAL[0], CATEGORICAL[1]
    bw = 0.38
    raw_max = 0.0

    for j in range(n_blocks):
        sl = slice(j * K, (j + 1) * K)
        own = tab['weights'][j, sl].astype(float)
        comp = tab['weights'][t, sl].astype(float)
        if not (np.isfinite(own).all() and np.isfinite(comp).all()):
            continue
        x = np.arange(K)
        # Each task routes over a different number of experts, so put both
        # series on the "multiple of uniform share" scale before comparing.
        aK_own = tab['active_Ks'][j]
        aK_comp = tab['active_Ks'][t]
        own_u = own * aK_own
        comp_u = comp * aK_comp
        raw_max = max(raw_max, float(own_u.max()), float(comp_u.max()))

        # ---- top: weight relative to uniform share -----------------
        ax = axes[0][j]
        ax.bar(x - bw / 2 - 0.01, own_u, bw, color=C_OWN,
               edgecolor=SURFACE, linewidth=1.2)
        ax.bar(x + bw / 2 + 0.01, comp_u, bw, color=C_COMP,
               edgecolor=SURFACE, linewidth=1.2)
        ax.axhline(1.0, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)),
                   zorder=4)
        ax.set_title(f'T{j + 1}  {tab["tasks"][j]}', fontsize=8.5,
                     color=INK, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([str(j * K + i) for i in range(K)], fontsize=7.5)
        ax.grid(axis='y', color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        if j == 0:
            ax.set_ylabel('weight ÷ uniform share\n(1 = fair share)',
                          fontsize=8)
        # Block-level version of the same normalisation: a block's fair share
        # is 1/(number of blocks active), so this says whether the block as a
        # whole is favoured or starved, independently of how many blocks exist.
        own_blocks = j + 1
        comp_blocks = t + 1
        ax.annotate(
            f'block share {own.sum() * own_blocks:.2f} → '
            f'{comp.sum() * comp_blocks:.2f}',
            xy=(0.5, 0.97), xycoords='axes fraction', ha='center',
            va='top', fontsize=7, color=INK_2)

        # ---- bottom: renormalised within the block ----------------
        ax = axes[1][j]
        o = own / own.sum() if own.sum() > 0 else own
        c = comp / comp.sum() if comp.sum() > 0 else comp
        ax.bar(x - bw / 2 - 0.01, o, bw, color=C_OWN,
               edgecolor=SURFACE, linewidth=1.2)
        ax.bar(x + bw / 2 + 0.01, c, bw, color=C_COMP,
               edgecolor=SURFACE, linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(j * K + i) for i in range(K)], fontsize=7.5)
        ax.set_xlabel('expert index', fontsize=8)
        ax.grid(axis='y', color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 1.12)
        if j == 0:
            ax.set_ylabel('routing weight\n(renormalised in block)',
                          fontsize=8)

        # Mark the expert each side prefers, and quote the agreement. The
        # tau-removed cosine is the honest one (see gate.router_copy_similarity)
        # because the tau column is shared across tasks.
        b = sim.get(f'block{j}', {})
        if b:
            match = b.get('argmax_match_no_tau', b.get('argmax_match'))
            txt = (f'cos {b["cosine_no_tau"]:.3f}'
                   if 'cosine_no_tau' in b else f'cos {b["cosine"]:.3f}')
            txt += '   same top expert' if match else '   different top expert'
            ax.annotate(txt, xy=(0.5, 0.99), xycoords='axes fraction',
                        ha='center', va='top', fontsize=7.5,
                        color=INK if match else DIV_RED)
        star = int(np.argmax(o))
        ax.plot([star], [o[star] + 0.05], marker='v', markersize=5,
                color=C_OWN, clip_on=False)
        star_c = int(np.argmax(c))
        ax.plot([star_c], [c[star_c] + 0.05], marker='v', markersize=5,
                color=C_COMP, clip_on=False)

    for j in range(n_blocks):
        axes[0][j].set_ylim(0, max(raw_max * 1.28, 1.35))

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=C_OWN, edgecolor=SURFACE,
                      label='after its own primitive task'),
        plt.Rectangle((0, 0), 1, 1, facecolor=C_COMP, edgecolor=SURFACE,
                      label=f'after the composition task ({tab["tasks"][t]})'),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, ncol=2,
               loc='lower center', bbox_to_anchor=(0.5, -0.035))
    fig.suptitle(
        f'{tab["taskset"]} / {tab["seed_dir"]} — does the composition task '
        f'reuse each primitive the way that primitive did?',
        x=0.02, ha='left', fontsize=9.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return _save(fig, outdir, name)


# =======================================================================
# 4. Causal per-expert importance (leave-one-expert-out)
# =======================================================================

def causal_heatmap(tab: dict, loo: dict, outdir: Path,
                   name='router_vs_causal') -> Path:
    """Router weight (what the model *says* it uses) directly above
    leave-one-out prediction damage (what it *actually* needs), on a shared
    expert axis. Divergence between the two rows is the interesting result
    either way."""
    t = tab['num_tasks'] - 1
    aK = tab['active_Ks'][t]
    K = tab['K_per_task']
    w = tab['weights'][t, :aK]
    dmg = np.array([loo.get(f'drop_expert{k}', np.nan) for k in range(aK)])

    fig, axes = plt.subplots(
        2, 1, figsize=(0.62 * aK + 3.4, 4.2), sharex=True,
        gridspec_kw=dict(height_ratios=[1, 1], hspace=0.35))

    ax = axes[0]
    im0 = ax.imshow(w[None, :], cmap=CMAP_SEQ, vmin=0, vmax=float(w.max()),
                    aspect='auto')
    for k in range(aK):
        ax.text(k, 0, f'{w[k]:.2f}'.lstrip('0'), ha='center', va='center',
                fontsize=7, color='white' if w[k] > 0.55 * w.max() else INK)
    ax.set_yticks([0])
    ax.set_yticklabels(['router weight'], fontsize=8)
    fig.colorbar(im0, ax=ax, fraction=0.02, pad=0.015).outline.set_visible(False)

    ax = axes[1]
    finite = dmg[np.isfinite(dmg)]
    vmax = float(np.abs(finite).max()) if finite.size else 1.0
    im1 = ax.imshow(dmg[None, :], cmap=CMAP_DIV,
                    norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
                    aspect='auto')
    for k in range(aK):
        if np.isfinite(dmg[k]):
            ax.text(k, 0, f'{dmg[k]:+.0%}', ha='center', va='center',
                    fontsize=6.5, color=INK)
    ax.set_yticks([0])
    ax.set_yticklabels(['damage if removed'], fontsize=8)
    ax.set_xticks(range(aK))
    ax.set_xticklabels([str(k) for k in range(aK)], fontsize=7.5)
    ax.set_xlabel('expert index', labelpad=16)
    cb = fig.colorbar(im1, ax=ax, fraction=0.02, pad=0.015)
    cb.set_label('Δ latent-prediction MSE vs. unablated', fontsize=7.5)
    cb.outline.set_visible(False)

    for a in axes:
        for j in range(1, t + 1):
            a.axvline(j * K - 0.5, color=SURFACE, linewidth=2.5)
    _draw_block_strip(axes[1], aK, K, t + 1, y=0.85, h=0.3)
    axes[1].set_ylim(1.5, -0.5)

    axes[0].set_title(
        f'{tab["taskset"]} / {tab["seed_dir"]} — stated vs. actual reliance '
        f'on each expert, final task', loc='left', fontsize=9, color=INK_2)
    return _save(fig, outdir, name)


# =======================================================================
# 5. Ablation conditions
# =======================================================================

def ablation_bars(results: dict, tab: dict, outdir: Path, *, metric: str,
                  ylabel: str, name: str, higher_is_better=True,
                  err_metric: str | None = None) -> Path:
    """Headline ablation conditions as a bar chart against the unablated
    reference line."""
    order = [c for c in ['full', 'old_only', 'new_only', 'ctl_uniform',
                         'ctl_shuffle']
             if c in results]
    order += sorted(c for c in results
                    if c.startswith(('drop_block', 'keep_block')))
    vals = [results[c].get(metric, np.nan) for c in order]
    ref = results.get('full', {}).get(metric, np.nan)

    pretty = {'full': 'unablated', 'old_only': 'prior experts only',
              'new_only': 'new experts only', 'ctl_uniform': 'uniform routing',
              'ctl_shuffle': 'shuffled routing'}
    labels = [pretty.get(c, c.replace('drop_block', 'drop T').
                         replace('keep_block', 'only T')) for c in order]

    colors = []
    for c, v in zip(order, vals):
        if c == 'full':
            colors.append(INK_2)
        elif not np.isfinite(v) or not np.isfinite(ref):
            colors.append(GRID)
        else:
            worse = (v < ref) if higher_is_better else (v > ref)
            colors.append(DIV_RED if worse else DIV_BLUE)

    errs = ([results[c].get(err_metric, np.nan) for c in order]
            if err_metric else None)
    fig, ax = plt.subplots(figsize=(0.95 * len(order) + 2.4, 3.4))
    bars = ax.bar(range(len(order)), vals, 0.55, color=colors,
                  edgecolor=SURFACE, linewidth=1.2,
                  yerr=errs, ecolor=INK_2,
                  error_kw=dict(elinewidth=1.1, capsize=3, capthick=1.1))
    if np.isfinite(ref):
        ax.axhline(ref, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)))
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + 0.01 * (max(vals) - min(0, min(vals)) + 1e-9),
                    f'{v:.2f}', ha='center', fontsize=7.5, color=INK)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=7.5, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{tab["taskset"]} / {tab["seed_dir"]} — {ylabel} under '
                 f'routing ablation (final task)', loc='left', fontsize=9,
                 color=INK_2)
    ax.grid(axis='y', color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    return _save(fig, outdir, name)
