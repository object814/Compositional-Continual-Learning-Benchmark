"""Batched torch versions of the Meta-World reward utilities.

Same math as `metaworld/utils/reward_utils.py` (dm_control's `tolerance` with
the long-tail sigmoid, and the Hamacher t-norm), vectorised over worlds.
"""
import math

import torch

_DEFAULT_VALUE_AT_MARGIN = 0.1
_LONG_TAIL_SCALE = math.sqrt(1.0 / _DEFAULT_VALUE_AT_MARGIN - 1.0)


def tolerance(x, bounds=(0.0, 0.0), margin=0.0):
    """Long-tail `tolerance`: 1 inside `bounds`, decaying with distance / margin.

    `x` and `margin` may be tensors (broadcast together); a margin of 0 gives a
    hard 0/1 indicator, as in the original.
    """
    lower, upper = bounds
    x = torch.as_tensor(x)
    margin = torch.as_tensor(margin, dtype=x.dtype, device=x.device)
    in_bounds = (x >= lower) & (x <= upper)
    safe_margin = torch.where(margin > 0, margin, torch.ones_like(margin))
    d = torch.where(x < lower, lower - x, x - upper) / safe_margin
    soft = 1.0 / ((d * _LONG_TAIL_SCALE) ** 2 + 1.0)
    soft = torch.where(margin > 0, soft, torch.zeros_like(soft))
    return torch.where(in_bounds, torch.ones_like(soft), soft)


def hamacher_product(a, b):
    """(a * b) / (a + b - a * b), 0 when the denominator is 0."""
    denom = a + b - a * b
    safe = torch.where(denom > 0, denom, torch.ones_like(denom))
    return torch.where(denom > 0, a * b / safe, torch.zeros_like(denom))


def quat_xyzw_from_matrix(m):
    """Rotation matrices (..., 3, 3) -> quaternions (..., 4) in (x, y, z, w) order.

    Same algorithm (and sign convention) as scipy's
    `Rotation.from_matrix(m).as_quat()`, which the CPU env uses for the block.
    """
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    tr = m00 + m11 + m22
    decision = torch.stack([m00, m11, m22, tr], dim=-1)
    choice = decision.argmax(dim=-1)

    q = torch.empty(m.shape[:-2] + (4,), dtype=m.dtype, device=m.device)
    # choice == 3 (trace largest)
    q_tr = torch.stack([
        m[..., 2, 1] - m[..., 1, 2],
        m[..., 0, 2] - m[..., 2, 0],
        m[..., 1, 0] - m[..., 0, 1],
        1.0 + tr,
    ], dim=-1)
    q[:] = q_tr
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        qi = torch.empty_like(q_tr)
        qi[..., i] = 1.0 - tr + 2.0 * m[..., i, i]
        qi[..., j] = m[..., j, i] + m[..., i, j]
        qi[..., k] = m[..., k, i] + m[..., i, k]
        qi[..., 3] = m[..., k, j] - m[..., j, k]
        q = torch.where((choice == i).unsqueeze(-1), qi, q)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
