"""EWC (Elastic Weight Consolidation) manager for TD-MPC2's shared backbone.

Standard per-task EWC (Kirkpatrick et al. 2017):
  L_total = L_task(θ) + λ · Σ_t Σ_i F_{t,i} · (θ_i - θ*_{t,i})²

Backbone-only adaptation (mirrors the Dreamer EWC manager):
  The continual-learning split in `sequential_train.py` shares ONLY the
  encoder + dynamics (the "backbone") across tasks; per-task heads
  (reward / Qs / termination / pi + RunningScale + MPC `_prev_mean`) are
  re-initialised every task and live on disk per task. Forgetting can only
  happen inside the shared backbone, so EWC protects encoder + dynamics
  weights and ignores everything else.

Fisher computation reproduces the world-model loss from `TDMPC2._update`
(without stepping the optimizers). The policy backward pass operates on
`zs.detach()` and so never flows into encoder/dynamics — we skip it.

Penalty computation uses pre-flattened vectorised tensors (fast
O(N_backbone_params) per training step).
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from common import math as tdmpc_math


# ---------------------------------------------------------------------------
# Parameter selection — which params does EWC protect?
# ---------------------------------------------------------------------------

def _norm_name(name: str) -> str:
    """Strip torch.compile's `_orig_mod.` prefix(es) so name matching works
    whether or not compile is on."""
    return name.replace("._orig_mod.", ".").replace("_orig_mod.", "")


# ---------------------------------------------------------------------------
# World-model loss reproduction (mirrors TDMPC2._update, no opt step)
# ---------------------------------------------------------------------------

def _compute_world_model_loss(agent, obs, action, reward, terminated, task):
    """Re-run the world-model forward pass and return `total_loss`.

    Mirrors the body of `TDMPC2._update` up to (but not including) the
    `total_loss.backward()` call. Keep in sync with `tdmpc2.py` if the
    upstream loss formulation changes.
    """
    cfg = agent.cfg
    model = agent.model

    with torch.no_grad():
        next_z = model.encode(obs[1:], task)
        td_targets = agent._td_target(next_z, reward, terminated, task)

    model.train()

    zs = torch.empty(
        cfg.horizon + 1, cfg.batch_size, cfg.latent_dim, device=agent.device,
    )
    z = model.encode(obs[0], task)
    zs[0] = z
    consistency_loss = 0
    for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
        z = model.next(z, _action, task)
        consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * cfg.rho ** t
        zs[t + 1] = z

    _zs = zs[:-1]
    qs = model.Q(_zs, action, task, return_type='all')
    reward_preds = model.reward(_zs, action, task)
    if cfg.episodic:
        termination_pred = model.termination(zs[1:], task, unnormalized=True)

    reward_loss, value_loss = 0, 0
    for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(
        zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))
    ):
        reward_loss = reward_loss + tdmpc_math.soft_ce(
            rew_pred_unbind, rew_unbind, cfg
        ).mean() * cfg.rho ** t
        for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
            value_loss = value_loss + tdmpc_math.soft_ce(
                qs_unbind_unbind, td_targets_unbind, cfg
            ).mean() * cfg.rho ** t

    consistency_loss = consistency_loss / cfg.horizon
    reward_loss = reward_loss / cfg.horizon
    if cfg.episodic:
        termination_loss = F.binary_cross_entropy_with_logits(
            termination_pred, terminated
        )
    else:
        termination_loss = 0.
    value_loss = value_loss / (cfg.horizon * cfg.num_q)

    total_loss = (
        cfg.consistency_coef * consistency_loss
        + cfg.reward_coef * reward_loss
        + cfg.termination_coef * termination_loss
        + cfg.value_coef * value_loss
    )
    return total_loss


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class EWCManager:
    """EWC regularisation manager for the TD-MPC2 backbone (encoder + dynamics)."""

    def __init__(
        self,
        lambda_ewc: float = 1000.0,
        backbone_prefixes: tuple = ("model._encoder.", "model._dynamics."),
    ):
        self.lambda_ewc = lambda_ewc
        self.backbone_prefixes = backbone_prefixes

        # {task_idx: {'importance': {name: F}, 'task_param': {name: θ*}}}
        self.regularization_terms: Dict[int, dict] = {}
        self.num_tasks_consolidated: int = 0

        # Pre-flattened cache for fast per-step penalty.
        self._penalty_cache_valid = False
        self._cached_fishers: List[torch.Tensor] = []
        self._cached_params: List[torch.Tensor] = []
        self._param_name_to_slice: Dict[str, Tuple[int, int]] = {}
        self._tracked_param_names: List[str] = []

    # ------------------------------------------------------------------
    # Parameter selection
    # ------------------------------------------------------------------
    def _is_backbone_param(self, name: str, param: torch.Tensor) -> bool:
        if not param.requires_grad:
            return False
        norm = _norm_name(name)
        return any(norm.startswith(p) for p in self.backbone_prefixes)

    def collect_backbone_params(self, agent) -> List[Tuple[str, torch.Tensor]]:
        """Return `(name, param)` pairs for every backbone param of `agent`.

        Names are taken verbatim from `agent.named_parameters()` so callers
        can index into the live param tree.
        """
        return [
            (n, p) for n, p in agent.named_parameters()
            if self._is_backbone_param(n, p)
        ]

    # ------------------------------------------------------------------
    # Fisher computation (called once per task boundary)
    # ------------------------------------------------------------------
    def compute_fisher(self, agent, buffer, num_batches: int = 50,
                       device: str = 'cuda'):
        """Compute diagonal Fisher across backbone (encoder + dynamics) params.

        Each batch:
          1. Sample (obs, action, reward, terminated, task) from `buffer`.
          2. Forward through the world-model loss only; backward.
          3. Accumulate squared per-parameter gradients (FP32) into `importance`.

        We deliberately skip the policy backward pass: π is updated on
        `zs.detach()`, so its gradients never flow into encoder/dynamics,
        and we are protecting only those.
        """
        importance = {}
        task_param = {}
        for name, param in self.collect_backbone_params(agent):
            importance[name] = torch.zeros_like(
                param.data, dtype=torch.float32, device=device,
            )
            task_param[name] = param.data.detach().to(
                device=device, dtype=torch.float32,
            ).clone()

        was_training = agent.model.training
        agent.model.train()
        for _, p in self.collect_backbone_params(agent):
            p.requires_grad_(True)

        for _ in range(num_batches):
            obs, action, reward, terminated, task = buffer.sample()

            agent.optim.zero_grad(set_to_none=True)
            agent.pi_optim.zero_grad(set_to_none=True)
            total_loss = _compute_world_model_loss(
                agent, obs, action, reward, terminated, task,
            )
            total_loss.backward()
            for name, param in self.collect_backbone_params(agent):
                if param.grad is not None:
                    g = param.grad.detach().float()
                    importance[name] += (g ** 2) / num_batches

            agent.optim.zero_grad(set_to_none=True)
            agent.pi_optim.zero_grad(set_to_none=True)

        if not was_training:
            agent.model.eval()

        return importance, task_param

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------
    def consolidate(self, agent, importance, task_param, task_idx: int):
        """Snapshot Fisher + θ* for `task_idx` and rebuild penalty cache."""
        self.regularization_terms[task_idx] = {
            'importance': {k: v.clone() for k, v in importance.items()},
            'task_param': {k: v.clone() for k, v in task_param.items()},
        }
        self.num_tasks_consolidated += 1
        self._rebuild_penalty_cache(agent)

    def _rebuild_penalty_cache(self, agent):
        """Pre-flatten Fisher diagonals and θ* for fast per-step penalty."""
        if self.num_tasks_consolidated == 0:
            self._penalty_cache_valid = False
            return

        device = next(iter(
            next(iter(self.regularization_terms.values()))['importance'].values()
        )).device

        first_reg = next(iter(self.regularization_terms.values()))
        self._tracked_param_names = []
        self._param_name_to_slice = {}
        offset = 0
        for name, param in self.collect_backbone_params(agent):
            if name in first_reg['importance']:
                n = param.numel()
                self._tracked_param_names.append(name)
                self._param_name_to_slice[name] = (offset, offset + n)
                offset += n

        total_params = offset
        self._cached_fishers = []
        self._cached_params = []
        for tid in sorted(self.regularization_terms.keys()):
            reg = self.regularization_terms[tid]
            fisher_flat = torch.zeros(total_params, dtype=torch.float32, device=device)
            params_flat = torch.zeros(total_params, dtype=torch.float32, device=device)
            for name in self._tracked_param_names:
                s, e = self._param_name_to_slice[name]
                fisher_flat[s:e] = reg['importance'][name].flatten()
                params_flat[s:e] = reg['task_param'][name].flatten()
            self._cached_fishers.append(fisher_flat)
            self._cached_params.append(params_flat)

        self._penalty_cache_valid = True

    # ------------------------------------------------------------------
    # Per-step penalty (called in the training loop)
    # ------------------------------------------------------------------
    def penalty(self, agent) -> torch.Tensor:
        """Compute λ · Σ_t Σ_i F · (θ - θ*)² over all backbone params.

        Returns a scalar tensor on the agent's device. Returns 0 if no
        tasks have been consolidated yet.
        """
        device = agent.device
        if not self._penalty_cache_valid or self.num_tasks_consolidated == 0:
            return torch.tensor(0.0, device=device)

        param_dict = {_norm_name(n): p for n, p in agent.named_parameters()}

        reg_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        for name in self._tracked_param_names:
            norm = _norm_name(name)
            if norm not in param_dict:
                continue
            s, e = self._param_name_to_slice[name]
            param_flat = param_dict[norm].flatten().float()
            for fisher_flat, params_flat in zip(self._cached_fishers, self._cached_params):
                diff = param_flat - params_flat[s:e]
                reg_loss = reg_loss + (fisher_flat[s:e] * diff ** 2).sum()
        return self.lambda_ewc * reg_loss

    # ------------------------------------------------------------------
    # Save / Load (resume support)
    # ------------------------------------------------------------------
    def state_dict(self):
        return {
            'regularization_terms': {
                tid: {
                    'importance': {k: v.detach().cpu() for k, v in reg['importance'].items()},
                    'task_param': {k: v.detach().cpu() for k, v in reg['task_param'].items()},
                }
                for tid, reg in self.regularization_terms.items()
            },
            'num_tasks_consolidated': self.num_tasks_consolidated,
            'lambda_ewc': self.lambda_ewc,
            'backbone_prefixes': list(self.backbone_prefixes),
        }

    def load_state_dict(self, state, agent=None):
        self.regularization_terms = state['regularization_terms']
        self.num_tasks_consolidated = state['num_tasks_consolidated']
        self.lambda_ewc = state.get('lambda_ewc', self.lambda_ewc)
        if 'backbone_prefixes' in state:
            self.backbone_prefixes = tuple(state['backbone_prefixes'])
        self._penalty_cache_valid = False
        if agent is not None and self.num_tasks_consolidated > 0:
            self._rebuild_penalty_cache(agent)

    def to_device(self, device):
        """Move all stored Fishers and θ* tensors to `device`."""
        for tid, reg in self.regularization_terms.items():
            for k in reg['importance']:
                reg['importance'][k] = reg['importance'][k].to(device)
                reg['task_param'][k] = reg['task_param'][k].to(device)
        self._penalty_cache_valid = False
