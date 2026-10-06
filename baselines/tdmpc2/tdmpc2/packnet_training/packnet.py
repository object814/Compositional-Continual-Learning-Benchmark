"""PackNet manager (paper-faithful) for TD-MPC2's shared backbone.

Faithfully follows Mallya & Lazebnik, CVPR 2018:
  - Only WEIGHT tensors (ndim >= 2) of Linear/Conv layers are pruned and
    masked per task.
  - Bias and normalisation parameters (ndim == 1) are frozen after the
    first task's prune-retrain cycle and shared across all subsequent tasks.

Backbone-only adaptation (mirrors the Dreamer PackNet manager):
  `sequential_train.py` shares only the encoder + dynamics across tasks;
  per-task heads (reward / Qs / termination / pi + RunningScale + MPC
  `_prev_mean`) are re-initialised every task. Capacity carving therefore
  needs to happen only inside the shared backbone — heads are protected
  by the architectural split itself.

Parameter categories within `agent.named_parameters()`:
  PRUNABLE (ndim >= 2, name starts with one of `backbone_prefixes`):
      Linear / Conv weight matrices in encoder + dynamics.
    -> Full PackNet: prune by magnitude, retrain, freeze surviving,
                     per-task masks.
  SHARED   (ndim == 1, name starts with a backbone prefix):
      biases, LayerNorm/SimNorm weights.
    -> Frozen after the first task's prune-retrain cycle.
  IGNORED:
      Per-task heads (reward / Qs / termination / pi / task_emb), target /
      detach Q clones, and any non-trainable buffers.
"""

from typing import Dict, List

import torch


def _norm_name(name: str) -> str:
    """Strip torch.compile's `_orig_mod.` prefix(es)."""
    return name.replace("._orig_mod.", ".").replace("_orig_mod.", "")


class PackNetManager:
    """Paper-faithful PackNet manager for the TD-MPC2 backbone."""

    def __init__(
        self,
        prune_ratio: float = 0.75,
        backbone_prefixes: tuple = ("model._encoder.", "model._dynamics."),
    ):
        self.prune_ratio = prune_ratio
        self.backbone_prefixes = backbone_prefixes

        # --- Prunable (weight) masks ---
        # frozen_mask[name] : float tensor, 1.0 = frozen, 0.0 = free
        self.frozen_mask: Dict[str, torch.Tensor] = {}
        # task_masks[task_idx][name] : float tensor, 1.0 = active for that
        # task (frozen + surviving), 0.0 = pruned.
        self.task_masks: Dict[int, Dict[str, torch.Tensor]] = {}

        # --- Shared (bias/norm) state ---
        self._shared_param_names: List[str] = []
        self._shared_params_frozen: bool = False

        self.num_tasks_packed: int = 0

        # Retrain state
        self._retrain_mode: bool = False
        self._retrain_mask: Dict[str, torch.Tensor] = {}
        self._retrain_task_mask: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Parameter classification
    # ------------------------------------------------------------------
    def _is_backbone_param(self, name: str, param: torch.Tensor) -> bool:
        if not param.requires_grad:
            return False
        norm = _norm_name(name)
        return any(norm.startswith(p) for p in self.backbone_prefixes)

    def _is_prunable(self, name: str, param: torch.Tensor) -> bool:
        return self._is_backbone_param(name, param) and param.ndim >= 2

    def _is_shared(self, name: str, param: torch.Tensor) -> bool:
        """1-D backbone params (biases, norm weights) frozen after task 1."""
        return self._is_backbone_param(name, param) and param.ndim == 1

    # ------------------------------------------------------------------
    # Initialisation: discover shared params
    # ------------------------------------------------------------------
    def register_agent_params(self, agent):
        """Call once after the agent is created. Discovers shared (1D) params.

        Must be re-called for every fresh agent (i.e. once per task) so the
        manager and the live tensor identities stay in sync.
        """
        self._shared_param_names = []
        n_prunable_groups = 0
        n_prunable_params = 0
        n_shared_groups = 0
        n_shared_params = 0

        for name, param in agent.named_parameters():
            if self._is_prunable(name, param):
                n_prunable_groups += 1
                n_prunable_params += param.numel()
            elif self._is_shared(name, param):
                self._shared_param_names.append(name)
                n_shared_groups += 1
                n_shared_params += param.numel()

        print(f">>> PackNet: {n_prunable_groups} backbone prunable groups "
              f"({n_prunable_params:,} params, ndim>=2)")
        print(f">>> PackNet: {n_shared_groups} backbone shared groups "
              f"({n_shared_params:,} params, ndim==1, "
              f"frozen after task 1)")

    # ------------------------------------------------------------------
    # Gradient masking (called every training step, world-model optimizer)
    # ------------------------------------------------------------------
    def apply_gradient_mask(self, agent):
        """Zero gradients on frozen backbone weights and shared backbone params.

        Call AFTER `loss.backward()` and BEFORE `clip_grad_norm_` /
        `optimizer.step()`.
        """
        for name, param in agent.named_parameters():
            if param.grad is None:
                continue
            if not self._is_backbone_param(name, param):
                continue

            if self._is_prunable(name, param):
                if self._retrain_mode and name in self._retrain_mask:
                    param.grad.data[self._retrain_mask[name] == 0] = 0.0
                elif name in self.frozen_mask:
                    param.grad.data[self.frozen_mask[name].bool()] = 0.0

            elif self._shared_params_frozen and self._is_shared(name, param):
                param.grad.data.zero_()

    def apply_weight_mask(self, agent):
        """Re-zero pruned backbone weights after `optimizer.step()`.

        Only relevant during retrain mode.
        """
        if not self._retrain_mode:
            return
        for name, param in agent.named_parameters():
            if name in self._retrain_task_mask:
                param.data.mul_(self._retrain_task_mask[name])

    # ------------------------------------------------------------------
    # Pruning (called once after training a task)
    # ------------------------------------------------------------------
    def prune(self, agent, task_idx: int) -> Dict[str, torch.Tensor]:
        """Prune the current task's free backbone weight params by magnitude."""
        task_mask = {}

        for name, param in agent.named_parameters():
            if not self._is_prunable(name, param):
                continue

            frozen = self.frozen_mask.get(name, torch.zeros_like(param.data))
            free_mask = (1.0 - frozen).bool()
            num_free = int(free_mask.sum().item())

            if num_free > 0:
                free_magnitudes = param.data.abs()[free_mask]
                k = int(num_free * self.prune_ratio)
                if 0 < k < num_free:
                    threshold = torch.kthvalue(free_magnitudes, k).values.item()
                    survive = frozen.bool() | (
                        free_mask & (param.data.abs() > threshold)
                    )
                elif k >= num_free:
                    survive = frozen.bool()
                else:
                    survive = torch.ones_like(param.data, dtype=torch.bool)
            else:
                survive = frozen.bool()

            mask = survive.float()
            task_mask[name] = mask
            param.data.mul_(mask)

        n_total = sum(
            p.numel() for n, p in agent.named_parameters()
            if self._is_prunable(n, p)
        )
        n_frozen = sum(v.sum().item() for v in self.frozen_mask.values())
        n_surviving = sum(v.sum().item() for v in task_mask.values())
        n_pruned = n_total - n_surviving
        n_task_surviving = n_surviving - n_frozen
        print(
            f">>> PackNet: pruned task {task_idx + 1} backbone weights "
            f"(prune_ratio={self.prune_ratio:.2f}): "
            f"{int(n_total):,} total, "
            f"{int(n_frozen):,} previously-frozen, "
            f"{int(n_task_surviving):,} task-surviving, "
            f"{int(n_pruned):,} pruned (free for future)"
        )
        return task_mask

    # ------------------------------------------------------------------
    # Retrain mode (called between prune and freeze)
    # ------------------------------------------------------------------
    def start_retrain(self, task_mask: Dict[str, torch.Tensor]):
        self._retrain_mode = True
        self._retrain_task_mask = {k: v.clone() for k, v in task_mask.items()}
        self._retrain_mask = {}
        for name, mask in task_mask.items():
            frozen = self.frozen_mask.get(name, torch.zeros_like(mask))
            self._retrain_mask[name] = mask * (1.0 - frozen)

    def end_retrain(self):
        self._retrain_mode = False
        self._retrain_mask = {}
        self._retrain_task_mask = {}

    # ------------------------------------------------------------------
    # Freeze (called after retrain to finalise the task)
    # ------------------------------------------------------------------
    def freeze_task(self, task_mask: Dict[str, torch.Tensor], task_idx: int):
        self.task_masks[task_idx] = {k: v.clone() for k, v in task_mask.items()}

        for name, mask in task_mask.items():
            if name in self.frozen_mask:
                self.frozen_mask[name] = torch.max(self.frozen_mask[name], mask)
            else:
                self.frozen_mask[name] = mask.clone()

        if not self._shared_params_frozen:
            self._shared_params_frozen = True
            print(f">>> PackNet: freezing {len(self._shared_param_names)} "
                  f"backbone shared (bias/norm) groups after task {task_idx + 1}")

        self.num_tasks_packed += 1

        n_frozen_w = sum(v.sum().item() for v in self.frozen_mask.values())
        n_total_w = sum(v.numel() for v in self.frozen_mask.values())
        pct = 100 * n_frozen_w / max(n_total_w, 1)
        print(
            f">>> PackNet: frozen task {task_idx + 1}: "
            f"{int(n_frozen_w):,}/{int(n_total_w):,} backbone weight params "
            f"frozen ({pct:.1f}%), shared_frozen={self._shared_params_frozen}"
        )

    # ------------------------------------------------------------------
    # Eval-time mask application (backbone weights only)
    # ------------------------------------------------------------------
    def save_agent_weights(self, agent) -> Dict[str, torch.Tensor]:
        """Snapshot prunable backbone weights so they can be restored after eval."""
        saved = {}
        for name, param in agent.named_parameters():
            if self._is_prunable(name, param):
                saved[name] = param.data.clone()
        return saved

    def restore_agent_weights(self, agent, saved: Dict[str, torch.Tensor]):
        for name, param in agent.named_parameters():
            if name in saved:
                param.data.copy_(saved[name])

    def apply_eval_mask(self, agent, task_idx: int):
        """Apply task-specific backbone weight mask for evaluation.

        Caller MUST `save_agent_weights` before this and
        `restore_agent_weights` afterwards — this mutates `param.data`.
        """
        if task_idx not in self.task_masks:
            return
        mask = self.task_masks[task_idx]
        for name, param in agent.named_parameters():
            if name in mask:
                param.data.mul_(mask[name])

    # ------------------------------------------------------------------
    # Save / Load (resume support)
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "frozen_mask": {k: v.cpu() for k, v in self.frozen_mask.items()},
            "task_masks": {
                tid: {k: v.cpu() for k, v in masks.items()}
                for tid, masks in self.task_masks.items()
            },
            "num_tasks_packed": self.num_tasks_packed,
            "prune_ratio": self.prune_ratio,
            "shared_params_frozen": self._shared_params_frozen,
            "shared_param_names": self._shared_param_names,
            "backbone_prefixes": list(self.backbone_prefixes),
        }

    def load_state_dict(self, state: dict):
        self.frozen_mask = state["frozen_mask"]
        self.task_masks = state["task_masks"]
        self.num_tasks_packed = state["num_tasks_packed"]
        self.prune_ratio = state.get("prune_ratio", self.prune_ratio)
        self._shared_params_frozen = state.get("shared_params_frozen", False)
        self._shared_param_names = state.get("shared_param_names", [])
        if "backbone_prefixes" in state:
            self.backbone_prefixes = tuple(state["backbone_prefixes"])

    def to_device(self, device):
        self.frozen_mask = {k: v.to(device) for k, v in self.frozen_mask.items()}
        self.task_masks = {
            tid: {k: v.to(device) for k, v in masks.items()}
            for tid, masks in self.task_masks.items()
        }
        if self._retrain_mode:
            self._retrain_mask = {
                k: v.to(device) for k, v in self._retrain_mask.items()
            }
            self._retrain_task_mask = {
                k: v.to(device) for k, v in self._retrain_task_mask.items()
            }
