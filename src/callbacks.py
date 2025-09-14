from __future__ import annotations

import torch
from stable_baselines3.common.callbacks import BaseCallback


class UpdateMagnitudeCallback(BaseCallback):
    """Log parameter update magnitude per rollout.

    Records:
    - debug/update_delta_norm: ||theta_t - theta_{t-1}|| (L2 over all learnable params)
    - debug/param_norm_prev:  ||theta_{t-1}||
    - debug/update_ratio:     ||delta|| / ||theta_{t-1}||
    - debug/muon_lr, debug/adam_lr: average lr of param groups (if present)
    """

    def __init__(self, log_param_group_lrs: bool = True) -> None:
        super().__init__()
        self._prev_vec: torch.Tensor | None = None
        self._prev_norm: float | None = None
        self._log_param_group_lrs = log_param_group_lrs

    def _stack_learnable_params(self) -> tuple[torch.Tensor, float]:
        policy = self.model.policy  # type: ignore[attr-defined]
        with torch.no_grad():
            parts = [p.data.view(-1) for p in policy.parameters() if p.requires_grad]
            if not parts:
                vec = torch.tensor(0.0)
            else:
                vec = torch.cat(parts)
            norm = float(torch.linalg.norm(vec).item())
            return vec, norm

    def _on_rollout_end(self) -> bool:
        try:
            vec, norm = self._stack_learnable_params()
            # Log update ratio vs previous snapshot (after the previous training phase)
            if self._prev_vec is not None and self._prev_norm is not None and self._prev_norm > 0:
                delta = float(torch.linalg.norm(vec - self._prev_vec).item())
                ratio = delta / self._prev_norm
                self.model.logger.record("debug/update_delta_norm", delta)
                self.model.logger.record("debug/param_norm_prev", self._prev_norm)
                self.model.logger.record("debug/update_ratio", ratio)

            # Update snapshot for next cycle
            self._prev_vec = vec.detach().clone()
            self._prev_norm = max(norm, 1e-12)

            # Optionally, log per-group learning rates (Muon vs Adam)
            if self._log_param_group_lrs and hasattr(self.model.policy, "optimizer"):
                muon_lrs, adam_lrs = [], []
                for g in self.model.policy.optimizer.param_groups:  # type: ignore[attr-defined]
                    lr = float(g.get("lr", 0.0))
                    if g.get("use_muon", False):
                        muon_lrs.append(lr)
                    else:
                        adam_lrs.append(lr)
                if muon_lrs:
                    self.model.logger.record("debug/muon_lr", sum(muon_lrs) / len(muon_lrs))
                if adam_lrs:
                    self.model.logger.record("debug/adam_lr", sum(adam_lrs) / len(adam_lrs))

        except Exception as _:
            # Best-effort logging; never break training
            pass
        return True

    def _on_step(self) -> bool:  # required by BaseCallback abstract interface
        # We perform our logging at rollout end; keep stepping.
        return True
