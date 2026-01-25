from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple
from ..loss_weighting_strategies import LossWeightingStrategyBase
import math
import torch


class BalancedResidualDecayRate(LossWeightingStrategyBase):
    """
    Component-wise BRDR weighting.

    Based on the pointwise BRDR from Chen et al. (2024); the major differences are:
      - weights are updated per-epoch (not per-iteration), and
      - weights exist for loss components (not per collocation point).
    """

    def __init__(
        self,
        update_frequency: int = 1,
        use_gradients: bool = False,
        beta_c: float = 0.9,
        beta_w: float = 0.9,
        bias_correction: bool = True,
        use_effective_smoothing_for_skips: bool = True,
        min_weight: float = 0.01,
        max_weight: float = 100.0,
        epsilon: float = 1e-8,
    ):
        super().__init__(update_frequency, use_gradients)
        self.beta_c = float(beta_c)
        self.beta_w = float(beta_w)
        self.bias_correction = bool(bias_correction)
        self.use_effective_smoothing_for_skips = bool(use_effective_smoothing_for_skips)

        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.epsilon = float(epsilon)

        # EMA state of component 4th-moment (m4)
        self.ema_m4: Dict[str, float] = {}

        # For effective smoothing when some components are absent in an epoch
        self.last_update_epoch: Dict[str, int] = {}

        # "Epoch counter" for bias correction term (1 - beta_c^t)
        self.epoch_counter: int = 0

    # ------------------------------
    # Helpers
    # ------------------------------

    def _clip(self, x: float) -> float:
        return max(self.min_weight, min(self.max_weight, x))

    def compute_statistics(self, losses: List[float]) -> Optional[Dict[str, float]]:
        """Mean/std/count for a list of scalar losses (fallback path only)."""
        if not losses:
            return None
        t = torch.tensor(losses, dtype=torch.float32)
        return {
            "mean": float(t.mean()),
            "std": float(t.std(unbiased=False)) if t.numel() > 1 else 0.0,
            "count": int(t.numel()),
        }


    # ------------------------------
    # Core BRDR update
    # ------------------------------

    def compute_new_weights(
        self,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict],
        grad_norm_history: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, Dict]:
        """
        Args:
            loss_history:
                Historical scalar losses by component key. Used ONLY as a fallback
                when component_moments is not provided.

            current_weights:
                Hierarchical dict of current weights.
        """
        # Internal epoch counter
        self.epoch_counter += 1
        t = self.epoch_counter

        # --- Step 1: compute c_k (IRDR proxy) for each loss_key ---
        # c_k = m2_k / (sqrt(EMA(m4_k)_corr) + eps)
        c: Dict[str, float] = {}
        counts: Dict[str, float] = {}

        for loss_key, losses in loss_history.items():
            if not losses:
                continue
            stats = self.compute_statistics(losses)
            if stats is None:
                continue
            L = max(float(stats["mean"]), 0.0)
            # Approximation: treat L as m2, and L^2 as m4.
            m2 = L
            m4 = L * L
            cnt = float(stats["count"])

            # Handle skipping: effective smoothing if component wasn't updated for some epochs
            if self.use_effective_smoothing_for_skips:
                last_e = self.last_update_epoch.get(loss_key, t)
                delta = max(1, t - last_e)
                beta_c_eff = self.beta_c ** delta
                beta_w_eff = self.beta_w ** delta
            else:
                beta_c_eff = self.beta_c
                beta_w_eff = self.beta_w
                delta = 1

            self.last_update_epoch[loss_key] = t

            # EMA update for m4
            prev_ema = self.ema_m4.get(loss_key, m4)
            ema = beta_c_eff * prev_ema + (1.0 - beta_c_eff) * m4
            self.ema_m4[loss_key] = float(ema)

            # Bias correction (epoch-level)
            if self.bias_correction:
                denom_m4 = ema / max(1.0 - (self.beta_c ** t), self.epsilon)
            else:
                denom_m4 = ema

            denom = math.sqrt(max(denom_m4, 0.0)) + self.epsilon
            c_val = (m2 / denom) if denom > 0.0 else 0.0

            # Store
            c[loss_key] = float(c_val)
            counts[loss_key] = float(max(cnt, 1.0))

            # Store beta_w_eff for this key to use in the weight update

        if not c:
            return {k: v.copy() for k, v in current_weights.items()}

        # --- Step 2: normalization ---
        # Paper normalizes by mean(c) over sampled points; here we normalize over components.
        # Optional: count-weighted mean to avoid tiny components dominating.
        total_cnt = sum(counts.values())
        if total_cnt <= self.epsilon:
            mean_c = sum(c.values()) / max(len(c), 1)
        else:
            mean_c = sum(c[k] * counts[k] for k in c.keys()) / total_cnt

        if mean_c <= self.epsilon:
            w_ref = {k: 1.0 for k in c.keys()}
        else:
            w_ref = {k: (v / mean_c) for k, v in c.items()}

        # --- Step 3: EMA update weights with beta_w (or beta_w_eff) + clip ---
        new_weight_scalars: Dict[str, float] = {}

        for loss_key, w_target in w_ref.items():
            prev_w = self._get_previous_weight(loss_key, current_weights)

            if self.use_effective_smoothing_for_skips:
                last_e = self.last_update_epoch.get(loss_key, t)
                beta_w_eff = self.beta_w
            else:
                beta_w_eff = self.beta_w

            w_new = beta_w_eff * prev_w + (1.0 - beta_w_eff) * float(w_target)
            new_weight_scalars[loss_key] = self._clip(float(w_new))

        # --- Step 4: reconstruct hierarchical dict ---
        return self._reconstruct_weight_dict(new_weight_scalars, current_weights)
