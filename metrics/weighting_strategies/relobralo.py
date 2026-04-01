from __future__ import annotations

from typing import Dict, List, Optional
import random
import torch

from ..loss_weighting_strategies import LossWeightingStrategyBase


class ReLoBRaLo(LossWeightingStrategyBase):
    """
    Relative Loss Balancing with Random Lookback. 
    
    Based on (2021) "Multi-Objective Loss Balancing for Physics-Informed Deep Learning" 
    https://arxiv.org/abs/2110.09813
    """

    def __init__(
        self,
        update_frequency: int = 1,
        use_gradients: bool = False,
        alpha: float = 0.9,            # EMA factor in Eq. (11)
        tau: float = 1.0,              # temperature τ in λ_bal
        rho_prob: float = 0.999,       # P(ρ=1) (Bernoulli); "random lookback" switch
        eps: float = 1e-12,            # numerical stability
        normalize_sum_to_m: bool = True,   # softmax*m already sums to m; keep True unless you want to disable
        min_weight: Optional[float] = None, # not in paper; set to None to be strictly paper-faithful
        max_weight: Optional[float] = None  # not in paper; set to None to be strictly paper-faithful
    ):
        super().__init__(update_frequency, use_gradients)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        if tau <= 0.0:
            raise ValueError("tau must be > 0.")
        if not (0.0 <= rho_prob <= 1.0):
            raise ValueError("rho_prob must be in [0, 1].")

        self.alpha = float(alpha)
        self.tau = float(tau)
        self.rho_prob = float(rho_prob)
        self.eps = float(eps)
        self.normalize_sum_to_m = bool(normalize_sum_to_m)
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Internal state across timesteps t
        self._L0: Dict[str, float] = {}        # initial loss per component: L_i(0)
        self._L_prev: Dict[str, float] = {}    # previous loss per component: L_i(t-1)
        self._lambda_prev: Dict[str, float] = {}  # previous scaling λ_i(t-1)

    @staticmethod
    def _mean_loss(values: List[float]) -> float:
        if not values:
            return float("nan")
        s = 0.0
        for v in values:
            s += float(v)
        return s / max(len(values), 1)

    def _stable_softmax(self, x: torch.Tensor) -> torch.Tensor:
        x = x - torch.max(x)
        ex = torch.exp(x)
        return ex / (torch.sum(ex) + self.eps)

    def _lambda_bal(
        self,
        loss_keys: List[str],
        L_t: Dict[str, float],
        L_tprime: Dict[str, float],
    ) -> Dict[str, float]:
        """
        λ_bal_i(t,t') = m * softmax( L_i(t) / (τ * L_i(t')) )
        """
        m = len(loss_keys)
        logits = []
        for key in loss_keys:
            num = max(L_t[key], self.eps)
            den = max(self.tau * L_tprime[key], self.eps)
            logits.append(num / den)

        logits_t = torch.tensor(logits, dtype=torch.float32)
        probs = self._stable_softmax(logits_t)
        scaled = probs * float(m)

        out: Dict[str, float] = {}
        for i, key in enumerate(loss_keys):
            out[key] = float(scaled[i].item())
        return out

    def compute_new_weights(
        self,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict],
        grad_norm_history: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Dict]:
        """
        Steps:
          1) Compute L_i(t) for all loss keys (base, channel, sub-component)
          2) Compute λ_bal(t, t-1) and λ_bal(t, 0) for all keys
          3) Sample ρ and compute λ_hist for all keys
          4) EMA combine into λ(t) for all keys
          5) Reconstruct hierarchical weight dictionary
        """
        # Get all loss keys from history
        all_loss_keys = list(loss_history.keys())
        m = len(all_loss_keys)
        if m == 0:
            return {k: v.copy() for k, v in current_weights.items()}

        # --- Step 1: Build L_i(t) for all loss keys (unweighted) ---
        L_t: Dict[str, float] = {}
        for loss_key in all_loss_keys:
            vals = loss_history.get(loss_key, [])
            
            # Unweight the losses by dividing by current weight
            curr_weight = self._get_previous_weight(loss_key, current_weights)
            if curr_weight > self.eps:
                unweighted_vals = [v / curr_weight for v in vals]
            else:
                unweighted_vals = vals
            
            mu = self._mean_loss(unweighted_vals)
            if mu != mu:  # NaN check
                # No/invalid data: fallback
                if loss_key in self._L_prev:
                    mu = self._L_prev[loss_key]
                else:
                    mu = 1.0
            L_t[loss_key] = float(mu)

            # Initialize L0 the first time we see this key
            if loss_key not in self._L0:
                self._L0[loss_key] = float(mu)

        # --- Step 2: Ensure we have L_prev and lambda_prev initialized ---
        for loss_key in all_loss_keys:
            if loss_key not in self._L_prev:
                self._L_prev[loss_key] = L_t[loss_key]

        # Initialize λ(t-1) from current_weights if we haven't stored it yet
        if not self._lambda_prev:
            uniform = float(m) / float(m)  # =1.0 each
            for loss_key in all_loss_keys:
                prev_w = self._get_previous_weight(loss_key, current_weights)
                self._lambda_prev[loss_key] = float(prev_w) if prev_w != 1.0 else uniform

        # Ensure all keys have lambda_prev (for newly added keys)
        for loss_key in all_loss_keys:
            if loss_key not in self._lambda_prev:
                prev_w = self._get_previous_weight(loss_key, current_weights)
                self._lambda_prev[loss_key] = float(prev_w)

        # --- Step 3: Compute λ_bal(t, t-1) and λ_bal(t, 0) ---
        L_tminus1 = {key: self._L_prev[key] for key in all_loss_keys}
        L_0 = {key: self._L0[key] for key in all_loss_keys}

        lambda_bal_recent = self._lambda_bal(all_loss_keys, L_t, L_tminus1)  # (t, t-1)
        lambda_bal_start = self._lambda_bal(all_loss_keys, L_t, L_0)         # (t, 0)

        # --- Step 4: Sample ρ ~ Bernoulli(rho_prob) and compute λ_hist ---
        rho = 1.0 if random.random() < self.rho_prob else 0.0

        lambda_hist: Dict[str, float] = {}
        for loss_key in all_loss_keys:
            lambda_hist[loss_key] = rho * self._lambda_prev[loss_key] + (1.0 - rho) * lambda_bal_start[loss_key]

        # --- Step 5: EMA combine into λ(t) per Eq. (11) ---
        lambda_t: Dict[str, float] = {}
        for loss_key in all_loss_keys:
            lambda_t[loss_key] = self.alpha * lambda_hist[loss_key] + (1.0 - self.alpha) * lambda_bal_recent[loss_key]

        # --- Step 6: Optional: enforce sum-to-m ---
        if self.normalize_sum_to_m:
            s = sum(lambda_t.values())
            if s > self.eps:
                scale = float(m) / s
                for loss_key in all_loss_keys:
                    lambda_t[loss_key] *= scale

        # --- Step 7: Optional: clip (NOT in paper) ---
        if self.min_weight is not None or self.max_weight is not None:
            lo = self.min_weight if self.min_weight is not None else -float("inf")
            hi = self.max_weight if self.max_weight is not None else float("inf")
            for loss_key in all_loss_keys:
                lambda_t[loss_key] = float(max(lo, min(hi, lambda_t[loss_key])))

            if self.normalize_sum_to_m:
                s = sum(lambda_t.values())
                if s > self.eps:
                    scale = float(m) / s
                    for loss_key in all_loss_keys:
                        lambda_t[loss_key] *= scale

        # --- Step 8: Reconstruct hierarchical weight dictionary ---
        new_weights = self._reconstruct_weight_dict(lambda_t, current_weights)

        # --- Step 9: Update internal state for next call ---
        for loss_key in all_loss_keys:
            self._L_prev[loss_key] = L_t[loss_key]
            self._lambda_prev[loss_key] = lambda_t[loss_key]

        return new_weights
