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
        alpha: float = 0.9,            # EMA factor in Eq. (11)
        tau: float = 1.0,              # temperature τ in λ_bal
        rho_prob: float = 0.999,       # P(ρ=1) (Bernoulli); "random lookback" switch
        eps: float = 1e-12,            # numerical stability
        normalize_sum_to_m: bool = True,   # softmax*m already sums to m; keep True unless you want to disable
        min_weight: Optional[float] = None, # not in paper; set to None to be strictly paper-faithful
        max_weight: Optional[float] = None  # not in paper; set to None to be strictly paper-faithful
    ):
        super().__init__(update_frequency)
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
        component_names: List[str],
        L_t: Dict[str, float],
        L_tprime: Dict[str, float],
    ) -> Dict[str, float]:
        """
        λ_bal_i(t,t') = m * softmax( L_i(t) / (τ * L_i(t')) )
        """
        m = len(component_names)
        logits = []
        for name in component_names:
            num = max(L_t[name], self.eps)
            den = max(self.tau * L_tprime[name], self.eps)
            logits.append(num / den)

        logits_t = torch.tensor(logits, dtype=torch.float32)
        probs = self._stable_softmax(logits_t)
        scaled = probs * float(m)

        out: Dict[str, float] = {}
        for i, name in enumerate(component_names):
            out[name] = float(scaled[i].item())
        return out

    def compute_new_weights(
        self,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """
        Paper-faithful ReLoBRaLo update (Eq. (11)).
        Preserves schedule structure from current_weights entries.
        """
        component_names = list(current_weights.keys())
        m = len(component_names)
        if m == 0:
            return {}

        # 1) Build L_i(t) for all components (mean over the provided list for this update call).
        #    If a component has no data this step, we fall back to previous if available, else 1.0.
        L_t: Dict[str, float] = {}
        for name in component_names:
            vals = loss_history.get(name, [])
            mu = self._mean_loss(vals)
            if mu != mu:  # NaN check
                # No/invalid data: fallback
                if name in self._L_prev:
                    mu = self._L_prev[name]
                else:
                    mu = 1.0
            L_t[name] = float(mu)

            # Initialize L0 the first time we see this component
            if name not in self._L0:
                self._L0[name] = float(mu)

        # 2) Ensure we have L_prev and lambda_prev initialized
        for name in component_names:
            if name not in self._L_prev:
                self._L_prev[name] = L_t[name]

        # Initialize λ(t-1) from current_weights if we haven't stored it yet.
        # If base_weight is missing, default to uniform weights summing to m.
        if not self._lambda_prev:
            uniform = float(m) / float(m)  # =1.0 each
            for name in component_names:
                bw = current_weights.get(name, {}).get("base_weight", uniform)
                self._lambda_prev[name] = float(bw)

        # 3) Compute λ_bal(t, t-1) and λ_bal(t, 0)
        L_tminus1 = {name: self._L_prev[name] for name in component_names}
        L_0 = {name: self._L0[name] for name in component_names}

        lambda_bal_recent = self._lambda_bal(component_names, L_t, L_tminus1)  # (t, t-1)
        lambda_bal_start = self._lambda_bal(component_names, L_t, L_0)         # (t, 0)

        # 4) Sample ρ ~ Bernoulli(rho_prob) and compute λ_hist
        rho = 1.0 if random.random() < self.rho_prob else 0.0

        lambda_hist: Dict[str, float] = {}
        for name in component_names:
            lambda_hist[name] = rho * self._lambda_prev[name] + (1.0 - rho) * lambda_bal_start[name]

        # 5) EMA combine into λ(t) per Eq. (11)
        lambda_t: Dict[str, float] = {}
        for name in component_names:
            lambda_t[name] = self.alpha * lambda_hist[name] + (1.0 - self.alpha) * lambda_bal_recent[name]

        # 6) Optional: enforce sum-to-m (should already be close; this keeps it exact if desired)
        if self.normalize_sum_to_m:
            s = sum(lambda_t.values())
            if s > self.eps:
                scale = float(m) / s
                for name in component_names:
                    lambda_t[name] *= scale

        # 7) Optional: clip (NOT in paper; leave None for strict paper behavior)
        if self.min_weight is not None or self.max_weight is not None:
            lo = self.min_weight if self.min_weight is not None else -float("inf")
            hi = self.max_weight if self.max_weight is not None else float("inf")
            for name in component_names:
                lambda_t[name] = float(max(lo, min(hi, lambda_t[name])))

            if self.normalize_sum_to_m:
                s = sum(lambda_t.values())
                if s > self.eps:
                    scale = float(m) / s
                    for name in component_names:
                        lambda_t[name] *= scale

        # 8) Write back in framework format: preserve schedule structure, update 'base_weight'
        new_weights: Dict[str, Dict] = {}
        for name in component_names:
            entry = current_weights[name].copy()
            entry["base_weight"] = float(lambda_t[name])
            new_weights[name] = entry

        # 9) Update internal state for next call
        for name in component_names:
            self._L_prev[name] = L_t[name]
            self._lambda_prev[name] = lambda_t[name]

        return new_weights