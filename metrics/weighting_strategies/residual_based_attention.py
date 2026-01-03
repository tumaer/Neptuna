from typing import Dict, List, Optional
from ..loss_weighting_strategies import LossWeightingStrategyBase
import torch


class ResidualBasedAttention(LossWeightingStrategyBase):
    """
    Residual-Based Attention (RBA) for adaptive loss weighting. 
    
    Based on "Residual-based Attention and Connection to Information 
    Bottleneck Theory in PINNs" 
    https://arxiv.org/abs/2307.00379 
    """

    def __init__(
        self,
        update_frequency: int = 1,
        use_gradients: bool = False,
        gamma: float = 0.9,                # decay factor (paper's gamma)
        eta_star: float = 1.0,             # update step size (paper's eta*)
        residual_mode: str = "mean",       # 'mean', 'max', 'std'
        add_constant: float = 0.0,         # optional +c after update (paper mentions variants)
        eps: float = 1e-8,                 # numerical stability
        min_weight: float = 0.01,
        max_weight: float = 100.0,
        normalize_weights: bool = True,    # normalize sum to num_components (optional, not in paper)
    ):
        super().__init__(update_frequency, use_gradients)

        if not (0.0 <= gamma < 1.0):
            raise ValueError("gamma should be in [0, 1).")

        self.gamma = float(gamma)
        self.eta_star = float(eta_star)
        self.residual_mode = residual_mode
        self.add_constant = float(add_constant)
        self.eps = float(eps)

        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.normalize_weights = bool(normalize_weights)

        self.running_weights: Dict[str, float] = {}

    def compute_statistics(self, losses: List[float]) -> Optional[Dict[str, float]]:
        if not losses:
            return None
        t = torch.tensor(losses, dtype=torch.float32)
        return {
            "mean": float(t.mean()),
            "std": float(t.std(unbiased=False)),
            "min": float(t.min()),
            "max": float(t.max()),
            "count": int(t.numel()),
        }

    def compute_residual_proxy(self, losses: List[float]) -> float:
        """
        Per-component residual proxy R_c derived from loss history.

        This is a DESIGN CHOICE for component-level adaptation. Common choices:
        - mean: average loss over the epoch/steps
        - max: maximum observed loss
        - std: variability proxy (here: mean + std to keep scale > 0)
        """
        if not losses:
            return 1.0

        stats = self.compute_statistics(losses)
        if stats is None:
            return 1.0

        if self.residual_mode == "mean":
            return max(stats["mean"], 0.0)
        elif self.residual_mode == "max":
            return max(stats["max"], 0.0)
        elif self.residual_mode == "std":
            return max(stats["mean"] + stats["std"], 0.0)
        else:
            raise ValueError(f"Unknown residual_mode: {self.residual_mode}")

    def compute_new_weights(
        self,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """
        Compute updated component weights using paper-consistent RBA dynamics:

            w_c <- gamma * w_c + eta_star * (R_c / max_j R_j)

        where R_c is the per-component residual proxy.
        """
        if not current_weights:
            return {}

        component_names = list(current_weights.keys())

        # 1) Compute residual proxies R_c for each component
        residual_proxy: Dict[str, float] = {}
        for name in component_names:
            hist = loss_history.get(name, [])
            residual_proxy[name] = float(self.compute_residual_proxy(hist))

        # 2) Normalize by max over components (paper uses max over points)
        max_r = max(residual_proxy.values()) if residual_proxy else 1.0
        max_r = max(max_r, self.eps)

        # 3) RBA-style update: decay previous running weight + normalized residual
        new_weights: Dict[str, Dict] = {}
        for name in component_names:
            prev_w = self.running_weights.get(
                name,
                float(current_weights[name].get("base_weight", 1.0)),
            )

            normalized_r = residual_proxy[name] / max_r

            updated_w = self.gamma * prev_w + self.eta_star * normalized_r

            if self.add_constant != 0.0:
                updated_w = updated_w + self.add_constant

            updated_w = max(self.min_weight, min(self.max_weight, float(updated_w)))

            self.running_weights[name] = float(updated_w)

            new_weights[name] = current_weights[name].copy()
            new_weights[name]["base_weight"] = float(updated_w)

        # 4) Optional normalization so total weight stays comparable across #components
        if self.normalize_weights and new_weights:
            total = sum(v["base_weight"] for v in new_weights.values())
            n = len(new_weights)
            if total > self.eps:
                factor = n / total
                for name in new_weights:
                    w = new_weights[name]["base_weight"] * factor
                    w = max(self.min_weight, min(self.max_weight, float(w)))
                    new_weights[name]["base_weight"] = w
                    self.running_weights[name] = w

        return new_weights