from typing import Dict, List, Optional
from ..loss_weighting_strategies import LossWeightingStrategyBase
import torch
import math


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
        gamma: float = 0.999,             # paper uses close to 1 in examples
        eta_star: float = 0.01,           # paper example value
        residual_mode: str = "mean",      # 'mean' or 'max' over epoch history of the *loss*
        add_constant: float = 0.0,        # optional +c after update (paper mentions variants)
        eps: float = 1e-12,               # numerical stability for sqrt and divides
        min_weight: float = 0.0,          # allow 0 if you want paper-like init
        max_weight: float = 1e6,          # optional safety cap
        normalize_weights: bool = False,  # not in paper; keep off by default
        clip_weights: bool = False,       # keep off by default (paper relies on recurrence bound)
    ):
        super().__init__(update_frequency, use_gradients)

        if not (0.0 <= gamma < 1.0):
            raise ValueError("gamma should be in [0, 1).")
        if eta_star < 0.0:
            raise ValueError("eta_star should be >= 0.")

        self.gamma = float(gamma)
        self.eta_star = float(eta_star)
        self.residual_mode = residual_mode
        self.add_constant = float(add_constant)
        self.eps = float(eps)

        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.normalize_weights = bool(normalize_weights)
        self.clip_weights = bool(clip_weights)

        self.running_lambda: Dict[str, float] = {}

    @staticmethod
    def _safe_float(x) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    def compute_statistics(self, values: List[float]) -> Optional[Dict[str, float]]:
        if not values:
            return None
        t = torch.tensor(values, dtype=torch.float32)
        return {
            "mean": float(t.mean()),
            "max": float(t.max()),
            "min": float(t.min()),
            "count": int(t.numel()),
        }

    def _loss_stat(self, loss_history: Dict[str, List[float]], key: str) -> float:
        """
        Returns a nonnegative scalar summary of the component loss over the epoch.
        Assumes losses are >= 0 (MSE-style). Clamps negative due to numeric noise.
        """
        hist = loss_history.get(key, [])
        if not hist:
            return 0.0

        stats = self.compute_statistics(hist)
        if stats is None:
            return 0.0

        if self.residual_mode == "mean":
            v = stats["mean"]
        elif self.residual_mode == "max":
            v = stats["max"]
        else:
            raise ValueError(f"Unknown residual_mode: {self.residual_mode}")

        return max(float(v), 0.0)

    def _residual_proxy_from_loss(self, loss_value: float) -> float:
        """
        If component loss L ~= <r^2>, then sqrt(L) ~= RMS(|r|).
        This matches the paper's update signal being proportional to |r|.
        """
        return math.sqrt(max(loss_value, 0.0) + self.eps)

    def _get_prev_lambda(self, key: str, current_weights: Dict[str, Dict]) -> float:
        """
        Try to initialize lambda from existing framework weight w via lambda=sqrt(w),
        otherwise start at 0 (paper-like).
        """
        if key in self.running_lambda:
            return self.running_lambda[key]

        prev_w = self._get_previous_weight(key, current_weights)
        prev_w = max(float(prev_w), 0.0)
        if prev_w > 0.0:
            return math.sqrt(prev_w)
        return 0.0

    def compute_new_weights(
        self,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict],
        grad_norm_history: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Dict]:
        # Flat list of keys that were tracked this epoch
        all_keys = list(loss_history.keys())
        if not all_keys:
            return {k: v.copy() for k, v in current_weights.items()}

        # 1) Compute residual-magnitude proxies R_c (from loss summaries)
        residual_proxy: Dict[str, float] = {}
        for k in all_keys:
            loss_stat = self._loss_stat(loss_history, k)
            residual_proxy[k] = float(self._residual_proxy_from_loss(loss_stat))

        # 2) Normalize by max proxy across components (analogue of max_i |r_i|)
        max_r = max(residual_proxy.values()) if residual_proxy else 0.0
        max_r = max(max_r, self.eps)

        # 3) Update lambda_c using paper-style recurrence, then map to w_c=lambda_c^2
        new_weight_scalars: Dict[str, float] = {}
        for k in all_keys:
            prev_lambda = self._get_prev_lambda(k, current_weights)
            normalized_r = residual_proxy[k] / max_r  # in [0,1] (approximately)

            # paper: lambda <- gamma*lambda + eta_star*normalized_abs_residual
            lam = self.gamma * float(prev_lambda) + self.eta_star * float(normalized_r)

            if self.add_constant != 0.0:
                lam = lam + self.add_constant

            # Ensure nonnegative lambda (paper's lambda_i is effectively nonnegative)
            lam = max(0.0, float(lam))

            # Convert to framework weight
            w = lam * lam

            if self.clip_weights:
                w = max(self.min_weight, min(self.max_weight, float(w)))

            self.running_lambda[k] = float(lam)
            new_weight_scalars[k] = float(w)

        # 4) Optional normalization (not in paper)
        if self.normalize_weights:
            total = sum(new_weight_scalars.values())
            n = len(new_weight_scalars)
            if total > self.eps:
                factor = n / total
                for k in all_keys:
                    w = new_weight_scalars[k] * factor
                    if self.clip_weights:
                        w = max(self.min_weight, min(self.max_weight, float(w)))
                    new_weight_scalars[k] = float(w)
                    # update lambda to remain consistent with w
                    self.running_lambda[k] = math.sqrt(max(float(w), 0.0))

        # 5) Reconstruct hierarchical dict
        return self._reconstruct_weight_dict(new_weight_scalars, current_weights)

    def _get_previous_weight(self, loss_key: str, current_weights: Dict[str, Dict]) -> float:
        base_name, sub_name, channel_idx = self._parse_hierarchical_key(loss_key)

        if base_name not in current_weights:
            return 0.0

        config = current_weights[base_name]

        # Base component weight
        if sub_name is None and channel_idx is None:
            return float(config.get("base_weight", 0.0))

        # Per-channel weight
        if channel_idx is not None:
            if "channel_weights" in config:
                channel_weights = config["channel_weights"]
                if channel_idx < len(channel_weights):
                    return float(channel_weights[channel_idx])
            return 0.0

        # Per-sub-component weight
        if sub_name is not None:
            if "component_weights" in config:
                component_weights = config["component_weights"]
                return float(component_weights.get(sub_name, 0.0))
            return 0.0

        return 0.0

    def _reconstruct_weight_dict(
        self,
        new_weight_scalars: Dict[str, float],
        current_weights: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        new_weights: Dict[str, Dict] = {}

        for base_name, config in current_weights.items():
            new_config = config.copy()

            # Update base weight if present
            if base_name in new_weight_scalars:
                new_config["base_weight"] = float(new_weight_scalars[base_name])

            # Update channel weights if present
            if "channel_weights" in config:
                channel_weights = config["channel_weights"].clone()
                for ch_idx in range(len(channel_weights)):
                    ch_key = f"{base_name}/channel_{ch_idx}"
                    if ch_key in new_weight_scalars:
                        channel_weights[ch_idx] = float(new_weight_scalars[ch_key])
                new_config["channel_weights"] = channel_weights

            # Update component weights if present
            if "component_weights" in config:
                component_weights = config["component_weights"].copy()
                for sub_name in list(component_weights.keys()):
                    comp_key = f"{base_name}/{sub_name}"
                    if comp_key in new_weight_scalars:
                        component_weights[sub_name] = float(new_weight_scalars[comp_key])
                new_config["component_weights"] = component_weights

            new_weights[base_name] = new_config

        return new_weights