from typing import Dict, List, Optional

from ..loss_weighting_strategies import LossWeightingStrategyBase


class LogOnly(LossWeightingStrategyBase):
    """
    Log-only loss weighting strategy.

    This strategy never changes any weights. It exists to keep the adaptive
    weighting pipeline active so loss/grad statistics are collected and logged.
    """

    def __init__(self, update_frequency: int = 1, use_gradients: bool = False):
        super().__init__(update_frequency=update_frequency, use_gradients=use_gradients)

    def compute_new_weights(
        self,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict],
        grad_stats_history: Optional[Dict[str, Dict[str, List[float]]]] = None,
    ) -> Dict[str, Dict]:
        # Return a shallow copy to avoid in-place mutation.
        return {k: v.copy() for k, v in current_weights.items()}
