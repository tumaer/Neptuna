from .l1_loss import L1Loss
from .l2_loss import L2Loss
from .ssim import SSIM
from .ms_ssim import MSSSIM
from .pearson_correlation_loss import PearsonCorrelationLoss
from .sinkhorn_divergence import SinkhornDivergence
from .h1_semi_norm import H1SemiNorm
from .h2_semi_norm import H2SemiNorm

__all__ = [
    'L1Loss',
    'L2Loss',
    'SSIM',
    'MSSSIM',
    'PearsonCorrelationLoss',
    'SinkhornDivergence',
    'H1SemiNorm',
    'H2SemiNorm',
]