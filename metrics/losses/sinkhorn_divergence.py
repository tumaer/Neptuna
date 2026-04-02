import numpy as np
from functools import partial
from abc import ABC, abstractmethod
from typing import Literal, List, Optional, Dict, Union, Tuple
import torch
import torch.nn as nn
from torch.nn.functional import conv1d, avg_pool1d, avg_pool2d, avg_pool3d, interpolate
from ..loss_framework import LossComponent, WeightSchedule, NormalizationHelper

try:  # Import the keops library, www.kernel-operations.io
    # from pykeops.torch import generic_logsumexp, LazyTensor
    # from pykeops.torch.cluster import (
    #     grid_cluster,
    #     cluster_ranges_centroids,
    #     sort_clusters,
    #     from_matrix,
    # )

    keops_available = True
except:
    keops_available = False


# Adapted from geomloss:
# https://github.com/jeanfeydy/geomloss

class SinkhornDivergence(LossComponent):
    """
    Sinkhorn divergence loss component for optimal transport-based comparison.
    """
    def __init__(
        self,
        norm_helper: NormalizationHelper,
        weight: float = 1.0,
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        p: int = 2,
        blur: float = None,
        reach: float = None,
        axes: tuple = None,
        scaling: float = 0.5,
        cost=None,
        debias: bool = True,
        normalization: Literal['none', 'range', 'variance', 'std', 'norm', 'root_norm'] = 'none',
        epsilon: float = 1e-8,

        **kwargs,
    ):

        super().__init__(weight=weight, name=name, data_dim=data_dim, field_names=field_names, norm_helper=norm_helper)
        
        # Sinkhorn-specific parameters
        self.p = p
        self.blur = blur
        self.reach = reach
        self.axes = axes
        self.scaling = scaling
        self.cost = cost
        self.debias = debias
        self.kwargs = kwargs
        self.normalization = normalization
        self.epsilon = epsilon

    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        input_frames: Optional[torch.Tensor],
        return_detailed: bool = False,
        keep_bc_dims: bool = False,
        preserve_component_grads: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute Sinkhorn divergence between predictions and labels.
        
        Args:
            model: The neural network model (unused but kept for interface compatibility)
            predictions: Predicted distribution/measure
            labels: Target distribution/measure
            
        Returns:
            Weighted Sinkhorn divergence loss
        """
        
        original_shape = predictions.shape
        
        # Get weight tensor with proper broadcasting
        weight_tensor = self.weight_schedule.get_loss_weight(original_shape).to(predictions.device)
        
        # Apply weights to inputs (scale by sqrt to preserve Sinkhorn properties)
        weight_sqrt = torch.sqrt(weight_tensor)
        predictions_weighted = predictions * weight_sqrt
        labels_weighted = labels * weight_sqrt
        
        if keep_bc_dims:
            B, T, C = predictions_weighted.shape[:3]
            per_bc = []
            for b in range(B):
                per_c = []
                for c in range(C):
                    pred_bc = predictions_weighted[b:b+1, :, c:c+1, ...]
                    label_bc = labels_weighted[b:b+1, :, c:c+1, ...]

                    pred_bc, label_bc, axes = prepare_for_sinkhorn(pred_bc, label_bc)
                    axes_to_use = self.axes if self.axes is not None else axes

                    divergence = sinkhorn_divergence(
                        a=pred_bc,
                        b=label_bc,
                        p=self.p,
                        blur=self.blur,
                        reach=self.reach,
                        axes=axes_to_use,
                        scaling=self.scaling,
                        cost=self.cost,
                        debias=self.debias,
                        potentials=False,
                        **self.kwargs,
                    )
                    per_c.append(divergence.mean())  # reduce over frames
                per_bc.append(torch.stack(per_c, dim=0))
            loss = torch.stack(per_bc, dim=0)  # (B, C)
        else:
            # Prepare tensors for Sinkhorn (handles reshaping, non-negativity, etc.)
            predictions_weighted, labels_weighted, axes = prepare_for_sinkhorn(
                predictions_weighted, labels_weighted
            )
            
            axes_to_use = self.axes if self.axes is not None else axes

            divergence = sinkhorn_divergence(
                a=predictions_weighted,
                b=labels_weighted,
                p=self.p,
                blur=self.blur,
                reach=self.reach,
                axes=axes_to_use,
                scaling=self.scaling,
                cost=self.cost,
                debias=self.debias,
                potentials=False,
                **self.kwargs,
            )
            
            loss = divergence.mean()
        
        if not return_detailed:
            return loss
        
        # Sinkhorn divergence doesn't support detailed breakdown
        return loss, {}


# Custom method for reshaping tensors for Sinkhorn divergence
# Needs to be equilateral (square/cubic), size power of 2, non-negative values
def prepare_for_sinkhorn(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    target_size: Optional[int] = 256,
    preserve_aspect_ratio: bool = True,
    normalization: str = "minmax",  # "minmax", "softplus", or "separate_channels"
) -> tuple[torch.Tensor, torch.Tensor, tuple]:
    """
    Prepare arbitrary-shaped tensors for Sinkhorn divergence computation.
    
    Args:
        normalization: How to handle negative values:
            - "minmax": Shift to [0, inf) by subtracting global min
            - "softplus": Apply softplus transformation
            - "separate_channels": Treat positive/negative parts as separate channels
    
    Returns:
        predictions, labels, axes: Resampled tensors and appropriate axes specification
    """
    original_shape = predictions.shape
    B, F, C = original_shape[:3]
    spatial_dims = original_shape[3:]
    D = len(spatial_dims)
    
    # Determine target size (power of 2)
    if target_size is None:
        max_dim = max(spatial_dims)
        target_size = 2 ** int(np.ceil(np.log2(max_dim)))
        target_size = max(32, min(256, target_size))
    
    # Compute aspect ratios to preserve physical distances
    if preserve_aspect_ratio:
        max_spatial = max(spatial_dims)
        axes = tuple(
            [0.0, spatial_dims[d] / max_spatial] 
            for d in range(D)
        )
    else:
        axes = tuple([0.0, 1.0] for _ in range(D))
    
    # Reshape for processing
    predictions = predictions.reshape(B * F, C, *spatial_dims)
    labels = labels.reshape(B * F, C, *spatial_dims)
    
    # Interpolate to square/cubic grid
    if D == 1:
        target_shape = (target_size,)
        mode = 'linear'
    elif D == 2:
        target_shape = (target_size, target_size)
        mode = 'bilinear'
    elif D == 3:
        target_shape = (target_size, target_size, target_size)
        mode = 'trilinear'
    else:
        raise ValueError(f"Sinkhorn loss only supports 1D/2D/3D data, got {D}D")
    
    if spatial_dims != target_shape:
        predictions = interpolate(
            predictions,
            size=target_shape,
            mode=mode,
            align_corners=False
        )
        labels = interpolate(
            labels,
            size=target_shape,
            mode=mode,
            align_corners=False
        )
    
    # Handle negative values based on normalization strategy
    eps = 1e-8
    
    if normalization == "minmax":
        # Shift to non-negative by subtracting global minimum
        # This preserves relative differences
        global_min = min(predictions.min().item(), labels.min().item())
        if global_min < 0:
            predictions = predictions - global_min
            labels = labels - global_min
        predictions = predictions + eps
        labels = labels + eps
        
    elif normalization == "softplus":
        # Smooth approximation of ReLU: softplus(x) ≈ max(0, x) for |x| >> 1
        # softplus(x) = log(1 + exp(x))
        predictions = torch.nn.functional.softplus(predictions) + eps
        labels = torch.nn.functional.softplus(labels) + eps
        
    elif normalization == "separate_channels":
        # Split into positive and negative parts as separate channels
        # This doubles the channel dimension but preserves sign information
        pred_pos = torch.clamp(predictions, min=0.0) + eps
        pred_neg = torch.clamp(-predictions, min=0.0) + eps
        label_pos = torch.clamp(labels, min=0.0) + eps
        label_neg = torch.clamp(-labels, min=0.0) + eps
        
        predictions = torch.cat([pred_pos, pred_neg], dim=1)  # (B*F, 2*C, ...)
        labels = torch.cat([label_pos, label_neg], dim=1)
        C = 2 * C  # Update channel count
        
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    
    # Normalize to probability distributions
    # Each sample (B*F, C, spatial) should sum to 1
    predictions = predictions / predictions.reshape(B * F, C, -1).sum(dim=-1, keepdim=True).reshape(B * F, C, *([1] * D))
    labels = labels / labels.reshape(B * F, C, -1).sum(dim=-1, keepdim=True).reshape(B * F, C, *([1] * D))
    
    return predictions, labels, axes



def extrapolate(f_ba, g_ab, eps, damping, C_xy, b_log, C_xy_fine):
    return upsample(f_ba)

upsample_mode = {
    1: "linear",
    2: "bilinear",
    3: "trilinear",
}

def kernel_truncation(
    C_xy,
    C_yx,
    C_xy_fine,
    C_yx_fine,
    f_ba,
    g_ab,
    eps,
    truncate=None,
    cost=None,
    verbose=False,
):
    return C_xy_fine, C_yx_fine


def sinkhorn_divergence(
    a,
    b,
    p=2,
    blur=None,
    reach=None,
    axes=None,
    scaling=0.5,
    cost=None,
    debias=True,
    potentials=False,
    verbose=False,
    **kwargs,
):
    r"""Sinkhorn divergence between measures supported on 1D/2D/3D grids.

    Args:
        a ((B, Nx), (B, Nx, Ny) or (B, Nx, Ny, Nz) Tensor): Weights :math:`\alpha_i`
            for the first measure, with a batch dimension.

        b ((B, Nx), (B, Nx, Ny) or (B, Nx, Ny, Nz) Tensor): Weights :math:`\beta_j`
            for the second measure, with a batch dimension.

        p (int, optional): Exponent of the ground cost function
            :math:`C(x_i,y_j)`, which is equal to
            :math:`\tfrac{1}{p}\|x_i-y_j\|^p` if it is not provided
            explicitly through the `cost` optional argument.
            Defaults to 2.

        blur (float or None, optional): Target value for the blurring scale
            of the "point spread function" or Gibbs kernel
            :math:`K_{i,j} = \exp(-C(x_i,y_j)/\varepsilon) = \exp(-\|x_i-y_j\|^p / p \text{blur}^p).
            In the Sinkhorn algorithm, the temperature :math:`\varepsilon`
            is computed as :math:`\text{blur}^p`.
            Defaults to None: we pick the smallest pixel size across
            the Nx, Ny and Nz dimensions (if applicable).

        axes (tuple of pairs of floats or None (= [0, 1)^(1/2/3)), optional):
            Dimensions of the image domain, specified through a 1/2/3-uple
            of [vmin, vmax] bounds.
            For instance, if the batched 2D images correspond to sampled
            measures on [-10, 10) x [-3, 5), you may use "axes = ([-10, 10], [-3, 5])".
            The (implicit) pixel coordinates are computed using a "torch.linspace(...)"
            across each dimension: along any given axis, the spacing between two pixels
            is equal to "(vmax - vmin) / npixels".

            Defaults to None: we assume that the signal / image / volume
            is sampled on the unit interval [0, 1) / square [0, 1)^2 / cube [0, 1)^3.

        scaling (float in (0, 1), optional): Ratio between two successive
            values of the blur radius in the epsilon-scaling annealing descent.
            Defaults to 0.5.

        cost (function or None, optional): ...
            Defaults to None: we use a Euclidean cost
            :math:`C(x_i,y_j) = \tfrac{1}{p}\|x_i-y_j\|^p`.

        debias (bool, optional): Should we used the "de-biased" Sinkhorn divergence
            :math:`\text{S}_{\varepsilon, \rho}(\al,\be)` instead
            of the "raw" entropic OT cost
            :math:`\text{OT}_{\varepsilon, \rho}(\al,\be)`?
            This slows down the OT solver but guarantees that our approximation
            of the Wasserstein distance will be positive and definite
            - up to convergence of the Sinkhorn loop.
            For a detailed discussion of the influence of this parameter,
            see e.g. Fig. 3.21 in Jean Feydy's PhD thesis.
            Defaults to True.

        potentials (bool, optional): Should we return the optimal dual potentials
            instead of the cost value?
            Defaults to False.

    Returns:
        (B,) Tensor or pair of (B, Nx, ...), (B, Nx, ...) Tensors: If `potentials` is True,
            we return a pair of (B, Nx, ...), (B, Nx, ...) Tensors that encode the optimal
            dual vectors, respectively supported by :math:`x_i` and :math:`y_j`.
            Otherwise, we return a (B,) Tensor of values for the Sinkhorn divergence.
    """

    if blur is None:
        blur = 1 / a.shape[-1]

    # Pre-compute a multiscale decomposition (=Binary/Quad/OcTree)
    # of the input measures, stored as logarithms
    a_s, b_s = pyramid(a)[1:], pyramid(b)[1:]
    a_logs = list(map(log_dens, a_s))
    b_logs = list(map(log_dens, b_s))

    # By default, our cost function :math:`C(x_i,y_j)` is a halved,
    # squared Euclidean distance (p=2) or a simple Euclidean distance (p=1):
    depth = len(a_logs)
    if cost is None:
        C_s = [p] * depth  # Dummy "cost matrices"
    else:
        raise NotImplementedError()

    # Diameter of the configuration:
    diameter = 1
    # Target temperature epsilon:
    eps = blur**p
    # Strength of the marginal constraints:
    rho = None if reach is None else reach**p

    # Schedule for the multiscale descent, with ε-scaling:
    """
    sigma = diameter
    for n in range(depth):
        for _ in range(scaling_N):  # Number of steps per scale
            eps_list.append(sigma ** p)

            # Decrease the kernel radius, making sure that
            # the radius sigma is divided by two at every scale until we reach
            # the target value, "blur":
            scale = max(sigma * (2 ** (-1 / scaling_N)), blur)

    jumps = [scaling_N * (i + 1) - 1 for i in range(depth - 1)]
    """
    if scaling < 0.5:
        raise ValueError(
            f"Scaling value of {scaling} is too small: please use a number in [0.5, 1)."
        )

    diameter, eps, eps_list, rho = scaling_parameters(
        None, None, p, blur, reach, diameter, scaling
    )

    # List of pixel widths:
    pyramid_scales = [diameter / a.shape[-1] for a in a_s]
    if verbose:
        print("Pyramid scales:", pyramid_scales)

    current_scale = pyramid_scales.pop(0)
    jumps = []
    for i, eps in enumerate(eps_list[1:]):
        if current_scale**p > eps:
            jumps.append(i + 1)
            current_scale = pyramid_scales.pop(0)

    if verbose:
        print("Temperatures: ", eps_list)
        print("Jumps: ", jumps)

    assert (
        len(jumps) == len(a_s) - 1
    ), "There's a bug in the multicale pre-processing..."

    # Use an optimal transport solver to retrieve the dual potentials:
    f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(
        softmin_grid,
        a_logs,
        b_logs,
        C_s,
        C_s,
        C_s,
        C_s,
        eps_list,
        rho,
        jumps=jumps,
        kernel_truncation=kernel_truncation,
        extrapolate=extrapolate,
        debias=debias,
    )

    # Optimal transport cost:
    return sinkhorn_cost(
        eps,
        rho,
        a,
        b,
        f_aa,
        g_bb,
        g_ab,
        f_ba,
        batch=True,
        debias=debias,
        potentials=potentials,
    )



BATCH, CHANNEL, HEIGHT, WIDTH, DEPTH = 0, 1, 2, 3, 4

subsample = {
    1: (lambda x: 2 * avg_pool1d(x, 2)),
    2: (lambda x: 4 * avg_pool2d(x, 2)),
    3: (lambda x: 8 * avg_pool3d(x, 2)),
}

def log_dens(α):
    α_log = α.log()
    α_log[α <= 0] = -10000.0
    return α_log

def dimension(I):
    """Returns 2 if we are working with 2D images and 3 for volumes."""
    return I.dim() - 2

def pyramid(I):
    D = dimension(I)
    I_s = [I]

    for i in range(int(np.log2(I.shape[HEIGHT]))):
        I = subsample[D](I)
        I_s.append(I)

    I_s.reverse()
    return I_s

def upsample(I):
    D = dimension(I)
    return interpolate(I, scale_factor=2, mode=upsample_mode[D], align_corners=False)

def softmin_grid(eps, C_xy, h_y):
    r"""Soft-C-transform, implemented using seperable KeOps operations.

    This routine implements the (soft-)C-transform
    between dual vectors, which is the core computation for
    Auction- and Sinkhorn-like optimal transport solvers.

    If `eps` is a float number, `C_xy` is a tuple of axes dimensions
    and `h_y` encodes a dual potential :math:`h_j` that is supported by the 1D/2D/3D grid
    points :math:`y_j`'s, then `softmin_tensorized(eps, C_xy, h_y)` returns a dual potential
    `f` for ":math:`f_i`", supported by the :math:`x_i`'s, that is equal to:

    .. math::
        f_i \gets - \varepsilon \log \sum_{j=1}^{\text{M}} \exp
        \big[ h_j - C(x_i, y_j) / \varepsilon \big]~.

    For more detail, see e.g. Section 3.3 and Eq. (3.186) in Jean Feydy's PhD thesis.

    Args:
        eps (float, positive): Temperature :math:`\varepsilon` for the Gibbs kernel
            :math:`K_{i,j} = \exp(-C(x_i, y_j) / \varepsilon)`.

        C_xy (): Encodes the implicit cost matrix :math:`C(x_i,y_j)`.

        h_y ((B, Nx), (B, Nx, Ny) or (B, Nx, Ny, Nz) Tensor):
            Grid of logarithmic "dual" values, with a batch dimension.
            Most often, this image will be computed as `h_y = b_log + g_j / eps`,
            where `b_log` is an array of log-weights :math:`\log(\beta_j)`
            for the :math:`y_j`'s and :math:`g_j` is a dual variable
            in the Sinkhorn algorithm, so that:

            .. math::
                f_i \gets - \varepsilon \log \sum_{j=1}^{\text{M}} \beta_j
                \exp \tfrac{1}{\varepsilon} \big[ g_j - C(x_i, y_j) \big]~.

    Returns:
        (B, Nx), (B, Nx, Ny) or (B, Nx, Ny, Nz) Tensor: Dual potential `f` of values
            :math:`f_i`, supported by the points :math:`x_i`.
    """
    D = dimension(h_y)
    if D == 1:
        B, K, N = h_y.shape[BATCH], h_y.shape[CHANNEL], h_y.shape[HEIGHT]
    else:
        B, K, N = h_y.shape[BATCH], h_y.shape[CHANNEL], h_y.shape[WIDTH]

    if not keops_available:
        raise ImportError("This routine depends on the pykeops library.")

    x = torch.arange(N).type_as(h_y) / N
    p = C_xy
    if p == 1:
        x = x / eps
    elif p == 2:
        x = x / np.sqrt(2 * eps)
    else:
        raise NotImplementedError()

    def softmin(a_log):
        a_log = a_log.contiguous()
        # print(a_log.shape)
        a_log_j = LazyTensor(a_log.view(-1, 1, N, 1))
        x_i = LazyTensor(x.view(1, N, 1, 1))
        x_j = LazyTensor(x.view(1, 1, N, 1))

        if p == 1:
            kA_log_ij = a_log_j - (x_i - x_j).abs()  # (B * N, N, N, 1)
        elif p == 2:
            kA_log_ij = a_log_j - (x_i - x_j) ** 2  # (B * N, N, N, 1)

            # kA_log_ij =  (x_i - x_j)**2 - g_j

        # print(kA_log_ij)
        kA_log = kA_log_ij.logsumexp(dim=2)  # (B * N, N, 1)

        if D == 2:
            return kA_log.view(B, K, N, N)
        elif D == 3:
            return kA_log.view(B, K, N, N, N)

    if D == 2:
        h_y = softmin(h_y)  # Act on lines
        h_y = softmin(h_y.permute([0, 1, 3, 2])).permute([0, 1, 3, 2])  # Act on columns

    elif D == 3:
        h_y = softmin(h_y)  # Act on dim 4
        h_y = softmin(h_y.permute([0, 1, 2, 4, 3])).permute(
            [0, 1, 2, 4, 3]
        )  # Act on dim 3
        h_y = softmin(h_y.permute([0, 1, 4, 3, 2])).permute(
            [0, 1, 4, 3, 2]
        )  # Act on dim 2

    return -eps * h_y

def scal(a, f, batch=False):
    if batch:
        B = a.shape[0]
        return (a.reshape(B, -1) * f.reshape(B, -1)).sum(1)
    else:
        return torch.dot(a.reshape(-1), f.reshape(-1))
# ==============================================================================
#                             Utility functions
# ==============================================================================


def dampening(eps, rho):
    """Dampening factor for entropy+unbalanced OT with KL penalization of the marginals."""
    return 1 if rho is None else 1 / (1 + eps / rho)


def log_weights(a):
    """Returns the log of the input, with values clamped to -100k to avoid numerical bugs."""
    a_log = a.log()
    a_log[a <= 0] = -100000
    return a_log


class UnbalancedWeight(torch.nn.Module):
    """Applies the correct scaling to the dual variables in the Sinkhorn divergence formula.

    Remarkably, the exponentiated potentials should be scaled
    by "rho + eps/2" in the forward pass and "rho + eps" in the backward.
    For an explanation of this surprising "inconsistency"
    between the forward and backward formulas,
    please refer to Proposition 12 (Dual formulas for the Sinkhorn costs)
    in "Sinkhorn divergences for unbalanced optimal transport",
    Sejourne et al., https://arxiv.org/abs/1910.12958.
    """

    def __init__(self, eps, rho):
        super(UnbalancedWeight, self).__init__()
        self.eps, self.rho = eps, rho

    def forward(self, x):
        return (self.rho + self.eps / 2) * x

    def backward(self, g):
        return (self.rho + self.eps) * g


# ==============================================================================
#                            eps-scaling heuristic
# ==============================================================================


def max_diameter(x, y):
    """Returns a rough estimation of the diameter of a pair of point clouds.

    This quantity is used as a maximum "starting scale" in the epsilon-scaling
    annealing heuristic.

    Args:
        x ((N, D) Tensor): First point cloud.
        y ((M, D) Tensor): Second point cloud.

    Returns:
        float: Upper bound on the largest distance between points `x[i]` and `y[j]`.
    """
    mins = torch.stack((x.min(dim=0)[0], y.min(dim=0)[0])).min(dim=0)[0]
    maxs = torch.stack((x.max(dim=0)[0], y.max(dim=0)[0])).max(dim=0)[0]
    diameter = (maxs - mins).norm().item()
    return diameter


def epsilon_schedule(p, diameter, blur, scaling):
    r"""Creates a list of values for the temperature "epsilon" across Sinkhorn iterations.

    We use an aggressive strategy with an exponential cooling
    schedule: starting from a value of :math:`\text{diameter}^p`,
    the temperature epsilon is divided
    by :math:`\text{scaling}^p` at every iteration until reaching
    a minimum value of :math:`\text{blur}^p`.

    Args:
        p (integer or float): The exponent of the Euclidean distance
            :math:`\|x_i-y_j\|` that defines the cost function
            :math:`\text{C}(x_i,y_j) =\tfrac{1}{p} \|x_i-y_j\|^p`.

        diameter (float, positive): Upper bound on the largest distance between
            points :math:`x_i` and :math:`y_j`.

        blur (float, positive): Target value for the entropic regularization
            (":math:`\varepsilon = \text{blur}^p`").

        scaling (float, in (0,1)): Ratio between two successive
            values of the blur scale.

    Returns:
        list of float: list of values for the temperature epsilon.
    """
    eps_list = (
        [diameter**p]
        + [
            np.exp(e)
            for e in np.arange(
                p * np.log(diameter), p * np.log(blur), p * np.log(scaling)
            )
        ]
        + [blur**p]
    )
    return eps_list


def scaling_parameters(x, y, p, blur, reach, diameter, scaling):
    r"""Turns high-level arguments into numerical values for the Sinkhorn loop."""
    if diameter is None:
        D = x.shape[-1]
        diameter = max_diameter(x.view(-1, D), y.view(-1, D))

    eps = blur**p
    rho = None if reach is None else reach**p
    eps_list = epsilon_schedule(p, diameter, blur, scaling)
    return diameter, eps, eps_list, rho


# ==============================================================================
#                              Sinkhorn divergence
# ==============================================================================


def sinkhorn_cost(
    eps, rho, a, b, f_aa, g_bb, g_ab, f_ba, batch=False, debias=True, potentials=False
):
    r"""Returns the required information (cost, etc.) from a set of dual potentials.

    Args:
        eps (float): Target (i.e. final) temperature.
        rho (float or None (:math:`+\infty`)): Strength of the marginal constraints.

        a ((..., N) Tensor, nonnegative): Weights for the "source" measure on the points :math:`x_i`.
        b ((..., M) Tensor, nonnegative): Weights for the "target" measure on the points :math:`y_j`.
        f_aa ((..., N) Tensor)): Dual potential for the "a <-> a" problem.
        g_bb ((..., M) Tensor)): Dual potential for the "b <-> b" problem.
        g_ab ((..., M) Tensor)): Dual potential supported by :math:`y_j` for the "a <-> b" problem.
        f_ba ((..., N) Tensor)): Dual potential supported by :math:`x_i`  for the "a <-> a" problem.
        batch (bool, optional): Are we working in batch mode? Defaults to False.
        debias (bool, optional): Are we working with the "debiased" or the "raw" Sinkhorn divergence?
            Defaults to True.
        potentials (bool, optional): Shall we return the dual vectors instead of the cost value?
            Defaults to False.

    Returns:
        Tensor or pair of Tensors: if `potentials` is True, we return a pair
            of (..., N), (..., M) Tensors that encode the optimal dual vectors,
            respectively supported by :math:`x_i` and :math:`y_j`.
            Otherwise, we return a (,) or (B,) Tensor of values for the Sinkhorn divergence.
    """

    if potentials:  # Just return the dual potentials
        if debias:  # See Eq. (3.209) in Jean Feydy's PhD thesis.
            # N.B.: This formula does not make much sense in the unbalanced mode
            #       (i.e. if reach is not None).
            return f_ba - f_aa, g_ab - g_bb
        else:  # See Eq. (3.207) in Jean Feydy's PhD thesis.
            return f_ba, g_ab

    else:  # Actually compute the Sinkhorn divergence
        if (
            debias
        ):  # UNBIASED Sinkhorn divergence, S_eps(a,b) = OT_eps(a,b) - .5*OT_eps(a,a) - .5*OT_eps(b,b)
            if rho is None:  # Balanced case:
                # See Eq. (3.209) in Jean Feydy's PhD thesis.
                return scal(a, f_ba - f_aa, batch=batch) + scal(
                    b, g_ab - g_bb, batch=batch
                )
            else:
                # Unbalanced case:
                # See Proposition 12 (Dual formulas for the Sinkhorn costs)
                # in "Sinkhorn divergences for unbalanced optimal transport",
                # Sejourne et al., https://arxiv.org/abs/1910.12958.
                return scal(
                    a,
                    UnbalancedWeight(eps, rho)(
                        (-f_aa / rho).exp() - (-f_ba / rho).exp()
                    ),
                    batch=batch,
                ) + scal(
                    b,
                    UnbalancedWeight(eps, rho)(
                        (-g_bb / rho).exp() - (-g_ab / rho).exp()
                    ),
                    batch=batch,
                )

        else:  # Classic, BIASED entropized Optimal Transport OT_eps(a,b)
            if rho is None:  # Balanced case:
                # See Eq. (3.207) in Jean Feydy's PhD thesis.
                return scal(a, f_ba, batch=batch) + scal(b, g_ab, batch=batch)
            else:
                # Unbalanced case:
                # See Proposition 12 (Dual formulas for the Sinkhorn costs)
                # in "Sinkhorn divergences for unbalanced optimal transport",
                # Sejourne et al., https://arxiv.org/abs/1910.12958.
                # N.B.: Even if this quantity is never used in practice,
                #       we may want to re-check this computation...
                return scal(
                    a, UnbalancedWeight(eps, rho)(1 - (-f_ba / rho).exp()), batch=batch
                ) + scal(
                    b, UnbalancedWeight(eps, rho)(1 - (-g_ab / rho).exp()), batch=batch
                )


# ==============================================================================
#                              Sinkhorn loop
# ==============================================================================


def sinkhorn_loop(
    softmin,
    a_logs,
    b_logs,
    C_xxs,
    C_yys,
    C_xys,
    C_yxs,
    eps_list,
    rho,
    jumps=[],
    kernel_truncation=None,
    truncate=5,
    cost=None,
    extrapolate=None,
    debias=True,
    last_extrapolation=True,
):
    r"""Implements the (possibly multiscale) symmetric Sinkhorn loop,
    with the epsilon-scaling (annealing) heuristic.

    This is the main "core" routine of GeomLoss. It is written to
    solve optimal transport problems efficiently in all the settings
    that are supported by the library: (generalized) point clouds,
    images and volumes.

    This algorithm is described in Section 3.3.3 of Jean Feydy's PhD thesis,
    "Geometric data analysis, beyond convolutions" (Universite Paris-Saclay, 2020)
    (https://www.jeanfeydy.com/geometric_data_analysis.pdf).
    Algorithm 3.5 corresponds to the case where `kernel_truncation` is None,
    while Algorithm 3.6 describes the full multiscale algorithm.

    Args:
        softmin (function): This routine must implement the (soft-)C-transform
            between dual vectors, which is the core computation for
            Auction- and Sinkhorn-like optimal transport solvers.
            If `eps` is a float number, `C_xy` encodes a cost matrix :math:`C(x_i,y_j)`
            and `g` encodes a dual potential :math:`g_j` that is supported by the points
            :math:`y_j`'s, then `softmin(eps, C_xy, g)` must return a dual potential
            `f` for ":math:`f_i`", supported by the :math:`x_i`'s, that is equal to:

            .. math::
                f_i \gets - \varepsilon \log \sum_{j=1}^{\text{M}} \exp
                \big[ g_j - C(x_i, y_j) / \varepsilon \big]~.

            For more detail, see e.g. Section 3.3 and Eq. (3.186) in Jean Feydy's PhD thesis.

        a_logs (list of Tensors): List of log-weights :math:`\log(\alpha_i)`
            for the first input measure at different resolutions.

        b_logs (list of Tensors): List of log-weights :math:`\log(\beta_i)`
            for the second input measure at different resolutions.

        C_xxs (list): List of objects that encode the cost matrices
            :math:`C(x_i, x_j)` between the samples of the first input
            measure at different scales.
            These will be passed to the `softmin` function as second arguments.

        C_yys (list): List of objects that encode the cost matrices
            :math:`C(y_i, y_j)` between the samples of the second input
            measure at different scales.
            These will be passed to the `softmin` function as second arguments.

        C_xys (list): List of objects that encode the cost matrices
            :math:`C(x_i, y_j)` between the samples of the first and second input
            measures at different scales.
            These will be passed to the `softmin` function as second arguments.

        C_yxs (list): List of objects that encode the cost matrices
            :math:`C(y_i, x_j)` between the samples of the second and first input
            measures at different scales.
            These will be passed to the `softmin` function as second arguments.

        eps_list (list of float): List of successive values for the temperature
            :math:`\varepsilon`. The number of iterations in the loop
            is equal to the length of this list.

        rho (float or None): Strength of the marginal constraints for unbalanced OT.
            None stands for :math:`\rho = +\infty`, i.e. balanced OT.

        jumps (list, optional): List of iteration numbers where we "jump"
            from a coarse resolution to a finer one by looking
            one step further in the lists `a_logs`, `b_logs`, `C_xxs`, etc.
            Count starts at iteration 0.
            Defaults to [] - single-scale mode without jumps.

        kernel_truncation (function, optional): Implements the kernel truncation trick.
            Defaults to None.

        truncate (int, optional): Optional argument for `kernel_truncation`.
            Defaults to 5.

        cost (string or function, optional): Optional argument for `kernel_truncation`.
            Defaults to None.

        extrapolate (function, optional): Function.
            If
            `f_ba` is a dual potential that is supported by the :math:`x_i`'s,
            `g_ab` is a dual potential that is supported by the :math:`y_j`'s,
            `eps` is the current value of the temperature :math:`\varepsilon`,
            `damping` is the current value of the damping coefficient for unbalanced OT,
            `C_xy` encodes the cost matrix :math:`C(x_i, y_j)` at the current
            ("coarse") resolution,
            `b_log` denotes the log-weights :math:`\log(\beta_j)`
            that are supported by the :math:`y_j`'s at the coarse resolution,
            and
            `C_xy_fine` encodes the cost matrix :math:`C(x_i, y_j)` at the next
            ("fine") resolution,
            then
            `extrapolate(f_ba, g_ab, eps, damping, C_xy, b_log, C_xy_fine)`
            will be used to compute the new values of the dual potential
            `f_ba` on the point cloud :math:`x_i` at a finer resolution.
            Defaults to None - it is not needed in single-scale mode.

        debias (bool, optional): Should we used the "de-biased" Sinkhorn divergence
            :math:`\text{S}_{\varepsilon, \rho}(\al,\be)` instead
            of the "raw" entropic OT cost
            :math:`\text{OT}_{\varepsilon, \rho}(\al,\be)`?
            This slows down the OT solver but guarantees that our approximation
            of the Wasserstein distance will be positive and definite
            - up to convergence of the Sinkhorn loop.
            For a detailed discussion of the influence of this parameter,
            see e.g. Fig. 3.21 in Jean Feydy's PhD thesis.
            Defaults to True.

        last_extrapolation (bool, optional): Should we perform a last, "full"
            Sinkhorn iteration before returning the dual potentials?
            This allows us to retrieve correct gradients without having
            to backpropagate trough the full Sinkhorn loop.
            Defaults to True.

    Returns:
        4-uple of Tensors: The four optimal dual potentials
            `(f_aa, g_bb, g_ab, f_ba)` that are respectively
            supported by the first, second, second and first input measures
            and associated to the "a <-> a", "b <-> b",
            "a <-> b" and "a <-> b" optimal transport problems.
    """

    # Number of iterations, specified by our epsilon-schedule
    Nits = len(eps_list)

    # The multiscale algorithm may loop over several representations
    # of the input measures.
    # In this routine, the convention is that "myvars" denotes
    # the list of "myvar" across different scales.
    if type(a_logs) is not list:
        # The "single-scale" use case is simply encoded
        # using lists of length 1.

        # Logarithms of the weights:
        a_logs, b_logs = [a_logs], [b_logs]

        # Cost "matrices" C(x_i, y_j) and C(y_i, x_j):
        C_xys, C_yxs = [C_xys], [C_yxs]  # Used for the "a <-> b" problem.

        # Cost "matrices" C(x_i, x_j) and C(y_i, y_j):
        if debias:  # Only used for the "a <-> a" and "b <-> b" problems.
            C_xxs, C_yys = [C_xxs], [C_yys]

    # N.B.: We don't let users backprop through the Sinkhorn iterations
    #       and branch instead on an explicit formula "at convergence"
    #       using some "advanced" PyTorch syntax at the end of the loop.
    #       This acceleration "trick" relies on the "envelope theorem":
    #       it works very well if users are only interested in the gradient
    #       of the Sinkhorn loss, but may not produce correct results
    #       if one attempts to compute order-2 derivatives,
    #       or differentiate "non-standard" quantities that
    #       are defined using the optimal dual potentials.
    #
    #       We may wish to alter this behaviour in the future.
    #       For reference on the question, see Eq. (3.226-227) in
    #       Jean Feydy's PhD thesis and e.g.
    #       "Super-efficiency of automatic differentiation for
    #       functions defined as a minimum", Ablin, Peyré, Moreau (2020)
    #       https://arxiv.org/pdf/2002.03722.pdf.
    torch.autograd.set_grad_enabled(False)

    # Line 1 (in Algorithm 3.6 from Jean Feydy's PhD thesis) ---------------------------

    # We start at the coarsest resolution available:
    k = 0  # Scale index
    eps = eps_list[k]  # First value of the temperature (typically, = diameter**p)

    # Damping factor: equal to 1 for balanced OT,
    # < 1 for unbalanced OT with KL penalty on the marginal constraints.
    # For reference, see Table 1 in "Sinkhorn divergences for unbalanced
    # optimal transport", Sejourne et al., https://arxiv.org/abs/1910.12958.
    damping = dampening(eps, rho)

    # Load the measures and cost matrices at the current scale:
    a_log, b_log = a_logs[k], b_logs[k]
    C_xy, C_yx = C_xys[k], C_yxs[k]  # C(x_i, y_j), C(y_i, x_j)
    if debias:  # Info for the "a <-> a" and "b <-> b" problems
        C_xx, C_yy = C_xxs[k], C_yys[k]  # C(x_i, x_j), C(y_j, y_j)

    # Line 2 ---------------------------------------------------------------------------
    # Start with a decent initialization for the dual vectors:
    # N.B.: eps is really large here, so the log-sum-exp behaves as a sum
    #       and the softmin is basically
    #       a convolution with the cost function (i.e. the limit for eps=+infty).
    #       The algorithm was originally written with this convolution
    #       - but in this implementation, we use "softmin" for the sake of simplicity.
    g_ab = damping * softmin(eps, C_yx, a_log)  # a -> b
    f_ba = damping * softmin(eps, C_xy, b_log)  # b -> a
    if debias:
        f_aa = damping * softmin(eps, C_xx, a_log)  # a -> a
        g_bb = damping * softmin(eps, C_yy, b_log)  # a -> a

    # Lines 4-5: eps-scaling descent ---------------------------------------------------
    for i, eps in enumerate(eps_list):  # See Fig. 3.25-26 in Jean Feydy's PhD thesis.
        # Line 6: update the damping coefficient ---------------------------------------
        damping = dampening(eps, rho)  # eps and damping change across iterations
        
        # Line 7: "coordinate ascent" on the dual problems -----------------------------
        # N.B.: As discussed in Section 3.3.3 of Jean Feydy's PhD thesis,
        #       we perform "symmetric" instead of "alternate" updates
        #       of the dual potentials "f" and "g".
        #       To this end, we first create buffers "ft", "gt"
        #       (for "f-tilde", "g-tilde") using the standard
        #       Sinkhorn formulas, and update both dual vectors
        #       simultaneously.
        ft_ba = damping * softmin(eps, C_xy, b_log + g_ab / eps)  # b -> a
        gt_ab = damping * softmin(eps, C_yx, a_log + f_ba / eps)  # a -> b

        # See Fig. 3.21 in Jean Feydy's PhD thesis to see the importance
        # of debiasing when the target "blur" or "eps**(1/p)" value is larger
        # than the average distance between samples x_i, y_j and their neighbours.
        if debias:
            ft_aa = damping * softmin(eps, C_xx, a_log + f_aa / eps)  # a -> a
            gt_bb = damping * softmin(eps, C_yy, b_log + g_bb / eps)  # b -> b

        # Symmetrized updates - see Fig. 3.24.b in Jean Feydy's PhD thesis:
        f_ba, g_ab = 0.5 * (f_ba + ft_ba), 0.5 * (g_ab + gt_ab)  # OT(a,b) wrt. a, b
        if debias:
            f_aa, g_bb = 0.5 * (f_aa + ft_aa), 0.5 * (g_bb + gt_bb)  # OT(a,a), OT(b,b)

        # Line 8: jump from a coarse to a finer scale ----------------------------------
        # In multi-scale mode, we work we increasingly detailed representations
        # of the input measures: this type of strategy is known as "multi-scale"
        # in computer graphics, "multi-grid" in numerical analysis,
        # "coarse-to-fine" in signal processing or "divide and conquer"
        # in standard complexity theory (e.g. for the quick-sort algorithm).
        #
        # In the Sinkhorn loop with epsilon-scaling annealing, our
        # representations of the input measures are fine enough to ensure
        # that the typical distance between any two samples x_i, y_j is always smaller
        # than the current value of "blur = eps**(1/p)".
        # As illustrated in Fig. 3.26 of Jean Feydy's PhD thesis, this allows us
        # to reach a satisfying level of precision while speeding up the computation
        # of the Sinkhorn iterations in the first few steps.
        #
        # In practice, different multi-scale representations of the input measures
        # are generated by the "parent" code of this solver and stored in the
        # lists a_logs, b_logs, C_xxs, etc.
        #
        # The switch between different scales is specified by the list of "jump" indices,
        # that is generated in conjunction with the list of temperatures "eps_list".
        #
        # N.B.: In single-scale mode, jumps = []: the code below is never executed
        #       and we retrieve "Algorithm 3.5" from Jean Feydy's PhD thesis.
        if i in jumps:
            if i == len(eps_list) - 1:  # Last iteration: just extrapolate!
                C_xy_fine, C_yx_fine = C_xys[k + 1], C_yxs[k + 1]
                if debias:
                    C_xx_fine, C_yy_fine = C_xxs[k + 1], C_yys[k + 1]

                last_extrapolation = False  # No need to re-extrapolate after the loop
                torch.autograd.set_grad_enabled(True)

            else:  # It's worth investing some time on kernel truncation...
                # The lines below implement the Kernel truncation trick,
                # described in Eq. (3.222-3.224) in Jean Feydy's PhD thesis and in
                # "Stabilized sparse scaling algorithms for entropy regularized transport
                #  problems", Schmitzer (2016-2019), (https://arxiv.org/pdf/1610.06519.pdf).
                #
                # A more principled and "controlled" variant is also described in
                # "Capacity constrained entropic optimal transport, Sinkhorn saturated
                #  domain out-summation and vanishing temperature", Benamou and Martinet
                #  (2020), (https://hal.archives-ouvertes.fr/hal-02563022/).
                #
                # On point clouds, this code relies on KeOps' block-sparse routines.
                # On grids, it is a "dummy" call: we do not perform any "truncation"
                # and rely instead on the separability of the Gaussian convolution kernel.

                # Line 9: a <-> b ------------------------------------------------------
                C_xy_fine, C_yx_fine = kernel_truncation(
                    C_xy,
                    C_yx,
                    C_xys[k + 1],
                    C_yxs[k + 1],
                    f_ba,
                    g_ab,
                    eps,
                    truncate=truncate,
                    cost=cost,
                )

                if debias:
                    # Line 10: a <-> a  ------------------------------------------------
                    C_xx_fine, _ = kernel_truncation(
                        C_xx,
                        C_xx,
                        C_xxs[k + 1],
                        C_xxs[k + 1],
                        f_aa,
                        f_aa,
                        eps,
                        truncate=truncate,
                        cost=cost,
                    )
                    # Line 11: b <-> b -------------------------------------------------
                    C_yy_fine, _ = kernel_truncation(
                        C_yy,
                        C_yy,
                        C_yys[k + 1],
                        C_yys[k + 1],
                        g_bb,
                        g_bb,
                        eps,
                        truncate=truncate,
                        cost=cost,
                    )

            # Line 12: extrapolation step ----------------------------------------------
            # We extra/inter-polate the values of the dual potentials from
            # the "coarse" to the "fine" resolution.
            #
            # On point clouds, we use the expressions of the dual potentials
            # detailed e.g. in Eqs. (3.194-3.195) of Jean Feydy's PhD thesis.
            # On images and volumes, we simply rely on (bi/tri-)linear interpolation.
            #
            # N.B.: the cross-updates below *must* be done in parallel!
            f_ba, g_ab = (
                extrapolate(f_ba, g_ab, eps, damping, C_xy, b_log, C_xy_fine),
                extrapolate(g_ab, f_ba, eps, damping, C_yx, a_log, C_yx_fine),
            )

            # Extrapolation for the symmetric problems:
            if debias:
                f_aa = extrapolate(f_aa, f_aa, eps, damping, C_xx, a_log, C_xx_fine)
                g_bb = extrapolate(g_bb, g_bb, eps, damping, C_yy, b_log, C_yy_fine)

            # Line 13: update the measure weights and cost "matrices" ------------------
            k = k + 1
            a_log, b_log = a_logs[k], b_logs[k]
            C_xy, C_yx = C_xy_fine, C_yx_fine
            if debias:
                C_xx, C_yy = C_xx_fine, C_yy_fine

    # As a very last step, we perform a final "Sinkhorn" iteration.
    # As detailed above (around "torch.autograd.set_grad_enabled(False)"),
    # this allows us to retrieve correct expressions for the gradient
    # without having to backprop through the whole Sinkhorn loop.
    torch.autograd.set_grad_enabled(True)

    if last_extrapolation:
        # The cross-updates should be done in parallel!
        f_ba, g_ab = (
            damping * softmin(eps, C_xy, (b_log + g_ab / eps).detach()),
            damping * softmin(eps, C_yx, (a_log + f_ba / eps).detach()),
        )

        if debias:
            f_aa = damping * softmin(eps, C_xx, (a_log + f_aa / eps).detach())
            g_bb = damping * softmin(eps, C_yy, (b_log + g_bb / eps).detach())

    if debias:
        return f_aa, g_bb, g_ab, f_ba
    else:
        return None, None, g_ab, f_ba