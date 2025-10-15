"""
Model Configuration Utilities for Deep Learning Architectures.

This module provides base configuration classes and utilities for managing
model parameters across different neural network architectures. It extends
the HuggingFace transformers and includes other models.

Key Features:
- Unified configuration interface for all model architectures
- Automatic coordinate feature handling for spatial models
- Flexible sequence information management for temporal models
- JSON serialization with OmegaConf integration
- Extensible base class for custom model configurations

Classes:
    PretrainedConfig: Base configuration class for all model architectures
    
Configuration Parameters:
    Common parameters supported across all model types:
    - in_channels/out_channels: Input and output channel specifications
    - dimension: Spatial dimensionality (1D, 2D, 3D)
    - grid_resolution: Spatial resolution for each dimension
    - sequence_info: Temporal sequence configuration [input_len, output_len, stride]
    - coord_features: Automatic coordinate feature inclusion
    - latent_channels: Hidden layer channel counts

Usage Patterns:
    Model configurations are typically created by specific model classes
    that inherit from PretrainedConfig and add architecture-specific
    parameters. The base class handles common functionality like:
    - Channel size calculations with sequence length consideration
    - Coordinate feature integration
    - JSON serialization for model saving/loading
    - Configuration validation and type checking

Notes:
    The configuration system automatically adjusts input channel counts
    when coordinate features are enabled (True by default), adding one channel per spatial
    dimension. This ensures consistent handling of spatial information
    across different model architectures.
"""
import torch
from transformers import PretrainedConfig as PretrainedConfig_
import json
from typing import List, Optional, Tuple, Union
from omegaconf import OmegaConf
from torch import nn


class PretrainedConfig(PretrainedConfig_):
    """
    Base configuration class for all neural network model architectures.
    
    This class extends the HuggingFace PretrainedConfig to provide a unified
    interface for configuring deep learning models used in scientific computing
    and other applications. It handles common parameters shared across different
    architectures and provides automatic feature computation.
    
    The class automatically calculates derived parameters such as effective
    input/output sizes when sequence information and coordinate features are
    considered, ensuring consistent configuration across different model types.
    
    Parameters
    ----------
    in_channels : int, default=1
        Number of input channels for the model. This represents the number
        of different physical quantities or features in the input data.
    out_channels : int, default=1
        Number of output channels for the model. This represents the number
        of different quantities the model should predict.
    dimension : int, default=1
        Spatial dimensionality of the input data (1D, 2D, or 3D).
        Used for coordinate feature generation and spatial operations.
    grid_resolution : Union[int, List[int], Tuple[int]], default=[160]
        Spatial resolution for each dimension. Can be:
        - int: Same resolution for all dimensions
        - List/Tuple: Different resolution per dimension [H, W] or [D, H, W]
    sequence_info : Optional[List[int]], default=[1, 1, 1]
        Temporal sequence configuration as [input_seq_len, output_seq_len, stride].
        - input_seq_len: Number of input time steps
        - output_seq_len: Number of output time steps  
        - stride: Temporal stride between consecutive frames
    coord_features : bool, default=True
        Whether to automatically include coordinate features in the model.
        When True, adds spatial coordinate channels (x, y, z) to input.
    latent_channels : int, default=32
        Number of latent/hidden channels in the model architecture.
        Controls the model's representational capacity.
    include_input_seq_len : bool, default=True
        Whether to include sequence length in input/output size calculations.
        Set to False for models that handle sequences internally.
    **kwargs : dict
        Additional keyword arguments passed to the parent PretrainedConfig class.
        
    Attributes
    ----------
    in_size : int
        Effective input size calculated as:
        (in_channels * input_seq_len) + (dimension if coord_features else 0)
    out_size : int
        Effective output size calculated as:
        out_channels * output_seq_len
    
    Methods
    -------
    to_json_string(use_diff=True)
        Serialize configuration to JSON string with OmegaConf integration.
    
    Notes
    -----
    Coordinate Features:
    - When enabled, coordinate features add one channel per spatial dimension
    - Coordinates are typically normalized to [-1, 1] range
    - Essential for models that need spatial awareness (e.g., neural operators)
    
    Sequence Handling:
    - Sequence information is used to calculate effective channel counts
    - Input sequences are flattened into channel dimensions
    - Useful for models processing temporal data
    
    Configuration Inheritance:
    - Specific model classes should inherit from this base class
    - Add architecture-specific parameters as needed
    - Override methods for custom behavior
    
    The configuration system ensures consistent parameter handling across
    different model architectures while providing flexibility for specialized
    requirements.
    """

    def __init__(
        self,
        in_channels: int = 1,       # Number of input_channels
        out_channels: int = 1,      # Number of output channels
        dimension: int = 1,
        grid_resolution: Union[int, List[int], Tuple[int]] = [160], # Input and Output spatial size (required )
        sequence_info: Optional[List[int]] = [1,1,1],
        coord_features: bool = True,
        latent_channels: int = 32,
        include_input_seq_len: bool = True,
        conditioning: str = None,
        norm: str = 'identity',
        num_cond_params: int = 0,
        norm_layer_eps: float = 1e-5,
        **kwargs
    ):
        super().__init__(**kwargs)
    
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimension = dimension
        self.sequence_info = sequence_info

        if include_input_seq_len:
            self.in_size = self.in_channels * self.sequence_info[0] 
            self.out_size = self.out_channels * self.sequence_info[1]

        self.grid_resolution = grid_resolution
        self.latent_channels = latent_channels

        self.num_cond_params = num_cond_params
        self.conditioning = conditioning
        self.norm = norm
        self.norm_layer_eps = norm_layer_eps
        if norm not in ['layer', 'batch', 'group', 'identity']:
            raise ValueError(f'{norm} norm is not in the specified list of allowed norms')

        self.coord_features = coord_features
        # Add relative coordinate feature
        if coord_features:
            self.in_size = self.in_size + dimension


    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serialize this configuration instance to a JSON string.
        
        This method provides JSON serialization with support for OmegaConf
        objects and other complex data types commonly used in configuration
        management. It can serialize either the full configuration or just
        the differences from the default configuration.
        
        Parameters
        ----------
        use_diff : bool, default=True
            If True, only serialize the difference between this config instance
            and the default PretrainedConfig(). If False, serialize the complete
            configuration including all default values.
            
        Returns
        -------
        str
            JSON string containing the configuration parameters. The string
            is formatted with proper indentation and sorted keys for readability.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()

        # NOTE: default dumpts List objects in the PretrainedConfig to json, since only json.dumps throws an error
        def default(o):
            return OmegaConf.to_container(o, resolve=True)
        return json.dumps(config_dict, indent=2, sort_keys=True, default=default) + "\n"

# -----------------------------------------------------------------------------
# Helper to propagate **kwargs through nn.Sequential
# -----------------------------------------------------------------------------
class SequentialWithKwargs(nn.Sequential):
    """nn.Sequential variant that forwards any additional keyword arguments
    to every sub-module. This makes it compatible with blocks whose forward
    signature is ``forward(x, **kwargs)`` (e.g. blocks containing
    `CustomNorm` layers that need conditioning parameters).
    """

    def forward(self, x, **kwargs): 
        """Forward that is tolerant of modules which do **not** accept the
        extra keyword arguments.  For each sub-module we first attempt to call
        it with ``**kwargs``; if this results in a *TypeError* complaining
        about unexpected keyword arguments we retry without them.  This allows
        mixing plain layers (e.g. ``nn.Conv``) with custom layers (e.g.
        ``CustomNorm``) that require the extra data.
        """

        for module in self:
            if kwargs:
                try:
                    x = module(x, **kwargs)
                    continue  # success
                except TypeError as e:
                    # Only swallow the error if it is about unexpected kwarg
                    # to keep other bugs visible.
                    if "unexpected keyword argument" not in str(e):
                        raise
            # Fallback: call without kwargs
            x = module(x)
        return x

# Adapted from https://github.com/camlab-ethz/poseidon
class ConditionalLayer(nn.Module):
    def __init__(self, input_dim, num_cond_params, channel_at_last_position):
        super().__init__()
        self.num_cond_params = num_cond_params
        # instead of using nn.Parameter like in LayerNorm, weight and bias are learned linear functions of time (-> they vary with time)
        self.weight = nn.Linear(num_cond_params, input_dim)
        self.bias = nn.Linear(num_cond_params, input_dim)
        self.single_input_dim = input_dim
        self.channel_at_last_position = channel_at_last_position
    def forward(self, x, **kwargs):

        if "conditioning_parameters" in kwargs:
            cond_params = kwargs["conditioning_parameters"]
        else:
            raise ValueError("There is no conditioning_parameter in the dataset.")  

        if self.channel_at_last_position:
            B, *spatial_dims, C = x.shape
            gamma = self.weight(cond_params).view(B, *[1] * len(spatial_dims), C)
            beta = self.bias(cond_params).view(B, *[1] * len(spatial_dims), C)
        else:
            B, C, *spatial_dims = x.shape
            gamma = self.weight(cond_params).view(B, C, *[1] * len(spatial_dims))
            beta = self.bias(cond_params).view(B, C, *[1] * len(spatial_dims))

        out = gamma * x + beta #Affine Transformation
        return out

def get_num_groups(num_channels: int, max_groups: int = 16) -> int:
    """
    Return the largest number of groups ≤ max_groups that divides num_channels.
    Falls back to 1 (like InstanceNorm) if none divide cleanly.
    """
    for g in reversed(range(1, max_groups + 1)):
        if num_channels % g == 0:
            return g
    return 1 

class BatchNormChannelLast(nn.Module):
    """Batch Normalization for tensors in **channel-last** format.

    This helper wraps the standard ``torch.nn.BatchNorm1d/2d/3d`` modules so
    they can be used with data where the channel dimension is at the *end* of
    the tensor, e.g. ``(N, L, C)``, ``(N, H, W, C)`` or ``(N, D, H, W, C)``.
    Internally the tensor is temporarily permuted to channel-first layout 
    i.e (N, C, L) or (N, C, H, W) or (N, C, D, H, W),
    normalized, and permuted back so the user interface remains unchanged.

    Parameters
    ----------
    dim : int
        Total rank of the input tensor (including the batch dimension).
        Accepted values:

        * 3 → 1-D data ``(N, L, C)``
        * 4 → 2-D data ``(N, H, W, C)``
        * 5 → 3-D data ``(N, D, H, W, C)``

    num_channels : int
        The size of the channel dimension *C*.

    **kwargs : dict
        Additional arguments passed straight to the underlying
        ``torch.nn.BatchNorm1d/2d/3d`` instance (e.g. ``eps``, ``momentum``).
    """

    def __init__(self, dim: int, num_channels: int, **kwargs):
        super().__init__()
        if dim == 3:
            self.bn = nn.BatchNorm1d(num_channels, **kwargs)
        elif dim == 4:
            self.bn = nn.BatchNorm2d(num_channels, **kwargs)
        elif dim == 5:
            self.bn = nn.BatchNorm3d(num_channels, **kwargs)
        else:
            raise ValueError(f"Unsupported dimension: {dim}")
        self.dim = dim

    def forward(self, x):
        # Save original shape
        orig_shape = x.shape
        if self.dim == 3:
            # x: (N, L, C) → (N, C, L)
            x = x.permute(0, 2, 1)
            x = self.bn(x)
            x = x.permute(0, 2, 1)
        elif self.dim == 4:
            # x: (N, H, W, C) → (N, C, H, W)
            x = x.permute(0, 3, 1, 2)
            x = self.bn(x)
            x = x.permute(0, 2, 3, 1)
        elif self.dim == 5:
            # x: (N, D, H, W, C) → (N, C, D, H, W)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.bn(x)
            x = x.permute(0, 2, 3, 4, 1)
        elif self.dim == 6:
            # ADDED FOR 3D kFNO
            # x: (N, D1, D2, D3, D4, C) → (N, C, D1, D2, D3*D4)
            N, D1, D2, D3, D4, C = x.shape
            x = x.permute(0, 5, 1, 2, 3, 4)
            x = x.reshape(N, C, D1, D2, D3*D4)
            x = self.bn(x)
            x = x.reshape(N, C, D1, D2, D3, D4)
            x = x.permute(0, 2, 3, 4, 5, 1)
        return x

class GroupNormChannelLast(nn.Module):
    """Group Normalization for channel-last formatted tensors.

    This is analogous to :class:`BatchNormChannelLast`, but applies
    :pyclass:`torch.nn.GroupNorm` instead of batch normalization.  The input
    is expected to have channels in the last dimension (e.g. ``N, L, C`` for
    1-D, ``N, H, W, C`` for 2-D, or ``N, D, H, W, C`` for 3-D data).  The
    tensor is temporarily permuted to channel-first layout, normalized, and
    then permuted back to the original layout so the public interface remains
    channel-last.
    """

    def __init__(self, dim: int, num_channels: int, num_groups: int | None = None, **kwargs):
        """Parameters
        ----------
        dim : int
            Total dimension of the input tensor (including batch).  Accepted
            values are 3 (``N, L, C``), 4 (``N, H, W, C``) and 5 (``N, D, H, W, C``).
        num_channels : int
            Number of channels *C*.
        num_groups : int, optional
            Number of groups to use for :class:`torch.nn.GroupNorm`.  If
            omitted, it is chosen automatically via ``get_num_groups`` so that
            ``num_channels`` is divisible by ``num_groups`` while not exceeding
            16.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`torch.nn.GroupNorm` (e.g. ``eps``).
        """
        super().__init__()

        # Automatically determine a suitable group count if not specified
        if num_groups is None:
            num_groups = get_num_groups(num_channels, max_groups=16)

        # Underlying GroupNorm instance (expects channel-first layout)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, **kwargs)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
        if self.dim == 3:  # (N, L, C) → (N, C, L)
            x = x.permute(0, 2, 1)
            x = self.gn(x)
            x = x.permute(0, 2, 1)
        elif self.dim == 4:  # (N, H, W, C) → (N, C, H, W)
            x = x.permute(0, 3, 1, 2)
            x = self.gn(x)
            x = x.permute(0, 2, 3, 1)
        elif self.dim == 5:  # (N, D, H, W, C) → (N, C, D, H, W)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.gn(x)
            x = x.permute(0, 2, 3, 4, 1)
        elif self.dim == 6:
            # ADDED FOR 3D kFNO
            x = x.permute(0, 5, 1, 2, 3, 4) # (N, D1, D2, D3, D4, C) → (N, C, D1, D2, D3, D4)
            x = self.gn(x)
            x = x.permute(0, 2, 3, 4, 5, 1)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")
        return x

class WrappedBatchNorm6D(nn.Module):
    """Helper for applying BatchNorm3D to 6D tensors with channel in front."""
    def __init__(self, num_channels, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm3d(num_channels, **kwargs)
    
    def forward(self, x):
        # x: (N, C, D1, D2, D3, D4) → reshape to (N, C, D1, D2, D3*D4)
        orig_shape = x.shape
        N, C = orig_shape[0], orig_shape[1]
        x = x.reshape(N, C, orig_shape[2], orig_shape[3], -1)
        x = self.bn(x)
        # Restore original shape
        x = x.reshape(orig_shape)
        return x

class CustomNorm(nn.Module):
    """Factory wrapper that chooses an appropriate normalization layer.

    The class supports three normalization types (``layer``, ``batch``,
    ``group``) as specified by *config.norm* and handles both channel–first and
    channel–last tensor layouts.  When *conditioning* is enabled in the
    *config*, a learnable, input-dependent affine transform is applied *before*
    the chosen normalization layer via :class:`ConditionalLayer`.

    Usage::

           CustomNorm(config, num_channels=48, array_length=4,
                       channel_at_last_position=True)

    Here *array_length* is the rank of the data tensor (including batch), so
    ``3 → (N, L, C)``, ``4 → (N, H, W, C)``, ``5 → (N, D, H, W, C)``.
    """

    def __init__(
        self,
        config,
        num_channels: int,
        array_length: int,
        channel_at_last_position: bool = False,
    ):
        super().__init__()

        # Basic validation
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if array_length not in (3, 4, 5, 6):
            raise ValueError("array_length must be 3, 4, or 5 (including batch dimension)")

        self.num_channels = num_channels
        self.array_length = array_length
        self.channel_at_last_position = channel_at_last_position
        self.conditioning = config.conditioning

        # Optional conditional affine transformation before normalization
        if self.conditioning:
            self.cond_layer = ConditionalLayer(
                num_channels,  # channels being scaled/shifted
                num_cond_params=config.num_cond_params,
                channel_at_last_position=channel_at_last_position,
            )

        # Select the actual normalization layer
        self.norm = self._build_norm_layer(config)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _build_norm_layer(self, config):
        """Return a concrete ``torch.nn.Module`` implementing the norm."""

        eps = config.norm_layer_eps  # convenience alias
        norm_type = config.norm.lower()

        # -------------------- LAYER NORM --------------------
        if norm_type == "layer":
            if self.channel_at_last_position:
                return nn.LayerNorm(self.num_channels, eps=eps)
            else:  # channel-first → use GroupNorm with 1 group (InstanceNorm)
                return nn.GroupNorm(1, self.num_channels, eps=eps)

        # -------------------- BATCH NORM --------------------
        if norm_type == "batch":
            if self.channel_at_last_position:
                return BatchNormChannelLast(
                    dim=self.array_length, num_channels=self.num_channels, eps=eps
                )
            # channel-first mapping by spatial rank
            bn_cls_map = {3: nn.BatchNorm1d, 4: nn.BatchNorm2d, 5: nn.BatchNorm3d, 6: nn.BatchNorm3d}
            try:
                bn_cls = bn_cls_map[self.array_length]
            except KeyError:
                raise ValueError(
                    f"Unsupported tensor rank {self.array_length} for batch norm"
                )
            if self.array_length == 6:
                return WrappedBatchNorm6D(self.num_channels, eps=eps)
            return bn_cls(self.num_channels, eps=eps)

        # -------------------- GROUP NORM --------------------
        if norm_type == "group":
            num_groups = get_num_groups(self.num_channels, max_groups=16)
            if self.channel_at_last_position:
                return GroupNormChannelLast(
                    dim=self.array_length,
                    num_channels=self.num_channels,
                    num_groups=num_groups,
                    eps=eps,
                )
            else:
                return nn.GroupNorm(num_groups=num_groups, num_channels=self.num_channels, eps=eps)

        # -------------------- IDENTITY (No Normalization) --------------------
        if norm_type == "identity":
            return nn.Identity()

        # ----------------------------------------------------
        raise ValueError(f"{config.norm} is not a supported normalization type")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # type: ignore[override]
        if self.conditioning:
            x = self.cond_layer(x, **kwargs)
        return self.norm(x)