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
    
# Adapted from https://github.com/camlab-ethz/poseidon
class ConditionalLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps # small constant to avoid division by zero
        # instead of using nn.Parameter like in LayerNorm, weight and bias are learned linear functions of time (-> they vary with time)
        self.weight = nn.Linear(1, dim)
        self.bias = nn.Linear(1, dim)

    def forward(self, x, cond_params = None):

        # ToDo: Append cond_params

        if cond_params is None:
            raise ValueError("cond_params must be provided for ConditionalLayerNorm")
        
        # x: [16, 1024, 48]
        # compute mean and variance of input over last dimension (like in LayerNorm)
        mean = x.mean(dim=-1, keepdim=True) # [16, 1024, 1]
        var = (x**2).mean(dim=-1, keepdim=True) - mean**2 # [16, 1024, 1]
        # Normalize input x (zero mean, unit variance)
        x = (x - mean) / (var + self.eps).sqrt()
        cond_params = cond_params.reshape(-1, 1).type_as(x) # [16, 1]
        weight = self.weight(cond_params).unsqueeze(1) #[16, 1, 48]
        bias = self.bias(cond_params).unsqueeze(1) # [16, 1, 48]
        if x.dim() == 4:
            weight = weight.unsqueeze(1)
            bias = bias.unsqueeze(1)
        return weight * x + bias     
