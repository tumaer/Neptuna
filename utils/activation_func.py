"""
PyTorch Activation Function Utilities.

This module provides a centralized registry and factory function for PyTorch
activation functions commonly used in neural networks. It offers a convenient
string-based interface for creating activation function instances, making it
easy to configure models through configuration files or command-line arguments.

The module supports all major PyTorch activation functions including:
- Basic activations: ReLU, LeakyReLU, ELU, SELU, etc.
- Smooth activations: SiLU, GELU, Sigmoid, Tanh, etc.
- Specialized activations: PReLU, Softplus, Threshold, etc.

Example Usage:
    >>> from utils.activation_func import get_activation
    >>> 
    >>> # Create a ReLU activation
    >>> relu = get_activation("relu")
    >>> 
    >>> # Create a LeakyReLU with default parameters
    >>> leaky_relu = get_activation("leaky_relu")
    >>> 
    >>> # Use in a model
    >>> import torch.nn as nn
    >>> model = nn.Sequential(
    ...     nn.Linear(10, 20),
    ...     get_activation("gelu"),
    ...     nn.Linear(20, 1)
    ... )

Available Activation Functions:
    - relu: Standard ReLU activation
    - leaky_relu: LeakyReLU with negative_slope=0.1
    - prelu: Parametric ReLU (learnable parameters)
    - relu6: ReLU clamped to maximum value of 6
    - elu: Exponential Linear Unit
    - selu: Scaled Exponential Linear Unit
    - silu: Sigmoid Linear Unit (Swish)
    - gelu: Gaussian Error Linear Unit
    - sigmoid: Standard sigmoid activation
    - logsigmoid: Logarithm of sigmoid
    - softplus: Smooth approximation of ReLU
    - softshrink: Soft shrinkage function
    - softsign: Softsign activation
    - tanh: Hyperbolic tangent
    - tanhshrink: Tanh shrinkage function
    - threshold: Threshold activation with threshold=1.0, value=1.0
    - hardtanh: Hard tanh (clamped tanh)
    - identity: Identity function (no transformation)
"""

import torch.nn as nn
ACT2FN = {
    "relu": nn.ReLU,
    "leaky_relu": (nn.LeakyReLU, {"negative_slope": 0.1}),
    "prelu": nn.PReLU,
    "relu6": nn.ReLU6,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "logsigmoid": nn.LogSigmoid,
    "softplus": nn.Softplus,
    "softshrink": nn.Softshrink,
    "softsign": nn.Softsign,
    "tanh": nn.Tanh,
    "tanhshrink": nn.Tanhshrink,
    "threshold": (nn.Threshold, {"threshold": 1.0, "value": 1.0}),
    "hardtanh": nn.Hardtanh,
    "identity": nn.Identity,
}

def get_activation(activation: str) -> nn.Module:
    """
    Create an activation function instance from a string identifier.
    
    This factory function provides a convenient way to instantiate PyTorch
    activation functions using string names. It handles both simple activations
    (instantiated with default parameters) and parameterized activations
    (instantiated with predefined parameter values).
    
    Parameters
    ----------
    activation : str
        String identifier for the desired activation function. Case-insensitive.
        Must be one of the keys in the ACT2FN dictionary.
    
    Returns
    -------
    torch.nn.Module
        An instantiated PyTorch activation function ready to use in neural networks.
        
    Raises
    ------
    KeyError
        If the specified activation function is not found in the registry.
        The error message includes a list of all available activation functions.
    
    Examples
    --------
    >>> # Create basic activations
    >>> relu = get_activation("relu")
    >>> gelu = get_activation("GELU")  # case-insensitive
    >>> 
    >>> # Create parameterized activations (uses predefined parameters)
    >>> leaky_relu = get_activation("leaky_relu")  # negative_slope=0.1
    >>> threshold = get_activation("threshold")    # threshold=1.0, value=1.0
    >>> 
    >>> # Use in a forward pass
    >>> import torch
    >>> x = torch.randn(10, 5)
    >>> activated = relu(x)
    >>> 
    >>> # Use in model definition
    >>> import torch.nn as nn
    >>> model = nn.Sequential(
    ...     nn.Linear(784, 256),
    ...     get_activation("relu"),
    ...     nn.Linear(256, 10),
    ...     get_activation("softmax")  # Note: softmax not in current registry
    ... )
    
    Notes
    -----
    Some activation functions in the registry come with predefined parameters:
    - LeakyReLU: negative_slope=0.1
    - Threshold: threshold=1.0, value=1.0
    
    If you need custom parameters for these functions, instantiate them directly
    from torch.nn rather than using this factory function.
    
    The function converts the input string to lowercase for case-insensitive matching.
    """
    try:
        activation = activation.lower()
        module = ACT2FN[activation]
        if isinstance(module, tuple):
            return module[0](**module[1])
        else:
            return module()
    except KeyError:
        raise KeyError(
            f"Activation function {activation} not found. Available options are: {list(ACT2FN.keys())}"
        )