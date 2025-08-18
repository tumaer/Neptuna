from typing import List, Optional, Tuple
import torch.nn as nn
from torch import Tensor
import numpy as np
import torch
from utils.grid_utils import oned_meshgrid, twod_meshgrid, threed_meshgrid
from utils.model_utils import PretrainedConfig
from utils.model_utils import CustomNorm
from utils.model_utils import SequentialWithKwargs

class DeepONetConfig(PretrainedConfig):
    """
    Args:
        branch_net_str : str
            Type of branch network (FFN, CNN, ResNet)
        query_idxs : Optional[Tensor]
            Indices for the trunk network
        branch_depth : int
            Depth of the branch network, not used for ResNet
        trunk_depth : int
            Depth of the trunk network
        width : int
            Dimension of the hidden layers, if CNN or ResNet is used for branch net, this argument is only applied to trunk net
        activation_fn : str
            Activation function
        act_on_output : bool
            Whether to apply activation function on the output of the branch network, only used for FFN
        kernel_size : Optional[int]
            Kernel size for the CNN branch network, not used for FFN or ResNet
        padding : Optional[int]
            Padding for the CNN branch network, not used for FFN and ResNet
        stride : Optional[int]
            Stride for the CNN branch network, not used for FNN and ResNet
        num_blocks (List[int]): 
            Number of blocks in each stage, only used for ResNet branch network
        ResNet_block : Optional[str]
            Type of ResNet block, only used for ResNet branch network
        out_ffn_depth : Optional[int]
            Output depth for the FFN branch network
        norm : str
            Normalization type, by default "layer"
        norm_layer_eps : float
            Epsilon for the normalization layer, by default 1e-5
    """
    def __init__(
        self,
        branch_net_str: Optional[str] = None,
        query_idxs: Optional[Tensor] = None,
        branch_depth: int = 4,
        trunk_depth: int = 4,
        width: int = 100,
        activation_fn_name: str = "gelu",
        act_on_output: bool = False,
        kernel_size: Optional[int] = 3,
        padding: Optional[int] = 1,
        stride: Optional[int] = 2,
        num_blocks: Optional[List[int]] = [1],
        ResNet_block: Optional[str] = "BasicBlock",
        out_ffn_depth: Optional[int] = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Assign directly; validation happens at model init
        self.branch_net_str = branch_net_str
        self.query_idxs = query_idxs
        self.branch_depth = branch_depth
        self.trunk_depth = trunk_depth
        self.width = width
        self.activation_fn_name = activation_fn_name
        self.act_on_output = act_on_output
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.num_blocks = num_blocks
        self.ResNet_block = ResNet_block
        self.out_ffn_depth = out_ffn_depth

        if self.coord_features:
            self.in_size = self.in_size - self.dimension


class FFN(nn.Module):
    """
    A general fully connected multi-layer neural network.
    """

    def __init__(
        self, 
        dims: List, 
        activation_fn: nn.Module = nn.GELU(),
        act_on_output: bool = False,
    ):
        super().__init__()

        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if act_on_output:
            layers.append(activation_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x

class FFNBranch(nn.Module):
    """
    A general fully connected multi-layer neural network.
    """

    def __init__(
        self, 
        config: DeepONetConfig,
        dims: List, 
        activation_fn: nn.Module = nn.GELU(),
        act_on_output: bool = False,
    ):
        super().__init__()

        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(CustomNorm(config=config, num_channels=dims[i + 1], array_length=3, channel_at_last_position=False))
            layers.append(activation_fn)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if act_on_output:
            layers.append(activation_fn)
        # Replace plain nn.Sequential with variant that forwards **kwargs to submodules
        self.layers = SequentialWithKwargs(*layers)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        x = self.layers(x, **kwargs)
        return x
    
class CnnBranch(nn.Module):
    def __init__(
        self, 
        config: DeepONetConfig,
        in_channels: int, 
        kernel_size: int, 
        padding: int, 
        dimension: int,
        grid_resolution:Tuple[int],
        latent_channels: int = 32,
        depth: int = 4,
        activation_fn: nn.Module = nn.ReLU(),
        stride : int = 1,
        coord_features: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.depth = depth
        self.stride = stride
        self.grid_resolution = grid_resolution
        self.dimension = dimension
        self.coord_features = coord_features
        # Add relative coordinate feature
        if coord_features:
            self.in_channels = self.in_channels + dimension
        
        if dimension == 1:
            Conv = nn.Conv1d
            Pool = nn.MaxPool1d
        elif dimension == 2:
            Conv = nn.Conv2d
            Pool = nn.MaxPool2d
        elif dimension == 3:
            Conv = nn.Conv3d
            Pool = nn.MaxPool3d
        else:
            raise ValueError(f"Unsupported dimension: {dimension}. Must be 1, 2, or 3.")
        self.in_conv = Conv(
            self.in_channels, latent_channels , kernel_size=3, padding=1)
        self.out_conv = Conv(
            latent_channels, latent_channels, kernel_size=3, padding=1)

        blocks = []
        for i in range(depth):
            blocks += [
                Conv(latent_channels, latent_channels, kernel_size, padding=padding, stride=stride),
                Pool(2),
                CustomNorm(config=config, num_channels=latent_channels, array_length=dimension+2, channel_at_last_position=False),
                activation_fn,
            ]
        self.blocks = SequentialWithKwargs(*blocks)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if self.coord_features: 
            if self.dimension == 1:
                coord_feat = oned_meshgrid(list(x.shape), x.device)
            elif self.dimension == 2:
                coord_feat = twod_meshgrid(list(x.shape), x.device)
            elif self.dimension == 3:
                coord_feat = threed_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.in_conv(x)  # (b, 16, h, w)
        x = self.blocks(x, **kwargs)  # (b, 32, h/16=4, w/16=4)
        x = self.out_conv(x)  # (b, 32, 4, 4)
        return x
    
    def calc_out_shape(self) -> Tuple[int, ...]:
        out = list(self.grid_resolution)
        for _ in range(self.depth):
            for i in range(len(out)):
                out[i] = (out[i] + 2 * self.padding - self.kernel_size) // self.stride + 1  # Conv
                out[i] = (out[i]  - 2) // 2 + 1  # Pool
                assert out[i] > 0, f"Output shape is non-positive: {out[i]}, reduce the number of blocks "
        return tuple(out)

class BasicBlockND4DeepONet(nn.Module):
    """
    A small ResNet-like block tailored for DeepONet branches.
    """
    expansion: int = 1
    def __init__(
        self,
        config,
        in_planes: int,
        planes: int,
        dimension: int, 
        stride: int = 1,
        activation_fn: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__()
        
        if dimension == 1:
            Conv = nn.Conv1d
            Pool = nn.MaxPool1d
            kernel_size = (3,)
            padding = (1,)
        elif dimension == 2:
            Conv = nn.Conv2d
            Pool = nn.MaxPool2d
            kernel_size = (3, 3)
            padding = (1, 1)
        elif dimension == 3:
            Conv = nn.Conv3d
            Pool = nn.MaxPool3d
            kernel_size = (3, 3, 3)
            padding = (1, 1, 1)
        else:
            raise ValueError(f"Unsupported dimension: {dimension}. Must be 1, 2, or 3.")
        
        #2X 3*3 convolutions Layers and corresponding GroupNorm
        self.conv1 = Conv(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.norm1 = CustomNorm(config=config, num_channels=planes, array_length=dimension+2, channel_at_last_position=False)
        self.maxpool = Pool(2)
        self.conv2 = Conv(planes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=True)
        self.norm2 = CustomNorm(config=config, num_channels=planes, array_length=dimension+2, channel_at_last_position=False)
        self.activation = activation_fn

        # Shortcut connection
        self.shortcut = SequentialWithKwargs()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = SequentialWithKwargs(
                Conv(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                CustomNorm(config=config, num_channels=self.expansion * planes, array_length=dimension+2, channel_at_last_position=False),
            )

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        out = x
        out = self.conv1(out)
        out = self.norm1(out, **kwargs)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out, **kwargs)
        out = self.activation(out)
        out = out + self.shortcut(x, **kwargs)
        out = self.maxpool(out)
        return out

def grid_to_points(value: Tensor) -> Tuple[Tensor, List[int]]:
	"""
	Convert from grid-based (1D, 2D, 3D) representation to point-based representation.

	Parameters
	----------
	value : Tensor
		Input tensor of shape (B, C, X, Y, Z).

	Returns
	-------
	Tuple
		- Tensor of shape (B, C*X*Y*Z).
	"""
	output = value.reshape(value.size(0), -1)  # Reshape to (B, C*X*Y*Z)
	return output

def points_to_grid(value: Tensor, shape: List[int]) -> Tensor:
	"""
	Convert from point-based representation back to grid-based (1D, 2D, 3D) representation.

	Parameters
	----------
	value : Tensor
		Input tensor of shape (B, C*X*Y*Z).
	shape : List[int]
		Original shape as [B, C, X, Y, Z].

	Returns
	-------
	Tensor
		Restored tensor of shape (B, C, X, Y, Z).
	"""
	output = value.reshape(shape)  # Reshape back to (B, C, X, Y, Z)
	return output

def calc_resnet_out_shape(
	in_shape: tuple,
	num_blocks: List[int],
	if_maxpool: bool = False,
	stride: int=1,
	kernel_size: int=3,
	padding: int=1,
):
	out = list(in_shape)
	count = 0
	for j in range(len(num_blocks)):
		count += num_blocks[j]
	for _ in range(count):
		for i in range(len(out)):
			if if_maxpool:
				out[i] = (out[i] + 2*1 - 3) // stride + 1
				out[i] = (out[i] + 2*1 - 3) // 1 + 1
				out[i] = (out[i] - 2) // 2 + 1
				assert out[i] > 0, f"Output shape is non-positive: {out[i]}, reduce the number of blocks "
			else:
				out[i] = (out[i] + 2 * padding - kernel_size) // stride + 1
				out[i] = (out[i] + 2 * padding - kernel_size) // 1 + 1
				assert out[i] > 0, f"Output shape is non-positive: {out[i]}, reduce the number of blocks "
	return tuple(out)

def linspace_int_list(int1: int, int2: int, int3: int, reverse: bool) -> list:
	assert int2 > 1, "branch and trunk depth must be greater than 1"
	arr = [int(round(x)) for x in np.linspace(int3, int1, int2)]
	arr[0] = int3
	arr[-1] = int1
	if reverse:
		return arr[::-1]
	else:
		return arr