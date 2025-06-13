from typing import List, Optional, Union, Callable, Tuple
from itertools import product
import torch
import torch.nn as nn
from torch import Tensor
from utils import activation_func
from .deeponet_utils import Ffn, CnnBranch, grid_to_points, points_to_grid, calc_resnet_out_shape, linspace_int_list
from models.ResNet.resnet import ResNet1D, ResNet2D, ResNet3D



class AutoDeepONet(nn.Module):
    """
    Auto-regressive DeepONet for CFD.

    Args:
        in_channels : int
            Number of input fields
        out_channels : int
            Number of output fields
        grid_resolution : Tuple[int]
            Resolution of the input fields
        dimension : int
            Dimension of the input fields (1D, 2D, or 3D)
        branch_net : str
            Type of branch network (FFN, CNN, ResNet)
        query_idxs : Optional[Tensor]
            Indices for the trunk network
        sequence_info (Listint]):
            sequence_info[0]: input_seq_len, 
            sequence_info[1]: label_seq_len,
            sequence_info[2]: input/output_sequence_stride 
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
            Padding for the CNN and ResNet branch network, not used for FFN
        hidden_channels : Optional[int]
            Number of hidden channels for the CNN and ResNet branch network, not used for FFN
        num_blocks (List[int]): 
            Number of blocks in each stage, only used for ResNet branch network
    """
    main_input_name = "input_data"
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_resolution: Tuple[int],
        dimension: int,
        branch_net: str = "FFN",
        query_idxs: Optional[Tensor] = None,
        sequence_info: Optional[List[int]] = [1,1,1],
        #num_label_samples: int = 1000,
        branch_depth: int = 4,
        trunk_depth: int = 4,
        width: int = 100,
        activation_fn_name: str = "gelu",
        #act_norm: bool = False,
        act_on_output: bool = False,        
        kernel_size: Optional[int] = 3, 
        padding: Optional[int] = 1, 
        stride: Optional[int] = 2,
        latent_channels: Optional[int] = 32,
        num_blocks: Optional[List[int]] = [1],
    ):

        super().__init__()
        self.in_size = in_channels * sequence_info[0] 
        self.out_size = out_channels * sequence_info[1]
        self.branch_depth = branch_depth
        self.trunk_depth = trunk_depth
        self.width = width
        self.dimension = dimension
        self.activation: nn.Module = activation_func.get_activation(activation_fn_name)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation_fn_name} not implemented")

        if isinstance(branch_net, str):
            if branch_net == "FFN":
                self.AutoDeepONet = self.build_AutoDeepONet()(
                    in_size = self.in_size,
                    out_size = self.out_size,
                    grid_resolution = grid_resolution,
                    activation_fn = self.activation,
                    branch_net = branch_net,
                    branch_depth = branch_depth,
                    trunk_depth = trunk_depth,
                    width = width,
                    act_on_output = act_on_output,
                    query_idxs = query_idxs,
                    num_blocks = num_blocks,
                )
            elif branch_net == "CNN":
                self.AutoDeepONet = self.build_AutoDeepONet()(
                    in_size = self.in_size,
                    out_size = self.out_size,
                    grid_resolution = grid_resolution,
                    activation_fn = self.activation,
                    branch_net = branch_net,
                    branch_depth = branch_depth,
                    trunk_depth = trunk_depth,
                    width = width,
                    query_idxs = query_idxs,
                    kernel_size = kernel_size,
                    padding = padding,
                    stride = stride,
                    latent_channels = latent_channels,
                )
            elif branch_net == "ResNet": #TODO:add kernal_size and stride to ResNet
                self.AutoDeepONet = self.build_AutoDeepONet()(
                    in_size = self.in_size,
                    out_size = self.out_size,
                    grid_resolution = grid_resolution,
                    activation_fn = self.activation,
                    branch_net = branch_net,
                    branch_depth = branch_depth,
                    trunk_depth = trunk_depth,
                    padding= padding,
                    latent_channels = latent_channels, 
                    #block = "BasicBlock",
                    #num_blocks = [1,1,1],
                )

    def forward(self, 
                input_data: Tensor,
                ) -> Tensor: 
        #reshape input into [batch, in_channel, grid_x, grid_y, ...]
        #NOTE: input and output fields need not be necessarily the same.
        batch, input_seq, input_fields, *spatial = input_data.shape
        input_data=input_data.reshape(batch, input_seq * input_fields, *spatial)
        
        x = self.AutoDeepONet(input_data)  # (B, C_out, X)
     
        return x
    
    def build_AutoDeepONet(self):
        """Get the ResNet encoder based on the model dimensionality"""
        if self.dimension == 1:
            return AutoDeepONet1D
        elif self.dimension == 2:
            return AutoDeepONet2D
        elif self.dimension == 3:
            return AutoDeepONet3D
        else:
            raise NotImplementedError(
                "Invalid dimensionality. Only 1D, 2D, 3D ResNet implemented"
            )
    
class AutoDeepONet1D(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        grid_resolution: Tuple[int],
        activation_fn: nn.Module,
        branch_net: str,
        branch_depth: int,
        trunk_depth: int,
        width: Optional[int] = 4,
        num_blocks: Optional[List[int]] = [1],
        act_on_output: Optional[bool] = False,
        kernel_size: Optional[int] = 3, 
        stride: Optional[int] = 1,
        out_ffn_depth: Optional[int] = 3,
        padding: Optional[int] = 1, 
        latent_channels: Optional[int] = 32,
        query_idxs: Optional[Tensor] = None,        
    ):
        super().__init__()
        self.out_size = out_size
        self.activation_fn = activation_fn
        self.branch_net_str = branch_net
        self.query_idxs = query_idxs
        self.width = width
        self.trunk_depth = trunk_depth        
        if isinstance(branch_net, str):
            if branch_net == "FFN":
                branch_dim = in_size * grid_resolution[0]
                self.branch_dims = [branch_dim] + [width] * branch_depth
                self.branch_net = Ffn(
                    dims = self.branch_dims,
                    activation_fn = activation_fn,
                    act_on_output = act_on_output,
                )
                self.trunk_dims = linspace_int_list(self.width, self.trunk_depth, 1, False)
                #self.trunk_dims = [1] + [self.width] * self.trunk_depth        
                self.trunk_net = Ffn(
                    dims = self.trunk_dims, 
                    activation_fn = activation_fn,
                )
            elif branch_net == "CNN":
                self.branch_net = CnnBranch(
                    in_channels = in_size,
                    kernel_size = kernel_size,
                    dimension = 1,
                    activation_fn= activation_fn,
                    padding = padding,
                    stride = stride,
                    latent_channels = latent_channels, 
                    depth = branch_depth,
                    grid_resolution = grid_resolution,
                )
                length_new = latent_channels * self.branch_net.calc_out_shape()[0]
                self.trunk_dims = linspace_int_list(length_new, self.trunk_depth, 1, False)
                #self.trunk_dims = [1] + [self.width] * self.trunk_depth + [length_new]
                self.trunk_net = Ffn(
                    dims = self.trunk_dims, 
                    activation_fn = self.activation_fn,
                )
                dims_outffn = linspace_int_list(length_new, out_ffn_depth, 1, True)  
                self.out_Ffn = Ffn(
                    dims = dims_outffn,
                    #dims = [length_new] * out_ffn_depth  + [1], 
                    activation_fn = self.activation_fn,
                    act_on_output = False,  
                )
                
            elif branch_net == "ResNet":
                self.branch_net = ResNet1D(
                    in_channels= in_size,
                    out_channels= latent_channels,
                    block = "BasicBlock",
                    num_blocks = num_blocks,
                    hidden_channels= latent_channels,#TODO:change this!!!!!
                    coord_features= False,
                    padding= padding,
                )
                length_new = latent_channels * calc_resnet_out_shape(
                    in_shape = grid_resolution,
                    num_blocks = num_blocks,
                )[0]
                self.trunk_dims = linspace_int_list(length_new, self.trunk_depth, 1, False)
                #self.trunk_dims = [1] + [self.width] * self.trunk_depth + [length_new]
                self.trunk_net = Ffn(
                    dims = self.trunk_dims, 
                    activation_fn = self.activation_fn,
                )
                dims_outffn = linspace_int_list(length_new, out_ffn_depth, 1, True)  
                self.out_Ffn = Ffn(
                    dims = dims_outffn,
                    #dims = [length_new] * out_ffn_depth  + [1], 
                    activation_fn = self.activation_fn,
                    act_on_output = False,  
                )
            else:
                raise NotImplementedError(f"Unknown block: {branch_net}")
        else:
            raise ValueError(f"Unknown block type: {branch_net}")    
        
        
        self.bias = nn.Parameter(torch.zeros(1))  
        
    def forward(self, 
                x: Tensor) -> Tensor:
        shape = list(x.size())  # Save original shape [B, C, H]        
        length = shape[2]
        if self.query_idxs is None:
            self.query_idxs = torch.arange(
                length, 
                dtype=torch.long, 
                device=x.device
                ).unsqueeze(1)  # (H, 1)
            self.query_idxs = self.query_idxs.repeat(self.out_size, 1)  # (H * C_out , 1)
        x_trunk = self.query_idxs.float()
        x_trunk[:, 0] = x_trunk[:, 0]  / length #normalize to [0,1]
        
        if isinstance(self.branch_net_str, str):
            if self.branch_net_str == "FFN":
                flat_x= grid_to_points(x)  # (B, C * H )
                x_branch = self.branch_net(flat_x) # (B, p)
                x_branch = x_branch.unsqueeze(1)  # (B, 1, p)
                x_trunk = self.trunk_net(x_trunk)  # (H * C_out , p)
                x_trunk = x_trunk.unsqueeze(0)  # (1, H * C_out, p)
                x = torch.sum(x_branch * x_trunk, dim=-1) + self.bias  # (B, H * C_out)
            elif self.branch_net_str in ["CNN", "ResNet"]:
                x_branch = self.branch_net(x) # (B, C_hidden , H_new)
                x_branch = x_branch.view(shape[0], -1)  # (B, C_hidden*H_new)
                x_branch = x_branch.unsqueeze(1)  # (B, 1, C_hidden*H_new)
                x_trunk = self.trunk_net(x_trunk)  # (H * C_out , C_hidden*H_new)
                x_trunk = x_trunk.unsqueeze(0)  # (1, H * C_out, C_hidden*H_new)
                x = x_branch * x_trunk # (B, H * C_out, C_hidden*H_new)
                x = self.out_Ffn(x)  # (B, H * C_out, 1)
                x = x.squeeze(-1)  # (B, H * C_out)

        shape[1] = self.out_size  
        x = points_to_grid(x, shape)  # (B, C_out , H)
        return x

class AutoDeepONet2D(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        grid_resolution: Tuple[int],
        activation_fn: nn.Module,
        branch_net: str,
        branch_depth: int,
        trunk_depth: int,
        width: Optional[int] = 4,
        num_blocks: Optional[List[int]] = [1],
        act_on_output: Optional[bool] = False,
        kernel_size: Optional[int] = 3, 
        stride: Optional[int] = 1,
        out_ffn_depth: Optional[int] = 3,
        padding: Optional[int] = 1, 
        latent_channels: Optional[int] = 32,
        query_idxs: Optional[Tensor] = None,        
    ):
        super().__init__()
        self.out_size = out_size
        self.activation_fn = activation_fn
        self.branch_net_str = branch_net
        self.query_idxs = query_idxs
        self.width = width
        self.trunk_depth = trunk_depth        
        if isinstance(branch_net, str):
            if branch_net == "FFN":
                branch_dim = in_size * grid_resolution[0] * grid_resolution[1]
                self.branch_dims = [branch_dim] + [width] * branch_depth
                self.branch_net = Ffn(
                    dims = self.branch_dims,
                    activation_fn = activation_fn,
                    act_on_output = act_on_output,
                )
                self.trunk_dims = linspace_int_list(self.width, self.trunk_depth, 2, False)
                #self.trunk_dims = [2] + [self.width] * self.trunk_depth        
                self.trunk_net = Ffn(
                    dims = self.trunk_dims, 
                    activation_fn = activation_fn,
                )
            elif branch_net == "CNN":
                self.branch_net = CnnBranch(
                    in_channels = in_size,
                    kernel_size = kernel_size,
                    dimension = 2,
                    activation_fn= activation_fn,
                    padding = padding,
                    stride = stride,
                    latent_channels = latent_channels, 
                    depth = branch_depth,
                    grid_resolution = grid_resolution,
                )
                length_new = latent_channels * self.branch_net.calc_out_shape()[0] * self.branch_net.calc_out_shape()[1] 
                self.trunk_dims = linspace_int_list(length_new, self.trunk_depth, 2, False)
                #self.trunk_dims = [2] + [self.width] * self.trunk_depth + [length_new]
                self.trunk_net = Ffn(
                    dims = self.trunk_dims, 
                    activation_fn = self.activation_fn,
                )
                dims_outffn = linspace_int_list(length_new, out_ffn_depth, 1, True)  
                self.out_Ffn = Ffn(
                    dims = dims_outffn,
                    #dims = [length_new] * out_ffn_depth  + [1], 
                    activation_fn = self.activation_fn,
                    act_on_output = False,  
                )
                
            elif branch_net == "ResNet":
                self.branch_net = ResNet2D(
                    in_channels= in_size,
                    out_channels= latent_channels,
                    block = "BasicBlock",
                    num_blocks = num_blocks,
                    hidden_channels= latent_channels,#TODO:change this!!!!!
                    coord_features= False,
                    padding= padding,
                )
                length_new = latent_channels * calc_resnet_out_shape(
                    in_shape = grid_resolution,
                    num_blocks = num_blocks)[0]* calc_resnet_out_shape(
                                                    in_shape = grid_resolution,
                                                    num_blocks = num_blocks)[1]
                self.trunk_dims = linspace_int_list(length_new, self.trunk_depth, 2, False)
                #self.trunk_dims = [2] + [self.width] * self.trunk_depth + [length_new]
                self.trunk_net = Ffn(
                    dims = self.trunk_dims, 
                    activation_fn = self.activation_fn,
                )
                dims_outffn = linspace_int_list(length_new, out_ffn_depth, 1, True)  
                self.out_Ffn = Ffn(
                    dims = dims_outffn,
                    #dims = [length_new] * out_ffn_depth  + [1], 
                    activation_fn = self.activation_fn,
                    act_on_output = False,  
                )
            else:
                raise NotImplementedError(f"Unknown block: {branch_net}")
        else:
            raise ValueError(f"Unknown block type: {branch_net}")    
        
        
        self.bias = nn.Parameter(torch.zeros(1))  
        
    def forward(self, 
                x: Tensor) -> Tensor:
        shape = list(x.size())  # [B, C, H, W]
        height, width = shape[2], shape[3]
        if self.query_idxs is None:
            self.query_idxs = torch.tensor(
                list(product(range(height), range(width))),
                dtype=torch.long,
                device=x.device,
            )  # (H * W, 2)
            self.query_idxs = self.query_idxs.repeat(self.out_size, 1)  # (H * W * C_out, 2)
        x_trunk = self.query_idxs.float()
        x_trunk[:, 0] = x_trunk[:, 0] / height  # normalize H
        x_trunk[:, 1] = x_trunk[:, 1] / width   # normalize W
        
        if isinstance(self.branch_net_str, str):
            if self.branch_net_str == "FFN":
                flat_x= grid_to_points(x)  # (B, C * H * W)
                x_branch = self.branch_net(flat_x) # (B, p)
                x_branch = x_branch.unsqueeze(1)  # (B, 1, p)
                x_trunk = self.trunk_net(x_trunk)  # (H * W * C_out , p)
                x_trunk = x_trunk.unsqueeze(0)  # (1, H * W * C_out, p)
                x = torch.sum(x_branch * x_trunk, dim=-1) + self.bias  # (B, H * W * C_out)
            elif self.branch_net_str in ["CNN", "ResNet"]:
                x_branch = self.branch_net(x)  # (B, C_hidden, H_new, W_new)
                x_branch = x_branch.view(shape[0], -1)  # (B, C_hidden*H_new*W_new)
                x_branch = x_branch.unsqueeze(1)  # (B, 1, C_hidden*H_new*W_new)
                x_trunk = self.trunk_net(x_trunk)  # (H*W*C_out, C_hidden*H_new*W_new)
                x_trunk = x_trunk.unsqueeze(0)  # (1, H*W*C_out, C_hidden*H_new*W_new)
                x = x_branch * x_trunk # (B, H*W*C_out, C_hidden*H_new*W_new)
                x = self.out_Ffn(x)   # (B, H*W*C_out, 1)
                x = x.squeeze(-1)  # (B, H*W*C_out)

        shape[1] = self.out_size  
        x = points_to_grid(x, shape)  # (B, C_out, H, W)
        return x


class AutoDeepONet3D(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        grid_resolution: Tuple[int],
        activation_fn: nn.Module,
        branch_net: str,
        branch_depth: int,
        trunk_depth: int,
        width: Optional[int] = 4,
        num_blocks: Optional[List[int]] = [1],
        act_on_output: Optional[bool] = False,
        kernel_size: Optional[int] = 3, 
        stride: Optional[int] = 1,
        out_ffn_depth: Optional[int] = 3,
        padding: Optional[int] = 1, 
        latent_channels: Optional[int] = 32,
        query_idxs: Optional[Tensor] = None,        
    ):
        super().__init__()
        self.out_size = out_size
        self.activation_fn = activation_fn
        self.branch_net_str = branch_net
        self.query_idxs = query_idxs
        self.width = width
        self.trunk_depth = trunk_depth        
        if isinstance(branch_net, str):
            if branch_net == "FFN":
                branch_dim = in_size * grid_resolution[0] * grid_resolution[1] * grid_resolution[2]
                self.branch_dims = [branch_dim] + [width] * branch_depth
                self.branch_net = Ffn(
                    dims = self.branch_dims,
                    activation_fn = activation_fn,
                    act_on_output = act_on_output,
                )
                self.trunk_dims = linspace_int_list(self.width, self.trunk_depth, 3, False)
                #self.trunk_dims = [3] + [self.width] * self.trunk_depth        
                self.trunk_net = Ffn(
                    dims = self.trunk_dims, 
                    activation_fn = activation_fn,
                )
            elif branch_net == "CNN":
                self.branch_net = CnnBranch(
                    in_channels = in_size,
                    kernel_size = kernel_size,
                    dimension = 3,
                    activation_fn= activation_fn,
                    padding = padding,
                    stride = stride,
                    latent_channels = latent_channels, 
                    depth = branch_depth,
                    grid_resolution = grid_resolution,
                )
                length_new = latent_channels * self.branch_net.calc_out_shape()[0] * self.branch_net.calc_out_shape()[1] * self.branch_net.calc_out_shape()[2]
                self.trunk_dims = linspace_int_list(length_new, self.trunk_depth, 3, False)
                #self.trunk_dims = [3] + [self.width] * self.trunk_depth + [length_new]
                self.trunk_net = Ffn(
                    dims = self.trunk_dims, 
                    activation_fn = self.activation_fn,
                )
                dims_outffn = linspace_int_list(length_new, out_ffn_depth, 1, True)  
                self.out_Ffn = Ffn(
                    dims = dims_outffn,
                    #dims = [length_new] * out_ffn_depth  + [1], 
                    activation_fn = self.activation_fn,
                    act_on_output = False,  
                )
                
            elif branch_net == "ResNet":
                self.branch_net = ResNet3D(
                    in_channels= in_size,
                    out_channels= latent_channels,
                    block = "BasicBlock",
                    num_blocks = num_blocks,
                    hidden_channels= latent_channels,#TODO:change this!!!!!
                    coord_features= False,
                    padding= padding,
                )
                length_new = latent_channels * calc_resnet_out_shape(
                    in_shape = grid_resolution,
                    num_blocks = num_blocks)[0]* calc_resnet_out_shape(
                                                    in_shape = grid_resolution,
                                                    num_blocks = num_blocks)[1] * calc_resnet_out_shape(
                                                    in_shape = grid_resolution,
                                                    num_blocks = num_blocks)[2]
                self.trunk_dims = linspace_int_list(length_new, self.trunk_depth, 3, False)
                #self.trunk_dims = [3] + [self.width] * self.trunk_depth + [length_new]
                self.trunk_net = Ffn(
                    dims = self.trunk_dims, 
                    activation_fn = self.activation_fn,
                )
                dims_outffn = linspace_int_list(length_new, out_ffn_depth, 1, True)  
                self.out_Ffn = Ffn(
                    dims = dims_outffn,
                    #dims = [length_new] * out_ffn_depth  + [1], 
                    activation_fn = self.activation_fn,
                    act_on_output = False,  
                )
            else:
                raise NotImplementedError(f"Unknown block: {branch_net}")
        else:
            raise ValueError(f"Unknown block type: {branch_net}")    
        
        
        self.bias = nn.Parameter(torch.zeros(1))  

    def forward(self, x: Tensor) -> Tensor:
        shape = list(x.size())  # [B, C, D, H, W]
        depth, height, width = shape[2], shape[3], shape[4]
        if self.query_idxs is None:
            self.query_idxs = torch.tensor(
                list(product(range(depth), range(height), range(width))),
                dtype=torch.long,
                device=x.device,
            )  # (D * H * W, 3)
            self.query_idxs = self.query_idxs.repeat(self.out_size, 1)  # (D*H*W*C_out, 3)
        x_trunk = self.query_idxs.float()
        x_trunk[:, 0] = x_trunk[:, 0] / depth   # normalize D
        x_trunk[:, 1] = x_trunk[:, 1] / height  # normalize H
        x_trunk[:, 2] = x_trunk[:, 2] / width   # normalize W

        if isinstance(self.branch_net_str, str):
            if self.branch_net_str == "FFN":
                flat_x = x.reshape(shape[0], -1)  # (B, C*D*H*W)
                x_branch = self.branch_net(flat_x)  # (B, p)
                x_branch = x_branch.unsqueeze(1)  # (B, 1, p)
                x_trunk = self.trunk_net(x_trunk)  # (D*H*W*C_out, p)
                x_trunk = x_trunk.unsqueeze(0)  # (1, D*H*W*C_out, p)
                x = torch.sum(x_branch * x_trunk, dim=-1) + self.bias  # (B, D*H*W*C_out)
            elif self.branch_net_str in ["CNN", "ResNet"]:
                x_branch = self.branch_net(x)  # (B, C_hidden, D_new, H_new, W_new)
                x_branch = x_branch.view(shape[0], -1)  # (B, C_hidden*D_new*H_new*W_new)
                x_branch = x_branch.unsqueeze(1)  # (B, 1, C_hidden*D_new*H_new*W_new)
                x_trunk = self.trunk_net(x_trunk)  # (D*H*W*C_out, C_hidden*D_new*H_new*W_new)
                x_trunk = x_trunk.unsqueeze(0)  # (1, D*H*W*C_out, C_hidden*D_new*H_new*W_new)
                x = x_branch * x_trunk  # (B, D*H*W*C_out, C_hidden*D_new*H_new*W_new)
                x = self.out_Ffn(x)  # (B, D*H*W*C_out, 1)
                x = x.squeeze(-1)  # (B, D*H*W*C_out)

        shape[1] = self.out_size
        x = x.view(shape[0], self.out_size, depth, height, width)  # (B, C_out, D, H, W)
        return x    
 