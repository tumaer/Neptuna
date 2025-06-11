from typing import List, Optional, Union, Callable, Tuple
from itertools import product
import torch
import torch.nn as nn
from torch import Tensor
from utils import activation_func
from .deeponet_utils import Ffn, CnnBranch
from models.ResNet.resnet import ResNet1D, ResNet2D, ResNet3D

def grid_to_points(value: Tensor) -> Tuple[Tensor, List[int]]:
    """
    Convert from grid-based (XD) representation to point-based representation.

    Parameters
    ----------
    value : Tensor
        Input tensor of shape (B, C, X).

    Returns
    -------
    Tuple
        - Tensor of shape (B, C*X).
        - Original shape as a list [B, C, X].
    """
    output = value.reshape(value.size(0), -1)  # Reshape to (B, C*X)
    return output

def points_to_grid(value: Tensor, shape: List[int]) -> Tensor:
    """
    Convert from point-based representation back to grid-based representation.

    Parameters
    ----------
    value : Tensor
        Input tensor of shape (B, C*X).
    shape : List[int]
        Original shape as [B, C, X].

    Returns
    -------
    Tensor
        Restored tensor of shape (B, C, X).
    """
    output = value.reshape(shape)  # Reshape back to (B, C, X)
    return output



class AutoDeepONet(nn.Module):
    """
    Auto-regressive DeepONet for CFD.

    Args:
        in_channels : int
            Number of input fields
        out_channels : int
            Number of output fields
        resolution : Tuple[int]
            Resolution of the input fields
        dimension : int
            Dimension of the input fields (1D, 2D, or 3D)
        branch_net : str
            Type of branch network (FFN, CNN, ResNet)
        query_idxs : Optional[Tensor]
            Indices for the trunk network
        sequence_info (List[List[int]]):
            sequence_info[0][0]: input_seq_len, sequence_info[0][1]: label_seq_len,
            sequence_info[0][2]: input_sequence_stride, sequence_info[0][3]: label_sequence_stride  
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
    
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resolution: Tuple[int],
        dimension: int,
        branch_net: str = "FFN",
        query_idxs: Optional[Tensor] = None,
        sequence_info: Optional[List[List[int]]] = [[1,1,1,1]],
        #num_label_samples: int = 1000,
        branch_depth: int = 4,
        trunk_depth: int = 4,
        width: int = 100,
        activation_fn: str = "gelu",
        #act_norm: bool = False,
        act_on_output: bool = False,        
        kernel_size: Optional[int] = 3, 
        padding: Optional[int] = 1, 
        hidden_channels: Optional[int] = 32,
    ):

        super().__init__()
        self.in_channels = in_channels * sequence_info[0][0] 
        self.out_channels = out_channels * sequence_info[0][1]
        self.branch_depth = branch_depth
        self.trunk_depth = trunk_depth
        self.width = width
        self.dimension = dimension
        self.activation: nn.Module = activation_func.get_activation(activation_fn)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation_fn} not implemented")

        if isinstance(branch_net, str):
            if branch_net == "FFN":
                self.AutoDeepONet = self.build_AutoDeepONet()(
                    in_channels = self.in_channels,
                    out_channels = self.out_channels,
                    resolution = resolution,
                    activation_fn = self.activation,
                    branch_net = branch_net,
                    branch_depth = branch_depth,
                    trunk_depth = trunk_depth,
                    width = width,
                    act_on_output = act_on_output,
                    query_idxs = query_idxs,
                )
            elif branch_net == "CNN":
                self.AutoDeepONet = self.build_AutoDeepONet()(
                    in_channels = self.in_channels,
                    out_channels = self.out_channels,
                    resolution = resolution,
                    activation_fn = self.activation,
                    branch_net = branch_net,
                    branch_depth = branch_depth,
                    trunk_depth = trunk_depth,
                    width = width,
                    query_idxs = query_idxs,
                    kernel_size = kernel_size,
                    padding = padding,
                    hidden_channels = hidden_channels,
                )
            elif branch_net == "ResNet":
                self.AutoDeepONet = self.build_AutoDeepONet()(
                    in_channels = self.in_channels,
                    out_channels = self.out_channels,
                    resolution = resolution,
                    activation_fn = self.activation,
                    branch_net = branch_net,
                    branch_depth = branch_depth,
                    trunk_depth = trunk_depth,
                    padding= padding,
                    #block = "BasicBlock",
                    #num_blocks = [1,1,1],
                )

    def forward(self, 
                input_data: Tensor,
                labels: Tensor) -> Tensor: #NOTE: Vimp: forward SHOULD always have the arguments EXACTLY named as "input_data" and "labels", 
                                           #else the data collator will remove them. 
        
        #reshape input into [batch, in_channel, grid_x, grid_y, ...]
        #NOTE: input and output fields need not be necessarily the same.
        batch, input_seq, input_fields, *spatial = input_data.shape
        input_data=input_data.reshape(batch, input_seq * input_fields, *spatial)
        
        y = self.AutoDeepONet(input_data)  # (B, C_out, X)
     
        # Reshape the prediction to match the labels shape
        batch, output_seq, output_fields, *spatial = labels.shape
        y = y.reshape(batch, output_seq, output_fields, *spatial)
        
        return y, labels
    
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
        in_channels: int,
        out_channels: int,
        resolution: Tuple[int],
        activation_fn: nn.Module,
        branch_net: str,
        branch_depth: int,
        trunk_depth: int,
        width: Optional[int] = 4,
        act_on_output: Optional[bool] = False,
        kernel_size: Optional[int] = 3, 
        padding: Optional[int] = 1, 
        hidden_channels: Optional[int] = 32,
        query_idxs: Optional[Tensor] = None,        
    ):
        super().__init__()
        self.out_channels = out_channels
        self.activation_fn = activation_fn
        self.branch_net_str = branch_net
        self.query_idxs = query_idxs
        self.width = width
        self.trunk_depth = trunk_depth        
        if isinstance(branch_net, str):
            if branch_net == "FFN":
                branch_dim = in_channels * resolution[0]
                self.branch_dims = [branch_dim] + [width] * branch_depth
                self.branch_net = Ffn(
                    dims = self.branch_dims,
                    activation_fn = activation_fn,
                    act_on_output = act_on_output,
                )
            elif branch_net == "CNN":
                self.branch_net = CnnBranch(
                    in_channels = in_channels,
                    kernel_size = kernel_size,
                    dimension = 1,
                    activation_fn= activation_fn,
                    padding = padding,
                    hidden_channels = hidden_channels,
                    depth = branch_depth,
                )
            elif branch_net == "ResNet":
                self.branch_net = ResNet1D(
                    in_channels= in_channels,
                    out_channels= hidden_channels,
                    block = "BasicBlock",
                    num_blocks = [1],
                    hidden_channels= hidden_channels,
                    coord_features= False,
                    padding= padding,
                )
            else:
                raise NotImplementedError(f"Unknown block: {branch_net}")
        else:
            raise ValueError(f"Unknown block type: {branch_net}")    
        
        self.trunk_dims = [1] + [width] * trunk_depth
        
        self.trunk_net = Ffn(
            dims = self.trunk_dims, 
            activation_fn = activation_fn,
        )
        
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
            self.query_idxs = self.query_idxs.repeat(self.out_channels, 1)  # (H * C_out , 1)
        x_trunk = self.query_idxs.float()
        x_trunk[:, 0] = x_trunk[:, 0]  / length #normalize to [0,1]
        
        if isinstance(self.branch_net_str, str):
            if self.branch_net_str == "FFN":
                flat_x= grid_to_points(x)  # (B, C * H )
                x_branch = self.branch_net(flat_x) # (B, p)
                x_branch = x_branch.unsqueeze(1)  # (B, 1, p)
                x_trunk = self.trunk_net(x_trunk)  # (H * C_out , p)
                x_trunk = x_trunk.unsqueeze(0)  # (1, H * C_out, p)
                y = torch.sum(x_branch * x_trunk, dim=-1) + self.bias  # (B, H * C_out)
            elif self.branch_net_str in ["CNN", "ResNet"]:
                x_branch = self.branch_net(x) # (B, C_hidden , H_new)
                x_branch = x_branch.view(shape[0], -1)  # (B, C_hidden*H_new)
                x_branch = x_branch.unsqueeze(1)  # (B, 1, C_hidden*H_new)
                length_new = x_branch.size(2)
                self.trunk_dims = [1] + [self.width] * self.trunk_depth + [length_new]
                self.trunk_net = Ffn(
                    dims = self.trunk_dims, 
                    activation_fn = self.activation_fn,
                ).to(x.device)
                x_trunk = self.trunk_net(x_trunk)  # (X * C_out , C_hidden*H_new)
                x_trunk = x_trunk.unsqueeze(0)  # (1, X * C_out, C_hidden*H_new)
                self.out_Ffn = Ffn(
                    dims = [length_new]*3  + [1], # 3 fix value??
                    activation_fn = self.activation_fn,
                    act_on_output = False,  
                ).to(x.device)  # should i do this here? 
                y = x_branch * x_trunk # (B, H * C_out, C_hidden*H_new)
                y = self.out_Ffn(y)  # (B, H * C_out, 1)
                y = y.squeeze(-1)  # (B, H * C_out)

        shape[1] = self.out_channels   
        y = points_to_grid(y, shape)  # (B, C_out , H)
        return y


class AutoDeepONet2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resolution: Tuple[int, int],
        activation_fn: nn.Module,
        branch_net: str,
        branch_depth: int,
        trunk_depth: int,
        width: Optional[int] = 4,
        act_on_output: Optional[bool] = False,
        kernel_size: Optional[int] = 3, 
        padding: Optional[int] = 1, 
        hidden_channels: Optional[int] = 32,
        query_idxs: Optional[Tensor] = None,        
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.activation_fn = activation_fn
        self.branch_net_str = branch_net
        self.query_idxs = query_idxs
        self.width = width
        self.trunk_depth = trunk_depth        
        if isinstance(branch_net, str):
            if branch_net == "FFN":
                branch_dim = in_channels * resolution[0] * resolution[1]
                self.branch_dims = [branch_dim] + [width] * branch_depth
                self.branch_net = Ffn(
                    dims = self.branch_dims,
                    activation_fn = activation_fn,
                    act_on_output = act_on_output,
                )
            elif branch_net == "CNN":
                self.branch_net = CnnBranch(
                    in_channels = in_channels,
                    kernel_size = kernel_size,
                    dimension = 2,
                    activation_fn= activation_fn,
                    padding = padding,
                    hidden_channels = hidden_channels,
                    depth = branch_depth,
                )
            elif branch_net == "ResNet":
                self.branch_net = ResNet2D(
                    in_channels= in_channels,
                    out_channels= hidden_channels,
                    block = "BasicBlock",
                    num_blocks = [1],
                    hidden_channels= hidden_channels,
                    coord_features= False,
                    padding= padding,
                )
            else:
                raise NotImplementedError(f"Unknown block: {branch_net}")
        else:
            raise ValueError(f"Unknown block type: {branch_net}")    
        
        self.trunk_dims = [2] + [width] * trunk_depth
        
        self.trunk_net = Ffn(
            dims = self.trunk_dims, 
            activation_fn = activation_fn,
        )
        
        self.bias = nn.Parameter(torch.zeros(1))  
        
    def forward(self, x: Tensor) -> Tensor:
        shape = list(x.size())  # [B, C, H, W]
        height, width = shape[2], shape[3]
        if self.query_idxs is None:
            self.query_idxs = torch.tensor(
                list(product(range(height), range(width))),
                dtype=torch.long,
                device=x.device,
            )  # (H * W, 2)
            self.query_idxs = self.query_idxs.repeat(self.out_channels, 1)  # (H * W * C_out, 2)
        x_trunk = self.query_idxs.float()
        x_trunk[:, 0] = x_trunk[:, 0] / height  # normalize H
        x_trunk[:, 1] = x_trunk[:, 1] / width   # normalize W

        if isinstance(self.branch_net_str, str):
            if self.branch_net_str == "FFN":
                flat_x = x.reshape(shape[0], -1)  # (B, C*H*W)
                x_branch = self.branch_net(flat_x)  # (B, p)
                x_branch = x_branch.unsqueeze(1)  # (B, 1, p)
                x_trunk = self.trunk_net(x_trunk)  # (H*W*C_out, p)
                x_trunk = x_trunk.unsqueeze(0)  # (1, H*W*C_out, p)
                y = torch.sum(x_branch * x_trunk, dim=-1) + self.bias  # (B, H*W*C_out)
            elif self.branch_net_str in ["CNN", "ResNet"]:
                x_branch = self.branch_net(x)  # (B, C_hidden, H_new, W_new)
                x_branch = x_branch.view(shape[0], -1)  # (B, C_hidden*H_new*W_new)
                x_branch = x_branch.unsqueeze(1)  # (B, 1, C_hidden*H_new*W_new)
                length_new = x_branch.size(2)
                self.trunk_dims = [2] + [self.width] * self.trunk_depth + [length_new]
                self.trunk_net = Ffn(
                    dims=self.trunk_dims,
                    activation_fn=self.activation_fn,
                ).to(x.device)
                x_trunk = self.trunk_net(x_trunk)  # (H*W*C_out, C_hidden*H_new*W_new)
                x_trunk = x_trunk.unsqueeze(0)  # (1, H*W*C_out, C_hidden*H_new*W_new)
                self.out_Ffn = Ffn(
                    dims=[length_new]*3 + [1],
                    activation_fn=self.activation_fn,
                    act_on_output=False,
                ).to(x.device)
                y = x_branch * x_trunk  # (B, H*W*C_out, C_hidden*H_new*W_new)
                y = self.out_Ffn(y)  # (B, H*W*C_out, 1)
                y = y.squeeze(-1)  # (B, H*W*C_out)

        shape[1] = self.out_channels
        y = y.view(shape[0], self.out_channels, height, width)  # (B, C_out, H, W)
        return y
    

#not tested for CNN and ResNet branch nets
class AutoDeepONet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resolution: Tuple[int, int, int],
        activation_fn: nn.Module,
        branch_net: str,
        branch_depth: int,
        trunk_depth: int,
        width: Optional[int] = 4,
        act_on_output: Optional[bool] = False,
        kernel_size: Optional[int] = 3, 
        padding: Optional[int] = 1, 
        hidden_channels: Optional[int] = 32,
        query_idxs: Optional[Tensor] = None,        
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.activation_fn = activation_fn
        self.branch_net_str = branch_net
        self.query_idxs = query_idxs
        self.width = width
        self.trunk_depth = trunk_depth        
        if isinstance(branch_net, str):
            if branch_net == "FFN":
                branch_dim = in_channels * resolution[0] * resolution[1] * resolution[2]
                self.branch_dims = [branch_dim] + [width] * branch_depth
                self.branch_net = Ffn(
                    dims = self.branch_dims,
                    activation_fn = activation_fn,
                    act_on_output = act_on_output,
                )
            elif branch_net == "CNN":
                self.branch_net = CnnBranch(
                    in_channels = in_channels,
                    kernel_size = kernel_size,
                    dimension = 3,
                    activation_fn= activation_fn,
                    padding = padding,
                    hidden_channels = hidden_channels,
                    depth = branch_depth,
                )
            elif branch_net == "ResNet":
                self.branch_net = ResNet3D(
                    in_channels= in_channels,
                    out_channels= hidden_channels,
                    block = "BasicBlock",
                    num_blocks = [1],
                    hidden_channels= hidden_channels,
                    coord_features= False,
                    padding= padding,
                )
            else:
                raise NotImplementedError(f"Unknown block: {branch_net}")
        else:
            raise ValueError(f"Unknown block type: {branch_net}")    
        
        self.trunk_dims = [3] + [width] * trunk_depth
        
        self.trunk_net = Ffn(
            dims = self.trunk_dims, 
            activation_fn = activation_fn,
        )
        
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
            self.query_idxs = self.query_idxs.repeat(self.out_channels, 1)  # (D*H*W*C_out, 3)
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
                y = torch.sum(x_branch * x_trunk, dim=-1) + self.bias  # (B, D*H*W*C_out)
            elif self.branch_net_str in ["CNN", "ResNet"]:
                x_branch = self.branch_net(x)  # (B, C_hidden, D_new, H_new, W_new)
                x_branch = x_branch.view(shape[0], -1)  # (B, C_hidden*D_new*H_new*W_new)
                x_branch = x_branch.unsqueeze(1)  # (B, 1, C_hidden*D_new*H_new*W_new)
                length_new = x_branch.size(2)
                self.trunk_dims = [3] + [self.width] * self.trunk_depth + [length_new]
                self.trunk_net = Ffn(
                    dims=self.trunk_dims,
                    activation_fn=self.activation_fn,
                ).to(x.device)
                x_trunk = self.trunk_net(x_trunk)  # (D*H*W*C_out, C_hidden*D_new*H_new*W_new)
                x_trunk = x_trunk.unsqueeze(0)  # (1, D*H*W*C_out, C_hidden*D_new*H_new*W_new)
                self.out_Ffn = Ffn(
                    dims=[length_new]*3 + [1],
                    activation_fn=self.activation_fn,
                    act_on_output=False,
                ).to(x.device)
                y = x_branch * x_trunk  # (B, D*H*W*C_out, C_hidden*D_new*H_new*W_new)
                y = self.out_Ffn(y)  # (B, D*H*W*C_out, 1)
                y = y.squeeze(-1)  # (B, D*H*W*C_out)

        shape[1] = self.out_channels
        y = y.view(shape[0], self.out_channels, depth, height, width)  # (B, C_out, D, H, W)
        return y    