from itertools import product
import torch
import torch.nn as nn
from torch import Tensor
from models.ResNet.resnet_utils import ResNetConfig
from utils import activation_func
from utils.grid_utils import oned_meshgrid, twod_meshgrid, threed_meshgrid
from .deeponet_utils import FFN, CnnBranch, grid_to_points, points_to_grid, calc_resnet_out_shape, linspace_int_list
from models.ResNet.resnet import ResNet1D, ResNet2D, ResNet3D
from transformers import PreTrainedModel

class AutoDeepONet(PreTrainedModel):
    """
    Auto-regressive DeepONet for CFD.
    """
    main_input_name = "input_data"  
    conditioning_input_name = "conditioning_input_data"
    def __init__(self, config):
        super().__init__(config)
        
        self.config = config

        activation_fn: nn.Module = activation_func.get_activation(config.activation_fn_name)
        if activation_fn is None:
            raise NotImplementedError(f"Activation {config.activation_fn_name} not implemented")

        if isinstance(config.branch_net_str, str):
            if config.branch_net_str == "FFN":
                self.AutoDeepONet = self.build_AutoDeepONet()(config=config, activation_fn=activation_fn)
            elif config.branch_net_str == "CNN":
                self.AutoDeepONet = self.build_AutoDeepONet()(config=config, activation_fn=activation_fn)
            elif config.branch_net_str == "ResNet": #TODO:add kernal_size and stride to ResNet
                self.AutoDeepONet = self.build_AutoDeepONet()(config=config, activation_fn=activation_fn)

    def forward(self, 
                input_data: Tensor,
                **kwargs) -> Tensor: 

        if "conditioning_input_data" in kwargs:
            #NOTE: Conditioning data can be passed into a conv network before concatination with input_data.
            conditioning_input_data = kwargs["conditioning_input_data"]
            input_data = torch.cat([input_data, conditioning_input_data], dim=2)
        else:
            conditioning_input_data = None

        batch, input_seq, input_fields, *spatial = input_data.shape
        input_data=input_data.reshape(batch, input_seq * input_fields, *spatial)
        
        x = self.AutoDeepONet(input_data)  # (B, C_out, X)
     
        return x
    
    def build_AutoDeepONet(self):
        """Get the ResNet encoder based on the model dimensionality"""
        if self.config.dimension == 1:
            return AutoDeepONet1D
        elif self.config.dimension == 2:
            return AutoDeepONet2D
        elif self.config.dimension == 3:
            return AutoDeepONet3D
        else:
            raise NotImplementedError(
                "Invalid dimensionality. Only 1D, 2D, 3D ResNet implemented"
            )
    
class AutoDeepONet1D(PreTrainedModel):
    def __init__(self, config, activation_fn: nn.Module):
        super().__init__(config)    

        self.activation_fn = activation_fn
        
        if isinstance(config.branch_net_str, str):
            if config.branch_net_str == "FFN":
                branch_dim = (config.in_size + config.dimension if config.coord_features else 0) * config.grid_resolution[0]
                self.branch_dims = [branch_dim] + [config.width] * config.branch_depth
                self.branch_net = FFN(
                    dims = self.branch_dims,
                    activation_fn = self.activation_fn,
                    act_on_output = config.act_on_output
                )
                self.trunk_dims = linspace_int_list(config.width, config.trunk_depth, 1, False)
                #self.trunk_dims = [1] + [self.width] * self.trunk_depth        
                self.trunk_net = FFN(
                    dims = self.trunk_dims, 
                    activation_fn = self.activation_fn
                )
            elif config.branch_net_str == "CNN":
                self.branch_net = CnnBranch(
                    in_channels = config.in_size,
                    kernel_size = config.kernel_size,
                    dimension = 1,
                    activation_fn = self.activation_fn,
                    padding = config.padding,
                    stride = config.stride,
                    latent_channels = config.latent_channels, 
                    depth = config.branch_depth,
                    grid_resolution = config.grid_resolution,
                )
                length_new = config.latent_channels * self.branch_net.calc_out_shape()[0]
                self.trunk_dims = linspace_int_list(length_new, config.trunk_depth, 1, False)
                #self.trunk_dims = [1] + [self.width] * self.trunk_depth + [length_new]
                self.trunk_net = FFN(
                    dims = self.trunk_dims, 
                    activation_fn = self.activation_fn
                )
                dims_outffn = linspace_int_list(length_new, config.out_ffn_depth, 1, True)  
                self.out_Ffn = FFN(
                    dims = dims_outffn,
                    #dims = [length_new] * out_ffn_depth  + [1], 
                    activation_fn = self.activation_fn,
                    act_on_output = False
                )
                
            elif config.branch_net_str == "ResNet":
                resNetConfig = ResNetConfig(
                    in_size= config.in_size,
                    out_size= config.latent_channels,
                    block = config.ResNet_block,
                    num_blocks = config.num_blocks,
                    latent_channels= config.latent_channels,
                    coord_features= config.coord_features,
                    dimension = config.dimension,
                    padding = 0,
                    include_input_seq_len = False
                    )
                self.branch_net = ResNet1D(config=resNetConfig)
                length_new = config.latent_channels * calc_resnet_out_shape(
                    in_shape = config.grid_resolution,
                    num_blocks = config.num_blocks,
                    if_maxpool = False if config.ResNet_block == "BasicBlock" else True
                )[0]
                assert config.trunk_depth > 1, "Trunk depth must be greater than 1 for DeepONet with ResNet branch net"
                self.trunk_dims = linspace_int_list(length_new, config.trunk_depth, 1, False)
                #self.trunk_dims = [1] + [self.width] * self.trunk_depth + [length_new]
                self.trunk_net = FFN(
                    dims = self.trunk_dims, 
                    activation_fn = self.activation_fn
                )
                dims_outffn = linspace_int_list(length_new, config.out_ffn_depth, 1, True)  
                self.out_Ffn = FFN(
                    dims = dims_outffn,
                    #dims = [length_new] * out_ffn_depth  + [1], 
                    activation_fn = self.activation_fn,
                    act_on_output = False
                )
            else:
                raise NotImplementedError(f"Unknown block: {config.branch_net_str}")
        else:
            raise ValueError(f"Unknown block type: {config.branch_net_str}")    
        
        
        self.bias = nn.Parameter(torch.zeros(1))  
        
    def forward(self, 
                x: Tensor) -> Tensor:
        shape = list(x.size())  # Save original shape [B, C, H]        
        length = shape[2]
        if self.config.query_idxs is None:
            self.query_idxs = torch.arange(
                length, 
                dtype=torch.long, 
                device=x.device
                ).unsqueeze(1)  # (H, 1)
            self.query_idxs = self.query_idxs.repeat(self.config.out_size, 1)  # (H * C_out , 1)
        x_trunk = self.query_idxs.float()
        x_trunk[:, 0] = x_trunk[:, 0]  / length #normalize to [0,1]
        
        if isinstance(self.config.branch_net_str, str):
            if self.config.branch_net_str == "FFN":
                if self.config.coord_features: 
                    coord_feat = oned_meshgrid(list(x.shape), x.device)
                    x = torch.cat((x, coord_feat), dim=1)
                flat_x = grid_to_points(x)  # (B, C * H )
                x_branch = self.branch_net(flat_x) # (B, p)
                x_branch = x_branch.unsqueeze(1)  # (B, 1, p)
                x_trunk = self.trunk_net(x_trunk)  # (H * C_out , p)
                x_trunk = x_trunk.unsqueeze(0)  # (1, H * C_out, p)
                x = torch.sum(x_branch * x_trunk, dim=-1) + self.bias  # (B, H * C_out)
            elif self.config.branch_net_str in ["CNN", "ResNet"]:
                x_branch = self.branch_net(x) # (B, C_hidden , H_new)
                x_branch = x_branch.view(shape[0], -1)  # (B, C_hidden*H_new)
                x_branch = x_branch.unsqueeze(1)  # (B, 1, C_hidden*H_new)
                x_trunk = self.trunk_net(x_trunk)  # (H * C_out , C_hidden*H_new)
                x_trunk = x_trunk.unsqueeze(0)  # (1, H * C_out, C_hidden*H_new)
                x = x_branch * x_trunk # (B, H * C_out, C_hidden*H_new)
                x = self.out_Ffn(x)  # (B, H * C_out, 1)
                x = x.squeeze(-1)  # (B, H * C_out)

        shape[1] = self.config.out_size  
        x = points_to_grid(x, shape)  # (B, C_out , H)
        return x

class AutoDeepONet2D(PreTrainedModel):
    def __init__(self, config, activation_fn: nn.Module):
        super().__init__(config)

        self.activation_fn = activation_fn
            
        if isinstance(config.branch_net_str, str):
            if config.branch_net_str == "FFN":
                branch_dim = (config.in_size + config.dimension if config.coord_features else 0) * config.grid_resolution[0] * config.grid_resolution[1]
                self.branch_dims = [branch_dim] + [config.width] * config.branch_depth
                self.branch_net = FFN(
                    dims = self.branch_dims,
                    activation_fn = self.activation_fn,
                    act_on_output = config.act_on_output
                )
                self.trunk_dims = linspace_int_list(self.config.width, self.config.trunk_depth, 2, False)
                #self.trunk_dims = [2] + [self.width] * self.trunk_depth        
                self.trunk_net = FFN(
                    dims = self.trunk_dims, 
                    activation_fn = self.activation_fn
                )
            elif config.branch_net_str == "CNN":
                self.branch_net = CnnBranch(
                    in_channels = config.in_size,
                    kernel_size = config.kernel_size,
                    dimension = 2,
                    activation_fn= self.activation_fn,
                    padding = config.padding,
                    stride = config.stride,
                    latent_channels = config.latent_channels, 
                    depth = config.branch_depth,
                    grid_resolution = config.grid_resolution,
                )
                length_new = config.latent_channels * self.branch_net.calc_out_shape()[0] * self.branch_net.calc_out_shape()[1] 
                self.trunk_dims = linspace_int_list(length_new, self.config.trunk_depth, 2, False)
                #self.trunk_dims = [2] + [self.width] * self.trunk_depth + [length_new]
                self.trunk_net = FFN(
                    dims = self.trunk_dims, 
                    activation_fn = self.activation_fn,
                )
                dims_outffn = linspace_int_list(length_new, config.out_ffn_depth, 1, True)  
                self.out_Ffn = FFN(
                    dims = dims_outffn,
                    #dims = [length_new] * out_ffn_depth  + [1], 
                    activation_fn = self.activation_fn,
                    act_on_output = False,  
                )
                
            elif config.branch_net_str == "ResNet":
                resNetConfig = ResNetConfig(
                    in_size= config.in_size,
                    out_size= config.latent_channels,
                    block = config.ResNet_block,
                    num_blocks = config.num_blocks,
                    latent_channels= config.latent_channels,
                    coord_features= config.coord_features,
                    dimension = config.dimension,
                    padding= 0,
                    include_input_seq_len = False
                )
                self.branch_net = ResNet2D(config=resNetConfig)
                length_new = config.latent_channels * calc_resnet_out_shape(
                    in_shape = config.grid_resolution,
                    num_blocks = config.num_blocks,
                    if_maxpool = False if config.ResNet_block == "BasicBlock" else True)[0]* calc_resnet_out_shape(
                                                    in_shape = config.grid_resolution,
                                                    num_blocks = config.num_blocks,
                                                    if_maxpool = False if config.ResNet_block == "BasicBlock" else True)[1]
                self.trunk_dims = linspace_int_list(length_new, self.config.trunk_depth, 2, False)
                #self.trunk_dims = [2] + [self.width] * self.trunk_depth + [length_new]
                self.trunk_net = FFN(
                    dims = self.trunk_dims, 
                    activation_fn = self.activation_fn,
                )
                dims_outffn = linspace_int_list(length_new, config.out_ffn_depth, 1, True)  
                self.out_Ffn = FFN(
                    dims = dims_outffn,
                    #dims = [length_new] * out_ffn_depth  + [1], 
                    activation_fn = self.activation_fn,
                    act_on_output = False,  
                )
            else:
                raise NotImplementedError(f"Unknown block: {config.branch_net_str}")
        else:
            raise ValueError(f"Unknown block type: {config.branch_net_str}")    
        
        
        self.bias = nn.Parameter(torch.zeros(1))  
        
    def forward(self, 
                x: Tensor) -> Tensor:
        shape = list(x.size())  # [B, C, H, W]
        height, width = shape[2], shape[3]
        if self.config.query_idxs is None:
            self.query_idxs = torch.tensor(
                list(product(range(height), range(width))),
                dtype=torch.long,
                device=x.device,
            )  # (H * W, 2)
            self.query_idxs = self.query_idxs.repeat(self.config.out_size, 1)  # (H * W * C_out, 2)
        x_trunk = self.query_idxs.float()
        x_trunk[:, 0] = x_trunk[:, 0] / height  # normalize H
        x_trunk[:, 1] = x_trunk[:, 1] / width   # normalize W
        
        if isinstance(self.config.branch_net_str, str):
            if self.config.branch_net_str == "FFN":
                if self.config.coord_features: 
                    coord_feat = twod_meshgrid(list(x.shape), x.device)
                    x = torch.cat((x, coord_feat), dim=1)
                flat_x= grid_to_points(x)  # (B, C * H * W)
                x_branch = self.branch_net(flat_x) # (B, p)
                x_branch = x_branch.unsqueeze(1)  # (B, 1, p)
                x_trunk = self.trunk_net(x_trunk)  # (H * W * C_out , p)
                x_trunk = x_trunk.unsqueeze(0)  # (1, H * W * C_out, p)
                x = torch.sum(x_branch * x_trunk, dim=-1) + self.bias  # (B, H * W * C_out)
            elif self.config.branch_net_str in ["CNN", "ResNet"]:
                x_branch = self.branch_net(x)  # (B, C_hidden, H_new, W_new)
                x_branch = x_branch.view(shape[0], -1)  # (B, C_hidden*H_new*W_new)
                x_branch = x_branch.unsqueeze(1)  # (B, 1, C_hidden*H_new*W_new)
                x_trunk = self.trunk_net(x_trunk)  # (H*W*C_out, C_hidden*H_new*W_new)
                x_trunk = x_trunk.unsqueeze(0)  # (1, H*W*C_out, C_hidden*H_new*W_new)
                x = x_branch * x_trunk # (B, H*W*C_out, C_hidden*H_new*W_new)
                x = self.out_Ffn(x)   # (B, H*W*C_out, 1)
                x = x.squeeze(-1)  # (B, H*W*C_out)

        shape[1] = self.config.out_size  
        x = points_to_grid(x, shape)  # (B, C_out, H, W)
        return x


class AutoDeepONet3D(PreTrainedModel):
    def __init__(self, config, activation_fn: nn.Module):
        super().__init__(config) 

        self.activation_fn = activation_fn
            
        if isinstance(config.branch_net_str, str):
            if config.branch_net_str == "FFN":
                branch_dim = (config.in_size + config.dimension if config.coord_features else 0) * config.grid_resolution[0] * config.grid_resolution[1] * config.grid_resolution[2]
                self.branch_dims = [branch_dim] + [config.width] * config.branch_depth
                self.branch_net = FFN(
                    dims = self.branch_dims,
                    activation_fn = self.activation_fn,
                    act_on_output = config.act_on_output,
                )
                self.trunk_dims = linspace_int_list(self.config.width, self.config.trunk_depth, 3, False)
                #self.trunk_dims = [3] + [self.width] * self.trunk_depth        
                self.trunk_net = FFN(
                    dims = self.trunk_dims, 
                    activation_fn = self.activation_fn,
                )
            elif config.branch_net_str == "CNN":
                self.branch_net = CnnBranch(
                    in_channels = config.in_size,
                    kernel_size = config.kernel_size,
                    dimension = 3,
                    activation_fn= config.activation_fn,
                    padding = config.padding,
                    stride = config.stride,
                    latent_channels = config.latent_channels, 
                    depth = config.branch_depth,
                    grid_resolution = config.grid_resolution,
                )
                length_new = config.latent_channels * self.branch_net.calc_out_shape()[0] * self.branch_net.calc_out_shape()[1] * self.branch_net.calc_out_shape()[2]
                self.trunk_dims = linspace_int_list(length_new, self.config.trunk_depth, 3, False)
                #self.trunk_dims = [3] + [self.width] * self.trunk_depth + [length_new]
                self.trunk_net = FFN(
                    dims = self.trunk_dims, 
                    activation_fn = self.activation_fn
                )
                dims_outffn = linspace_int_list(length_new, config.out_ffn_depth, 1, True)  
                self.out_Ffn = FFN(
                    dims = dims_outffn,
                    #dims = [length_new] * out_ffn_depth  + [1], 
                    activation_fn = self.activation_fn,
                    act_on_output = False
                )
                
            elif config.branch_net_str == "ResNet":
                resNetConfig = ResNetConfig(
                    in_size= config.in_size,
                    out_size= config.latent_channels,
                    block = config.ResNet_block,
                    num_blocks = config.num_blocks,
                    latent_channels= config.latent_channels,#TODO:change this!!!!!
                    coord_features= config.coord_features,
                    dimension = config.dimension,
                    padding= 0,
                    include_input_seq_len = False
                )
                self.branch_net = ResNet3D(config=resNetConfig)
                length_new = config.latent_channels * calc_resnet_out_shape(
                    in_shape = config.grid_resolution,
                    num_blocks = config.num_blocks,
                    if_maxpool = False if config.ResNet_block == "BasicBlock" else True)[0]* calc_resnet_out_shape(
                                                    in_shape = config.grid_resolution,
                                                    num_blocks = config.num_blocks,
                                                    if_maxpool = False if config.ResNet_block == "BasicBlock" else True)[1] * calc_resnet_out_shape(
                                                    in_shape = config.grid_resolution,
                                                    num_blocks = config.num_blocks,
                                                    if_maxpool = False if config.ResNet_block == "BasicBlock" else True)[2]
                self.trunk_dims = linspace_int_list(length_new, self.config.trunk_depth, 3, False)
                #self.trunk_dims = [3] + [self.width] * self.trunk_depth + [length_new]
                self.trunk_net = FFN(
                    dims = self.trunk_dims, 
                    activation_fn = self.activation_fn
                )
                dims_outffn = linspace_int_list(length_new, config.out_ffn_depth, 1, True)  
                self.out_Ffn = FFN(
                    dims = dims_outffn,
                    #dims = [length_new] * out_ffn_depth  + [1], 
                    activation_fn = self.activation_fn,
                    act_on_output = False
                )
            else:
                raise NotImplementedError(f"Unknown block: {config.branch_net_str}")
        else:
            raise ValueError(f"Unknown block type: {config.branch_net_str}")    
        
        
        self.bias = nn.Parameter(torch.zeros(1))  

    def forward(self, x: Tensor) -> Tensor:
        shape = list(x.size())  # [B, C, D, H, W]
        depth, height, width = shape[2], shape[3], shape[4]
        if self.config.query_idxs is None:
            self.query_idxs = torch.tensor(
                list(product(range(depth), range(height), range(width))),
                dtype=torch.long,
                device=x.device,
            )  # (D * H * W, 3)
            self.query_idxs = self.query_idxs.repeat(self.config.out_size, 1)  # (D*H*W*C_out, 3)
        x_trunk = self.query_idxs.float()
        x_trunk[:, 0] = x_trunk[:, 0] / depth   # normalize D
        x_trunk[:, 1] = x_trunk[:, 1] / height  # normalize H
        x_trunk[:, 2] = x_trunk[:, 2] / width   # normalize W

        if isinstance(self.config.branch_net_str, str):
            if self.config.branch_net_str == "FFN":
                if self.config.coord_features: 
                    coord_feat = threed_meshgrid(list(x.shape), x.device)
                    x = torch.cat((x, coord_feat), dim=1)
                flat_x = x.reshape(shape[0], -1)  # (B, C*D*H*W)
                x_branch = self.branch_net(flat_x)  # (B, p)
                x_branch = x_branch.unsqueeze(1)  # (B, 1, p)
                x_trunk = self.trunk_net(x_trunk)  # (D*H*W*C_out, p)
                x_trunk = x_trunk.unsqueeze(0)  # (1, D*H*W*C_out, p)
                x = torch.sum(x_branch * x_trunk, dim=-1) + self.bias  # (B, D*H*W*C_out)
            elif self.config.branch_net_str in ["CNN", "ResNet"]:
                x_branch = self.branch_net(x)  # (B, C_hidden, D_new, H_new, W_new)
                x_branch = x_branch.view(shape[0], -1)  # (B, C_hidden*D_new*H_new*W_new)
                x_branch = x_branch.unsqueeze(1)  # (B, 1, C_hidden*D_new*H_new*W_new)
                x_trunk = self.trunk_net(x_trunk)  # (D*H*W*C_out, C_hidden*D_new*H_new*W_new)
                x_trunk = x_trunk.unsqueeze(0)  # (1, D*H*W*C_out, C_hidden*D_new*H_new*W_new)
                x = x_branch * x_trunk  # (B, D*H*W*C_out, C_hidden*D_new*H_new*W_new)
                x = self.out_Ffn(x)  # (B, D*H*W*C_out, 1)
                x = x.squeeze(-1)  # (B, D*H*W*C_out)

        shape[1] = self.config.out_size
        x = x.view(shape[0], self.config.out_size, depth, height, width)  # (B, C_out, D, H, W)
        return x    
 