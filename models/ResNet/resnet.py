import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils import activation_func
from .resnet_utils import getblock, make_layer
from utils.grid_utils import oned_meshgrid, twod_meshgrid, threed_meshgrid
from transformers import PreTrainedModel


class ResNet(PreTrainedModel):
    """Class to support ResNet like feedforward architectures"""

    main_input_name = "input_data"
    conditioning_input_name = "conditioning_input_data"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
               
        activation: nn.Module = activation_func.get_activation(config.activation_fn_name)
        if activation is None:
            raise NotImplementedError(f"Activation {config.activation_fn_name} not implemented")
        
        self.resnet = self.build_resnet()(
            config=config,
            activation_fn=activation
        )
           
    def build_resnet(self):
        """Get the ResNet encoder based on the model dimensionality"""
        if self.config.dimension == 1:
            return ResNet1D
        elif self.config.dimension == 2:
            return ResNet2D
        elif self.config.dimension == 3:
            return ResNet3D
        else:
            raise NotImplementedError(
                "Invalid dimensionality. Only 1D, 2D, 3D ResNet implemented"
            )
    
    def forward(self, 
                input_data: Tensor,
                **kwargs) -> Tensor: 

        if "conditioning_input_data" in kwargs:
            #NOTE: Conditioning data can be passed into a conv network before concatination with input_data.
            conditioning_input_data = kwargs["conditioning_input_data"]
            input_data = torch.cat([input_data, conditioning_input_data], dim=2)
        else:
            conditioning_input_data = None        

        batch, input_seq, input_channels, *spatial = input_data.shape
        x=input_data.reshape(batch, input_seq * input_channels, *spatial)
        
        return self.resnet(x, **kwargs)
                                                   
class ResNet1D(PreTrainedModel):
    def __init__(self, config, activation_fn: nn.Module = nn.GELU()) -> None:
        super().__init__(config)

        self.activation = activation_fn

        self.conv_in1 = nn.Conv1d(
            config.in_size,
            config.latent_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv1d(
            config.latent_channels,
            config.latent_channels,
            kernel_size=1,
            bias=True,
        )
        
        self.block = getblock(config.block)
        
        self.layers = nn.ModuleList(
            [
                make_layer(
                    self.block,
                    config,
                    config.latent_channels,
                    config.latent_channels,
                    config.num_blocks[i],
                    stride = config.stride,
                    dimension = 1,
                    activation_fn = self.activation,
                )
                for i in range(len(config.num_blocks))
            ]
        )
        
        self.conv_out1 = nn.Conv1d(
            config.latent_channels,
            config.latent_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv1d(
            config.latent_channels,
            config.out_size,
            kernel_size=1,
            bias=True,
        )           

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if x.dim() != 3:
            raise ValueError(
                "Only 3D tensors [batch, in_channels, grid_x] accepted for 1D ResNet"
            )
        
        #add coordinate-feature map
        if self.config.coord_features: 
            coord_feat = oned_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
        
        #encoder    
        x = self.activation(self.conv_in1(x)) 
        x = self.activation(self.conv_in2(x)) 
        if self.config.padding > 0:
            x = F.pad(x, [0, self.config.padding])
            
        #main part
        for layer in self.layers:
            x = layer(x, **kwargs)
            
        #decoder    
        if self.config.padding > 0:
            x = x[..., : -self.config.padding]

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)
                        
        return x

class ResNet2D(PreTrainedModel):
    def __init__(self, config, activation_fn: nn.Module = nn.GELU()) -> None:
        super().__init__(config)
        self.activation = activation_fn

        self.conv_in1 = nn.Conv2d(
            config.in_size,
            config.latent_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv2d(
            config.latent_channels,
            config.latent_channels,
            kernel_size=1,
            bias=True,
        )
        
        self.block = getblock(config.block)
        
        self.layers = nn.ModuleList(
            [
                make_layer(
                    self.block,
                    config,
                    config.latent_channels,
                    config.latent_channels,
                    config.num_blocks[i],
                    stride = config.stride,
                    dimension = 2,
                    activation_fn = self.activation,
                #     norm = config.norm,
                #  norm_layer_eps = config.norm_layer_eps,
                )
                for i in range(len(config.num_blocks))
            ]
        )
        
        self.conv_out1 = nn.Conv2d(
            config.latent_channels,
            config.latent_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv2d(
            config.latent_channels,
            config.out_size,
            kernel_size=1,
            bias=True,
        )           
           

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                "Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D ResNet"
            )
        
        #add feature map
        if self.config.coord_features: 
            coord_feat = twod_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
        
        #encoder    
        x = self.activation(self.conv_in1(x.float())) 
        x = self.activation(self.conv_in2(x.float())) 
        if self.config.padding > 0:
            x = F.pad(x, [0, self.config.padding, 0, self.config.padding])
            
        #main part
        for layer in self.layers:
            x = layer(x, **kwargs)
            
        #decoder    
        if self.config.padding > 0:
            x = x[..., : -self.config.padding, : -self.config.padding]

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)
                        
        return x

class ResNet3D(PreTrainedModel):

    def __init__(self, config, activation_fn: nn.Module = nn.GELU()) -> None:
        super().__init__(config)
        self.activation = activation_fn 
            
        self.conv_in1 = nn.Conv3d(
            config.in_size,
            config.latent_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv3d(
            config.latent_channels,
            config.latent_channels,
            kernel_size=1,
            bias=True,
        )
        
        self.block = getblock(config.block)
        
        self.layers = nn.ModuleList(
            [
                make_layer(
                    self.block,
                    config,
                    config.latent_channels,
                    config.latent_channels,
                    config.num_blocks[i],
                    stride = config.stride,
                    dimension = 3,
                    activation_fn = self.activation,
                    # norm = config.norm,
                    # norm_layer_eps = config.norm_layer_eps,
                )
                for i in range(len(config.num_blocks))
            ]
        )
        
        self.conv_out1 = nn.Conv3d(
            config.latent_channels,
            config.latent_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv3d(
            config.latent_channels,
            config.out_size,
            kernel_size=1,
            bias=True,
        )                  

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if x.dim() != 5:
            raise ValueError(
                "Only 5D tensors [batch, in_channels, grid_x, grid_y, grid_z] accepted for 3D ResNet"
            )
        
        #add feature map
        if self.config.coord_features: 
            coord_feat = threed_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
        
        #encoder    
        x = self.activation(self.conv_in1(x.float())) 
        x = self.activation(self.conv_in2(x.float())) 
        # 3D padding: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        if self.config.padding > 0:
            x = F.pad(x, (0, self.config.padding, 0, self.config.padding, 0, self.config.padding))
            
        #main part
        for layer in self.layers:
            x = layer(x, **kwargs)
            
        #decoder    
        if self.config.padding > 0:
            x = x[..., : -self.config.padding, : -self.config.padding, : -self.config.padding]
        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)
                        
        return x


    
