import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils import activation_func
from typing import Optional, List
from .resnet_utils import BasicBlockND,DilatedBasicBlockND, getblock, make_layer
from utils.feature_utils import oned_meshgrid, twod_meshgrid, threed_meshgrid


class ResNet(nn.Module):
    """Class to support ResNet like feedforward architectures

    Args:
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        block (str): 
            BasicBlock, Dilblock only for now
        num_blocks (List[int]): 
            Number of blocks in each stage
        sequence_info (List[List[int]]):
            sequence_info[0][0]: input_seq_len, sequence_info[0][1]: label_seq_len,
            sequence_info[0][2]: input_sequence_stride, sequence_info[0][3]: label_sequence_stride  
        latent_channels (int): 
            Number of channels in the latent space
        dimension : int
            Model dimensionality (supports 1,2,3)
        activation_fn : str
            Activation function, by default "gelu"
        coord_features : bool, optional
            Use coordinate grid as additional feature map, by default True
        norm (bool): 
            Whether to use normalization
        n_groups : int
            Number of groups for GroupNorm, by default 1 (equivalent with LayerNorm)
    """
    main_input_name = "input_data"
    conditioning_input_name = "conditioning_input_data"
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block: str,
        num_blocks: list,
        dimension: int,
        sequence_info: Optional[List[int]] = [1,1,1],
        latent_channels: int = 64,
        activation_fn_name: str = "gelu",
        coord_features: bool = True,
        norm: bool = True,
        padding: int = 9,
        n_groups: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.in_size = in_channels * sequence_info[0] 
        self.out_size = out_channels * sequence_info[1]
        self.latent_channels = latent_channels
        self.normalization = norm
        self.coord_features = coord_features
        self.dimension = dimension
               
        self.activation: nn.Module = activation_func.get_activation(activation_fn_name)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation_fn_name} not implemented")
        
        self.resnet = self.build_resnet()(
            in_size=self.in_size,
            out_size=self.out_size,
            block=block,
            num_blocks=num_blocks,    
            latent_channels=self.latent_channels,
            activation_fn=self.activation,
            coord_features=self.coord_features,
            norm=self.normalization,
            padding=padding,
            n_groups=n_groups,
            stride=stride,
        )
           
    def build_resnet(self):
        """Get the ResNet encoder based on the model dimensionality"""
        if self.dimension == 1:
            return ResNet1D
        elif self.dimension == 2:
            return ResNet2D
        elif self.dimension == 3:
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
        input_data=input_data.reshape(batch, input_seq * input_channels, *spatial)
        
        y = self.resnet(input_data)
        
        return y
                                                   
class ResNet1D(nn.Module):
    """    Args:
        in_size : int
            Number of input channels
        out_size : int
            Number of output channels
        block : str
            BasicBlock,Dilblock only for now
        num_blocks (List[int]): 
            Number of blocks in each stage
        latent_channels (int): 
            Number of channels in the hidden layers
        activation_fn : str
            Activation function, by default "gelu"
        coord_features : bool, optional
            Use coordinate grid as additional feature map, by default True
        padding : int
            Padding for the input tensor
        norm (bool): 
            Whether to use normalization
        n_groups : int
            Number of groups for GroupNorm, by default 1 (equivalent with LayerNorm)
    """
    def __init__(
        self,
        in_size: int,
        out_size: int,
        block: str ,
        num_blocks: list,
        latent_channels: int = 64,
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
        padding: int = 9,
        norm: bool = True,
        n_groups: int = 1,
        stride: int = 1,
        ) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.latent_channels = latent_channels
        self.activation = activation_fn                
        self.coord_features = coord_features
        self.padding = padding   
        self.num_blocks = num_blocks
        self.normalization = norm
        
        # Add relative coordinate feature
        if self.coord_features:
            self.in_size = self.in_size + 1
            
        self.conv_in1 = nn.Conv1d(
            self.in_size,
            self.latent_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv1d(
            self.latent_channels,
            self.latent_channels,
            kernel_size=1,
            bias=True,
        )
        
        self.block = getblock(block)
        
        self.layers = nn.ModuleList(
            [
                make_layer(
                    self.block,
                    self.latent_channels,
                    self.latent_channels,
                    num_blocks[i],
                    stride = stride,
                    dimension = 1,
                    activation_fn = self.activation,
                    norm = self.normalization,
                    n_groups = n_groups,
                )
                for i in range(len(num_blocks))
            ]
        )
        
        self.conv_out1 = nn.Conv1d(
            self.latent_channels,
            self.latent_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv1d(
            self.latent_channels,
            self.out_size,
            kernel_size=1,
            bias=True,
        )           

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(
                "Only 3D tensors [batch, in_channels, grid_x] accepted for 1D ResNet"
            )
        
        #add feature map
        if self.coord_features: 
            coord_feat = oned_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
        
        #encoder    
        x = self.activation(self.conv_in1(x.float())) 
        x = self.activation(self.conv_in2(x.float())) 
        if self.padding > 0:
            x = F.pad(x, [0, self.padding])
            
        #main part
        for layer in self.layers:
            x = layer(x)
            
        #decoder    
        if self.padding > 0:
            x = x[..., : -self.padding]

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)
                        
        return x

class ResNet2D(nn.Module):
    """    Args:
        in_size : int
            Number of input channels
        out_size : int
            Number of output channels
        block : str
            BasicBlock,Dilblock only for now
        num_blocks (List[int]): 
            Number of blocks in each stage
        latent_channels (int): 
            Number of channels in the hidden layers
        activation_fn : str
            Activation function, by default "gelu"
        coord_features : bool, optional
            Use coordinate grid as additional feature map, by default True
        padding : int
            Padding for the input tensor
        norm (bool): 
            Whether to use normalization
        n_groups : int
            Number of groups for GroupNorm, by default 1 (equivalent with LayerNorm)
    """
    def __init__(
        self,
        in_size: int,
        out_size: int,
        block: str ,
        num_blocks: list,
        latent_channels: int = 64,
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
        padding: int = 9,
        norm: bool = True,
        n_groups: int = 1,
        stride: int = 1,
        ) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.latent_channels = latent_channels
        self.activation = activation_fn                
        self.coord_features = coord_features
        self.padding = padding   
        self.num_blocks = num_blocks
        self.normalization = norm
        
        # Add relative coordinate feature
        if self.coord_features:
            self.in_size = self.in_size + 2
            
        self.conv_in1 = nn.Conv2d(
            self.in_size,
            self.latent_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv2d(
            self.latent_channels,
            self.latent_channels,
            kernel_size=1,
            bias=True,
        )
        
        self.block = getblock(block)
        
        self.layers = nn.ModuleList(
            [
                make_layer(
                    self.block,
                    self.latent_channels,
                    self.latent_channels,
                    num_blocks[i],
                    stride = stride,
                    dimension = 2,
                    activation_fn = self.activation,
                    norm = self.normalization,
                    n_groups = n_groups,
                )
                for i in range(len(num_blocks))
            ]
        )
        
        self.conv_out1 = nn.Conv2d(
            self.latent_channels,
            self.latent_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv2d(
            self.latent_channels,
            self.out_size,
            kernel_size=1,
            bias=True,
        )           
           

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                "Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D ResNet"
            )
        
        #add feature map
        if self.coord_features: 
            coord_feat = twod_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
        
        #encoder    
        x = self.activation(self.conv_in1(x.float())) 
        x = self.activation(self.conv_in2(x.float())) 
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
            
        #main part
        for layer in self.layers:
            x = layer(x)
            
        #decoder    
        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)
                        
        return x

class ResNet3D(nn.Module):
    """    Args:
        in_size : int
            Number of input channels
        out_size : int
            Number of output channels
        block : str
            BasicBlock,Dilblock only for now
        num_blocks (List[int]): 
            Number of blocks in each stage
        latent_channels (int): 
            Number of channels in the hidden layers
        activation_fn : str
            Activation function, by default "gelu"
        coord_features : bool, optional
            Use coordinate grid as additional feature map, by default True
        padding : int
            Padding for the input tensor
        norm (bool): 
            Whether to use normalization
        n_groups : int
            Number of groups for GroupNorm, by default 1 (equivalent with LayerNorm)
    """
    def __init__(
        self,
        in_size: int,
        out_size: int,
        block: str ,
        num_blocks: list,
        latent_channels: int = 64,
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
        padding: int = 9,
        norm: bool = True,
        n_groups: int = 1,
        stride: int = 1,
        ) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.latent_channels = latent_channels
        self.activation = activation_fn                
        self.coord_features = coord_features
        self.padding = padding   
        self.num_blocks = num_blocks
        self.normalization = norm
        
        # Add relative coordinate feature
        if self.coord_features:
            self.in_size = self.in_size + 3
            
        self.conv_in1 = nn.Conv3d(
            self.in_size,
            self.latent_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv3d(
            self.latent_channels,
            self.latent_channels,
            kernel_size=1,
            bias=True,
        )
        
        self.block = getblock(block)
        
        self.layers = nn.ModuleList(
            [
                make_layer(
                    self.block,
                    self.latent_channels,
                    self.latent_channels,
                    num_blocks[i],
                    stride = stride,
                    dimension = 3,
                    activation_fn = self.activation,
                    norm = self.normalization,
                    n_groups = n_groups,
                )
                for i in range(len(num_blocks))
            ]
        )
        
        self.conv_out1 = nn.Conv3d(
            self.latent_channels,
            self.latent_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv3d(
            self.latent_channels,
            self.out_size,
            kernel_size=1,
            bias=True,
        )                  

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 5:
            raise ValueError(
                "Only 5D tensors [batch, in_channels, grid_x, grid_y, grid_z] accepted for 3D ResNet"
            )
        
        #add feature map
        if self.coord_features: 
            coord_feat = threed_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
        
        #encoder    
        x = self.activation(self.conv_in1(x.float())) 
        x = self.activation(self.conv_in2(x.float())) 
        # 3D padding: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        if self.padding > 0:
            x = F.pad(x, (0, self.padding, 0, self.padding, 0, self.padding))
            
        #main part
        for layer in self.layers:
            x = layer(x)
            
        #decoder    
        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding, : -self.padding]
        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)
                        
        return x


    
