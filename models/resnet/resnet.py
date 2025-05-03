import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils import activation_func
from typing import Optional, Union, Tuple, List, Callable
from .resnet_utils import BasicBlock1D,BasicBlock2D,BasicBlock3D,DilatedBasicBlock1D,DilatedBasicBlock2D,DilatedBasicBlock3D
from utils.feature_utils import oned_meshgrid, twod_meshgrid, threed_meshgrid

def make_layer(
    block: Callable,
    in_planes: int,
    out_planes: int,
    num_blocks: int,
    stride: int,
    activation_fn: nn.Module = nn.GELU(),
    norm: bool = True,
) -> nn.Sequential:
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        layers.append(
            block(
                in_planes,
                out_planes,
                stride,
                activation_fn,
                norm,
            )
        )
        in_planes = out_planes * block.expansion
    return nn.Sequential(*layers)

class ResNet(nn.Module):
    """Class to support ResNet like feedforward architectures

    Args:
        in_channels : int
            Number of input fields
        out_channels : int
            Number of output fields
        block (str): 
            BasicBlock, Dilblock only for now
        num_blocks (List[int]): 
            Number of blocks in each stage
        sequence_info (List[List[int]]):
            sequence_info[0][0]: input_seq_len, sequence_info[0][1]: label_seq_len,
            sequence_info[0][2]: input_sequence_stride, sequence_info[0][3]: label_sequence_stride  
        hidden_channels (int): 
            Number of channels in the hidden layers
        dimension : int
            Model dimensionality (supports 2)
        activation_fn : str
            Activation function, by default "gelu"
        coord_features : bool, optional
            Use coordinate grid as additional feature map, by default True
        norm (bool): 
            Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block: str,
        num_blocks: list,
        sequence_info: Optional[List[List[int]]] = [[1,1,1,1]],
        hidden_channels: int = 64,
        dimension: int = 2,
        activation_fn: str = "gelu",
        coord_features: bool = True,
        norm: bool = True,
        padding: int = 9,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels * sequence_info[0][0] 
        self.out_channels = out_channels * sequence_info[0][1]
        self.hidden_channels = hidden_channels
        self.normalization = norm
        self.coord_features = coord_features
        self.dimension = dimension
               
        self.activation: nn.Module = activation_func.get_activation(activation_fn)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation_fn} not implemented")
        
        self.resnet = self.build_resnet()(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            block=block,
            num_blocks=num_blocks,     #is self.xx necessary??
            hidden_channels=self.hidden_channels,
            activation_fn=self.activation,
            coord_features=self.coord_features,
            norm=self.normalization,
            padding=padding,
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
                labels: Tensor) -> Tensor: #NOTE: Vimp: forward SHOULD always have the arguments EXACTLY named as "input_data" and "labels", 
                                           #else the data collator will remove them. 
        #reshape input into [batch, in_channel, grid_x, grid_y, ...]
        #NOTE: input and output fields need not be necessarily the same.
        batch, input_seq, input_fields, *spatial = input_data.shape
        input_data=input_data.reshape(batch, input_seq * input_fields, *spatial)
        
        y = self.resnet(input_data)
        
        # Reshape the prediction to match the labels shape
        batch, output_seq, output_fields, *spatial = labels.shape
        y = y.reshape(batch, output_seq, output_fields, *spatial)

        return y,labels
                                                   
class ResNet1D(nn.Module):
    """    Args:
        in_channels : int
            Number of input fields
        out_channels : int
            Number of output fields
        block : str
            BasicBlock,Dilblock only for now
        num_blocks (List[int]): 
            Number of blocks in each stage
        hidden_channels (int): 
            Number of channels in the hidden layers
        activation_fn : str
            Activation function, by default "gelu"
        coord_features : bool, optional
            Use coordinate grid as additional feature map, by default True
        padding : int
            Padding for the input tensor
        norm (bool): 
            Whether to use normalization
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block: str ,
        num_blocks: list,
        hidden_channels: int = 64,
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
        padding: int = 9,
        norm: bool = True,
        ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.activation = activation_fn                
        self.coord_features = coord_features
        self.padding = padding   
        self.num_blocks = num_blocks
        self.normalization = norm
        
        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 1
            
        self.conv_in1 = nn.Conv1d(
            self.in_channels,
            self.hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv1d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=1,
            bias=True,
        )
        
        self.block = self.getblock(block)
        
        self.layers = nn.ModuleList(
            [
                make_layer(
                    self.block,
                    self.hidden_channels,
                    self.hidden_channels,
                    num_blocks[i],
                    stride = 1,
                    activation_fn = self.activation,
                    norm = self.normalization,
                )
                for i in range(len(num_blocks))
            ]
        )
        
        self.conv_out1 = nn.Conv1d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv1d(
            self.hidden_channels,
            self.out_channels,
            kernel_size=1,
            bias=True,
        )       
    
    def getblock(self,block):
        """Get the ResNet block based on the model dimensionality"""
        if isinstance(block, str):
            if block == "BasicBlock":
                    return BasicBlock1D
            elif block == "DilatedBasicBlock":
                    return DilatedBasicBlock1D
            else:
                raise NotImplementedError(f"Unknown block: {block}")
        else:
            raise ValueError(f"Unknown block type: {block}")
           

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
        in_channels : int
            Number of input channels
        hidden_channels (int): 
            Number of channels in the hidden layers
        activation_fn : nn.Module, optional
            Activation function, by default nn.GELU
        coord_features : bool, optional
            Use coordinate grid as additional feature map, by default True
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block: str ,
        num_blocks: list,
        hidden_channels: int = 64,
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
        padding: int = 9,
        norm: bool = True,
        ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.activation = activation_fn                
        self.coord_features = coord_features
        self.padding = padding   
        self.num_blocks = num_blocks
        self.normalization = norm
        
        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 2
            
        self.conv_in1 = nn.Conv2d(
            self.in_channels,
            self.hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=1,
            bias=True,
        )
        
        self.block = self.getblock(block)
        
        self.layers = nn.ModuleList(
            [
                make_layer(
                    self.block,
                    self.hidden_channels,
                    self.hidden_channels,
                    num_blocks[i],
                    stride = 1,
                    activation_fn = self.activation,
                    norm = self.normalization,
                )
                for i in range(len(num_blocks))
            ]
        )
        
        self.conv_out1 = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv2d(
            self.hidden_channels,
            self.out_channels,
            kernel_size=1,
            bias=True,
        )       
    
    def getblock(self,block):
        """Get the ResNet block based on the model dimensionality"""
        if isinstance(block, str):
            if block == "BasicBlock":
                    return BasicBlock2D
            elif block == "DilatedBasicBlock":
                    return DilatedBasicBlock2D
            else:
                raise NotImplementedError(f"Unknown block: {block}")
        else:
            raise ValueError(f"Unknown block type: {block}")
           

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
        in_channels : int
            Number of input channels
        hidden_channels (int): 
            Number of channels in the hidden layers
        activation_fn : nn.Module, optional
            Activation function, by default nn.GELU
        coord_features : bool, optional
            Use coordinate grid as additional feature map, by default True
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block: str ,
        num_blocks: list,
        hidden_channels: int = 64,
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
        padding: int = 9,
        norm: bool = True,
        ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.activation = activation_fn                
        self.coord_features = coord_features
        self.padding = padding   
        self.num_blocks = num_blocks
        self.normalization = norm
        
        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 3
            
        self.conv_in1 = nn.Conv3d(
            self.in_channels,
            self.hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv3d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=1,
            bias=True,
        )
        
        self.block = self.getblock(block)
        
        self.layers = nn.ModuleList(
            [
                make_layer(
                    self.block,
                    self.hidden_channels,
                    self.hidden_channels,
                    num_blocks[i],
                    stride = 1,
                    activation_fn = self.activation,
                    norm = self.normalization,
                )
                for i in range(len(num_blocks))
            ]
        )
        
        self.conv_out1 = nn.Conv3d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv3d(
            self.hidden_channels,
            self.out_channels,
            kernel_size=1,
            bias=True,
        )       
    
    def getblock(self,block):
        if isinstance(block, str):
            if block == "BasicBlock":
                    return BasicBlock3D
            elif block == "DilatedBasicBlock":
                    return DilatedBasicBlock3D
            else:
                raise NotImplementedError(f"Unknown block: {block}")
        else:
            raise ValueError(f"Unknown block type: {block}")
           

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


    
