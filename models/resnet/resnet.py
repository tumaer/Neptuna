import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils import activation_func
from typing import Optional, Union, Tuple, List, Callable
from .resnet_utils import BasicBlock1D,BasicBlock2D,BasicBlock3D,DilatedBasicBlock1D,DilatedBasicBlock2D,DilatedBasicBlock3D

class ResNet(nn.Module):
    """Class to support ResNet like feedforward architectures

    Args:
        in_fields : int
            Number of input fields
        out_fields : int
            Number of output fields
        block (Callable): 
            BasicBlock only for now
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
        in_fields: int,
        out_fields: int,
        block: Callable,
        num_blocks: list,
        sequence_info: Optional[List[List[int]]] = [[1,1,1,1]],
        hidden_channels: int = 64,
        dimension: int = 2,
        activation_fn: str = "gelu",
        coord_features: bool = True,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_fields * sequence_info[0][0] 
        self.out_channels = out_fields * sequence_info[0][1]
        self.in_planes = hidden_channels
        self.normalization = norm
        self.coord_features = coord_features
        self.dimension = dimension
        
        if isinstance(block, str):
            if block == "BasicBlock2D":
                block = BasicBlock2D
            elif block == "BasicBlock1D":
                block = BasicBlock1D
            elif block == "BasicBlock3D":
                block = BasicBlock3D
            elif block == "DilatedBasicBlock2D":
                block = DilatedBasicBlock2D
            elif block == "DilatedBasicBlock1D":
                block = DilatedBasicBlock1D
            elif block == "DilatedBasicBlock3D":
                block = DilatedBasicBlock3D
        else:
            raise ValueError(f"Unknown block type: {block}")
        
        self.activation: nn.Module = activation_func.get_activation(activation_fn)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation_fn} not implemented")
        
        ResNetEncoder = self.getResNetEncoder()
        self.encoder = ResNetEncoder(
            in_channels = self.in_channels,
            hidden_channels = self.in_planes,
            activation_fn = self.activation,
            coord_features = self.coord_features,
        )

        self.layers = nn.ModuleList(
            [
                self._make_layer(
                    block,
                    self.in_planes,
                    num_blocks[i],
                    stride = 1,
                    activation_fn = self.activation,
                    norm = self.normalization,
                )
                for i in range(len(num_blocks))
            ]
        )
        
        ResNetDecoder = self.getResNetDecoder()
        self.decoder = ResNetDecoder(
            out_channels = self.out_channels,
            hidden_channels = self.in_planes,
            activation_fn = self.activation,
        )
        
    def getResNetEncoder(self):
        """Get the ResNet encoder based on the model dimensionality"""
        if self.dimension == 1:
            return ResNet1DEncoder
        elif self.dimension == 2:
            return ResNet2DEncoder
        elif self.dimension == 3:
            return ResNet3DEncoder
        else:
            raise NotImplementedError(
                "Invalid dimensionality. Only 1D, 2D, 3D ResNet implemented"
            )

    def _make_layer(
        self,
        block: Callable,
        planes: int,
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
                    self.in_planes,
                    planes,
                    stride,
                    activation_fn,
                    norm,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def getResNetDecoder(self):
        """Get the ResNet decoder based on the model dimensionality"""
        if self.dimension == 2:
            return ResNet2DDecoder
        elif self.dimension == 1:
            return ResNet1DDecoder
        elif self.dimension == 3:
            return ResNet3DDecoder
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
        orig_shape = input_data.shape
        batch, input_seq, input_fields, *spatial = input_data.shape
        input_data=input_data.reshape(batch, input_seq * input_fields, *spatial)
        
        #encoder
        y = self.encoder(input_data)
        
        #main part
        for layer in self.layers:
            y = layer(y)
        
        #decoder
        y = self.decoder(y)
        
        # Reshape the prediction to match the labels shape
        batch, output_seq, output_fields, *spatial = labels.shape
        y = y.reshape(batch, output_seq, output_fields, *spatial)

        return y,labels
                                                   


class ResNet2DEncoder(nn.Module):
    """2D ResNet encoder for ResNet

    Args:
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
        hidden_channels: int = 64,
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.in_planes = hidden_channels
        self.activation = activation_fn                
        self.coord_features = coord_features
        self.padding = 9   #TODO: add padding to the model, hard code for now
        
        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 2
            
        self.conv_in1 = nn.Conv2d(
            self.in_channels,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv2d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )

            
    def meshgrid(self, shape: List[int], device: torch.device) -> Tensor:
        """Creates 2D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x, size_y = shape[0], shape[2], shape[3]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1) 
        return torch.cat((grid_x, grid_y), dim=1)
    
    
    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                "Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D ResNet"
            )
        
        if self.coord_features: 
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
            
        x = self.activation(self.conv_in1(x.float())) #x[8, 128, 128, 128]
        x = self.activation(self.conv_in2(x.float())) #x[8, 128, 128, 128]
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
                        
        return x
    
class ResNet3DEncoder(nn.Module):
    """3D ResNet encoder for ResNet

    Args:
        in_channels : int
            Number of input channels
        hidden_channels : int
            Number of channels in the hidden layers
        activation_fn : nn.Module, optional
            Activation function, by default nn.GELU
        coord_features : bool, optional
            Use coordinate grid as additional feature map, by default True
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.in_planes = hidden_channels
        self.activation = activation_fn
        self.coord_features = coord_features
        self.padding = 9  #TODO: add padding to the model, hard code for now
        
        if self.coord_features:
            self.in_channels += 3

        self.conv_in1 = nn.Conv3d(
            self.in_channels,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv3d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )

    def meshgrid(self, shape: List[int], device: torch.device) -> Tensor:
        """Creates 3D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x, size_y, size_z = shape[0], shape[2], shape[3], shape[4]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_z = torch.linspace(0, 1, size_z, dtype=torch.float32, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        return torch.cat((grid_x, grid_y, grid_z), dim=1)
    
    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 5:
            raise ValueError(
                "Only 5D tensors [batch, in_channels, size_x, size_y, size_z] accepted for 3D ResNet"
            )

        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.activation(self.conv_in1(x.float()))
        x = self.activation(self.conv_in2(x.float()))
        # 3D padding: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        if self.padding > 0:
            x = F.pad(x, (0, self.padding, 0, self.padding, 0, self.padding))

        return x
    
class ResNet1DEncoder(nn.Module):
    """1D ResNet encoder for ResNet

    Args:
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
        hidden_channels: int = 64,
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.in_planes = hidden_channels
        self.activation = activation_fn                
        self.coord_features = coord_features
        self.padding = 9   #TODO: add padding to the model, hard code for now
        
        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 1
            
        self.conv_in1 = nn.Conv1d(
            self.in_channels,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv1d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )

            
    def meshgrid(self, shape: List[int], device: torch.device) -> Tensor:
        """Creates 1D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x = shape[0], shape[2]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1)
        return grid_x
    
    
    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(
                "Only 3D tensors [batch, in_channels, grid_x accepted for 1D ResNet"
            )
        
        if self.coord_features: 
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
            
        x = self.activation(self.conv_in1(x.float())) #x[8, 128, 128, 128]
        x = self.activation(self.conv_in2(x.float())) #x[8, 128, 128, 128]
        if self.padding > 0:
            x = F.pad(x, [0, self.padding])
                        
        return x
    
        
class ResNet2DDecoder(nn.Module):
    """2D ResNet decoder for ResNet

    Args:
        out_channels : int
            Number of output channels
        hidden_channels (int): 
            Number of channels in the hidden layers
        activation_fn : nn.Module, optional
            Activation function, by default nn.GELU
    """
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int = 64,
        activation_fn: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__()        
        self.out_channels = out_channels
        self.in_planes = hidden_channels
        self.activation = activation_fn
        self.padding = 9   #TODO: add padding to the model, hard code for now
       
        self.conv_out1 = nn.Conv2d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv2d(
            self.in_planes,
            self.out_channels,
            kernel_size=1,
            bias=True,
        )

        
    def forward(self, x: Tensor) -> Tensor:
            
        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)
                        
        return x


class ResNet1DDecoder(nn.Module):
    """1D ResNet decoder for ResNet

    Args:
        out_channels : int
            Number of output channels
        hidden_channels (int): 
            Number of channels in the hidden layers
        activation_fn : nn.Module, optional
            Activation function, by default nn.GELU
    """
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int = 64,
        activation_fn: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__()        
        self.out_channels = out_channels
        self.in_planes = hidden_channels
        self.activation = activation_fn
        self.padding = 9   #TODO: add padding to the model, hard code for now
       
        self.conv_out1 = nn.Conv1d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv1d(
            self.in_planes,
            self.out_channels,
            kernel_size=1,
            bias=True,
        )

        
    def forward(self, x: Tensor) -> Tensor:
            
        if self.padding > 0:
            x = x[..., : -self.padding]

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)
                        
        return x
    
class ResNet3DDecoder(nn.Module):
    """3D ResNet decoder for ResNet

    Args:
        out_channels : int
            Number of output channels
        hidden_channels : int
            Number of channels in the hidden layers
        activation_fn : nn.Module, optional
            Activation function, by default nn.GELU
    """
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int = 64,
        activation_fn: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.in_planes = hidden_channels
        self.activation = activation_fn
        self.padding = 9 #TODO: add padding to the model, hard code for now

        self.conv_out1 = nn.Conv3d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv3d(
            self.in_planes,
            self.out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding, : -self.padding]

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)
        return x