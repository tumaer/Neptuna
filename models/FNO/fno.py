from typing import  Union
from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .fno_utils import Conv1dFCLayer, SpectralConv1d
from .fno_utils import Conv2dFCLayer, SpectralConv2d 
from .fno_utils import Conv3dFCLayer, SpectralConv3d
from .fno_utils import FullyConnected
from utils import activation_func
from typing import Optional, Union, Tuple, List

from utils.feature_utils import oned_meshgrid, twod_meshgrid, threed_meshgrid
# ===================================================================
# ===================================================================
# nD FNO main class
# ===================================================================
# ===================================================================
class FNO(nn.Module):
    """Fourier neural operator (FNO) model.

    Note
    ----
    The FNO architecture supports options for 1D, 2D, 3D and 4D fields which can
    be controlled using the `dimension` parameter.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    decoder_layers : int, optional
        Number of decoder layers, by default 1
    decoder_layer_size : int, optional
        Number of neurons in decoder layers, by default 32
    decoder_activation_fn : str, optional
        Activation function for decoder, by default "silu"
    dimension : int
        Model dimensionality (supports 1, 2, 3).
    latent_channels : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding : int, optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : str, optional
        Activation function, by default "gelu"
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True

    Example
    -------
    >>> # define the 2d FNO model
    >>> model = modulus.models.fno.FNO(
    ...     in_channels=4,
    ...     out_channels=3,
    ...     decoder_layers=2,
    ...     decoder_layer_size=32,
    ...     dimension=2,
    ...     latent_channels=32,
    ...     num_fno_layers=2,
    ...     padding=0,
    ... )
    >>> input = torch.randn(32, 4, 32, 32) #(N, C, H, W)
    >>> output = model(input)
    >>> output.size()
    torch.Size([32, 3, 32, 32])

    Note
    ----
    Reference: Li, Zongyi, et al. "Fourier neural operator for parametric
    partial differential equations." arXiv preprint arXiv:2010.08895 (2020).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sequence_info: Optional[List[List[int]]] = [[1,1,1,1]],
        decoder_layers: int = 1,
        decoder_layer_size: int = 32,
        decoder_activation_fn_name: str = "silu",
        dimension: int = 2,
        latent_channels: int = 32,
        num_fno_layers: int = 4,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: int = 8,
        padding_type: str = "constant",
        activation_fn_name: str = "gelu",
        coord_features: bool = True,
    ) -> None:
        super().__init__()
        self.num_fno_layers = num_fno_layers
        self.num_fno_modes = num_fno_modes
        self.padding = padding
        self.padding_type = padding_type
        self.activation_fn = activation_func.get_activation(activation_fn_name)

        if self.activation_fn is None:
            raise NotImplementedError(f"Activation {activation_fn_name} not implemented")

        self.coord_features = coord_features
        self.dimension = dimension
        self.sequence_info = sequence_info
        
        model_in_channels= in_channels*sequence_info[0][0]
        model_out_channels= out_channels*self.sequence_info[0][1]
        
        self.fno = self.build_FNO()(
            in_channels=model_in_channels,
            out_channels=model_out_channels,
            num_fno_layers=self.num_fno_layers,
            fno_layer_size=latent_channels,
            num_fno_modes=self.num_fno_modes,
            padding=self.padding,
            padding_type=self.padding_type,
            activation_fn=self.activation_fn,
            coord_features=self.coord_features,
            decoder_activation_fn_name=decoder_activation_fn_name,
            decoder_layers=decoder_layers,
            decoder_layer_size=decoder_layer_size,
        )

    def build_FNO(self):
        """Get the FNO encoder based on the model dimensionality"""
        if self.dimension == 1:
            return FNO1D
        elif self.dimension == 2:
            return FNO2D
        elif self.dimension == 3:
            return FNO3D
        else:
            raise NotImplementedError(
                "Invalid dimensionality. Only 1D ,2D and 3D FNO implemented"
            )

    def forward(self, 
                input_data: Tensor,
                labels: Tensor) -> Tensor: #NOTE: Vimp: forward SHOULD always have the arguments EXACTLY named as "input_data" and "labels", 
                                           #else the data collator will remove them. 
        
        #reshape input into [batch, in_channel, grid_x, grid_y, ...]
        #NOTE: input and output fields need not be necessarily the same.
        batch, input_seq, input_fields, *spatial = input_data.shape
        input_data=input_data.reshape(batch, input_seq * input_fields, *spatial)

        # Fourier encoder
        y_latent = self.fno(input_data)

        # Reshape to pointwise inputs if not a conv FC model
        y_shape = y_latent.shape
        y_latent, y_shape = self.fno.grid_to_points(y_latent)

        # Decoder
        decoder_net = (self.fno.decoder_net()).to(y_latent.device)
        y = decoder_net(y_latent)

        # Convert back into grid
        y = self.fno.points_to_grid(y, y_shape)

        # Reshape the prediction to match the labels shape
        batch, output_seq, output_fields, *spatial = labels.shape
        y = y.reshape(batch, output_seq, output_fields, *spatial)

        return y,labels
# ===================================================================
# ===================================================================

# ===================================================================
# 1D FNO 
# ===================================================================
class FNO1D(nn.Module):
    """1D FNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_fno_layers: int = 4,
        fno_layer_size: int = 32,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
        decoder_activation_fn_name: str = "silu",
        decoder_layers: int = 1,
        decoder_layer_size: int = 32,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.activation_fn = activation_fn
        self.decoder_activation_fn_name = decoder_activation_fn_name
        self.decoder_layers = decoder_layers
        self.decoder_layer_size = decoder_layer_size

        # Add relative coordinate feature
        self.coord_features = coord_features
        if self.coord_features:
            self.in_channels = self.in_channels + 1

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding]
        self.pad = padding[:1]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes]

        # build lift
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def decoder_net(self) -> None:
        return FullyConnected(
            in_features=self.fno_width,
            layer_size=self.decoder_layer_size,
            out_features=self.out_channels,
            num_layers=self.decoder_layers,
            activation_fn=self.decoder_activation_fn_name,
        )

    def build_lift_network(self) -> None:
        """construct network for lifting variables to latent space."""
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            Conv1dFCLayer(self.in_channels, int(self.fno_width / 2))
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            Conv1dFCLayer(int(self.fno_width / 2), self.fno_width)
        )

    def build_fno(self, num_fno_modes: List[int]) -> None:
        """construct FNO block.
        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions

        """
        # Build Neural Fourier Operators
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                SpectralConv1d(self.fno_width, self.fno_width, num_fno_modes[0])
            )
            self.conv_layers.append(nn.Conv1d(self.fno_width, self.fno_width, 1))

    def forward(self, x: Tensor) -> Tensor:
        if self.coord_features:
            coord_feat = oned_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)
        # (left, right)
        x = F.pad(x, (0, self.pad[0]), mode=self.padding_type)
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)

        x = x[..., : self.ipad[0]]
        return x

    def grid_to_points(self, value: Tensor) -> Tuple[Tensor, List[int]]:
        """converting from grid based (image) to point based representation

        Parameters
        ----------
        value : Meshgrid tensor

        Returns
        -------
        Tuple
            Tensor, meshgrid shape
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: Tensor, shape: List[int]) -> Tensor:
        """converting from point based to grid based (image) representation

        Parameters
        ----------
        value : Tensor
            Tensor
        shape : List[int]
            meshgrid shape

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        output = value.reshape(shape[0], shape[2], value.size(-1))
        return torch.permute(output, (0, 2, 1))


# ===================================================================
# 2D FNO Encoder
# ===================================================================
class FNO2D(nn.Module):
    """2D Spectral encoder for FNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_fno_layers: int = 4,
        fno_layer_size: int = 32,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
        decoder_activation_fn_name: str = "silu",
        decoder_layers: int = 1,
        decoder_layer_size: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        
        self.activation_fn = activation_fn
        self.decoder_activation_fn_name = decoder_activation_fn_name
        self.decoder_layers = decoder_layers
        self.decoder_layer_size = decoder_layer_size
        
        # Add relative coordinate feature
        self.coord_features = coord_features
        if self.coord_features:
            self.in_channels = self.in_channels + 2

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding, padding]
        padding = padding + [0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:2]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes, num_fno_modes]

        # build lift
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def decoder_net(self) -> None:
        return FullyConnected(
            in_features=self.fno_width,
            layer_size=self.decoder_layer_size,
            out_features=self.out_channels,
            num_layers=self.decoder_layers,
            activation_fn=self.decoder_activation_fn_name,
        )

    def build_lift_network(self) -> None:
        """construct network for lifting variables to latent space."""
        # Initial lift network
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            Conv2dFCLayer(self.in_channels, int(self.fno_width / 2))
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            Conv2dFCLayer(int(self.fno_width / 2), self.fno_width)
        )

    def build_fno(self, num_fno_modes: List[int]) -> None:
        """construct FNO block.
        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions

        """
        # Build Neural Fourier Operators
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                SpectralConv2d(
                    self.fno_width, self.fno_width, num_fno_modes[0], num_fno_modes[1]
                )
            )
            self.conv_layers.append(nn.Conv2d(self.fno_width, self.fno_width, 1))

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                "Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D FNO"
            )

        if self.coord_features: #TODO: Do this for ALL the models
            coord_feat = twod_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)
        # (left, right, top, bottom)
        x = F.pad(x, (0, self.pad[1], 0, self.pad[0]), mode=self.padding_type)
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)

        # remove padding
        x = x[..., : self.ipad[0], : self.ipad[1]]

        return x

    def grid_to_points(self, value: Tensor) -> Tuple[Tensor, List[int]]:
        """converting from grid based (image) to point based representation

        Parameters
        ----------
        value : Meshgrid tensor

        Returns
        -------
        Tuple
            Tensor, meshgrid shape
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 3, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: Tensor, shape: List[int]) -> Tensor:
        """converting from point based to grid based (image) representation

        Parameters
        ----------
        value : Tensor
            Tensor
        shape : List[int]
            meshgrid shape

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        output = value.reshape(shape[0], shape[2], shape[3], value.size(-1))
        return torch.permute(output, (0, 3, 1, 2))
    

# ===================================================================
# 3D FNO Encoder
# ===================================================================
class FNO3D(nn.Module):
    """3D Spectral encoder for FNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_fno_layers: int = 4,
        fno_layer_size: int = 32,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
        decoder_activation_fn_name: str = "silu",
        decoder_layers: int = 1,
        decoder_layer_size: int = 32,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        
        self.activation_fn = activation_fn
        self.decoder_activation_fn_name = decoder_activation_fn_name
        self.decoder_layers = decoder_layers
        self.decoder_layer_size = decoder_layer_size
        # Add relative coordinate feature
        self.coord_features = coord_features
        if self.coord_features:
            self.in_channels = self.in_channels + 3

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding, padding, padding]
        padding = padding + [0, 0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:3]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes, num_fno_modes, num_fno_modes]

        # build lift
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def decoder_net(self) -> None:
        return FullyConnected(
            in_features=self.fno_width,
            layer_size=self.decoder_layer_size,
            out_features=self.out_channels,
            num_layers=self.decoder_layers,
            activation_fn=self.decoder_activation_fn_name,
        )

    def build_lift_network(self) -> None:
        """construct network for lifting variables to latent space."""
        # Initial lift network
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            Conv3dFCLayer(self.in_channels, int(self.fno_width / 2))
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            Conv3dFCLayer(int(self.fno_width / 2), self.fno_width)
        )

    def build_fno(self, num_fno_modes: List[int]) -> None:
        """construct FNO block.
        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions

        """
        # Build Neural Fourier Operators
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                SpectralConv3d(
                    self.fno_width,
                    self.fno_width,
                    num_fno_modes[0],
                    num_fno_modes[1],
                    num_fno_modes[2],
                )
            )
            self.conv_layers.append(nn.Conv3d(self.fno_width, self.fno_width, 1))

    def forward(self, x: Tensor) -> Tensor:
        if self.coord_features:
            coord_feat = threed_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)
        # (left, right, top, bottom, front, back)
        x = F.pad(
            x,
            (0, self.pad[2], 0, self.pad[1], 0, self.pad[0]),
            mode=self.padding_type,
        )
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)

        x = x[..., : self.ipad[0], : self.ipad[1], : self.ipad[2]]
        return x

    def grid_to_points(self, value: Tensor) -> Tuple[Tensor, List[int]]:
        """converting from grid based (image) to point based representation

        Parameters
        ----------
        value : Meshgrid tensor

        Returns
        -------
        Tuple
            Tensor, meshgrid shape
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 3, 4, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: Tensor, shape: List[int]) -> Tensor:
        """converting from point based to grid based (image) representation

        Parameters
        ----------
        value : Tensor
            Tensor
        shape : List[int]
            meshgrid shape

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        output = value.reshape(shape[0], shape[2], shape[3], shape[4], value.size(-1))
        return torch.permute(output, (0, 4, 1, 2, 3))