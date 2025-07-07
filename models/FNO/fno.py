from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .fno_utils import ConvNdFCLayer, SpectralConvNd
from .fno_utils import FullyConnected, build_lift_network, build_fno
from utils import activation_func
from typing import Optional, Union, Tuple, List
from models.model_utils import cfd_PreTrainedModel
from utils.feature_utils import oned_meshgrid, twod_meshgrid, threed_meshgrid

class FNO(cfd_PreTrainedModel):
    """Fourier neural operator (FNO) model.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    sequence_info : List[int], optional
        Configuration for input/output sequences [input_seq_len, output_seq_len, stride], by default [1,1,1]
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
        padding_type options: 'constant', 'reflect', 'replicate' or 'circular'
    activation_fn : str, optional
        Activation function, by default "gelu"
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    """
    main_input_name = "input_data"
    conditioning_input_name = "conditioning_input_data"
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        sequence_info: Optional[List[int]] = [1,1,1],
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
        super().__init__(config)
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
        
        in_size= in_channels*sequence_info[0]
        out_size= out_channels*self.sequence_info[1]
        
        self.fno = self.build_FNO()(
            config=config,
            in_size=in_size,
            out_size=out_size,
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
                **kwargs) -> Tensor: 
        
        if "conditioning_input_data" in kwargs:
            #NOTE: Conditioning data can be passed into a conv network before concatination with input_data.
            conditioning_input_data = kwargs["conditioning_input_data"]
            input_data = torch.cat([input_data, conditioning_input_data], dim=2)
        else:
            conditioning_input_data = None

        batch, input_seq, input_channels, *spatial = input_data.shape
        input_data=input_data.reshape(batch, input_seq * input_channels, *spatial)

        # Fourier encoder
        x_latent = self.fno(input_data)

        # Reshape to pointwise inputs if not a conv FC model
        x_shape = x_latent.shape
        x_latent, x_shape = self.fno.grid_to_points(x_latent)

        # Decoder
        decoder_net = (self.fno.decoder_net()).to(x_latent.device)
        x = decoder_net(x_latent)

        # Convert back into grid
        x = self.fno.points_to_grid(x, x_shape)

        return x

class FNO1D(cfd_PreTrainedModel):
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
        config,
        in_size: int,
        out_size: int,
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
        super().__init__(config)

        self.in_size = in_size
        self.out_size = out_size
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.activation_fn = activation_fn
        self.decoder_activation_fn_name = decoder_activation_fn_name
        self.decoder_layers = decoder_layers
        self.decoder_layer_size = decoder_layer_size

        # Add relative coordinate feature
        self.coord_features = coord_features
        if self.coord_features:
            self.in_size = self.in_size + 1

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding]
        self.pad = padding[:1]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes]

        # build lift
        self.lift_network = build_lift_network(
            in_channels=self.in_size,
            fno_width=self.fno_width,
            activation_fn=self.activation_fn,
            dimension=1,
        )
        # build main part
        self.spconv_layers,self.conv_layers = build_fno(
            fno_width=self.fno_width,
            num_fno_modes=num_fno_modes,
            num_fno_layers=self.num_fno_layers,
            dimension=1,
        )

    def decoder_net(self) -> nn.Module:
        return FullyConnected(
            in_features=self.fno_width,
            layer_size=self.decoder_layer_size,
            out_features=self.out_size,
            num_layers=self.decoder_layers,
            activation_fn=self.decoder_activation_fn_name,
        )

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


class FNO2D(cfd_PreTrainedModel):
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
        config,
        in_size: int,
        out_size: int,
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
        super().__init__(config)
        self.in_size = in_size
        self.out_size = out_size
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        
        self.activation_fn = activation_fn
        self.decoder_activation_fn_name = decoder_activation_fn_name
        self.decoder_layers = decoder_layers
        self.decoder_layer_size = decoder_layer_size
        
        # Add relative coordinate feature
        self.coord_features = coord_features
        if self.coord_features:
            self.in_size = self.in_size + 2

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
        self.lift_network = build_lift_network(
            in_channels=self.in_size,
            fno_width=self.fno_width,
            activation_fn=self.activation_fn,
            dimension=2,
        )
        # build main part
        self.spconv_layers,self.conv_layers = build_fno(
            fno_width=self.fno_width,
            num_fno_modes=num_fno_modes,
            num_fno_layers=self.num_fno_layers,
            dimension=2,
        )

    def decoder_net(self) -> nn.Module:
        return FullyConnected(
            in_features=self.fno_width,
            layer_size=self.decoder_layer_size,
            out_features=self.out_size,
            num_layers=self.decoder_layers,
            activation_fn=self.decoder_activation_fn_name,
        )

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
    

class FNO3D(cfd_PreTrainedModel):
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
        config,
        in_size: int,
        out_size: int,
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
        super().__init__(config)

        self.in_size = in_size
        self.out_size = out_size
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        
        self.activation_fn = activation_fn
        self.decoder_activation_fn_name = decoder_activation_fn_name
        self.decoder_layers = decoder_layers
        self.decoder_layer_size = decoder_layer_size
        # Add relative coordinate feature
        self.coord_features = coord_features
        if self.coord_features:
            self.in_size = self.in_size + 3

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
        self.lift_network = build_lift_network(
            in_channels=self.in_size,
            fno_width=self.fno_width,
            activation_fn=self.activation_fn,
            dimension=3,
        )
        # build main part
        self.spconv_layers,self.conv_layers = build_fno(
            fno_width=self.fno_width,
            num_fno_modes=num_fno_modes,
            num_fno_layers=self.num_fno_layers,
            dimension=3,
        )

    def decoder_net(self) -> nn.Module:
        return FullyConnected(
            in_features=self.fno_width,
            layer_size=self.decoder_layer_size,
            out_features=self.out_size,
            num_layers=self.decoder_layers,
            activation_fn=self.decoder_activation_fn_name,
        )

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