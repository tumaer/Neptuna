from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .fno_utils import FullyConnected, build_lift_network, build_fno
from utils import activation_func
from typing import Tuple, List
from transformers import PreTrainedModel
from utils.grid_utils import oned_meshgrid, twod_meshgrid, threed_meshgrid
from utils.model_utils import CustomNorm
from .fno_utils import FNOConfig

class FNO(PreTrainedModel):
    """Fourier neural operator (FNO) model."""

    main_input_name = "input_data"
    conditioning_input_name = "conditioning_input_data"
    config_class = FNOConfig
    
    def __init__(self, config) -> None:
        super().__init__(config)

        activation_fn = activation_func.get_activation(config.activation_fn_name)
        if activation_fn is None:
            raise NotImplementedError(f"Activation {config.activation_fn_name} not implemented")

        self.config = config
        self.fno = self.build_FNO()(config=config, activation_fn=activation_fn)

        self.decoder_net = self.fno.decoder_net()

    def build_FNO(self):
        """Get the FNO encoder based on the model dimensionality"""
        if self.config.dimension == 1:
            return FNO1D
        elif self.config.dimension == 2:
            return FNO2D
        elif self.config.dimension == 3:
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
        x_latent = self.fno(input_data, **kwargs)

        # Reshape to pointwise inputs if not a conv FC model
        x_shape = x_latent.shape
        x_latent, x_shape = self.fno.grid_to_points(x_latent)

        # Decoder
        x = self.decoder_net(x_latent)

        # Convert back into grid
        x = self.fno.points_to_grid(x, x_shape)

        return x

class FNO1D(PreTrainedModel):
    """1D FNO"""

    def __init__(self, config, activation_fn: nn.Module) -> None:
        super().__init__(config)

        self.activation_fn = activation_fn

        # Padding values for spectral conv
        if isinstance(config.padding, int):
            padding = [config.padding]
        self.pad = padding[:1]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = config.padding_type

        if isinstance(config.num_fno_modes, int):
            num_fno_modes = [config.num_fno_modes]

        # build lift
        self.lift_network = build_lift_network(
            in_channels=config.in_size,
            fno_width=config.latent_channels,
            activation_fn=self.activation_fn,
            dimension=1,
        )

        self.norm = CustomNorm(config=config, 
                               num_channels=config.latent_channels,
                               array_length=3,
                               channel_at_last_position=False)
       
        # build main part
        self.spconv_layers,self.conv_layers = build_fno(
            fno_width=config.latent_channels,
            num_fno_modes=num_fno_modes,
            num_fno_layers=config.num_fno_layers,
            dimension=1,
        )

    def decoder_net(self) -> nn.Module:
        return FullyConnected(
            in_features=self.config.latent_channels,
            layer_size=self.config.decoder_layer_size,
            out_features=self.config.out_size,
            num_layers=self.config.decoder_layers,
            activation_fn=self.config.decoder_activation_fn_name,
        )

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if self.config.coord_features:
            coord_feat = oned_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)
        # (left, right)
        x = F.pad(x, (0, self.pad[0]), mode=self.padding_type)
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = conv(x) + w(x)
                x = self.norm(x, **kwargs)
                x = self.activation_fn(x)
            else:
                x = conv(x) + w(x)
                x = self.norm(x, **kwargs)

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


class FNO2D(PreTrainedModel):
    """2D Spectral encoder for FNO"""

    def __init__(self, config, activation_fn: nn.Module) -> None:
        super().__init__(config)

        self.activation_fn = activation_fn

        # Padding values for spectral conv
        if isinstance(config.padding, int):
            padding = [config.padding, config.padding]
        padding = padding + [0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:2]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = config.padding_type

        if isinstance(config.num_fno_modes, int):
            num_fno_modes = [config.num_fno_modes, config.num_fno_modes]

        # build lift
        self.lift_network = build_lift_network(
            in_channels=config.in_size,
            fno_width=config.latent_channels,
            activation_fn=self.activation_fn,
            dimension=2,
        )

        self.norm = CustomNorm(config=config, 
                               num_channels=config.latent_channels,
                               array_length=4, #len(x.shape for 2D datasets)
                               channel_at_last_position=False)

        # build main part
        self.spconv_layers,self.conv_layers = build_fno(
            fno_width=config.latent_channels,
            num_fno_modes=num_fno_modes,
            num_fno_layers=config.num_fno_layers,
            dimension=2,
        )

    def decoder_net(self) -> nn.Module:
        return FullyConnected(
            in_features=self.config.latent_channels,
            layer_size=self.config.decoder_layer_size,
            out_features=self.config.out_size,
            num_layers=self.config.decoder_layers,
            activation_fn=self.config.decoder_activation_fn_name,
        )

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                "Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D FNO"
            )

        if self.config.coord_features: #TODO: Do this for ALL the models
            coord_feat = twod_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)
        # (left, right, top, bottom)
        x = F.pad(x, (0, self.pad[1], 0, self.pad[0]), mode=self.padding_type)
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:   
                x = conv(x) + w(x)
                x = self.norm(x, **kwargs)
                x = self.activation_fn(x)
            else:
                x = conv(x) + w(x)
                x = self.norm(x, **kwargs)
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
    

class FNO3D(PreTrainedModel):
    """3D Spectral encoder for FNO"""

    def __init__(self, config, activation_fn: nn.Module) -> None:
        super().__init__(config)

        self.activation_fn = activation_fn

        # Padding values for spectral conv
        if isinstance(config.padding, int):
            padding = [config.padding, config.padding, config.padding]
        padding = padding + [0, 0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:3]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = config.padding_type

        if isinstance(config.num_fno_modes, int):
            num_fno_modes = [config.num_fno_modes, config.num_fno_modes, config.num_fno_modes]

        # build lift
        self.lift_network = build_lift_network(
            in_channels=config.in_size,
            fno_width=config.latent_channels,
            activation_fn=self.activation_fn,
            dimension=3,
        )

        self.norm = CustomNorm(config=config, 
                               num_channels=config.latent_channels,
                               array_length=5, #len(x.shape for 3D datasets)
                               channel_at_last_position=False)

        # build main part
        self.spconv_layers,self.conv_layers = build_fno(
            fno_width=config.latent_channels,
            num_fno_modes=num_fno_modes,
            num_fno_layers=config.num_fno_layers,
            dimension=3,
        )

    def decoder_net(self) -> nn.Module:
        return FullyConnected(
            in_features=self.config.latent_channels,
            layer_size=self.config.decoder_layer_size,
            out_features=self.config.out_size,
            num_layers=self.config.decoder_layers,
            activation_fn=self.config.decoder_activation_fn_name,
        )

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if self.config.coord_features:
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
                x = conv(x) + w(x)
                x = self.norm(x, **kwargs)
                x = self.activation_fn(x)
            else:
                x = conv(x) + w(x)
                x = self.norm(x, **kwargs)

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