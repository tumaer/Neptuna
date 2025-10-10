from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .kfno_utils import FullyConnected, build_lift_network, build_fno
from utils import activation_func
from typing import Tuple, List
from transformers import PreTrainedModel
from utils.grid_utils import oned_meshgrid, twod_meshgrid, threed_meshgrid
from utils.model_utils import CustomNorm
from .kfno_utils import kFNOConfig

class kFNO(PreTrainedModel):
    """Fourier neural operator (FNO) model."""

    main_input_name = "input_data"
    conditioning_input_name = "conditioning_input_data"
    config_class = kFNOConfig
    
    def __init__(self, config) -> None:
        super().__init__(config)

        activation_fn = activation_func.get_activation(config.activation_fn_name)
        if activation_fn is None:
            raise NotImplementedError(f"Activation {config.activation_fn_name} not implemented")

        self.config = config
        self.fno = self.build_kFNO()(config=config, activation_fn=activation_fn)

        self.decoder_net = self.fno.decoder_net()

    def build_kFNO(self):
        """Get the FNO encoder based on the model dimensionality"""
        if self.config.dimension == 1:
            return kFNO1D
        elif self.config.dimension == 2:
            return kFNO2D
        elif self.config.dimension == 3:
            return kFNO3D
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

        # Main kFNO sequence: L, H, K, Q blocks
        x_latent = self.fno(input_data, **kwargs)

        # P block (decoder applied individually to each output frame)
        outputs = []
        for i in range(x_latent.shape[1]):
            frame = x_latent[:, i]
            
            # Reshape to pointwise inputs
            frame_points, _ = self.fno.grid_to_points(frame)
            
            # Decoder
            decoded_frame = self.decoder_net(frame_points)
            
            # Calculate output shape for a single frame
            batch_size = input_data.shape[0]
            out_channels = self.config.out_channels
            spatial_dim = frame.shape[-1]
            output_shape = (batch_size, out_channels, spatial_dim)
            
            # Reshape back to grid
            grid_frame = self.fno.points_to_grid(decoded_frame, output_shape)
            outputs.append(grid_frame)

        x = torch.stack(outputs, dim=1)

        return x

class kFNO1D(PreTrainedModel):
    """1D kFNO"""

    def __init__(self, config, activation_fn: nn.Module) -> None:
        super().__init__(config)

        self.activation_fn = activation_fn
        self.num_repeats = config.out_size // config.out_channels
        self.share_A_weights = config.share_A_weights
        self.share_Q_weights = config.share_Q_weights
        
        # Padding values for spectral conv
        if isinstance(config.padding, int):
            padding = [config.padding]
        self.pad = padding[:1]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = config.padding_type

        # Prepare modes for spectral conv
        if isinstance(config.num_fno_modes, int):
            num_fno_modes = [config.num_fno_modes]

        # Standard norm layer
        self.norm = CustomNorm(config=config, 
                               num_channels=config.latent_channels,
                               array_length=3,
                               channel_at_last_position=False)

        # Build model components
        # 1. Lift network (L block)
        self.lift_network = build_lift_network(
            in_channels=config.in_size,
            fno_width=config.latent_channels,
            activation_fn=self.activation_fn,
            dimension=1,
        )

        # 2. Latent Koopman encoder (H block)
        self.H_spconv_layers, self.H_conv_layers = build_fno(
            fno_width=config.latent_channels,
            num_fno_modes=num_fno_modes,
            num_fno_layers=config.num_H_layers,
            dimension=1,
        )

        # 3. Koopman operator (A block) with optional weight sharing
        if self.share_A_weights:
            self.A_spconv_layers, self.A_conv_layers = build_fno(
                fno_width=config.latent_channels,
                num_fno_modes=num_fno_modes,
                num_fno_layers=config.num_A_layers,
                dimension=1,
            )
        else:
            self.A_spconv_blocks = nn.ModuleList()
            self.A_conv_blocks = nn.ModuleList()
            for _ in range(self.num_repeats):
                A_spconv, A_conv = build_fno(
                    fno_width=config.latent_channels,
                    num_fno_modes=num_fno_modes,
                    num_fno_layers=config.num_A_layers,
                    dimension=1,
                )
                self.A_spconv_blocks.append(A_spconv)
                self.A_conv_blocks.append(A_conv)

        # 4. Koopman decoder/mixing (Q block) with separate or coupled processing
        # and optional weight sharing
        if self.share_Q_weights:
            if config.Q_type == "separate":
                self.Q_spconv_layers, self.Q_conv_layers = build_fno(
                    fno_width=config.latent_channels,
                    num_fno_modes=num_fno_modes,
                    num_fno_layers=config.num_Q_layers,
                    dimension=1,
                )
                self.Q_norm = CustomNorm(
                    config=config, 
                    num_channels=config.latent_channels,
                    array_length=3,
                    channel_at_last_position=False
                )
            elif config.Q_type == "coupled":
                temporal_fno_modes = max(1, min(self.num_repeats // 2, int(round(0.35 * self.num_repeats))))

                self.Q_spconv_layers, self.Q_conv_layers = build_fno(
                    fno_width=config.latent_channels,
                    num_fno_modes=[config.num_fno_modes, temporal_fno_modes],
                    num_fno_layers=config.num_Q_layers,
                    dimension=2,
                )
                self.Q_norm = CustomNorm(config=config, 
                            num_channels=config.latent_channels,
                            array_length=4,  
                            channel_at_last_position=False)
        else:
            self.Q_spconv_blocks = nn.ModuleList()
            self.Q_conv_blocks = nn.ModuleList()
            self.Q_norms = nn.ModuleList()
            
            for _ in range(self.num_repeats):
                if config.Q_type == "separate":
                    Q_spconv, Q_conv = build_fno(
                        fno_width=config.latent_channels,
                        num_fno_modes=num_fno_modes,
                        num_fno_layers=config.num_Q_layers,
                        dimension=1,
                    )
                    self.Q_spconv_blocks.append(Q_spconv)
                    self.Q_conv_blocks.append(Q_conv)
                    self.Q_norms.append(CustomNorm(
                        config=config, 
                        num_channels=config.latent_channels,
                        array_length=3,
                        channel_at_last_position=False
                    ))
                elif config.Q_type == "coupled":
                    temporal_fno_modes = max(1, min(self.num_repeats // 2, int(round(0.35 * self.num_repeats))))

                    Q_spconv, Q_conv = build_fno(
                        fno_width=config.latent_channels,
                        num_fno_modes=[config.num_fno_modes, temporal_fno_modes],
                        num_fno_layers=config.num_Q_layers,
                        dimension=2,
                    )
                    self.Q_spconv_blocks.append(Q_spconv)
                    self.Q_conv_blocks.append(Q_conv)
                    self.Q_norms.append(CustomNorm(config=config, 
                                num_channels=config.latent_channels,
                                array_length=4,  
                                channel_at_last_position=False))

    def decoder_net(self) -> nn.Module:
        return FullyConnected(
            in_features=self.config.latent_channels,
            layer_size=self.config.decoder_layer_size,
            out_features=self.config.out_channels,
            num_layers=self.config.decoder_layers,
            activation_fn=self.config.decoder_activation_fn_name,
        )

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if self.config.coord_features:
            coord_feat = oned_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)

        x = F.pad(x, (0, self.pad[0]), mode=self.padding_type)

        # Apply H block
        x = self.apply_fno_block(
            x, 
            self.H_spconv_layers, 
            self.H_conv_layers,
            self.norm, 
            **kwargs
        )

        x_prev = x
    
        # For the coupled case, collect all A-block outputs first
        if self.config.Q_type == "coupled":
            a_outputs = []
            for i in range(self.num_repeats):
                if self.share_A_weights:
                    a_out = self.apply_fno_block(
                        x_prev, 
                        self.A_spconv_layers, 
                        self.A_conv_layers,
                        self.norm,
                        use_activation=not self.config.linear_A,
                        **kwargs
                    )
                else:
                    a_out = self.apply_fno_block(
                        x_prev, 
                        self.A_spconv_blocks[i], 
                        self.A_conv_blocks[i],
                        self.norm,
                        use_activation=not self.config.linear_A,
                        **kwargs
                    )
                a_outputs.append(a_out)
                x_prev = a_out
            
            # Stack all A outputs along a new dimension
            a_stacked = torch.stack(a_outputs, dim=-1)  # [batch, channels, spatial_dim, time]
            
            # Process through the d+1 Q block
            if self.share_Q_weights:
                q_out = self.apply_fno_block(
                    a_stacked, 
                    self.Q_spconv_layers, 
                    self.Q_conv_layers,
                    self.Q_norm,
                    use_activation=True, 
                    **kwargs
                )
            else:
                q_norm = self.Q_norms[0]
                q_out = self.apply_fno_block(
                    a_stacked, 
                    self.Q_spconv_blocks[0], 
                    self.Q_conv_blocks[0],
                    q_norm,
                    use_activation=True,
                    **kwargs
                )
            
            # Split back into separate frames
            outputs = [q_out[..., i] for i in range(self.num_repeats)]
            
        else:  # For the separate case, process sequentially
            outputs = []
            for i in range(self.num_repeats):
                # Apply A block
                if self.share_A_weights:
                    a_out = self.apply_fno_block(
                        x_prev, 
                        self.A_spconv_layers, 
                        self.A_conv_layers,
                        self.norm,
                        use_activation=not self.config.linear_A,
                        **kwargs
                    )
                else:
                    a_out = self.apply_fno_block(
                        x_prev, 
                        self.A_spconv_blocks[i], 
                        self.A_conv_blocks[i],
                        self.norm,
                        use_activation=not self.config.linear_A,
                        **kwargs
                    )
                
                # Apply Q block for separate case
                if self.share_Q_weights:
                    q_out = self.apply_fno_block(
                        a_out, 
                        self.Q_spconv_layers, 
                        self.Q_conv_layers,
                        self.Q_norm,
                        use_activation=True,
                        **kwargs
                    )
                else:
                    q_out = self.apply_fno_block(
                        a_out, 
                        self.Q_spconv_blocks[i], 
                        self.Q_conv_blocks[i],
                        self.Q_norms[i],
                        use_activation=True,
                        **kwargs
                    )
                    
                outputs.append(q_out)
                x_prev = a_out
    
        # Stack the outputs
        x = torch.stack(outputs, dim=1)  # [batch, time_steps, ...]

        x = x[..., : self.ipad[0]]
        return x

    def apply_fno_block(self, x, spconv_layers, conv_layers, norm_layer, use_activation=True, **kwargs):
        """
        Helper method to process x through a block of FNO layers with optional activation
        and configurable skip connection
        """
        x_input = x
        
        for k, (conv, w) in enumerate(zip(conv_layers, spconv_layers)):
            if k < len(conv_layers) - 1:
                x = conv(x) + w(x)
                x = norm_layer(x, **kwargs)
                if use_activation:
                    x = self.activation_fn(x)
            else:
                x = conv(x) + w(x)
                x = norm_layer(x, **kwargs)
        
        if hasattr(self.config, 'skip_percentage') and self.config.skip_percentage > 0:
            skip_pct = max(0.0, min(1.0, self.config.skip_percentage))
            x = (1 - skip_pct) * x + skip_pct * x_input
        
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


class kFNO2D(PreTrainedModel):
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
    

class kFNO3D(PreTrainedModel):
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