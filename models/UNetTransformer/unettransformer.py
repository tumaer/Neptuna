import math
import torch
from torch import Tensor
import torch.nn as nn
from utils import activation_func

from transformers import PreTrainedModel
from .unettransformer_utils import ConservativeDownsampling, ECAttention, MultiHeadECA, ResidualBlockND, SEAttention, MultiHeadSEAttention, MiddleBlockND, DownsampleND, UpsampleND, UNetTransformerConfig
from utils.grid_utils import twod_meshgrid
from utils.model_utils import CustomNorm
#from .dca import DCA
import itertools

class UNetTransformer(PreTrainedModel): 

    main_input_name = "input_data"
    conditioning_input_name = "conditioning_input_data"
    config_class = UNetTransformerConfig
    
    def __init__(self, config):
        
        super().__init__(config)
        super().post_init()

        self.config = config

        activation = activation_func.get_activation(config.activation_fn_name)
        if activation is None:
            raise NotImplementedError(f"Activation {config.activation_fn_name} not implemented")
        
        self.unet_transformer = self.build_UNetTransformer()(config=config, activation=activation)
       
    def build_UNetTransformer(self):
        """Get the appropriate upsampler based on the dimension."""
        if self.config.dimension == 2:
            return UNetTransformer2D
        else:
            raise NotImplementedError(f"UNetTransformer not implemented for dimension {self.config.dimension}")

    
    ### Main Forward function ###
    def forward(self, input_data: Tensor, **kwargs) -> Tensor:
        
        if "conditioning_input_data" in kwargs:
            #NOTE: Conditioning data can be passed into a conv network before concatination with input_data.
            conditioning_input_data = kwargs["conditioning_input_data"]
            input_data = torch.cat([input_data, conditioning_input_data], dim=2)
        else:
            conditioning_input_data = None

        batch, input_seq, input_channels, *spatial = input_data.shape
        x = input_data.reshape(batch, input_seq * input_channels, *spatial)
        return self.unet_transformer(x, **kwargs)


class UNetTransformer2D(PreTrainedModel):
    """2D U-Net Transformer"""
    def __init__(self, config, activation: nn.Module):
        super().__init__(config)

        torch.autograd.set_detect_anomaly(True)

        if (config.grid_resolution[0] & -config.grid_resolution[0]).bit_length() <= config.num_grids - 1 and (config.grid_resolution[1] & -config.grid_resolution[1]).bit_length() <= config.num_grids - 1:
            raise ValueError("Grid resolution must be divisible by 2^num_grids. Reduce num_grids or adjust grid_resolution.")

        if config.downsample_method == "average":
            self.grids_creation = nn.ModuleList([nn.Identity()]+[
                nn.AvgPool2d(kernel_size=2**i, stride=2**i)
                for i in range(1, config.num_grids)
            ])
        elif config.downsample_method == "conservative":
            self.grids_creation = ConservativeDownsampling(config)
        else:
            raise NotImplementedError(f"Invalid downsampling method: {config.downsample_method}.")

        self.activation = activation
        # Number of resolutions (depth of unet)
        unet_depth = len(config.channel_multiplier)
        
        out_channels_list = [
            config.latent_channels * math.prod(config.channel_multiplier[:i])
            for i in range(config.num_grids)
        ]
        # Project image into feature map
        self.grid_conv = nn.ModuleList([
            nn.Conv2d(config.in_size, out_channels_list[i], kernel_size=(3, 3), padding=(1, 1))
            for i in range(config.num_grids)
            ])
        self.conv_down = nn.ModuleList([
            nn.Conv2d(2*out_channels_list[i+1], out_channels_list[i+1], kernel_size=(3, 3), padding=(1, 1))
            for i in range(config.num_grids-1)
            ])

        width = config.grid_resolution[0]
        height = config.grid_resolution[1]
        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels_down = in_channels_down = config.latent_channels
        channels_list = [in_channels_down]
        # For each resolution
        for i in range(unet_depth):
            # Number of output channels at this resolution
            out_channels_down = in_channels_down * config.channel_multiplier[i]
            # Add `n_blocks`
            for _ in range(config.n_blocks):
                down.append(
                    ResidualBlockND(
                        config=config,
                        in_channels=in_channels_down,
                        out_channels=out_channels_down,
                        dim=2,
                        activation=config.activation_fn_name,
                        shift=(i%2==1),
                        input_resolution=(width // (2**i), height // (2**i)),
                    )
                )
                in_channels_down = out_channels_down
                channels_list.append(in_channels_down)
            # Down sample at all resolutions except the last
            if i < unet_depth - 1:
                down.append(DownsampleND(n_channels=in_channels_down, dim=2))
                if config.attention_concat_type is not None:
                    if config.attention_concat_type == "se":
                        down.append(SEAttention(channels=2*in_channels_down))
                    elif config.attention_concat_type == "m-se":
                        down.append(MultiHeadSEAttention(channels=2*in_channels_down, num_heads=config.num_heads))
                    elif config.attention_concat_type == 'ec':
                        down.append(ECAttention(channels=2*in_channels_down))
                    elif config.attention_concat_type == 'm-ec':
                        down.append(MultiHeadECA(channels=2*in_channels_down, num_heads=config.num_heads))
                    else:
                        raise NotImplementedError(f"Attention concat type {config.attention_concat_type} not implemented.")
                channels_list.append(in_channels_down)

            

        # Combine the set of modules
        self.down = nn.ModuleList(down)
        
        out_channels_mid = out_channels_down
        self.middle = MiddleBlockND(config=config,
                                    n_channels=out_channels_mid, 
                                    dim=2, 
                                    activation=config.activation_fn_name, 
                                    input_resolution=(width // (2**(unet_depth-1)), height // (2**(unet_depth-1)))
                                    )
        
        patch_size = 8 # for 512: 16 # 4 for paper patch size -> 28 patches
        self.use_dca = config.use_dca
        if config.use_dca:
            self.dca = DCA(n=1, # number of DCA blocks stacked
                        features = channels_list, 
                        strides=[patch_size // e for e in [y for x in itertools.accumulate(config.channel_multiplier, lambda a, b: a * b) for y in (x, x)]], 
                        patch=width // patch_size, 
                        spatial_att=True, 
                        channel_att=True, 
                        spatial_head=[4] * len(channels_list), 
                        channel_head=[1] * len(channels_list),
                        )

        # #### Second half of U-Net - increasing resolution
        up = []
        attn = []
        # Number of channels
        in_channels_up = out_channels_mid
        # For each resolution
        for i in reversed(range(unet_depth)):
            # `n_blocks` at the same resolution
            out_channels_up = in_channels_up
            for _ in range(config.n_blocks):
                up.append(
                    ResidualBlockND(
                        config=config,
                        in_channels=in_channels_up + out_channels_up,
                        out_channels=out_channels_up,
                        dim=2,
                        activation=config.activation_fn_name,
                        shift=(i%2==1),
                        input_resolution=(width // (2**i), height // (2**i))
                    )
                )
                if config.attention_concat_all:
                    if config.attention_concat_type == "se":
                        attn.append(SEAttention(channels=in_channels_up + out_channels_up))
                    elif config.attention_concat_type == "m-se":
                        attn.append(MultiHeadSEAttention(channels=in_channels_up + out_channels_up, num_heads=config.num_heads))
                    elif config.attention_concat_type == 'ec':
                        attn.append(ECAttention(channels=in_channels_up + out_channels_up))
                    elif config.attention_concat_type == 'm-ec':
                        attn.append(MultiHeadECA(channels=in_channels_up + out_channels_up, num_heads=config.num_heads))
                    else:
                        raise NotImplementedError(f"Attention concat type {config.attention_concat_type} not implemented.")
            # Final block to reduce the number of channels
            out_channels_up = in_channels_up // config.channel_multiplier[i]
            up.append(ResidualBlockND(config=config,
                                in_channels=in_channels_up + out_channels_up, 
                                out_channels=out_channels_up, 
                                dim=2, 
                                activation=config.activation_fn_name, 
                                input_resolution=(width // (2**i), height // (2**i))
                                ))
            if config.attention_concat_all:
                if config.attention_concat_type == "se":
                    attn.append(SEAttention(channels=in_channels_up + out_channels_up))
                elif config.attention_concat_type == "m-se":
                    attn.append(MultiHeadSEAttention(channels=in_channels_up + out_channels_up, num_heads=config.num_heads))
                elif config.attention_concat_type == 'ec':
                    attn.append(ECAttention(channels=in_channels_up + out_channels_up))
                elif config.attention_concat_type == 'm-ec':
                    attn.append(MultiHeadECA(channels=in_channels_up + out_channels_up, num_heads=config.num_heads))
                else:
                    raise NotImplementedError(f"Attention concat type {config.attention_concat_type} not implemented.")
            in_channels_up = out_channels_up
            # Up sample at all resolutions except last
            if i > 0:
                up.append(UpsampleND(n_channels=in_channels_up, dim=2))
                if config.attention_concat_all:
                    attn.append(nn.Identity())

        # Combine the set of modules
        self.up = nn.ModuleList(up)
        if config.attention_concat_all:
            self.attn = nn.ModuleList(attn)
        self.attention_concat_all = config.attention_concat_all

        self.norm = CustomNorm(config=config, num_channels=config.latent_channels, array_length=4, channel_at_last_position=False)
        self.final = nn.Conv2d(in_channels_up, config.out_size, kernel_size=(3, 3), padding=(1, 1))



    def forward(self, x: torch.Tensor, **kwargs):
        if x.dim() != 4:
            raise ValueError(
                "Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D UNet"
            )
        
        if self.config.downsample_method == "conservative":
            grids = self.grids_creation(x, **kwargs) # list of [8, 3, 256/2^i, 64/2^i]
        else:
            grids = []
            for i in range(self.config.num_grids):
                grid = self.grids_creation[i](x) # [8, 3, 256/2^i, 64/2^i]

                if self.config.coord_features:
                    coord_feat = twod_meshgrid(list(grid.shape), grid.device)
                    downsampled = torch.cat((grid, coord_feat), dim=1) # [8, 3+2, 256/2^i, 64/2^i]

                grids.append(downsampled)


        x = self.grid_conv[0](grids[0])

        h = [x]
        for i,m in enumerate(self.down):
            if isinstance(m, MultiHeadSEAttention) or isinstance(m, SEAttention) or isinstance(m, MultiHeadECA) or isinstance(m, ECAttention):
                j = (i - 2) // 3 + 1
                grid = self.grid_conv[j](grids[j])
                # Apply attention to the downsampled grid and the grid of same resolution 
                x = torch.cat((x, grid), dim=1)
            x = m(x, **kwargs)

            if isinstance(m, MultiHeadSEAttention) or isinstance(m, SEAttention) or isinstance(m,  MultiHeadECA) or isinstance(m, ECAttention):
                x = self.conv_down[(i - 2) // 3](x)
            else:
                h.append(x)
                
                
        if self.use_dca:
            h = self.dca(h)
        x = self.middle(x, **kwargs)

        for i, m in enumerate(self.up):
            if isinstance(m, UpsampleND):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                if self.attention_concat_all:
                    x = self.attn[i](x, **kwargs)
                x = m(x, **kwargs)

        x = self.norm(x, **kwargs)
        x = self.activation(x)
        x = self.final(x)

        return x