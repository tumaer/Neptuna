### Model taken from Poseidon: Efficient Foundation Models for PDEs. 
# Github: https://github.com/camlab-ethz/poseidon?tab=readme-ov-file 
# Paper: https://arxiv.org/abs/2405.19101

import math
from transformers import PreTrainedModel
from transformers.models.swinv2.modeling_swinv2 import (
    Swinv2EncoderOutput,
)
import torch
from torch import nn, Tensor
from typing import Optional, Union, Tuple, List
import collections
from utils.grid_utils import twod_meshgrid
from .scot_utils import ScOTOutput, ScOTEmbeddings, ScOTPatchRecovery, ScOTPatchMerging, ScOTPatchUnmerging, ScOTLayer, ConvNeXtBlock, ResNetBlock
from utils.model_utils import CustomNorm
from .scot_utils import ScOTConfig
class ScOTEncodeStage(nn.Module):
    def __init__(
        self,
        config,
        dim,
        input_resolution,
        depth,
        num_heads,
        drop_path,
        downsample,
        pretrained_window_size=0,
    ):
        super().__init__()
        self.config = config
        self.dim = dim #48
        window_size = ( # (16, 16)
            config.window_size
            if isinstance(config.window_size, collections.abc.Iterable)
            else (config.window_size, config.window_size)
        )
        self.blocks = nn.ModuleList(
            [
                ScOTLayer(
                    config=config,
                    dim=dim, #48
                    input_resolution=input_resolution, #(32, 32)
                    num_heads=num_heads, #3
                    shift_size=( # every second layer [0, 0], otherwise half of window_size: 16/2 = 8
                        [0, 0]
                        if (i % 2 == 0)
                        else [window_size[0] // 2, window_size[1] // 2]
                    ),
                    drop_path=drop_path[i], #[0.0, 0.0, 0.0, 0.0]
                    pretrained_window_size=pretrained_window_size, #0
                )
                for i in range(depth) #4
            ]
        )

        # patch merging layer
        if downsample is not None: # ScOTPatchMerging in every stage except last one

            self.downsample = downsample(
                config, input_resolution, dim=dim, norm_layer=CustomNorm
            )
        else:
            self.downsample = None

        self.pointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        
        height, width = input_dimensions
        inputs = hidden_states

        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                layer_head_mask,
                output_attentions,
                always_partition,
                **kwargs
            )

            hidden_states = layer_outputs[0] # 16, 1024, 48

        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)# 16, 16
            hidden_states = self.downsample(
                hidden_states_before_downsampling + inputs, input_dimensions, **kwargs # why residual connection here? or is it sth else?
            )
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (
            hidden_states, # 16, 256, 96
            hidden_states_before_downsampling, # 16, 1024, 48
            output_dimensions,# 32, 32, 16, 16
        )

        if output_attentions: # False
            stage_outputs += layer_outputs[1:]
        return stage_outputs


class ScOTDecodeStage(nn.Module):
    def __init__(
        self,
        config,
        dim, #384
        input_resolution, #(4, 4)
        depth, #4
        num_heads, # 24,
        drop_path, # [0, 0, 0, 0]
        upsample, # PatchUnmerging
        upsampled_size, # (8, 8)
        pretrained_window_size=0,
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.dim = dim # 384
        window_size = ( # (16, 16)
            config.window_size
            if isinstance(config.window_size, collections.abc.Iterable)
            else (config.window_size, config.window_size)
        )
        self.blocks = nn.ModuleList(
            [
                ScOTLayer(
                    config=config,
                    dim=dim, # 384
                    input_resolution=input_resolution, # 4
                    num_heads=num_heads, # 24
                    shift_size=(
                        [0, 0]
                        if (i % 2 == 0)
                        else [window_size[0] // 2, window_size[1] // 2] # 8
                    ),
                    drop_path=drop_path[depth - 1 - i],  # reversed!
                    pretrained_window_size=pretrained_window_size,
                )
                for i in reversed(range(depth))  # reversed !
            ]
        )

        if upsample is not None: # upsample in every layer except last one

            self.upsample = upsample(config, input_resolution, dim=dim, norm_layer=CustomNorm) # PatchUnmerging
            self.upsampled_size = upsampled_size
        else:
            self.upsample = None

        self.pointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        height, width = input_dimensions

        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                layer_head_mask,
                output_attentions,
                always_partition,
                **kwargs
            )

            hidden_states = layer_outputs[0]

        hidden_states_before_upsampling = hidden_states
        if self.upsample is not None:
            height_upsampled, width_upsampled = self.upsampled_size
            output_dimensions = (height, width, height_upsampled, width_upsampled)
            hidden_states = self.upsample(
                hidden_states_before_upsampling,
                (height_upsampled, width_upsampled),
                **kwargs
            )
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (
            hidden_states,
            hidden_states_before_upsampling,
            output_dimensions,
        )

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


class ScOTEncoder(nn.Module):
    """
    This is just a Swinv2Encoder with changed dpr.
    We just have to change the drop path rate since we also have a decoder by default.
    """

    def __init__(self, config, grid_size, pretrained_window_sizes=(0, 0, 0, 0)):
        super().__init__()
        self.num_layers = len(config.depths) # 4
        self.config = config
        if self.config.pretrained_window_sizes is not None: # used to support pretrained window masks
            pretrained_window_sizes = config.pretrained_window_sizes
        drop_rates_encode_decode = torch.linspace( # Create linearly increasing sequence of drop probabilities ranging from 0 to drop_path_rate
            0, config.drop_path_rate, 2 * sum(config.depths) # twice the number of total blocks bc of encoder AND decoder
        ) # drop_path_rate = 0.0; depths = [4, 4, 4, 4]
        # 2 * (4 + 4 + 4 + 4) = 32 (total blocks in encoder AND decoder) -> array of 0s of length 32

        dpr = [ # array (instead of tensor) of the encoder part of drop_rates_encode_decode
            x.item()
            for x in drop_rates_encode_decode[: drop_rates_encode_decode.shape[0] // 2] # only first half (encoder) for drop path rates
        ]
        self.layers = nn.ModuleList(
            [
                ScOTEncodeStage(
                    config=config,
                    dim=int(config.latent_channels * 2**i_layer), # 48, 96, 192, 384 doubles at each layer
                    input_resolution=(
                        grid_size[0] // (2**i_layer), # 32, 16, 8, 4 half at each layer
                        grid_size[1] // (2**i_layer),
                    ),
                    depth=config.depths[i_layer], #[4, 4, 4, 4]
                    num_heads=config.num_heads[i_layer], # [3, 6, 12, 24]
                    drop_path=dpr[ # each ScOTEncodeStage receives its own slice of drop path schedule
                        sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1]) # -> dpr[0: 4]
                    ], # if used, should increase linearly with depth -> residual connection is randomly dropped during training
                    downsample=(
                        ScOTPatchMerging if (i_layer < self.num_layers - 1) else None # downsample in every stage except the last
                    ),
                    pretrained_window_size=pretrained_window_sizes[i_layer], # [0, 0, 0, 0]
                )
                for i_layer in range(self.num_layers) # 4
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor, # [16, 1024, 48] input
        input_dimensions: Tuple[int, int], # [32, 32]
        head_mask: Optional[torch.FloatTensor] = None, #[None, None, None, None]
        output_attentions: Optional[bool] = False, # False
        output_hidden_states: Optional[bool] = False, # True
        output_hidden_states_before_downsampling: Optional[bool] = False, # True
        always_partition: Optional[bool] = False, # False
        **kwargs
    ) -> Union[Tuple, Swinv2EncoderOutput]:
        all_hidden_states = () if output_hidden_states else None # ()
        all_reshaped_hidden_states = () if output_hidden_states else None # ()
        all_self_attentions = () if output_attentions else None # None

        if output_hidden_states: # True # save hidden_states in all_hidden_states to return them later
            batch_size, _, hidden_size = hidden_states.shape #48, 16
            # rearrange b (h w) c -> b c h w
            reshaped_hidden_state = hidden_states.view( # [16, 32, 32, 48]
                batch_size, *input_dimensions, hidden_size
            )
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2) # [16, 48, 32, 32]
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers): # 4 layers
            layer_head_mask = head_mask[i] if head_mask is not None else None # None

            if self.gradient_checkpointing and self.training: # False, True
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    input_dimensions,
                    layer_head_mask,
                    output_attentions,
                    **kwargs
                )
            else: # call forward of ScOTEncodeStage
                layer_outputs = layer_module(
                    hidden_states,
                    input_dimensions,
                    layer_head_mask,
                    output_attentions,
                    always_partition,
                    **kwargs
                )

            # hidden_states before: [16, 1024, 48]
            hidden_states = layer_outputs[0] # [16, 256, 96]
            hidden_states_before_downsampling = layer_outputs[1] # [16, 1024, 48]
            output_dimensions = layer_outputs[2] # (32, 32, 16, 16)
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_downsampling: # True, True # save hidden_states before and after downsampling to return later
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                # rearrange b (h w) c -> b c h w
                # here we use the original (not downsampled) height and width
                reshaped_hidden_state = hidden_states_before_downsampling.view(
                    batch_size,
                    *(output_dimensions[0], output_dimensions[1]),
                    hidden_size,
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_downsampling: # only save hidden_states after downsampling
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange b (h w) c -> b c h w
                reshaped_hidden_state = hidden_states.view(
                    batch_size, *input_dimensions, hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions: # False
                all_self_attentions += layer_outputs[3:]

        return Swinv2EncoderOutput( # Just a return data class
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


class ScOTDecoder(nn.Module):
    """Here we do reverse encoder."""

    def __init__(self, config, grid_size, pretrained_window_sizes=(0, 0, 0, 0)):
        super().__init__()
        self.num_layers = len(config.depths) # 4
        self.config = config
        if self.config.pretrained_window_sizes is not None:
            pretrained_window_sizes = config.pretrained_window_sizes # (0, 0, 0, 0)
        drop_rates_encode_decode = torch.linspace(
            0, config.drop_path_rate, 2 * sum(config.depths) # same as for encoder
        )
        dpr = [
            x.item()
            for x in drop_rates_encode_decode[drop_rates_encode_decode.shape[0] // 2 :] # only second parth (decoder) used for drop path rates
        ]
        self.layers = nn.ModuleList(
            [
                ScOTDecodeStage(
                    config=config,
                    dim=int(config.latent_channels * 2**i_layer), # 384, 192, 96, 48 halves at each decode stage
                    input_resolution=(
                        grid_size[0] // (2**i_layer), # 4, 8, 16, 32 doubles at each decode stage
                        grid_size[1] // (2**i_layer),
                    ),
                    depth=config.depths[i_layer], # 4
                    num_heads=config.num_heads[i_layer], # [24, 12, 6, 3]
                    drop_path=dpr[ # here not used, since all 0 (used for drop_path)
                        sum(config.depths[i_layer + 1 :]) : sum(config.depths[i_layer:])
                    ],
                    upsample=ScOTPatchUnmerging if i_layer > 0 else None, # Upsample between stages (not after last one)
                    upsampled_size=(
                        grid_size[0] // (2 ** (i_layer - 1)), #[8, 16, 32]
                        grid_size[1] // (2 ** (i_layer - 1)),
                    ),
                    pretrained_window_size=pretrained_window_sizes[i_layer], # (0, 0, 0, 0)
                )
                for i_layer in reversed(range(self.num_layers)) # reversed! 3 -> 0
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor, #[16, 16, 384]
        input_dimensions: Tuple[int, int], #(4,4)
        skip_states: List[torch.FloatTensor], # list(3)
        head_mask: Optional[torch.FloatTensor] = None, # [None, None, None, None]
        output_attentions: Optional[bool] = False, # False
        output_hidden_states: Optional[bool] = False, # False
        output_hidden_states_before_upsampling: Optional[bool] = False, # False
        always_partition: Optional[bool] = False, # False
        **kwargs
    ) -> Union[Tuple, Swinv2EncoderOutput]:
        all_hidden_states = () if output_hidden_states else None # None
        all_reshaped_hidden_states = () if output_hidden_states else None # None
        all_self_attentions = () if output_attentions else None # None

        if output_hidden_states: # False
            batch_size, _, hidden_size = hidden_states.shape
            # rearrange b (h w) c -> b c h w
            reshaped_hidden_state = hidden_states.view(
                batch_size, *input_dimensions, hidden_size
            )
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None # None

            if i != 0 and skip_states[len(skip_states) - i] is not None:
                # residual connection
                hidden_states = hidden_states + skip_states[len(skip_states) - i]
            if self.gradient_checkpointing and self.training: # False, True
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    input_dimensions,
                    layer_head_mask,
                    output_attentions,
                    **kwargs
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    input_dimensions,
                    layer_head_mask,
                    output_attentions,
                    always_partition,
                    **kwargs
                )

            hidden_states = layer_outputs[0] # [16, 64, 192]
            hidden_states_before_upsampling = layer_outputs[1] # [16, 16, 384]
            output_dimensions = layer_outputs[2] # (4, 4, 8, 8)

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_upsampling: # False, False
                batch_size, _, hidden_size = hidden_states_before_upsampling.shape
                # rearrange b (h w) c -> b c h w
                # here we use the original (not downsampled) height and width
                reshaped_hidden_state = hidden_states_before_upsampling.view(
                    batch_size,
                    *(output_dimensions[0], output_dimensions[1]),
                    hidden_size,
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_upsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_upsampling:# False
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange b (h w) c -> b c h w
                reshaped_hidden_state = hidden_states.view(
                    batch_size, *input_dimensions, hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions: # False
                all_self_attentions += layer_outputs[3:]

        return Swinv2EncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )

    

class ScOT(PreTrainedModel):

    main_input_name = "input_data"
    conditioning_input_name = "conditioning_input_data"
    config_class = ScOTConfig
    base_model_prefix = "scot" # will it work?
     
    def __init__(self, config):
        super().__init__(config)

        self.dimension = config.dimension

        self.scot = self.build_ScOT()(config)

        self.post_init()

    def build_ScOT(self):
        if self.dimension == 2:
            return ScOT2D
        else:
            raise NotImplementedError("Invalid dimensionality. Only 2D ScOT implemented.")

    

    def forward(self, input_data: Tensor, **kwargs) -> Tensor:        
        if "conditioning_input_data" in kwargs:
            #NOTE: Conditioning data can be passed into a conv network before concatination with input_data.
            conditioning_input_data = kwargs["conditioning_input_data"]
            input_data = torch.cat([input_data, conditioning_input_data], dim=2)
        else:
            conditioning_input_data = None
        
        batch, input_seq, channels, *spatial = input_data.shape
        x= input_data.reshape(batch, input_seq * channels, *spatial)
        return self.scot(x, **kwargs)

class ScOT2D(PreTrainedModel):
    """Inspired by https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/swinv2/modeling_swinv2.py#L1129"""

    def __init__(self, config, use_mask_token=False):
        super().__init__(config)

        self.config = config
        self.num_layers_encoder = len(config.depths)
        self.num_layers_decoder = len(config.depths)
        self.num_features = int(config.latent_channels * 2 ** (self.num_layers_encoder - 1) * config.sequence_info[1]) # the channel size at the final stage of the encoder


        self.embeddings = ScOTEmbeddings(config, use_mask_token=use_mask_token) # creates patch embeddings from input
        self.encoder = ScOTEncoder(config, self.embeddings.patch_grid) # processes embedded input, extracting features
        self.decoder = ScOTDecoder(config, self.embeddings.patch_grid) # mirrors encoder to reconstruct representation
        self.patch_recovery = ScOTPatchRecovery(config) # reconstructs final output

        if config.residual_model == "convnext":
            res_model = ConvNeXtBlock
        elif config.residual_model == "resnet":
            res_model = ResNetBlock
        else:
            raise ValueError("residual_model must be 'convnext' or 'resnet'")

        self.residual_blocks = nn.ModuleList(
            [
                (
                    nn.ModuleList(
                        [
                            res_model(config, (self.embeddings.patch_grid[0] // 2 ** i, self.embeddings.patch_grid[1] // 2 ** i), config.latent_channels * 2**i)
                            for _ in range(depth)
                        ]
                    )
                    if depth > 0
                    else nn.ModuleList([nn.Identity()])
                )
                for i, depth in enumerate(config.skip_connections)
            ]
        )

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layers[layer].attention.prune_heads(heads)
        for layer, heads in reversed(heads_to_prune.items()):
            self.decoder.layers[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_data: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[Tuple, ScOTOutput]:

        output_attentions = self.config.output_attentions # False
        output_hidden_states = self.config.output_hidden_states # False

        # calculate 5D tensor used by attention mechanism to selectively include / exclude attention heads
        # [batch_size, num_heads, seq_len, seq_len] -> [num_layers, batch_size, num_heads, seq_len, seq_len]
        # num_layers: different mask at each layer, batch_size: one mask per input sample, num_heads: mask per attention head, seq_len: attention query / key positions
        head_mask = self.get_head_mask( # [None] * 8
            head_mask, self.num_layers_encoder + self.num_layers_decoder
        )

        if isinstance(head_mask, list):
            head_mask_encoder = head_mask[: self.num_layers_encoder]
            head_mask_decoder = head_mask[self.num_layers_encoder :]
        else:
            head_mask_encoder, head_mask_decoder = head_mask.split(
                [self.num_layers_encoder, self.num_layers_decoder]
            )


        if self.config.coord_features:
            coord_feat = twod_meshgrid(list(input_data.shape), input_data.device)
            input_data = torch.cat((input_data, coord_feat), dim=1)

        embedding_output, input_dimensions = self.embeddings(
            input_data, bool_masked_pos=bool_masked_pos, **kwargs
        )

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask_encoder,
            output_attentions=output_attentions,
            output_hidden_states=True,
            output_hidden_states_before_downsampling=True, 
            **kwargs
        )

        skip_states = list(encoder_outputs[1][1:])

        for i in range(len(skip_states)):
            for block in self.residual_blocks[i]: # 2 ConvNext blocks (last skip layer: identity)
                if isinstance(block, nn.Identity):
                    skip_states[i] = block(skip_states[i])
                else: # is not Identity
                    skip_states[i] = block(skip_states[i], **kwargs)

        input_dim_x = math.ceil(input_dimensions[0] / (2 ** (len(self.config.depths) - 1)))
        input_dim_y = math.ceil(input_dimensions[1] / (2 ** (len(self.config.depths) - 1)))

        decoder_output = self.decoder(
            skip_states[-1],
            (input_dim_x, input_dim_y),
            skip_states=skip_states[:-1],
            head_mask=head_mask_decoder,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs
        )

        sequence_output = decoder_output[0] # [16, 1024, 48]
        prediction = self.patch_recovery(sequence_output)


        return prediction
