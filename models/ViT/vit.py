from typing import Optional, Tuple, Union
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.vit.modeling_vit import ViTPreTrainedModel

from utils.model_utils import CustomNorm
from .vit_utils import ViTEmbeddings, ViTEncoder
import torch
from torch import nn, Tensor
from .vit_utils import ViTConfig
from utils.grid_utils import twod_meshgrid


class ViTModel(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config)

        res_dim = config.grid_resolution[0] // config.patch_size * config.grid_resolution[1] // config.patch_size + 1
        self.layernorm = CustomNorm(config=config, num_channels=config.latent_channels, array_length=3, channel_at_last_position=True)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output, **kwargs)

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    

class ViT(PreTrainedModel):

    main_input_name = "input_data"
    conditioning_input_name = "conditioning_input_data"

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        if config.dimension != 2:
            raise ValueError(
                f"ViT model only supports 2D inputs, but got dimension={config.dimension}"
            )
        
        if config.grid_resolution[0] % config.patch_size != 0 or config.grid_resolution[1] % config.patch_size != 0:
            raise ValueError("Specified patch_size does not fit to dataset. grid_resolution is not divisable by patch_size.")

        self.vit = ViT2D(config=config)
        
        

    def forward(self, input_data: Tensor, **kwargs) -> Tensor:

        if input_data is None:
            raise ValueError("input_data cannot be None")
        
        batch, input_seq, channels, *spatial = input_data.shape
        input_data = input_data.reshape(batch, input_seq * channels, *spatial)

        if self.config.coord_features:
            coord_feat = twod_meshgrid(list(input_data.shape), input_data.device)
            input_data = torch.cat((input_data, coord_feat), dim=1)

        y = self.vit(input_data, **kwargs)

        return y


class ViT2D(ViTModel):
    def __init__(self, config: ViTConfig) -> None:

        super().__init__(config)
        self.config = config

        self.vit = ViTModel(config, use_mask_token=True)

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=config.latent_channels,
                out_channels=config.encoder_stride**2 * config.out_channels,
                kernel_size=1,
            ),
            nn.PixelShuffle(config.encoder_stride),
        )

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_data: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tensor:

        # ToDo: add head_mask in case necessary
        bool_masked_pos = None
        head_mask = None
        
        if input_data is None:
            raise ValueError("You have to specify input_data")

        if bool_masked_pos is not None and (self.config.patch_size != self.config.encoder_stride):
            raise ValueError(
                "When `bool_masked_pos` is provided, `patch_size` must be equal to `encoder_stride` to ensure that "
                "the reconstructed image has the same dimensions as the input. "
                f"Got `patch_size` = {self.config.patch_size} and `encoder_stride` = {self.config.encoder_stride}."
            )

        outputs = self.vit(
            input_data, #[6, 5, 160, 160]
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
            interpolate_pos_encoding=self.config.interpolate_pos_encoding,
            return_dict=False,
            **kwargs
        )

        sequence_output = outputs[0] #[6, 101, 768]

        # Reshape to (batch_size, num_channels, height, width)
        sequence_output = sequence_output[:, 1:] #[6, 100, 768]
        batch_size, sequence_length, num_channels = sequence_output.shape #6, 100, 768
        height = self.config.grid_resolution[0] // self.config.patch_size
        width = self.config.grid_resolution[1] // self.config.patch_size
        assert height * width == sequence_length, "Something went wrong with the sequence length"
        sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width) # [6, 768, 10, 10]

        # Reconstruct pixel values
        reconstructed_input_data = self.decoder(sequence_output) # [6, 1, 160, 160]

        return reconstructed_input_data