from typing import Optional, Union
from transformers import PreTrainedModel
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTPatchEmbeddings, ViTEncoder, ViTPooler, ViTModel, BaseModelOutputWithPooling
import torch
from torch import nn, Tensor
from models.ViT.vit_utils import ViTConfig
from utils.grid_utils import twod_meshgrid


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

        self.vit = ViT2D(config=config)
        
        

    def forward(self, input_data: Tensor) -> Tensor:

        if input_data is None:
            raise ValueError("input_data cannot be None")
        
        batch, input_seq, channels, *spatial = input_data.shape
        input_data = input_data.reshape(batch, input_seq * channels, *spatial)

        if self.config.coord_features:
            coord_feat = twod_meshgrid(list(input_data.shape), input_data.device)
            input_data = torch.cat((input_data, coord_feat), dim=1)

        y = self.vit(input_data)

        return y


class ViT2D(ViTModel):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):

        super().__init__(config)
        self.config = config

        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None
    ) -> Tensor:

        output_attentions = self.config.output_attentions
        self.config.output_hidden_states
        bool_masked_pos = None
        
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=self.config.interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=self.config.output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        # pooled_output shape: [6, 876] [batch, pooler_output_size] -> remove pooler
        # ToDo: patch unembedding (has currently shape [6, 101, 768] [batch, x * y patches, hidden_size])

        return sequence_output
        """return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )"""

