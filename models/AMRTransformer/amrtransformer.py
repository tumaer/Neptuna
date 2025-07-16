from typing import Optional
from transformers import PreTrainedModel
import torch
from torch import nn, Tensor
from models.AMRTransformer.amrtransformer_utils import Normalizer, SinusoidalPositionalEncoding3D
from utils.grid_utils import twod_meshgrid


class AMRTransformer(PreTrainedModel):

    main_input_name = "input_data"
    conditioning_input_name = "conditioning_input_data"

    def __init__(self, config):
        super().__init__(config)
        
        self.dimension = config.dimension
        self.amrtransformer = self.build_AMRTransformer()(config)

    def build_AMRTransformer(self):
        if self.dimension == 2:
            return AMRTransformer2D
        else:
            raise NotImplementedError("Invalid dimensionality. Only 2D AMRTransformer implemented.")

    def forward(self, input_data: Tensor) -> Tensor:

        if input_data is None:
            raise ValueError("input_data cannot be None")
        
        #batch, input_seq, channels, *spatial = input_data.shape
        #input_data = input_data.reshape(batch, input_seq * channels, *spatial)

        # input_data shape: (B, N, dim, k, k)
        # Expected inputs shape: (B, N, k, k, dim)
        input_data = input_data.permute(0, 1, 4, 2, 3)

        y = self.amrtransformer(input_data)

        return y



class AMRTransformer2D(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        encoder_layer = nn.TransformerEncoderLayer(config.d_model, config.num_heads, config.dim_feedforward, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.linear1 = nn.Linear(config.in_channels * config.patch_size * config.patch_size + config.n_case_params, config.d_model)  # dim * patch_size * patch_size + ncp
        self.linear2 = nn.Linear(config.d_model, config.out_channels * config.patch_size * config.patch_size)

        self.positional_encoding_3d = SinusoidalPositionalEncoding3D(config.d_model)

        self._output_normalizer = Normalizer(size=config.out_channels, name='output_normalizer', device='cpu') # ToDo !!!
        self._node_normalizer = Normalizer(size=config.in_channels, name='node_normalizer', device='cpu')

    def forward(
            self,
            inputs: Tensor,
            case_params: Tensor
    ) -> Tensor:
        """
        Args:
        - inputs shape: (B, N, k, k, dim)
        - case_params: (B, 3)

        Returns:
            (B, out_chan, h, w) or (B, out_chan, h, w), loss
        """

        # inputs shape: (B, N, k, k, dim)

        # ToDo: !!! look into case parameters
        max_size_para = case_params[:, -1]
        case_params = case_params[:, :-1]
        # temp_inputs = inputs.clone()
        # mask = mask.unsqueeze(1)  # (B, 1, h, w)

        depth = inputs[:, :, :, :, -3].mean(dim=(2, 3))  # shape: (B, N)
        x = inputs[:, :, :, :, -2].mean(dim=(2, 3))  # shape: (B, N)
        y = inputs[:, :, :, :, -1].mean(dim=(2, 3))  # shape: (B, N)
        x = x / max_size_para.unsqueeze(1)  # shape: (B, N)
        y = y / max_size_para.unsqueeze(1)  # shape: (B, N)
        pos_encoded = self.positional_encoding_3d(x, y, depth)  # shape: (B, N, d_model)

        inputs = inputs[:, :, :, :, :-3]  # (B, N, k, k, dim-3)

        B, N, k, _, dim = inputs.shape

        inputs = self._node_normalizer(inputs.reshape(B * N * k * k, -1))

        # residual = residual.reshape(B*N, -1)  # (B, N, out_c*k*k)
        inputs = inputs.reshape(B, N, dim * k * k)  # shape: (B, N, dim*k*k)

        # Add case params as additional channels
        case_params = case_params.unsqueeze(1).expand(-1, N, -1)  # (B, N, c)
        inputs_combined = torch.cat(
            [inputs, case_params], dim=2
        )  # (B, N, dim*k*k+c)

        inputs_combined = self.linear1(inputs_combined)
        # inputs_combined = self.LN_input(inputs_combined)
        inputs_combined += pos_encoded

        inputs_transformed = self.transformer(inputs_combined)  # (B, N, d_model)
        inputs_final = self.linear2(inputs_transformed)  # (B, N, out_c*k*k)

        preds = inputs_final

        preds = preds.reshape(B, N, k, k, dim)

        return preds