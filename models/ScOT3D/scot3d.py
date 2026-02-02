import math
import torch
from torch import nn, Tensor
from utils.model_utils import PretrainedConfig
from transformers import PreTrainedModel
from typing import List, Optional, Tuple, Union
from transformers.pytorch_utils import meshgrid
from functools import reduce, lru_cache
from operator import mul
from transformers.models.swinv2.modeling_swinv2 import (
    Swinv2DropPath,
    Swinv2Intermediate,
    Swinv2Output,
    Swinv2EncoderOutput

)
from transformers.utils import ModelOutput
from dataclasses import dataclass

def twod_meshgrid(shape: List[int], device: torch.device) -> Tensor:
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

def threed_meshgrid(shape: List[int], device: torch.device) -> Tensor:
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

class ResNetBlock(nn.Module):
    def __init__(self, config, input_resolution, dim):
        super().__init__()
        kernel_size = 3
        self.input_resolution = input_resolution
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad) 
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad)
        self.bn1 = nn.BatchNorm3d(dim)
        self.bn2 = nn.BatchNorm3d(dim)

    def forward(self, x, **kwargs):
        # batch_size, sequence_length, hidden_size = x.shape

        input = x 
        # x = x.reshape(batch_size, self.input_resolution[0], self.input_resolution[1], self.input_resolution[2], hidden_size) 
        # x = x.permute(0, 4, 1, 2, 3).contiguous() 
        x = self.conv1(x) 
        x = self.bn1(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # x = x.permute(0, 2, 3, 4, 1).contiguous() 
        # x = x.reshape(batch_size, sequence_length, hidden_size) 
        x = x + input  
        return x

@dataclass
class ScOT3DOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class ScOT3DConfig(PretrainedConfig):

    model_type = 'scot3d'
    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
            self,
            image_size=(224, 224),
            patch_size=(2, 4, 4),
            # num_channels=3,
            # num_out_channels=1,
            # embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            skip_connections=[True, True, True],
            window_size=(2, 7, 7),
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            use_absolute_embeddings=False,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            p=1,  # for loss: 1 for l1, 2 for l2
            channel_slice_list_normalized_loss=None,  # if None will fall back to absolute loss otherwise normalized loss with split channels
            residual_model="convnext",  # "convnext" or "resnet"
            use_conditioning=False,
            learn_residual=False,  # learn the residual for time-dependent problems
            output_hidden_states: bool = False,
            output_attentions: bool = False,
            **kwargs,
            ):
        

        super().__init__(**kwargs)

        self.image_size = image_size #> from config
        self.patch_size = patch_size
        # self.num_channels = num_channels #> from config
        # self.num_out_channels = num_out_channels #> from config
        # self.embed_dim = embed_dim #> from config
        self.depths = depths
        self.num_heads = num_heads
        self.skip_connections = skip_connections
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale  
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.use_absolute_embeddings = use_absolute_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.p = p
        self.channel_slice_list_normalized_loss = channel_slice_list_normalized_loss
        self.residual_model = residual_model
        self.use_conditioning = use_conditioning
        self.learn_residual = learn_residual
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions

class ScOT3DPatchEmbeddings(nn.Module):
    """
    This class turns `input_data` of shape `(batch_size, in_channels, time_frames, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, hidden_size, time_frames, patched_height, patched_width)` to be consumed by a
    Transformer. >  according to the ref code. TODO: Do we need to change the order in tensor shape?
    """


    def __init__(self, config) -> None:
        super().__init__()

        resolution, patch_size = config.grid_resolution, config.patch_size
        in_channels, hidden_size = config.in_channels, config.latent_channels # by latent I mean embed dim here
        self.coord_features = config.coord_features # seems extra
        self.config = config

        seq_in = config.sequence_info[0]
        resolution = [seq_in, config.grid_resolution[0], config.grid_resolution[1]]
        # resolution = resolution.append(seq_in)

        # Sanity check for 3D patch size
        if len(patch_size) != 3:
            raise ValueError(f'Patch size must be a tuple of length 3 for 3D inputs. Got {patch_size}')
        
        # input with shape (batch_size, in_channels, time_frames, height, width) to (batch_size, hidden_size, time_frames_prime, height_prime, width_prime) 
        # basec on Pytorch Conv3D https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html

        self.projection = nn.Conv3d(
            in_channels + (2 if self.coord_features else 0), 
            hidden_size, 
            kernel_size=patch_size, 
            stride=patch_size)

        
        num_patches = (resolution[0] // patch_size[0]) * (resolution[1] // patch_size[1]) * (resolution[2] // patch_size[2])
        self.num_patches = num_patches
        self.resolution = resolution
        self.patch_size = patch_size
        self.in_channels = in_channels # seems extra, from config?
        self.grid_size = ( # number of patches in each dimension of the input
            resolution[0] // patch_size[0],
            resolution[1] // patch_size[1],
            resolution[2] // patch_size[2]
        ) 


    def maybe_pad(self, input_data, time_frames, height, width):
        # _, _, time, H, W = input_data.size()
        if width % self.patch_size[2] != 0:
            pad_values = (0, self.patch_size[2] - width % self.patch_size[2])
            input_data = nn.functional.pad(input_data, pad_values)
        if height % self.patch_size[1] != 0:
            pad_values = (0, 0, 0, self.patch_size[1] - height % self.patch_size[1])
            input_data = nn.functional.pad(input_data, pad_values)
        if time_frames % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            input_data = nn.functional.pad(input_data, pad_values)
        return input_data


    def forward(self, input_data):
        input_data = input_data.permute(0, 2, 1, 3, 4)
        _, in_channels, time_frames,  height, width = input_data.shape # B, C, T, X, Y


        if in_channels != (self.in_channels) + (2 if self.coord_features else 0):
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        
        # pad the input to be divisible by self.patch_size, if needed (zero padding)
        # input_data = self.maybe_pad(input_data, time_frames, height, width)


        embeddings = self.projection(input_data)    
        _, _, T, H, W = embeddings.shape
        output_dimensions = (T, H, W)
        embeddings = embeddings.flatten(2).transpose(1, 2) # Batch, Num_patches, Hidden_size >> prepare for norm

        return embeddings, output_dimensions

class ScOT3DEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token=False):
        super().__init__()

        self.patch_embeddings = ScOT3DPatchEmbeddings(config)   
        self.config = config
        num_patches = self.patch_embeddings.num_patches
        if config.use_absolute_embeddings: # absolute position information into the patch embeddings (spatial structure of trajectory)
            self.position_embeddings = nn.Parameter( # zeros of shape (1: broadcast the same positional embeddings across all inputs in the batch, number of patches, dimensionality of each token)
                torch.zeros(1, num_patches, config.latent_channels)
            )
        else:
            self.position_embeddings = None


        self.grid_size = self.patch_embeddings.grid_size
        self.mask_token = ( # None
            nn.Parameter(torch.zeros(1, 1, config.latent_channels))
            if use_mask_token
            else None
        )

        self.norm = nn.LayerNorm(config.latent_channels)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, input_data,
                bool_masked_pos: Optional[torch.BoolTensor] = None, # None
                **kwargs
                ):

        embeddings, output_dimensions = self.patch_embeddings(input_data) 
        embeddings = self.norm(embeddings) 
        embeddings = embeddings.transpose(1, 2).view(-1, self.config.latent_channels, self.grid_size[0], self.grid_size[1], self.grid_size[2])

        # batch_size, seq_len, _, _, _ = embeddings.size()
        # if bool_masked_pos is not None: # None # Boolean masked positions: Indicates which patches are masked (1) and which aren't (0)
        #     mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
        #     # replace the masked visual tokens by mask_tokens
        #     mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
        #     embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # if self.position_embeddings is not None: # None # adds positional information to patch embeddings: spatial structure of the patch (in the trajectory) (compare with position of pixels in image)
        #     embeddings = embeddings + self.position_embeddings

        # embeddings = self.dropout(embeddings)


        return embeddings, output_dimensions

class ScOT3DPatchRecovery(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        # why we repeat some parameters which are already passed to "self" in PatchEmbedding? Like they are the same!


        resolution, patch_size = config.grid_resolution, config.patch_size
        out_channels, hidden_size = config.out_channels, config.latent_channels # by latent I mean embed dim here
        self.config = config

        seq_in = config.sequence_info[0]
        resolution = [seq_in, config.grid_resolution[0], config.grid_resolution[1]]


        num_patches = (resolution[0] // patch_size[0]) * (resolution[1] // patch_size[1]) * (resolution[2] // patch_size[2])
        self.num_patches = num_patches
        self.resolution = resolution
        self.patch_size = patch_size

        self.grid_size = ( # number of patches in each dimension of the input
            resolution[0] // patch_size[0],
            resolution[1] // patch_size[1],
            resolution[2] // patch_size[2]
        ) 
      
        self.out_channels = out_channels

        # only to change the T

        self.T_out = config.sequence_info[1] # output sequence length


        self.change_T_out = nn.Conv3d(
            in_channels=resolution[0]//patch_size[0], # 3
            out_channels=self.T_out, # 1
            kernel_size=1,
            stride=1
        ) 
       

        self.projection = nn.ConvTranspose3d(
            in_channels=hidden_size, # 27
            out_channels=out_channels, # 1
            kernel_size=(self.T_out, patch_size[1], patch_size[2]), # (2, 4, 4)
            stride=(self.T_out, patch_size[1], patch_size[2]), # (2, 4, 4)
            # padding=(3, 0, 0),
            # output_padding=(1, 0, 0),
        )


        # the following is not done in Pangu # copied from Poseidon
        self.mixup = nn.Conv3d(
            out_channels , # 1
            out_channels , # 1
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )

    def maybe_crop(self, input_data, H, W):
        if input_data.shape[2] > H:
            input_data = input_data[:, :, :, :H, :]
        if input_data.shape[3] > W:
            input_data = input_data[:, :, :, :, :W]
        return input_data

    def forward(self, hidden_states): # Hidden_state shape : B, C, T, H, W : 8, 27, 3, 64, 64 >> Output shape 8, 1, 1, 256, 256 (out_channel =1 and out_seq_len=1)
        # hidden_states = hidden_states.transpose(1, 2) # [16, 48, 1024]
        # hidden_states = hidden_states.reshape(
        #     hidden_states.shape[0], hidden_states.shape[1], *self.grid_size
        # ) # [16, 48, 32, 32]

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous() # B, C, T, H, W >> B, T, C, H, W
        hidden_states = self.change_T_out(hidden_states) # B, T_out, C, H, W
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous() # B, C, T_out, H, W

        output = self.projection(hidden_states)
        output = self.maybe_crop(output, self.resolution[1], self.resolution[2]) # check if last two dimensions have the expected dim, otherwise crop
        return self.mixup(output)

class ScOT3DPatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """
    
    def __init__(self, config, input_resolution: Tuple[int], dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        

    def maybe_pad(self, input_feature, T, H, W): # we won't pad the T dim. It is only patched in the Embedding layer. 
        should_pad = (H % 2 == 1) or (W % 2 == 1) # False
        if should_pad:
            pad_values = (0, W % 2, 0, H % 2, 0, 0)
            input_feature = nn.functional.pad(input_feature, pad_values)

        return input_feature
    
    def forward(
        self,
        input_feature: torch.Tensor,
        input_dimensions: Tuple[int, int],
        **kwargs
    ) -> torch.Tensor:
        """
        Forward function.
        Input feature, tensor size (B, D, H, W, C). TODO: check the shape
        """


        # D, H, W = input_dimensions #32, 32
        # `dim` is height * width
        batch_size, in_channels, T, H, W = input_feature.shape # 16, 1024, 48

        # input_feature = input_feature.view(batch_size, T, H, W, in_channels) # 16, 32, 32, 48
        # pad input to be divisible by width and height, if needed
        input_feature = self.maybe_pad(input_feature, T, H, W) # here: no padding needed
        # [batch_size, T , H/2, W/2, in_channels] T is already patched.
        # splitting into 4 groups: (even rows, even cols), (odd rows, even cols), (even rows, odd cols), (odd rows, odd cols)
        input_feature_0 = input_feature[:, :, :, 0::2, 0::2]   #0::2: starting at 0, taking every second element
        # [batch_size, T, H/2, W/2, in_channels]
        input_feature_1 = input_feature[:, :, :, 1::2, 0::2] 
        # [batch_size, T, H/2, W/2, in_channels]
        input_feature_2 = input_feature[:, :, :, 0::2, 1::2] 
        # [batch_size, T, H/2, W/2, in_channels]
        input_feature_3 = input_feature[:, :, :, 1::2, 1::2]
        # [batch_size, T, H/2, W/2, 4*in_channels]
        input_feature = torch.cat(
            [input_feature_0, input_feature_1, input_feature_2, input_feature_3], 1
        )
        input_feature = input_feature.permute(0, 2, 3, 4, 1).contiguous().view(
            batch_size, -1, 4 * in_channels 
        )  # [batch_size, T * H/2 * W/2, 4*C]


        # TODO: It seems first norm and then reduction is more stable for Merging!
        input_feature = self.norm(input_feature) 
        input_feature = self.reduction(input_feature) #  4 * dim -> 2 * dim
        input_feature = input_feature.permute(0, 2, 1).contiguous().view(batch_size, 2*in_channels, T, H//2 , W //2)

        return input_feature
        
class ScOT3DPatchUnmerging(nn.Module):

    def __init__(
        self,
        config,
        input_resolution: Tuple[int],
        dim: int,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution # (4, 4)
        self.dim = dim # 384
        self.upsample = nn.Linear(dim, 2 * dim, bias=False) # 384 -> 768
        self.mixup = nn.Linear(dim // 2, dim // 2, bias=False) # 192 -> 192
        self.norm = nn.LayerNorm(dim//2) # 192

    def maybe_crop(self, input_feature, H, W):
        H_in, W_in = input_feature.shape[2], input_feature.shape[3] 
        if H_in > H:
            input_feature = input_feature[:, :, :H, :, :]
        if W_in > W:
            input_feature = input_feature[:, :, :, :W, :]
        return input_feature

    def forward(
        self,
        input_feature: torch.Tensor, # 16, 16, 384
        output_dimensions: Tuple[int, int], # (8, 8)
        **kwargs
    ) -> torch.Tensor:
        # output_height, output_width = output_dimensions # 8, 8
        batch_size, in_channels, T, H, W = input_feature.shape # 16, 16, 384
        # r = output_height / output_width # r is proportion factor
        # input_height = math.floor((seq_len * r) ** 0.5) # To calculate new height (double old one) -> sqrt(4 * area * proportion factor)
        # input_width = math.floor((seq_len / r) ** 0.5)

        input_feature = input_feature.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, T*H*W, in_channels)

        input_feature = self.upsample(input_feature) 

        input_feature = input_feature.reshape(
            batch_size, T, H, W, 2, 2, in_channels //2
        )
        input_feature = input_feature.permute(0, 1, 2, 4, 3, 5, 6).contiguous() 
        input_feature = input_feature.reshape(
            batch_size, T, 2*H, 2*W, in_channels //2
        ) 

        input_feature = self.maybe_crop(input_feature, 2*H, 2*W) # crop in case the feature does not have the specified dim
        input_feature = input_feature.reshape(batch_size, -1, in_channels // 2) 

        input_feature = self.mixup(self.norm(input_feature)) 
        input_feature = input_feature.permute(0, 2, 1).contiguous().view(batch_size, in_channels//2, T, H*2 , W*2)


        return input_feature

class ScOTAttention3D(nn.Module):

    """
    ScOT Attention Module including time as the third dimension.
    So by 3d window partition and creating a 3d meshgrid and a 3d mask, we enforce 3d attention. Everything else stays same. 
    TODO: Extend to 4D to account fro Channel dimension as well
    TODO: Add prune heads functionality
    """

    def __init__ (self,
                  config,
                  dim,
                  input_resolution: Tuple[int],
                  num_heads,
                  shift_size: Tuple[int],
                  window_size: Tuple[int],
                  pretrained_window_size: Optional[List[int]] = [0, 0, 0, 0]
                  ):
        super().__init__()

        self.config = config
        self.dim = dim
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size

        #### Class ScOTSelfAttenstion3D
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )
        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.attention_head_size * num_heads # extra!

        self.window_size = config.window_size # (Wd,WH,WW) TODO: sanity check addition

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1)))) # taken from SwinV2 in Huggingface

        # positional bias: (based on SwinV2)
        # 1. all possible normalized and log-transformed* relative positions for 3D window
        # 2. An MLP to generate continuous relative position bias
        # 3. create index tensor
        # 4. "during forward pass", get the relative position bias for each head using the index tensor

        # 1. 
        # Batch, Time(t), Channel(C), Height(H), Width(W)] 
        # 3Dwindow size: (Wt, Wh, Ww)
        relative_coords_t = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.int64).float()
        relative_coords_h = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.int64).float()
        relative_coords_w = torch.arange(-(self.window_size[2] - 1), self.window_size[2], dtype=torch.int64).float()
        relative_coords_table = (
            torch.stack(meshgrid([relative_coords_t, relative_coords_h, relative_coords_w], indexing="ij"))
            .permute(1, 2, 3, 0)
            .contiguous()
            .unsqueeze(0)
        )  # shape: [1, 2*Wt - 1, 2*Wh - 1, 2*Ww - 1, 3]

        # normalize to -1, 1
        if pretrained_window_size > 0:
            relative_coords_table[:, :, :, :, 0] /= pretrained_window_size - 1
            relative_coords_table[:, :, :, :, 1] /= pretrained_window_size - 1
            relative_coords_table[:, :, :, :, 2] /= pretrained_window_size - 1
        elif self.window_size:
            relative_coords_table[:, :, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, :, 1] /= self.window_size[1] - 1
            relative_coords_table[:, :, :, :, 2] /= self.window_size[2] - 1


        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        ) 
        # This log-scaled transform compresses large relative distances while preserving fine detail for nearby tokens, 
        # enabling Swin v2 to learn stable, resolution-agnostic relative position bias.


        # 2.
        self.continuous_position_bias_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
        )

        # set to same dtype as mlp weight
        relative_coords_table = relative_coords_table.to(next(self.continuous_position_bias_mlp.parameters()).dtype)
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)


        # 3. get pair-wise relative position index for each token inside the window
        # 3.A: create absolute coordinate for each token inside the window
        coords_t = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(meshgrid([coords_t, coords_h, coords_w], indexing="ij")) # shape: [3, Wt, Wh, Ww]

        # 3.B: compuet the relative coordinate
        coords_flatten = torch.flatten(coords, 1) # shape: [3, Wt*Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # shape: [3, Wt*Wh*Ww, Wt*Wh*Ww]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # shape: [Wt*Wh*Ww, Wt*Wh*Ww, 3]

        # 3.C: shift to start from 0 for indexing
        relative_coords[:, :, 0] += self.window_size[0] - 1  
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        # 3.D: combine to a single index (from 3D to 1D - example: 25 elements in 2D*)
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)


        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)




        #### Class ScOTSelfOutput3D
        self.dense = nn.Linear(dim, dim)
        # self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        """
        input features with shape of (num_windows*B, N, C)
        """
        
        B_, N, C = hidden_states.shape # B_: num_windows * batch_size, N: Wt*Wh*Ww, C: hidden size (dim)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        # shapes: (B_, num_heads, N, head_size)


        # cosine attention: Q · K^T 
        attention_scores = \
        nn.functional.normalize(query_layer, dim=-1) @ \
        nn.functional.normalize(key_layer, dim=-1).transpose(-2, -1).contiguous() # shape: (B_, num_heads, N, N) 

        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp() # shape: (num_heads, 1, 1)

        # scale the attention scores. Instead of dividing by sqrt(d), we multiply by a learnable logit_scale parameter
        attention_scores = attention_scores * logit_scale # shape: (B_, num_heads, N, N)


        # step 4 from Init
        relative_position_bias_table = self.continuous_position_bias_mlp(self.relative_coords_table).view(
            -1, self.num_attention_heads
        ) # shape: [(2*Wt-1)*(2*Wh-1)*(2*Ww-1), num_heads]

        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1
        ) # shape: [Wt*Wh*Ww, Wt*Wh*Ww, num_heads]

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        # shape: [num_heads, Wt*Wh*Ww, Wt*Wh*Ww]

        # final attention score with relative position bias, shape: (B_, num_heads, N, N)
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)



        if attention_mask is not None: # TODO: check
            # Apply the attention mask is (precomputed for all layers in Swinv2Model forward() function)
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                B_ // mask_shape, mask_shape, self.num_attention_heads, N, N
            ) 
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, N, N)

        #  Normalize the attention scores to probabilities. (the meaningful part), shape: (B_, num_heads, N, N)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # Apply dropout , shape: (B_, num_heads, N, N)
        attention_probs = self.dropout(attention_probs)

        # Attention @ Value > weighted sum of the values based on the attention probabilities
        output = (attention_probs @ value_layer).transpose(1, 2).reshape(B_, N, C) # shape: (B_, N, C)

        # OutputClass
        output = self.dense(output)
        output = self.dropout(output)

        outputs = (output, attention_probs) if output_attentions else (output,)

        return outputs

def get_attn_mask_3d(T, H, W, window_size, shift_size, dtype, device):

    """
    creats a mask and window it according to window size and shift size
    """

    if any(i > 0 for i in shift_size):
        img_mask = torch.zeros((1, 1, T, H, W), dtype=dtype, device=device)  # 1 1 T H W

        T_slices = (
            slice(0, -window_size[0]),
            slice(-window_size[0], -shift_size[0]),
            slice(-shift_size[0], None)
        )
        H_slices = (
            slice(0, -window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None)
        )
        W_slices = (
            slice(0, -window_size[2]),
            slice(-window_size[2], -shift_size[2]),
            slice(-shift_size[2], None)
        )

        count = 0

        for t in T_slices:
            for h in H_slices:
                for w in W_slices:
                    img_mask[:, :, t, h, w] = count
                    count += 1


        mask_windows = window_partition_3d(img_mask, window_size) # nW, ws[0]*ws[1]*ws[2], 1
        mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    else:
        attn_mask = None


    return attn_mask

def window_partition_3d(input, window_size): # normally gets imported from Huggingface SwinV2 for 2D
    """
    This function is responsible to create the windows and give a total shape of 
    [window count, each window element count, C or dim]
    
    Args:
        x: (B, C, T, H, W)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """

    B, C, T, H, W = input.shape
    input = input.permute(0, 2, 3, 4, 1).contiguous().view(B, T // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = input.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows
     
def window_reverse_3d(windows, window_size, T, H, W):
    """
    This function is responsible to reverse the windows back to the original shape.
    Returns:
        x: (B, C, T, H, W)
    """
    C = windows.shape[-1]
    windows = windows.view(-1, T // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], C)
    windows = windows.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous().view(-1, C, T, H, W)

    return windows

class ScOT3DLayer(nn.Module):

    """

    The main building block*
    Here we Window the input and apply SW-MSA or W-MSA based on the shift size.
    Args:   
        config: ScOT3DConfig
        dim (int): Number of hidden channels
        input_resolution (Tuple[int]): Input resolution
        num_heads (int): Number of attention heads
        shift_size (Tuple[int]): Shift size for SW-MSA

    """


    def __init__ (self,
                  config,
                  dim,
                  input_resolution: Tuple[int],
                  num_heads,
                  shift_size: Tuple[int],
                  pretrained_window_size: Optional[List[int]] = [0, 0, 0, 0],
                  drop_path=0.0, 
                  ):
        super().__init__()

        self.config = config
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.pretrained_window_size = pretrained_window_size
        self.set_shift_and_window_size_3d()
        self.attn = ScOTAttention3D(
            config=self.config, 
            dim=self.dim, 
            input_resolution=self.input_resolution, 
            num_heads=self.num_heads, 
            shift_size=self.shift_size, 
            window_size=self.window_size,
            pretrained_window_size=self.pretrained_window_size)
        self.norm_before = nn.LayerNorm(self.dim)
        self.drop_path = Swinv2DropPath(drop_path) if drop_path > 0.0 else nn.Identity() # 0 -> Identity
        self.intermediate = Swinv2Intermediate(config, dim)
        self.output = Swinv2Output(config, dim)
        self.norm_after = nn.LayerNorm(self.dim)

    def set_shift_and_window_size_3d(self):
        """
        TODO: complete it.
        """
        use_window_size = list(self.config.window_size)
        hidden_state_shape = list(self.input_resolution)
        if self.shift_size is not None:
            use_shift_size = list(self.shift_size)
        for i in range(len(hidden_state_shape)):
            if hidden_state_shape[i] <= use_window_size[i]:
                use_window_size[i] = hidden_state_shape[i]
                if self.shift_size is not None:
                    use_shift_size[i] = 0

        if self.shift_size is None:
            self.window_size = tuple(use_window_size)
        else:
            self.window_size = tuple(use_window_size)
            self.shift_size = tuple(use_shift_size)

    def maybe_pad(self, hidden_states, T, H, W):
        #  B, C, T, H, W = hidden_states.shape
        pad_Tr = (self.window_size[0] - T % self.window_size[0]) % self.window_size[0]
        pad_Hr = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1] 
        pad_Wr = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2] 
        pad_Tl = pad_Hl = pad_Wl = 0
        pad_values = (pad_Wl, pad_Wr, pad_Hl, pad_Hr, pad_Tl, pad_Tr ) # (pad_last_dim_left, pad_last_dim_right, pad_2nd_last_left, pad_2nd_last_right, ...)
        hidden_states = nn.functional.pad(hidden_states, pad_values)

        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: torch.Tensor, # embedded output
        input_dimensions: Tuple[int, int], # (32, 32)
        head_mask: Optional[torch.FloatTensor] = None, # None
        output_attentions: Optional[bool] = False,# False
        # always_partition: Optional[bool] = False, # False
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        shortcut = hidden_states
        
        B, C, T, H, W = hidden_states.shape
        # window_size, shift_size = self.set_shift_and_window_size_3d((T,H,W), self.window_size, self.shift_size)

        # Layer norm before attention
        hidden_states = hidden_states.flatten(2).transpose(1, 2) # shape: (B, T*H*W, C) # the correct channel dim in Decoder mode
        hidden_states = self.norm_before(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).view(B, C, T, H, W) # shape: (B, T, H, W, C)

        # pad the input if needed # first place I need 3d data
        hidden_states, pad_values = self.maybe_pad(hidden_states, T, H, W)
        _, _, Tp, Hp, Wp = hidden_states.shape

        # cyclic shift in-between (W-MSA and SW-MSA)
        if any(i > 0 for i in self.shift_size):
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(-3, -2, -1))
        else:
            shifted_hidden_states = hidden_states

        # partition windows
        hidden_states_windows = window_partition_3d(shifted_hidden_states, self.config.window_size) # shape: (num_windows*B, window_size*window_size, C)

        # Attention mask for SW-MSA
        attn_mask = get_attn_mask_3d(Tp, Hp, Wp, self.config.window_size, self.shift_size, hidden_states.dtype, hidden_states.device)

        # Apply attention
        attn_outputs = self.attn(
            hidden_states=hidden_states_windows,
            attention_mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0] # shape: (B_, N, C)


        # reconstruct and merge windows
        attn_output_reverse = window_reverse_3d(attn_output, self.config.window_size, Tp, Hp, Wp) # shape: (B, Dp, Hp, Wp, C)

        # reverse cyclic shift if needed
        if any(i > 0 for i in self.shift_size):
            attn_windows = torch.roll(attn_output_reverse, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(-3, -2, -1))
        else:
            attn_windows = attn_output_reverse

        # remove padding if needed
        was_padded = any(pad_values[i] > 0 for i in range(len(pad_values)))
        if was_padded:
            attn_windows = attn_windows[
                :,
                :,
                : T,
                : H,
                : W,
            ].contiguous()

        # Residual connection 1
        output = shortcut + self.drop_path(attn_windows)

        output = output.permute(0, 2, 3, 4, 1).contiguous()


        # Layer norm + MLP + Drop_path after attention + Residual connection 2
        output = output + self.drop_path(self.output(self.intermediate(self.norm_after(output))))

        output = output.permute(0, 4, 1, 2, 3).contiguous()

        outputs = (
            (output, attn_outputs[1])
            if output_attentions
            else (output,)
        )


        return outputs

class ScOT3DEncoderStage(nn.Module):

    def __init__(
            self,
            config,
            dim,
            input_resolution,
            depth,
            num_heads,
            drop_path,
            downsample,
            pretrained_window_size,
    ):
        super().__init__()
        self.config = config
        self.dim = dim

        self.blocks = nn.ModuleList(
            [
                ScOT3DLayer(
                    config=config,
                    dim=dim, 
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else (config.window_size[0] // 2, config.window_size[1] // 2, config.window_size[2] // 2), # tuple or tensor
                    pretrained_window_size=pretrained_window_size,
                    drop_path=drop_path[i]
                ) for i in range(depth)
            ]
        )


        # PatchMerging
        if downsample is not None: # ScOT3DPatchMerging in every stage except last one

            self.downsample = downsample(
                config, input_resolution, dim=dim, norm_layer=nn.LayerNorm
            )
        else:
            self.downsample = None

    def forward(
            self,
            hidden_states,
            input_dimensions: Tuple[int, int],
            head_mask: Optional[List[torch.FloatTensor]] = None,
            output_attentions: Optional[bool] = False,
            # always_partition: Optional[bool] = False,
            **kwargs
    ):
        T, H, W = input_dimensions
        inputs = hidden_states

        for i, block in enumerate(self.blocks):
            block_head_mask = head_mask[i] if head_mask is not None else None

            block_output = block(
                hidden_states=hidden_states,
                input_dimensions=input_dimensions,
                head_mask=block_head_mask,
                output_attentions=output_attentions,
                # always_partition=always_partition,
                **kwargs
            )

            hidden_states = block_output[0]


        hidden_states_before_downsampling = hidden_states

        if self.downsample is not None:
            

            hidden_states = self.downsample(
                # hidden_states_before_downsampling + inputs, # Check the existence of this addition: not originally
                hidden_states_before_downsampling,
                input_dimensions=input_dimensions, # why needed? no needed!!!!!!!!!
                **kwargs
            )


            _, _, _, Hds, Wds = hidden_states.shape
            # output_dimensions = (T, H, W, T, Hds, Wds) # why do I need it?
            output_dimensions = (T, Hds, Wds) # why not written like this?

        else:
            # output_dimensions = (T, H, W, T, H, W)
            output_dimensions = (T, H, W)


        stage_outputs = (
            hidden_states, 
            hidden_states_before_downsampling, 
            output_dimensions)
        

        if output_attentions:
            stage_outputs = stage_outputs + block_output[1:]


        return stage_outputs

class ScOT3DEncoder(nn.Module):

    def __init__(
            self,
            config,
            grid_size,
            pretrained_window_sizes=[0, 0, 0, 0]
    ):
        super().__init__()

        self.num_layers = len(config.depths) 
        self.config = config

        # drop path rate 
        drop_rates_encode_decode = torch.linspace( 
            0, config.drop_path_rate, 2 * sum(config.depths) 
        ) 
        dpr = [ 
            x.item()
            for x in drop_rates_encode_decode[: drop_rates_encode_decode.shape[0] // 2] # only first half (encoder) for drop path rates
        ]


        # Stages (or as In Ref. Layers)
        self.layers = nn.ModuleList(
            [
                ScOT3DEncoderStage(
                    config=config,
                    dim=int(config.latent_channels * 2**i), #important
                    input_resolution=(
                        grid_size[0], # will stay the same after Embedding 
                        grid_size[1] // (2**i),
                        grid_size[2] // (2**i),
                    ),
                    depth=config.depths[i], 
                    num_heads=config.num_heads[i], 
                    drop_path=dpr[ 
                        sum(config.depths[:i]) : sum(config.depths[: i + 1]) 
                    ], 
                    downsample=(
                        ScOT3DPatchMerging if (i < self.num_layers - 1) else None 
                    ),
                    pretrained_window_size=pretrained_window_sizes[i], 

                ) for i in range(self.num_layers)
            ]
        )

    def forward(
            self,
            hidden_states,
            input_dimensions: Tuple[int, int],
            head_mask: Optional[List[torch.FloatTensor]] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = True,
            output_hidden_states_before_downsample: Optional[bool] = True,
            # always_partition: Optional[bool] = False,
            **kwargs
    ):
        
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        # all_self_attentions = () if output_attentions else None
        all_self_attentions = None


        if output_hidden_states: # in this model we need explicit 3D patches.

            all_hidden_states += (hidden_states,)


            # B, _, C = hidden_states.shape
            # reshaped_hidden_states = hidden_states.view(B, *input_dimensions, C)
            # reshaped_hidden_states = reshaped_hidden_states.permute(0, 4, 1, 2, 3)
            # all_reshaped_hidden_states += (reshaped_hidden_states,)


        for i, layer_module in enumerate(self.layers):

            layer_haed_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                input_dimensions=input_dimensions,
                head_mask=layer_haed_mask,
                output_attentions=output_attentions,
                # always_partition=always_partition,
                **kwargs
            )



            hidden_states = layer_outputs[0] # shape: B, C, T, H, W = 8, 54, 3, 32, 32
            hidden_states_before_downsampling = layer_outputs[1] # shape: B, C, T, H, W = 8, 54, 3, 64, 64
            output_dimensions = layer_outputs[2]
            input_dimensions = (output_dimensions[-3], output_dimensions[-2], output_dimensions[-1]) 
            # input_dimensions = layer_outputs[2]

            if output_hidden_states and output_hidden_states_before_downsample:

                all_hidden_states += (hidden_states_before_downsampling,)


                # B, _, C = hidden_states_before_downsampling.shape
                # reshaped_hidden_states = hidden_states_before_downsampling.view(B, *(output_dimensions[0], output_dimensions[1], output_dimensions[2]), C)
                # reshaped_hidden_states = hidden_states_before_downsampling.view(B, *input_dimensions, C)
                # reshaped_hidden_states = reshaped_hidden_states.permute(0, 4, 1, 2, 3)
                # all_reshaped_hidden_states += (reshaped_hidden_states,)

            elif output_hidden_states and not output_hidden_states_before_downsample:

                all_hidden_states += (hidden_states,)
                
                # B, _, C = hidden_states.shape
                # reshaped_hidden_states = hidden_states.view(B, *input_dimensions, C)
                # reshaped_hidden_states = reshaped_hidden_states.permute(0, 4, 1, 2, 3)
                # all_reshaped_hidden_states += (reshaped_hidden_states,)


            if output_attentions:
                all_self_attentions += layer_outputs[3:] 


        return Swinv2EncoderOutput( # Just a return data class
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )

class ScOT3DDecoderStage(nn.Module):

    def __init__(
            self,
            config,
            dim,
            input_resolution,
            depth,
            num_heads,
            drop_path,
            upsample, # PatchUnmerging
            # upsampled_size,
            pretrained_window_size=0
    ):
        super().__init__()
        self.config = config
        self.dim = dim

        self.blocks = nn.ModuleList(
            [
                ScOT3DLayer(
                    config=config,
                    dim=dim, 
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else (config.window_size[0] // 2, config.window_size[1] // 2, config.window_size[2] // 2), # tuple or tensor
                    pretrained_window_size=pretrained_window_size,
                    drop_path=drop_path[depth - 1 - i],  # reversed!
                ) for i in reversed(range(depth))  # reversed !
            ]
        )


        # PatchUnmerging
        if upsample is not None: # upsample in every layer except last one

            self.upsample = upsample(config, input_resolution, dim=dim, norm_layer=nn.LayerNorm) # PatchUnmerging
            # self.upsampled_size = upsampled_size
        else:
            self.upsample = None

    def forward(
            self,
            hidden_states,
            input_dimensions: Tuple[int, int],
            head_mask: Optional[List[torch.FloatTensor]] = None,
            output_attentions: Optional[bool] = False,
            # always_partition: Optional[bool] = False,
            **kwargs
    ):
        T, H, W = input_dimensions
        inputs = hidden_states

        for i, block in enumerate(self.blocks):
            block_head_mask = head_mask[i] if head_mask is not None else None

            block_output = block(
                hidden_states=hidden_states,
                input_dimensions=input_dimensions,
                head_mask=block_head_mask,
                output_attentions=output_attentions,
                # always_partition=always_partition,
                **kwargs
            )

            hidden_states = block_output[0]


        hidden_states_before_upsampling = hidden_states

        if self.upsample is not None:
            

            hidden_states = self.upsample(
                # hidden_states_before_upsampling + inputs, # Check the existence of this addition
                hidden_states_before_upsampling,
                output_dimensions=input_dimensions, # WILL REMOVE
                **kwargs
            )


            _, _, _, Hus, Wus = hidden_states.shape
            output_dimensions = (T, Hus, Wus) 

        else:
            output_dimensions = (T, H, W)


        stage_outputs = (
            hidden_states, 
            hidden_states_before_upsampling, 
            output_dimensions)
        

        if output_attentions:
            stage_outputs = stage_outputs + block_output[1:]


        return stage_outputs

class ScOT3DDecoder(nn.Module):

    "reverse of the encoder"

    def __init__(
            self,
            config,
            grid_size,
            pretrained_window_sizes=[0, 0, 0, 0],
    ):
        super().__init__()

        self.num_layers = len(config.depths) 
        self.config = config

        # drop path rate 
        drop_rates_encode_decode = torch.linspace( 
            0, config.drop_path_rate, 2 * sum(config.depths) 
        ) 
        dpr = [ 
            x.item()
            for x in drop_rates_encode_decode[drop_rates_encode_decode.shape[0] // 2 :] # only second parth (decoder) used for drop path rates
        ]


        # Stages (or as In Ref. Layers)
        self.layers = nn.ModuleList(
            [
                ScOT3DDecoderStage(
                    config=config,
                    dim=int(config.latent_channels * 2**i), 
                    input_resolution=(
                        grid_size[0],
                        grid_size[1] // (2**i), 
                        grid_size[2] // (2**i),
                    ),
                    depth=config.depths[i], 
                    num_heads=config.num_heads[i], 
                    drop_path=dpr[ 
                        sum(config.depths[i + 1 :]) : sum(config.depths[i:])
                    ], 
                    upsample=ScOT3DPatchUnmerging if i > 0 else None, # Upsample between stages (not after last one)
                    # upsampled_size=(
                    #     grid_size[0] // (2 ** (i - 1)), #[8, 16, 32]
                    #     grid_size[1] // (2 ** (i - 1)),
                    # ),
                    pretrained_window_size=pretrained_window_sizes[i], 

                ) for i in reversed(range(self.num_layers)) # reversed! 3 -> 0
            ]
        )

    def forward(
            self,
            hidden_states,
            input_dimensions: Tuple[int, int], # this is cominjg wrong!
            skip_states: List[torch.FloatTensor],
            head_mask: Optional[List[torch.FloatTensor]] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = True,
            output_hidden_states_before_upsample: Optional[bool] =True,
            # always_partition: Optional[bool] = False,
            **kwargs
    ):
        
        # TODO: Do we need all_hidden_states here? it is Decoder.
        

        for i, layer_module in enumerate(self.layers):

            layer_head_mask = head_mask[i] if head_mask is not None else None


            if i!=0 and skip_states[len(skip_states) - i] is not None:

                hidden_states = hidden_states + skip_states[len(skip_states) - i]

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                input_dimensions=input_dimensions,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
                # always_partition=always_partition,
                **kwargs
            )



            hidden_states = layer_outputs[0] # shape: B, C, T, H. W : 8, 54, 3, 32, 32
            hidden_states_before_upsampling = layer_outputs[1] # shape: B, C, T, H. W : 8, 108, 3, 16, 16
            output_dimensions = layer_outputs[2]
            input_dimensions = (output_dimensions[-3], output_dimensions[-2], output_dimensions[-1]) # T, H, W

            # if output_hidden_states and output_hidden_states_before_upsample:

            #     all_hidden_states += (hidden_states_before_upsampling,)


            #     B, _, C = hidden_states_before_upsampling.shape
            #     # reshaped_hidden_states = hidden_states_before_upsampling.view(B, *(output_dimensions[0], output_dimensions[1], output_dimensions[2]), C)
            #     reshaped_hidden_states = hidden_states_before_upsampling.view(B, *input_dimensions, C)
            #     reshaped_hidden_states = reshaped_hidden_states.permute(0, 4, 1, 2, 3)
            #     all_reshaped_hidden_states += (reshaped_hidden_states,)

            # elif output_hidden_states and not output_hidden_states_before_upsample:

            #     all_hidden_states += (hidden_states,)
                
            #     # B, _, C = hidden_states.shape
            #     # reshaped_hidden_states = hidden_states.view(B, *input_dimensions, C)
            #     # reshaped_hidden_states = reshaped_hidden_states.permute(0, 4, 1, 2, 3)
            #     # all_reshaped_hidden_states += (reshaped_hidden_states,)


            if output_attentions:
                all_self_attentions += layer_outputs[3:] 


        all_hidden_states = None
        all_reshaped_hidden_states = None
        all_self_attentions = None


        return Swinv2EncoderOutput( # Just a return data class, Probably I can only pass last hidden state
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )

class ScOT3D(PreTrainedModel):
    main_input_name = "input_data"
    

    config_class = ScOT3DConfig

    def __init__(
            self,
            config,
            # use_mask_token=False,
    ):
        super().__init__(config)



        self.config = config
        self.num_layers_encoder = len(config.depths)
        self.num_layers_decoder = len(config.depths)

        self.embeddings = ScOT3DEmbeddings(config)
        self.encoder = ScOT3DEncoder(config, self.embeddings.grid_size)
        self.decoder = ScOT3DDecoder(config, self.embeddings.grid_size)
        self.patch_recovery = ScOT3DPatchRecovery(config)

        res_model = ResNetBlock
        self.residual_blocks = nn.ModuleList(
            [
                (
                    nn.ModuleList(
                        [
                            res_model(config, (self.embeddings.grid_size[0] // 2 ** i, self.embeddings.grid_size[1] // 2 ** i), config.latent_channels * 2**(i))
                            for _ in range(depth)
                        ]
                    )
                    if depth > 0
                    else nn.ModuleList([nn.Identity()])
                )
                for i, depth in enumerate(config.skip_connections)
            ]
        )

        self.post_init()

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

    def forward(
        self,
        input_data: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[Tuple, ScOT3DOutput]:

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
            coord_feat = threed_meshgrid(list(input_data.shape), input_data.device)
            input_data = torch.cat((input_data, coord_feat), dim=1)

        embedding_output, input_dimensions = self.embeddings(
            input_data, bool_masked_pos=bool_masked_pos, **kwargs
        )

        encoder_outputs = self.encoder( # embedding_output: torch.Size([8, 27, 3, 64, 64]) normed and re shaped
            embedding_output,
            input_dimensions,
            head_mask=head_mask_encoder,
            output_attentions=output_attentions,
            output_hidden_states=True,
            output_hidden_states_before_downsampling=True, 
            **kwargs
        )

        skip_states = list(encoder_outputs[1][1:]) # The outputs of Encoder Stage, ignore Embedding and these outputs are the ones before PatchMerging.

        for i in range(len(skip_states)):
            for block in self.residual_blocks[i]: # 2  blocks (last skip layer: identity)
                if isinstance(block, nn.Identity):
                    skip_states[i] = block(skip_states[i])
                else: # is not Identity
                    skip_states[i] = block(skip_states[i], **kwargs)

        input_dim_t = skip_states[-1].shape[-3]
        input_dim_x = skip_states[-1].shape[-2]
        input_dim_y = skip_states[-1].shape[-1]

        decoder_output = self.decoder(
            skip_states[-1],
            (input_dim_t, input_dim_x, input_dim_y),
            skip_states=skip_states[:-1],
            head_mask=head_mask_decoder,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs
        )

        sequence_output = decoder_output[0] # 8, 27, 3, 64, 64
        prediction = self.patch_recovery(sequence_output)

        prediction = prediction.permute(0, 2, 1, 3, 4).contiguous() # B, T, c_OUT, H, W


        return prediction

        

        
        







        


