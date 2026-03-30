"""
Copied directly and entirely from https://github.com/HaoZhongkai/DPOT/blob/main/models/dpot.py
and https://github.com/HaoZhongkai/DPOT/blob/main/models/dpot3d.py
"""
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils import activation_func
from .dpot_utils import DPOTConfig, TimeAggregator, PatchEmbed2D, Block2D
from .dpot_utils import PatchEmbed3D, Block3D
from transformers import PreTrainedModel

class DPOT(PreTrainedModel):
    """Denoising Pre-training Operator Transformer."""
    main_input_name = "input_data"
    conditioning_input_name = "conditioning_input_data"
    config_class = DPOTConfig
    
    def __init__(
        self,
        config: DPOTConfig,
    ):
        super().__init__(config)
        super().post_init()

        self.img_size = config.grid_resolution
        
        self.dpot = self.build_DPOT()(config=config)

    def build_DPOT(self):
        """Get the FNO encoder based on the model dimensionality"""
        if self.config.dimension == 2:
            return DPOTNet2D
        elif self.config.dimension == 3:
            return DPOTNet3D
        else:
            raise NotImplementedError(
                "Invalid dimensionality. Only 2D and 3D DPOT implemented"
            )

    def forward(
        self,
        input_data: Tensor,
    ):
        # DPOT looks for B, X, Y, [z], T, C
        # Neptuna inputs - B x T x C x H x W x D
        x = input_data
        orig_len = len(x.shape)
        original_shape = None

        if orig_len not in (5, 6):
            raise ValueError(
                f"DPOT expects 5D (2D data) or 6D (3D data) input: got shape {tuple(x.shape)}"
            )

        # Pad channel dimension up to 4 (Density, VelX, VelY, Pressure)
        # and unpad on return. Disallow more than 4 channels.
        orig_c = int(x.shape[2])
        if orig_c > 4:
            raise ValueError(
                f"DPOT only supports up to 4 input channels, got {orig_c} (shape {tuple(x.shape)})"
            )
        if orig_c < 4:
            pad_c = 4 - orig_c
            zeros_shape = (x.shape[0], x.shape[1], pad_c, *x.shape[3:])
            x = torch.cat([x, x.new_zeros(zeros_shape)], dim=2)

        if orig_len == 5:
            x = rearrange(x, "b t c h w -> b h w t c")
        elif orig_len == 6:
            if tuple(x.shape[-3:]) != tuple(self.img_size):
                T = x.shape[0]
                original_shape = x.shape[-3:]
                x = rearrange(x, "b t c h w d -> (t b) c h w d")
                x = F.interpolate(
                    x,
                    size=tuple(self.img_size),
                    mode="trilinear",
                    align_corners=False,
                )
                x = rearrange(x, "(t b) c h w d -> b t c h w d", t=T)
            x = rearrange(x, "b t c h w d -> b h w d t c")
        # RUN MODEL
        # [0] takes the spatial output x and drops cls_pred (the [1] element of the returned tuple).
        preds = self.dpot(x)
        # # RESHAPE OUTPUTS
        if orig_len == 5:
            preds = rearrange(preds, "b h w t c -> b t c h w")
        elif orig_len == 6:
            preds = rearrange(preds, "b h w d t c -> b t c h w d")
            if original_shape is not None:
                T = preds.shape[0]
                preds = rearrange(preds, "b t c h w d -> (b t) c h w d")
                preds = F.interpolate(
                    preds,
                    size=tuple(original_shape),
                    mode="trilinear",
                    align_corners=False,
                )
                preds = rearrange(preds, "(b t) c h w d -> b t c h w d", t=T)
        # Unpad channels back to original count
        if orig_c < 4:
            preds = preds[:, :, :orig_c, ...]
        return preds

class DPOTNet2D(nn.Module):
    def __init__(
        self,
        config: DPOTConfig,
    ):
        super().__init__()
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.in_timesteps = config.sequence_info[0]
        self.out_timesteps = config.sequence_info[1]
        self.n_blocks = config.n_blocks
        self.modes = config.modes
        self.num_features = self.latent_channels = config.latent_channels
        self.mlp_ratio = config.mlp_ratio
        self.act = activation_func.get_activation(config.act)
        self.patch_embed = PatchEmbed2D(
            img_size=config.grid_resolution,
            patch_size=config.patch_size,
            in_chans=config.in_channels + 3,
            latent_channels=config.out_channels * config.patch_size + 3,
            out_dim=config.latent_channels,
            act=config.act,
        )
        self.latent_size = self.patch_embed.out_size
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1, 
                config.latent_channels, 
                self.patch_embed.out_size[0], 
                self.patch_embed.out_size[1]
            )
        )
        self.normalize = config.normalize
        self.time_agg = config.time_agg
        self.n_cls = config.n_cls

        # Note - these aren't actually used anywhere, so just commenting them out for the non-square.
        h = 1  # img_size // patch_size
        w = 1  # h // 2 + 1

        self.blocks = nn.ModuleList(
            [
                Block2D(
                    mixing_type=config.mixing_type,
                    modes=config.modes,
                    width=config.latent_channels,
                    mlp_ratio=config.mlp_ratio,
                    channel_first=True,
                    n_blocks=config.n_blocks,
                    double_skip=False,
                    h=h,
                    w=w,
                    act=config.act,
                )
                for _ in range(config.depth)
            ]
        )

        if self.normalize:
            self.scale_feats_mu = nn.Linear(2 * config.in_channels, config.latent_channels)
            self.scale_feats_sigma = nn.Linear(2 * config.in_channels, config.latent_channels)

        self.cls_head = nn.Sequential(
            nn.Linear(config.latent_channels, config.latent_channels),
            self.act,
            nn.Linear(config.latent_channels, config.latent_channels),
            self.act,
            nn.Linear(config.latent_channels, config.n_cls),
        )

        self.time_agg_layer = TimeAggregator(
            config.in_channels, self.in_timesteps, config.latent_channels, config.time_agg
        )

        ### attempt load balancing for high resolution
        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=config.latent_channels,
                out_channels=config.out_layer_dim,
                kernel_size=config.patch_size,
                stride=config.patch_size,
            ),
            self.act,
            nn.Conv2d(
                in_channels=config.out_layer_dim,
                out_channels=config.out_layer_dim,
                kernel_size=1,
                stride=1,
            ),
            self.act,
            nn.Conv2d(
                in_channels=config.out_layer_dim,
                out_channels=self.out_channels * self.out_timesteps,
                kernel_size=1,
                stride=1,
            ),
        )

        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.mixing_type = config.mixing_type

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.002)  # .02
            if m.bias is not None:
                # if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, x):
        batchsize, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(x.device)
        return grid

    def get_grid_3d(self, x):
        batchsize, size_x, size_y, size_z = (
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3],
        )
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = (
            gridx.reshape(1, size_x, 1, 1, 1)
            .to(x.device)
            .repeat([batchsize, 1, size_y, size_z, 1])
        )
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = (
            gridy.reshape(1, 1, size_y, 1, 1)
            .to(x.device)
            .repeat([batchsize, size_x, 1, size_z, 1])
        )
        #gridz represents the time
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = (
            gridz.reshape(1, 1, 1, size_z, 1)
            .to(x.device)
            .repeat([batchsize, size_x, size_y, 1, 1])
        )

        grid = torch.cat((gridx, gridy, gridz), dim=-1)
        return grid

    ### in/out: B, X, Y, T, C
    def forward(self, x):
        B, _, _, T, _ = x.shape
        if self.normalize:
            mu, sigma = (
                x.mean(dim=(1, 2, 3), keepdim=True),
                x.std(dim=(1, 2, 3), keepdim=True) + 1e-6,
            )  # B,1,1,1,C
            x = (x - mu) / sigma
            scale_mu = (
                self.scale_feats_mu(torch.cat([mu, sigma], dim=-1))
                .squeeze(-2)
                .permute(0, 3, 1, 2)
            )  # -> B, C, 1, 1
            scale_sigma = (
                self.scale_feats_sigma(torch.cat([mu, sigma], dim=-1))
                .squeeze(-2)
                .permute(0, 3, 1, 2)
            )

        grid = self.get_grid_3d(x)
        x = torch.cat((x, grid), dim=-1).contiguous()  # B, X, Y, T, C+3
        x = rearrange(x, "b x y t c -> (b t) c x y")
        x = self.patch_embed(x)

        x = x + self.pos_embed
        x = rearrange(x, "(b t) c x y -> b x y t c", b=B, t=T)

        x = self.time_agg_layer(x) # B, X, Y, T, C --> B, X, Y, C

        x = rearrange(x, "b x y c -> b c x y")

        if self.normalize:
            x = scale_sigma * x + scale_mu  ### Ada_in layer

        for blk in self.blocks:
            x = blk(x)

        #commented out as we dont need class prediction during fine tuning
        # cls_token = x.mean(dim=(2, 3), keepdim=False)
        # cls_pred = self.cls_head(cls_token)

        x = self.out_layer(x).permute(0, 2, 3, 1)
        x = x.reshape(*x.shape[:3], self.out_timesteps, self.out_channels).contiguous()
        if self.normalize:
            x = x * sigma + mu

        return x  #, cls_pred

    def extra_repr(self) -> str:
        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)

        string_repr = ""
        for p in self.named_parameters():
            name = p[0].split(".")[0]
            if name not in named_modules:
                string_repr = (
                    string_repr
                    + "("
                    + name
                    + "): "
                    + "tensor("
                    + str(tuple(p[1].shape))
                    + ", requires_grad="
                    + str(p[1].requires_grad)
                    + ")\n"
                )

        return string_repr

class DPOTNet3D(nn.Module):
    def __init__(
        self,
        config: DPOTConfig,
    ):
        super().__init__()

        # self.num_classes = num_classes
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.in_timesteps = config.sequence_info[0]
        self.out_timesteps = config.sequence_info[1]

        self.n_blocks = config.n_blocks
        self.modes = config.modes
        self.num_features = self.latent_channels = (
            config.latent_channels  # num_features for consistency with other models
        )
        self.mlp_ratio = config.mlp_ratio
        self.act = activation_func.get_activation(config.act)
        self.patch_embed = PatchEmbed3D(
            img_size=config.grid_resolution,
            patch_size=config.patch_size,
            in_chans=config.in_channels + 4,
            embed_dim=config.out_channels * config.patch_size + 4,
            out_dim=config.latent_channels,
            act=config.act,
        )

        self.latent_size = self.patch_embed.out_size

        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                config.latent_channels,
                self.patch_embed.out_size[0],
                self.patch_embed.out_size[1],
                self.patch_embed.out_size[2],
            )
        )
        # self.pos_drop = nn.Dropout(p=drop_rate)
        self.normalize = config.normalize
        self.time_agg = config.time_agg
        self.n_cls = config.n_cls

        h = 1  # img_size // patch_size
        w = 1  # h // 2 + 1

        self.blocks = nn.ModuleList(
            [
                Block3D(
                    mixing_type=config.mixing_type,
                    modes=config.modes,
                    width=config.latent_channels,
                    mlp_ratio=config.mlp_ratio,
                    channel_first=True,
                    n_blocks=config.n_blocks,
                    double_skip=False,
                    h=h,
                    w=w,
                    act=config.act,
                )
                for _ in range(config.depth)
            ]
        )

        if self.normalize: #set to False as the input data is already normalized
            self.scale_feats_mu = nn.Linear(2 * config.in_channels, config.latent_channels)
            self.scale_feats_sigma = nn.Linear(2 * config.in_channels, config.latent_channels)
        
        self.cls_head = nn.Sequential(
            nn.Linear(config.latent_channels, config.latent_channels),
            self.act,
            nn.Linear(config.latent_channels, config.latent_channels),
            self.act,
            nn.Linear(config.latent_channels, config.n_cls),
        )

        self.time_agg_layer = TimeAggregator(
            config.in_channels, self.in_timesteps, config.latent_channels, config.time_agg
        )

        # self.norm = norm_layer(embed_dim)

        ### attempt load balancing for high resolution
        self.out_layer = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=config.latent_channels,
                out_channels=config.out_layer_dim,
                kernel_size=config.patch_size,
                stride=config.patch_size,
            ),
            self.act,
            nn.Conv3d(
                in_channels=config.out_layer_dim,
                out_channels=config.out_layer_dim,
                kernel_size=1,
                stride=1,
            ),
            self.act,
            nn.Conv3d(
                in_channels=config.out_layer_dim,
                out_channels=self.out_channels * self.out_timesteps,
                kernel_size=1,
                stride=1,
            ),
        )

        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.mixing_type = config.mixing_type

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.002)  # .02
            if m.bias is not None:
                # if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, x):
        batchsize, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(x.device)
        return grid

    def get_grid_4d(self, x):
        batchsize, size_x, size_y, size_z, size_t = (
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3],
            x.shape[4],
        )
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = (
            gridx.reshape(1, size_x, 1, 1, 1, 1)
            .to(x.device)
            .repeat([batchsize, 1, size_y, size_z, size_t, 1])
        )
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = (
            gridy.reshape(1, 1, size_y, 1, 1, 1)
            .to(x.device)
            .repeat([batchsize, size_x, 1, size_z, size_t, 1])
        )
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = (
            gridz.reshape(1, 1, 1, size_z, 1, 1)
            .to(x.device)
            .repeat([batchsize, size_x, size_y, 1, size_t, 1])
        )
        gridt = torch.tensor(np.linspace(0, 1, size_t), dtype=torch.float)
        gridt = (
            gridt.reshape(1, 1, 1, 1, size_t, 1)
            .to(x.device)
            .repeat([batchsize, size_x, size_y, size_z, 1, 1])
        )

        grid = torch.cat((gridx, gridy, gridz, gridt), dim=-1)
        return grid

    ### in/out: B, X, Y, Z, T, C
    def forward(self, x):
        B, _, _, _, T, _ = x.shape
        if self.normalize:
            mu, sigma = (
                x.mean(dim=(1, 2, 3, 4), keepdim=True),
                x.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-6,
            )  # B,1,1,1,1,C
            x = (x - mu) / sigma
            scale_mu = (
                self.scale_feats_mu(torch.cat([mu, sigma], dim=-1))
                .squeeze(-2)
                .permute(0, 4, 1, 2, 3)
            )  # -> B, C, 1, 1, 1
            scale_sigma = (
                self.scale_feats_sigma(torch.cat([mu, sigma], dim=-1))
                .squeeze(-2)
                .permute(0, 4, 1, 2, 3)
            )

        grid = self.get_grid_4d(x)
        x = torch.cat((x, grid), dim=-1).contiguous()  # B, X, Y, Z, T, C+4
        x = rearrange(x, "b x y z t c -> (b t) c x y z")
        x = self.patch_embed(x)

        x = x + self.pos_embed

        x = rearrange(x, "(b t) c x y z -> b x y z t c", b=B, t=T)

        x = self.time_agg_layer(x)

        # x = self.pos_drop(x)
        x = rearrange(x, "b x y z c -> b c x y z")

        if self.normalize:
            x = scale_sigma * x + scale_mu  ### Ada_in layer

        for blk in self.blocks:
            x = blk(x)

        x = self.out_layer(x).permute(0, 2, 3, 4, 1)
        x = x.reshape(*x.shape[:4], self.out_timesteps, self.out_channels).contiguous()

        if self.normalize:
            x = x * sigma + mu

        return x #, []

    def extra_repr(self) -> str:
        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)

        string_repr = ""
        for p in self.named_parameters():
            name = p[0].split(".")[0]
            if name not in named_modules:
                string_repr = (
                    string_repr
                    + "("
                    + name
                    + "): "
                    + "tensor("
                    + str(tuple(p[1].shape))
                    + ", requires_grad="
                    + str(p[1].requires_grad)
                    + ")\n"
                )

        return string_repr