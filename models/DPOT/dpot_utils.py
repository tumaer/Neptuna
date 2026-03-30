"""
Copied directly and entirely from https://github.com/HaoZhongkai/DPOT/blob/main/models/dpot.py
and https://github.com/HaoZhongkai/DPOT/blob/main/models/dpot3d.py
"""
import torch
import torch.fft
import torch.nn as nn
from collections import OrderedDict
from utils import activation_func
from utils.model_utils import PretrainedConfig
from einops import rearrange
import torch.nn.functional as F

class DPOTConfig(PretrainedConfig):
    def __init__(
        self,
        patch_size: int = 16,
        mixing_type: str = "afno",
        n_blocks: int = 4,
        out_layer_dim: int = 32,
        depth: int = 12,
        modes: int = 32,
        mlp_ratio: float = 1.0,
        n_cls: int = 12,
        normalize: bool = False,
        act: str = "gelu",
        time_agg: str = "exp_mlp",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.mixing_type = mixing_type
        self.n_blocks = n_blocks
        self.out_layer_dim = out_layer_dim
        self.depth = depth
        self.modes = modes
        self.mlp_ratio = mlp_ratio
        self.n_cls = n_cls
        self.normalize = normalize
        self.act = act
        self.time_agg = time_agg

# -----------------------------------------------------------------------------
# 2D -> 3D fine-tuning helper
# -----------------------------------------------------------------------------
def dpot_load_3d_components_from_2d(model, state_dict, components="all"):
    """
    Load compatible subsets of a 2D DPOT state_dict into a 3D DPOT model.

    This is intended for fine-tuning a 2D DPOT checkpoint on 3D data by:
    - loading transformer blocks (and adapting 2D MLP conv weights to 3D by
      unsqueezing the final spatial dim)
    - loading the time aggregation MLP (shape-compatible)

    Parameters
    ----------
    model:
        Target model module that exposes `blocks` and/or `time_agg_layer`.
        In this codebase that is the inner DPOT network (e.g. `DPOT(...).dpot`).
    state_dict:
        Source state dict (typically from a 2D `.pth` checkpoint) using the
        inner-module key space (e.g. `blocks.0.*`, `time_agg_layer.*`).
    components:
        "all" or list containing any of: "blocks", "time_agg".
        Note: other component names are accepted but currently ignored.
    """

    if not state_dict:
        raise ValueError("Empty state_dict provided.")

    # Strip DistributedDataParallel prefix if present
    first_key = next(iter(state_dict.keys()))
    if isinstance(first_key, str) and first_key.startswith("module."):
        new_state_dict = OrderedDict()
        for key, item in state_dict.items():
            new_state_dict[key.replace("module.", "", 1)] = item
        state_dict = new_state_dict

    if (components == "all") or (isinstance(components, (list, tuple, set)) and ("all" in components)):
        model.load_state_dict(state_dict)
        return

    for name in components:
        if name == "blocks" and hasattr(model, "blocks"):
            for i, block in enumerate(model.blocks):
                block_state_dict = OrderedDict(
                    {
                        k.replace(f"blocks.{i}.", "", 1): v
                        for k, v in state_dict.items()
                        if k.startswith(f"blocks.{i}.")
                    }
                )

                # reshape 2d conv param to 3d conv param
                for k, v in list(block_state_dict.items()):
                    if "mlp" in k and "weight" in k and hasattr(v, "ndim") and v.ndim == 4:
                        block_state_dict[k] = v.unsqueeze(-1)

                block.load_state_dict(block_state_dict)

        elif name == "time_agg" and hasattr(model, "time_agg_layer"):
            model.time_agg_layer.load_state_dict(
                OrderedDict(
                    {
                        k.replace("time_agg_layer.", "", 1): v
                        for k, v in state_dict.items()
                        if k.startswith("time_agg_layer.")
                    }
                )
            )
        else:
            print(f"Submodule does not exists：{name}")

    return

#TimeAggregator - A common class for both 2D and 3D datasets
class TimeAggregator(nn.Module):
    def __init__(self, n_channels, n_timesteps, out_channels, type="mlp"):
        super().__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.out_channels = out_channels
        self.type = type
        if self.type == "mlp":
            self.w = nn.Parameter(
                1
                / (n_timesteps * out_channels**0.5)
                * torch.randn(n_timesteps, out_channels, out_channels),
                requires_grad=True,
            )  # initialization could be tuned
        elif self.type == "exp_mlp":
            self.w = nn.Parameter(
                1
                / (n_timesteps * out_channels**0.5)
                * torch.randn(n_timesteps, out_channels, out_channels),
                requires_grad=True,
            )  # initialization could be tuned
            self.gamma = nn.Parameter(
                2 ** torch.linspace(-10, 10, out_channels).unsqueeze(0),
                requires_grad=True,
            )  # 1, C

    ##  B, X, Y, T, C
    def forward(self, x):
        if self.type == "mlp":
            x = torch.einsum("tij, ...ti->...j", self.w, x)
        elif self.type == "exp_mlp":
            t = torch.linspace(0, 1, x.shape[-2]).unsqueeze(-1).to(x.device)  # T, 1
            t_embed = torch.cos(t @ self.gamma)
            x = torch.einsum("tij,...ti->...j", self.w, x * t_embed)

        return x

class AFNO2D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    """

    def __init__(
        self,
        width=32,
        num_blocks=8,
        channel_first=False,
        sparsity_threshold=0.01,
        modes=32,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
        act="gelu",
    ):
        super().__init__()
        assert (
            width % num_blocks == 0
        ), f"hidden_size {width} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = width
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.channel_first = channel_first
        self.modes = modes
        self.hidden_size_factor = hidden_size_factor
        # self.scale = 0.02
        self.scale = 1 / (self.block_size * self.block_size * self.hidden_size_factor)

        self.act = activation_func.get_activation(act)

        self.w1 = nn.Parameter(
            self.scale
            * torch.rand(
                2,
                self.num_blocks,
                self.block_size,
                self.block_size * self.hidden_size_factor,
            )
        )
        self.b1 = nn.Parameter(
            self.scale
            * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            self.scale
            * torch.rand(
                2,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
                self.block_size,
            )
        )
        self.b2 = nn.Parameter(
            self.scale * torch.rand(2, self.num_blocks, self.block_size)
        )

    ### N, C, X, Y
    def forward(self, x, spatial_size=None):
        if self.channel_first:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)  ### ->N, X, Y, C
        else:
            B, H, W, C = x.shape
        x_orig = x

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        # x = torch.fft.rfft2(x, dim=(1, 2))

        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros(
            [
                B,
                x.shape[1],
                x.shape[2],
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o1_imag = torch.zeros(
            [
                B,
                x.shape[1],
                x.shape[2],
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        # total_modes = H*W // 2 + 1
        kept_modes = self.modes

        o1_real[:, :kept_modes, :kept_modes] = self.act(
            torch.einsum(
                "...bi,bio->...bo", x[:, :kept_modes, :kept_modes].real, self.w1[0]
            )
            - torch.einsum(
                "...bi,bio->...bo", x[:, :kept_modes, :kept_modes].imag, self.w1[1]
            )
            + self.b1[0]
        )

        o1_imag[:, :kept_modes, :kept_modes] = self.act(
            torch.einsum(
                "...bi,bio->...bo", x[:, :kept_modes, :kept_modes].imag, self.w1[0]
            )
            + torch.einsum(
                "...bi,bio->...bo", x[:, :kept_modes, :kept_modes].real, self.w1[1]
            )
            + self.b1[1]
        )

        o2_real[:, :kept_modes, :kept_modes] = (
            torch.einsum(
                "...bi,bio->...bo", o1_real[:, :kept_modes, :kept_modes], self.w2[0]
            )
            - torch.einsum(
                "...bi,bio->...bo", o1_imag[:, :kept_modes, :kept_modes], self.w2[1]
            )
            + self.b2[0]
        )

        o2_imag[:, :kept_modes, :kept_modes] = (
            torch.einsum(
                "...bi,bio->...bo", o1_imag[:, :kept_modes, :kept_modes], self.w2[0]
            )
            + torch.einsum(
                "...bi,bio->...bo", o1_real[:, :kept_modes, :kept_modes], self.w2[1]
            )
            + self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        ## for ab study
        # x = F.softshrink(x, lambd=self.sparsity_threshold)

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")

        x = x + x_orig
        if self.channel_first:
            x = x.permute(0, 3, 1, 2)  ### N, C, X, Y

        return x

class PatchEmbed2D(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        latent_channels=768,
        out_dim=128,
        act="gelu",
    ):
        super().__init__()
        # Make img_size into tuple if it is an integer
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.out_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.out_dim = out_dim
        self.act = activation_func.get_activation(act)

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, latent_channels, kernel_size=patch_size, stride=patch_size),
            self.act,
            nn.Conv2d(latent_channels, out_dim, kernel_size=1, stride=1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class Block2D(nn.Module):
    def __init__(
        self,
        mixing_type="afno",
        double_skip=True,
        width=32,
        n_blocks=4,
        mlp_ratio=1.0,
        channel_first=True,
        modes=32,
        drop=0.0,
        drop_path=0.0,
        act="gelu",
        h=14, #unused
        w=8, #unused
    ):
        super().__init__()
        # self.norm1 = norm_layer(width)
        # self.norm1 = torch.nn.LayerNorm([width])
        self.norm1 = torch.nn.GroupNorm(8, width)
        # self.norm1 = torch.nn.InstanceNorm2d(width,affine=True,track_running_stats=False)
        self.width = width
        self.modes = modes
        self.act = activation_func.get_activation(act)

        if mixing_type == "afno":
            self.filter = AFNO2D(
                width=width,
                num_blocks=n_blocks,
                sparsity_threshold=0.01,
                channel_first=channel_first,
                modes=modes,
                hard_thresholding_fraction=1,
                hidden_size_factor=1,
                act=act,
            )

        self.norm2 = torch.nn.GroupNorm(8, width)

        mlp_hidden_dim = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(
                in_channels=width, out_channels=mlp_hidden_dim, kernel_size=1, stride=1
            ),
            self.act,
            nn.Conv2d(
                in_channels=mlp_hidden_dim, out_channels=width, kernel_size=1, stride=1
            ),
        )

        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)

        x = x + residual

        return x

#---------------------------------3D---------------------------------
class AFNO3D(nn.Module):
    def __init__(
        self,
        width=32,
        num_blocks=8,
        channel_first=False,
        sparsity_threshold=0.01,
        modes=32,
        temporal_modes=8,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
        act="gelu",
    ):
        super().__init__()
        assert (
            width % num_blocks == 0
        ), f"hidden_size {width} should be divisble by num_blocks {num_blocks}"
        self.hidden_size = width
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.channel_first = channel_first
        self.modes = modes
        self.temporal_modes = temporal_modes
        self.hidden_size_factor = hidden_size_factor
        self.act = activation_func.get_activation(act)
        # self.scale = 0.02
        self.scale = 1 / (self.block_size * self.block_size * self.hidden_size_factor)

        self.w1 = nn.Parameter(
            self.scale
            * torch.rand(
                2,
                self.num_blocks,
                self.block_size,
                self.block_size * self.hidden_size_factor,
            )
        )
        self.b1 = nn.Parameter(
            self.scale
            * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            self.scale
            * torch.rand(
                2,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
                self.block_size,
            )
        )
        self.b2 = nn.Parameter(
            self.scale * torch.rand(2, self.num_blocks, self.block_size)
        )

    ### N, C, X, Y, Z
    def forward(self, x, spatial_size=None):
        if self.channel_first:
            x = rearrange(x, "b c x y z -> b x y z c")
        B, H, W, L, C = x.shape
        x_orig = x

        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")

        x = x.reshape(
            B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size
        )

        o1_real = torch.zeros(
            [
                B,
                x.shape[1],
                x.shape[2],
                x.shape[3],
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o1_imag = torch.zeros(
            [
                B,
                x.shape[1],
                x.shape[2],
                x.shape[3],
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        # total_modes = H*W // 2 + 1
        kept_modes = self.modes

        o1_real[:, :kept_modes, :kept_modes, : self.temporal_modes] = F.gelu(
            torch.einsum(
                "...bi,bio->...bo",
                x[:, :kept_modes, :kept_modes, : self.temporal_modes].real,
                self.w1[0],
            )
            - torch.einsum(
                "...bi,bio->...bo",
                x[:, :kept_modes, :kept_modes, : self.temporal_modes].imag,
                self.w1[1],
            )
            + self.b1[0]
        )

        o1_imag[:, :kept_modes, :kept_modes, : self.temporal_modes] = F.gelu(
            torch.einsum(
                "...bi,bio->...bo",
                x[:, :kept_modes, :kept_modes, : self.temporal_modes].imag,
                self.w1[0],
            )
            + torch.einsum(
                "...bi,bio->...bo",
                x[:, :kept_modes, :kept_modes, : self.temporal_modes].real,
                self.w1[1],
            )
            + self.b1[1]
        )

        o2_real[:, :kept_modes, :kept_modes, : self.temporal_modes] = (
            torch.einsum(
                "...bi,bio->...bo",
                o1_real[:, :kept_modes, :kept_modes, : self.temporal_modes],
                self.w2[0],
            )
            - torch.einsum(
                "...bi,bio->...bo",
                o1_imag[:, :kept_modes, :kept_modes, : self.temporal_modes],
                self.w2[1],
            )
            + self.b2[0]
        )

        o2_imag[:, :kept_modes, :kept_modes, : self.temporal_modes] = (
            torch.einsum(
                "...bi,bio->...bo",
                o1_imag[:, :kept_modes, :kept_modes, : self.temporal_modes],
                self.w2[0],
            )
            + torch.einsum(
                "...bi,bio->...bo",
                o1_real[:, :kept_modes, :kept_modes, : self.temporal_modes],
                self.w2[1],
            )
            + self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        # x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], C)
        x = torch.fft.irfftn(x, s=(H, W, L), dim=(1, 2, 3), norm="ortho")

        x = x + x_orig
        if self.channel_first:
            x = rearrange(x, "b x y t c -> b c x y t")
        return x

class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        out_dim=128,
        act="gelu",
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size, img_size)
        patch_size = (patch_size, patch_size, patch_size)
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (img_size[2] // patch_size[2])
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.out_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        self.out_dim = out_dim
        self.act = activation_func.get_activation(act)

        self.proj = nn.Sequential(
            nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            self.act,
            nn.Conv3d(embed_dim, out_dim, kernel_size=1, stride=1),
        )

    def forward(self, x):
        B, C, H, W, L = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1] and L == self.img_size[2]
        ), f"Input image size ({H}*{W}*{L}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x)
        return x

class Block3D(nn.Module):
    def __init__(
        self,
        mixing_type="afno",
        double_skip=True,
        width=32,
        n_blocks=4,
        mlp_ratio=1.0,
        channel_first=True,
        modes=32,
        drop=0.0,
        drop_path=0.0,
        act="gelu",
        h=14, #unused
        w=8, #unused
    ):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(8, width)
        self.width = width
        self.modes = modes
        self.act = activation_func.get_activation(act)

        if mixing_type == "afno":
            self.filter = AFNO3D(
                width=width,
                num_blocks=n_blocks,
                sparsity_threshold=0.01,
                channel_first=channel_first,
                modes=modes,
                hard_thresholding_fraction=1,
                hidden_size_factor=1,
                act=act,
            )

        self.norm2 = torch.nn.GroupNorm(8, width)

        mlp_hidden_dim = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv3d(
                in_channels=width, out_channels=mlp_hidden_dim, kernel_size=1, stride=1
            ),
            self.act,
            nn.Conv3d(
                in_channels=mlp_hidden_dim, out_channels=width, kernel_size=1, stride=1
            ),
        )

        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        # x = self.mlp(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.norm2(x)
        x = self.mlp(x)

        # x = self.drop_path(x)
        x = x + residual

        return x