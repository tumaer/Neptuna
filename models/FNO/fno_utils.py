from typing import Callable,List, Optional, Union, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from models.model_utils import cfd_PretrainedConfig
from utils.activation_func import get_activation
#from modulus>models>layers>fully_connected_layers.py

class FNOConfig(cfd_PretrainedConfig):
    """Fourier neural operator (FNO) model.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    sequence_info : List[int], optional
        Configuration for input/output sequences [input_seq_len, output_seq_len, stride], by default [1,1,1]
    decoder_layers : int, optional
        Number of decoder layers, by default 1
    decoder_layer_size : int, optional
        Number of neurons in decoder layers, by default 32
    decoder_activation_fn : str, optional
        Activation function for decoder, by default "silu"
    dimension : int
        Model dimensionality (supports 1, 2, 3).
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding : int, optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
        padding_type options: 'constant', 'reflect', 'replicate' or 'circular'
    activation_fn : str, optional
        Activation function, by default "gelu"
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    """

    model_type = "FNO"
    
    def __init__(
        self,
        decoder_layers: int = 1,
        decoder_layer_size: int = 32,
        decoder_activation_fn_name: str = "silu",
        num_fno_layers: int = 4,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn_name: str = "gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.decoder_layers = decoder_layers
        self.decoder_layer_size = decoder_layer_size
        self.decoder_activation_fn_name = decoder_activation_fn_name
        self.num_fno_layers = num_fno_layers
        self.num_fno_modes = num_fno_modes
        self.padding = padding
        self.padding_type = padding_type
        self.activation_fn_name = activation_fn_name
        

class ConvNdFCLayer(nn.Module):
    """Channel-wise FC like layer with 1,2,3D convolutions

    Parameters
    ----------
    in_channels : int
        Size of input features
    out_channels : int
        Size of output features
    dimension : int
        Model dimensionality (supports 1,2,3)
    activation_fn : Union[nn.Module, None], optional
        Activation function to use. Can be None for no activation, by default None
    activation_par : Union[nn.Parameter, None], optional
        Additional parameters for the activation function, by default None
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int, 
        activation_fn: Union[nn.Module, Callable[[Tensor], Tensor], None] = None, 
        activation_par: Union[nn.Parameter, None] = None,
        weight_norm: bool = False,
    ) -> None:
        super().__init__()
        if activation_fn is None:
            self.activation_fn = nn.Identity()
        else:
            self.activation_fn = activation_fn
        self.activation_par = activation_par
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if dimension == 1:
            Conv = nn.Conv1d
        elif dimension == 2:
            Conv = nn.Conv2d
        elif dimension == 3:
            Conv = nn.Conv3d
        else:
            raise ValueError(f"Unsupported dimension: {dimension}. Must be 1, 2, or 3.")
        self.conv = Conv(self.in_channels, self.out_channels, kernel_size=1, bias=True)
        self.reset_parameters()

        if weight_norm:
            raise NotImplementedError("Weight norm not supported for Conv FC layers")

    def apply_activation(self, x: Tensor) -> Tensor:
        """Applied activation / learnable activations

        Parameters
        ----------
        x : Tensor
            Input tensor
        """
        if self.activation_par is None:
            x = self.activation_fn(x)
        else:
            x = self.activation_fn(self.activation_par * x)
        return x
    
    def reset_parameters(self) -> None:
        """Reset layer weights"""
        nn.init.constant_(self.conv.bias, 0)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x
    
######################################################
# Spectral Side of the model
######################################################
# from modulus>models>layers>spectral_layers.py  
class SpectralConvNd(nn.Module):
    """1D Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : List[int]
        Number of Fourier modes to multiply, at most floor(N/2) + 1
    dimension : int
        Model dimensionality (supports 1,2,3)
    """

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        modes: List[int],
        dimension: int, 
        ):
        super().__init__()
        assert len(modes) == dimension, "Length of modes must match dimension"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.dimension = dimension
        self.scale = 1 / (in_channels * out_channels)
        if self.dimension == 1:
            self.weights = nn.ParameterList([
                nn.Parameter(torch.empty(in_channels, out_channels, modes[0], 2))
            ])
        elif self.dimension == 2:
            self.weights = nn.ParameterList([
                nn.Parameter(torch.empty(in_channels, out_channels, modes[0], modes[1], 2)),
                nn.Parameter(torch.empty(in_channels, out_channels, modes[0], modes[1], 2))
            ])
        elif self.dimension == 3:
            self.weights = nn.ParameterList([
                nn.Parameter(torch.empty(in_channels, out_channels, modes[0], modes[1], modes[2], 2)),
                nn.Parameter(torch.empty(in_channels, out_channels, modes[0], modes[1], modes[2], 2)),
                nn.Parameter(torch.empty(in_channels, out_channels, modes[0], modes[1], modes[2], 2)),
                nn.Parameter(torch.empty(in_channels, out_channels, modes[0], modes[1], modes[2], 2))
            ])
        else:
            raise ValueError(f"Unsupported dimension: {dimension}. Must be 1, 2, or 3.")
        self.reset_parameters()

    def compl_mul(
        self,
        input: Tensor,
        weights: Tensor,
        einsum_eq: str,
    ) -> Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor
        einsum_eq : str
            Einsum equation for multiplication

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        # 1D (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        # 2D (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        # 3D (batch, in_channel, x, y, z), (in_channel, out_channel, x, y, z) -> (batch, out_channel, x, y, z)
        cweights = torch.view_as_complex(weights)
        return torch.einsum(einsum_eq, input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        if self.dimension == 1:
            # Compute Fourier coeffcients up to factor of e^(- something constant)
            x_ft = torch.fft.rfft(x)            
            # Multiply relevant Fourier modes
            out_ft = torch.zeros(
                batchsize,
                self.out_channels,
                x.size(-1) // 2 + 1,
                device=x.device,
                dtype=torch.cfloat,
            )
            out_ft[:, :, : self.modes[0]] = self.compl_mul(
                x_ft[:, :, : self.modes[0]],
                self.weights[0],
                "bix,iox->box"
            )
            # Return to physical space
            x = torch.fft.irfft(out_ft, n=x.size(-1))
        elif self.dimension == 2:
            # Compute Fourier coeffcients up to factor of e^(- something constant)
            x_ft = torch.fft.rfft2(x)
            # Multiply relevant Fourier modes
            out_ft = torch.zeros(
                batchsize,
                self.out_channels,
                x.size(-2),
                x.size(-1) // 2 + 1,
                dtype=torch.cfloat,
                device=x.device,
            )
            out_ft[:, :, : self.modes[0], : self.modes[1]] = self.compl_mul(
                x_ft[:, :, : self.modes[0], : self.modes[1]],
                self.weights[0],
                "bixy,ioxy->boxy"
            )
            out_ft[:, :, -self.modes[0] :, : self.modes[1]] = self.compl_mul(
                x_ft[:, :, -self.modes[0] :, : self.modes[1]],
                self.weights[1],
                "bixy,ioxy->boxy"
            )
            # Return to physical space
            x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        elif self.dimension == 3:
            # Compute Fourier coeffcients up to factor of e^(- something constant)
            x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
            # Multiply relevant Fourier modes
            out_ft = torch.zeros(
                batchsize,
                self.out_channels,
                x.size(-3),
                x.size(-2),
                x.size(-1) // 2 + 1,
                dtype=torch.cfloat,
                device=x.device,
            )
            out_ft[:, :, : self.modes[0], : self.modes[1], : self.modes[2]] = self.compl_mul(
                x_ft[:, :, : self.modes[0], : self.modes[1], : self.modes[2]], 
                self.weights[0],
                "bixyz,ioxyz->boxyz"
            )
            out_ft[:, :, -self.modes[0] :, : self.modes[1], : self.modes[2]] = self.compl_mul(
                x_ft[:, :, -self.modes[0] :, : self.modes[1], : self.modes[2]], 
                self.weights[1],
                "bixyz,ioxyz->boxyz"
            )
            out_ft[:, :, : self.modes[0], -self.modes[1] :, : self.modes[2]]= self.compl_mul(
                x_ft[:, :, : self.modes[0], -self.modes[1] :, : self.modes[2]], 
                self.weights[2],
                "bixyz,ioxyz->boxyz"
            )
            out_ft[:, :, -self.modes[0] :, -self.modes[1] :, : self.modes[2]] = self.compl_mul(
                x_ft[:, :, -self.modes[0] :, -self.modes[1] :, : self.modes[2]], 
                self.weights[3],
                "bixyz,ioxyz->boxyz"
            )
            # Return to physical space
            x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        for w in self.weights:
            w.data = self.scale * torch.rand(w.data.shape)
#######################################################

#######################################################
# Decoder utils, same for 1D, 2D, 3D
#######################################################
# from modulus>models>mlp>fully_connected.py
class FCLayer(nn.Module):
    """Densely connected NN layer
    Parameters
    ----------
    in_features : int
        Size of input features
    out_features : int
        Size of output features
    activation_fn : Union[nn.Module, None], optional
        Activation function to use. Can be None for no activation, by default None
    weight_norm : bool, optional
        Applies weight normalization to the layer, by default False
    weight_fact : bool, optional
        Applies weight factorization to the layer, by default False
    activation_par : Union[nn.Parameter, None], optional
        Additional parameters for the activation function, by default None
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn: Union[nn.Module, Callable[[Tensor], Tensor], None] = None,
        weight_norm: bool = False,
        weight_fact: bool = False,
        activation_par: Union[nn.Parameter, None] = None,
    ) -> None:
        super().__init__()

        if activation_fn is None:
            self.activation_fn = nn.Identity()
        else:
            self.activation_fn = activation_fn
        self.weight_norm = weight_norm
        self.weight_fact = weight_fact
        self.activation_par = activation_par

        # Ensure weight_norm and weight_fact are not both True
        if weight_norm and weight_fact:
            raise ValueError(
                "Cannot apply both weight normalization and weight factorization together, please select one."
            )

        #if weight_norm:
        #    self.linear = WeightNormLinear(in_features, out_features, bias=True)
        #elif weight_fact:
        #    self.linear = WeightFactLinear(in_features, out_features, bias=True)
        #else:
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset fully connected weights"""
        if not self.weight_norm and not self.weight_fact:
            nn.init.constant_(self.linear.bias, 0)
            nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)

        if self.activation_par is None:
            x = self.activation_fn(x)
        else:
            x = self.activation_fn(self.activation_par * x)

        return x


class FullyConnected(nn.Module):
    """A densely-connected MLP architecture

    Parameters
    ----------
    in_features : int, optional
        Size of input features, by default 512
    layer_size : int, optional
        Size of every hidden layer, by default 512
    out_features : int, optional
        Size of output features, by default 512
    num_layers : int, optional
        Number of hidden layers, by default 6
    activation_fn : Union[str, List[str]], optional
        Activation function to use, by default 'silu'
    skip_connections : bool, optional
        Add skip connections every 2 hidden layers, by default False
    adaptive_activations : bool, optional
        Use an adaptive activation function, by default False
    weight_norm : bool, optional
        Use weight norm on fully connected layers, by default False
    weight_fact : bool, optional
        Use weight factorization on fully connected layers, by default False

    Example
    -------
    >>> model = modulus.models.mlp.FullyConnected(in_features=32, out_features=64)
    >>> input = torch.randn(128, 32)
    >>> output = model(input)
    >>> output.size()
    torch.Size([128, 64])
    """

    def __init__(
        self,
        in_features: int = 512,
        layer_size: int = 512,
        out_features: int = 512,
        num_layers: int = 6,
        activation_fn: Union[str, List[str]] = "silu",
        skip_connections: bool = False,
        adaptive_activations: bool = False,
        weight_norm: bool = False,
        weight_fact: bool = False,
    ) -> None:
        super().__init__()
        self.skip_connections = skip_connections

        if adaptive_activations:
            activation_par = nn.Parameter(torch.ones(1))
        else:
            activation_par = None

        if not isinstance(activation_fn, list):
            activation_fn = [activation_fn] * num_layers
        if len(activation_fn) < num_layers:
            activation_fn = activation_fn + [activation_fn[-1]] * (
                num_layers - len(activation_fn)
            )
        activation_fn = [get_activation(a) for a in activation_fn]

        self.layers = nn.ModuleList()

        layer_in_features = in_features
        for i in range(num_layers):
            self.layers.append(
                FCLayer(
                    layer_in_features,
                    layer_size,
                    activation_fn[i],
                    weight_norm,
                    weight_fact,
                    activation_par,
                )
            )
            layer_in_features = layer_size

        self.final_layer = FCLayer(
            in_features=layer_size,
            out_features=out_features,
            activation_fn=None,
            weight_norm=False,
            weight_fact=False,
            activation_par=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_skip: Optional[Tensor] = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.skip_connections and i % 2 == 0:
                if x_skip is not None:
                    x, x_skip = x + x_skip, x
                else:
                    x_skip = x

        x = self.final_layer(x)
        return x
    
def build_lift_network(
    in_channels: int,
    fno_width: int,
    activation_fn: nn.Module,
    dimension: int,
    ) -> nn.Sequential:
    """construct network for lifting variables to latent space."""
    # Initial lift network
    lift_network = torch.nn.Sequential()
    lift_network.append(
        ConvNdFCLayer(
            in_channels = in_channels,
            out_channels = int(fno_width / 2), 
            dimension = dimension,
            #activation_fn = activation_fn,
            )
    )
    lift_network.append(activation_fn) #insert the activation function into ConvNdFCLayer,no need to add it here
    lift_network.append(
        ConvNdFCLayer(
            in_channels=int(fno_width / 2), 
            out_channels=fno_width,
            dimension=dimension,
            )
    )
    return lift_network

def build_fno(
    fno_width: int,
    num_fno_modes: List[int],
    num_fno_layers: int,
    dimension: int,
    ) -> Tuple[nn.ModuleList, nn.ModuleList]:
    """construct FNO block.
    """
    # Build Neural Fourier Operators
    if dimension == 1:
        Conv = nn.Conv1d
    elif dimension == 2:
        Conv = nn.Conv2d
    elif dimension == 3:
        Conv = nn.Conv3d
    else:
        raise ValueError(f"Unsupported dimension: {dimension}. Must be 1, 2, or 3.")
    spconv_layers = nn.ModuleList()
    conv_layers = nn.ModuleList()
    for _ in range(num_fno_layers):
        spconv_layers.append(
            SpectralConvNd(
                in_channels=fno_width,
                out_channels=fno_width,
                modes=num_fno_modes,
                dimension=dimension,
            )
        )
        conv_layers.append(Conv(fno_width, fno_width, 1))
    return spconv_layers, conv_layers