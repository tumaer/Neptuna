from typing import Callable, Union
from typing import List, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor

#from modulus>models>layers>fully_connected_layers.py
class ConvFCLayer(nn.Module):
    """Base class for 1x1 Conv layer for image channels

    Parameters
    ----------
    activation_fn : Union[nn.Module, None], optional
        Activation function to use. Can be None for no activation, by default None
    activation_par : Union[nn.Parameter, None], optional
        Additional parameters for the activation function, by default None
    """

    def __init__(
        self,
        activation_fn: Union[nn.Module, Callable[[Tensor], Tensor], None] = None,
        activation_par: Union[nn.Parameter, None] = None,
    ) -> None:
        super().__init__()
        if activation_fn is None:
            self.activation_fn = nn.Identity
        else:
            self.activation_fn = activation_fn
        self.activation_par = activation_par

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

#from modulus>models>layers>fully_connected_layers.py
class Conv1dFCLayer(ConvFCLayer):
    """Channel-wise FC like layer with 1d convolutions

    Parameters
    ----------
    in_features : int
        Size of input features
    out_features : int
        Size of output features
    activation_fn : Union[nn.Module, None], optional
        Activation function to use. Can be None for no activation, by default None
    activation_par : Union[nn.Parameter, None], optional
        Additional parameters for the activation function, by default None
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn: Union[nn.Module, Callable[[Tensor], Tensor], None] = None,
        activation_par: Union[nn.Parameter, None] = None,
        weight_norm: bool = False,
    ) -> None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_features
        self.out_channels = out_features
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
        self.reset_parameters()

        if weight_norm:
            raise NotImplementedError("Weight norm not supported for Conv FC layers")

    def reset_parameters(self) -> None:
        """Reset layer weights"""
        nn.init.constant_(self.conv.bias, 0)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x

#from modulus>models>layers>fully_connected_layers.py
class Conv2dFCLayer(ConvFCLayer):
    """Channel-wise FC like layer with 2d convolutions

    Parameters
    ----------
    in_features : int
        Size of input features
    out_features : int
        Size of output features
    activation_fn : Union[nn.Module, None], optional
        Activation function to use. Can be None for no activation, by default None
    activation_par : Union[nn.Parameter, None], optional
        Additional parameters for the activation function, by default None
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn: Union[nn.Module, Callable[[Tensor], Tensor], None] = None,
        activation_par: Union[nn.Parameter, None] = None,
    ) -> None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset layer weights"""
        nn.init.constant_(self.conv.bias, 0)
        self.conv.bias.requires_grad = False
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x

#from modulus>models>layers>fully_connected_layers.py
class Conv3dFCLayer(ConvFCLayer):
    """Channel-wise FC like layer with 3d convolutions

    Parameters
    ----------
    in_features : int
        Size of input features
    out_features : int
        Size of output features
    activation_fn : Union[nn.Module, None], optional
        Activation function to use. Can be None for no activation, by default None
    activation_par : Union[nn.Parameter, None], optional
        Additional parameters for the activation function, by default None
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn: Union[nn.Module, Callable[[Tensor], Tensor], None] = None,
        activation_par: Union[nn.Parameter, None] = None,
    ) -> None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
        self.reset_parameters()

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
class SpectralConv1d(nn.Module):
    """1D Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply, at most floor(N/2) + 1
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            torch.empty(in_channels, out_channels, self.modes1, 2)
        )
        self.reset_parameters()

    def compl_mul1d(
        self,
        input: Tensor,
        weights: Tensor,
    ) -> Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bix,iox->box", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        bsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            bsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1],
            self.weights1,
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.data = self.scale * torch.rand(self.weights1.data.shape)

#from modulus>models>layers>spectral_layers.py    
class SpectralConv2d(nn.Module):
    """2D Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
    modes2 : int
        Number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            torch.empty(in_channels, out_channels, self.modes1, self.modes2, 2)
        )
        self.weights2 = nn.Parameter(
            torch.empty(in_channels, out_channels, self.modes1, self.modes2, 2)
        )
        self.reset_parameters()

    def compl_mul2d(self, input: Tensor, weights: Tensor) -> Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bixy,ioxy->boxy", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
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
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights1,
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2],
            self.weights2,
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.data = self.scale * torch.rand(self.weights1.data.shape)
        self.weights2.data = self.scale * torch.rand(self.weights2.data.shape)

#from modulus>models>layers>spectral_layers.py    
class SpectralConv3d(nn.Module):
    """3D Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
    modes2 : int
        Number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    modes3 : int
        Number of Fourier modes to multiply in third dimension, at most floor(N/2) + 1
    """

    def __init__(
        self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            torch.empty(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2
            )
        )
        self.weights2 = nn.Parameter(
            torch.empty(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2
            )
        )
        self.weights3 = nn.Parameter(
            torch.empty(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2
            )
        )
        self.weights4 = nn.Parameter(
            torch.empty(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2
            )
        )
        self.reset_parameters()

    def compl_mul3d(
        self,
        input: Tensor,
        weights: Tensor,
    ) -> Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        # (batch, in_channel, x, y, z), (in_channel, out_channel, x, y, z) -> (batch, out_channel, x, y, z)
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bixyz,ioxyz->boxyz", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
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
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.data = self.scale * torch.rand(self.weights1.data.shape)
        self.weights2.data = self.scale * torch.rand(self.weights2.data.shape)
        self.weights3.data = self.scale * torch.rand(self.weights3.data.shape)
        self.weights4.data = self.scale * torch.rand(self.weights4.data.shape)
#######################################################

#######################################################
# Decoder utils, same for 1D, 2D, 3D
#######################################################
#from modulus>models>layers>fully_connected_layers.py
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
            self.activation_fn = nn.Identity
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

# from modulus>models>mlp>fully_connected.py
class FullyConnected():
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
        activation_fn = [activation_fn.get_activation(a) for a in activation_fn]

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