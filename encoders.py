from typing import List, Union

import torch
from torch import nn
from typing_extensions import Literal

__all__ = ["get_mlp"]


def get_mlp(
    n_in: int,
    n_out: int,
    layers: List[int],
    layer_normalization: Union[None, Literal["bn"], Literal["gn"]] = None,
    output_normalization: Union[None, None] = None,
    output_normalization_kwargs=None,
    act_inf_param=0.01,
):
    """
    Creates an MLP.

    Args:
        n_in: Dimensionality of the input data
        n_out: Dimensionality of the output data
        layers: Number of neurons for each hidden layer
        layer_normalization: Normalization for each hidden layer.
            Possible values: bn (batch norm), gn (group norm), None
        output_normalization: (Optional) Normalization applied to output of network.
        output_normalization_kwargs: Arguments passed to the output normalization, e.g., the radius for the sphere.
    """
    modules: List[nn.Module] = []

    def add_module(n_layer_in: int, n_layer_out: int, last_layer: bool = False):
        modules.append(nn.Linear(n_layer_in, n_layer_out))
        # perform normalization & activation not in last layer
        if not last_layer:
            if layer_normalization == "bn":
                modules.append(nn.BatchNorm1d(n_layer_out))
            elif layer_normalization == "gn":
                modules.append(nn.GroupNorm(1, n_layer_out))
            modules.append(nn.LeakyReLU(negative_slope=act_inf_param))

        return n_layer_out

    if len(layers) > 0:
        n_out_last_layer = n_in
    else:
        assert n_in == n_out, "Network with no layers must have matching n_in and n_out"
        modules.append(layers.Lambda(lambda x: x))

    layers.append(n_out)

    for i, l in enumerate(layers):
        n_out_last_layer = add_module(n_out_last_layer, l, i == len(layers) - 1)

    if output_normalization_kwargs is None:
        output_normalization_kwargs = {}
    return nn.Sequential(*modules)


class CompositionEncMix(nn.Module):
    """
    A class representing a composition of mixing functions and encoders (only for numerical experiment).

    Args:
        mixing_fns (list): List of mixing functions.
        encoders (list): List of encoders.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        mixing_fns (list): List of mixing functions.
        encoders (list): List of encoders.
        H (torch.nn.ModuleList): List of composed functions.

    """

    def __init__(self, mixing_fns, encoders, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mixing_fns = mixing_fns
        self.encoders = encoders

        self.H = torch.nn.ModuleList()
        for i in range(len(self.mixing_fns)):
            f_i = self.mixing_fns[i]
            g_i = self.encoders[i]
            h_i = torch.nn.Sequential(*list(f_i) + list(g_i))  #: g_i(f_i(z)) # f: mixing function; g: encoder
            self.H.append(h_i)

    def forward(self, z, z3, S_k, n_views=4):
        """
        Forward pass of the composition encoder.

        Args:
            z (torch.Tensor): Input tensor.
            z3 (torch.Tensor): Another input tensor.
            S_k (list): List of indices.
            n_views (int): Number of views.

        Returns:
            tuple: A tuple containing the reconstructed tensors, the dictionary of intermediate outputs.

        """
        with torch.set_grad_enabled(True):
            z3_rec = []
            z_rec = []

            hzs = dict({})

            for i in range(n_views):
                z_rec_i = self.view_specific_forward(z, view_idx=i, S_k=S_k)
                z3_rec_i = self.view_specific_forward(z3, view_idx=i, S_k=S_k)
                z_rec.append(z_rec_i)
                z3_rec.append(z3_rec_i)
                hzs[i] = {"hz": z_rec_i}
        return z_rec, z3_rec, hzs

    def view_specific_forward(self, z, view_idx, S_k):
        """
        Forward pass for a specific view.

        Args:
            z (torch.Tensor): Input tensor.
            view_idx (int): Index of the view.
            S_k (list): List of indices.

        Returns:
            torch.Tensor: Reconstructed tensor.

        """
        z_rec_i = self.H[view_idx].forward(z[:, S_k[view_idx]])
        return z_rec_i


class ConvEncoder(nn.Module):
    """
    Convolutional Encoder module for image processing.

    Args:
        input_size (int): Number of input channels (default: 3).
        output_size (int): Number of output channels (default: 10).
        hidden_layers (list): List of hidden layer sizes (default: []).
    """

    def __init__(self, input_size=3, output_size=10, hidden_layers=[]) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_size, kernel_size=4, stride=2, out_channels=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, kernel_size=4, stride=2, out_channels=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, kernel_size=4, stride=2, out_channels=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, kernel_size=4, stride=2, out_channels=64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_size),
        )

    def forward(self, x):
        """
        Forward pass of the ConvEncoder module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded output tensor.
        """
        return self.encoder(x)


class TextEncoder2D(nn.Module):
    """
    2D ConvNet Text Encoder module.

    Args:
        input_size (int): The size of the input.
        output_size (int): The size of the output.
        sequence_length (int): The length of the input sequence.
        embedding_dim (int, optional): The dimension of the embedding. Defaults to 128.
        fbase (int, optional): The base factor for the number of filters in the convolutional layers. Defaults to 25.

    Raises:
        ValueError: If sequence_length is not between 24 and 31.

    Attributes:
        fbase (int): The base factor for the number of filters in the convolutional layers.
        embedding (nn.Linear): Linear layer for embedding the input.
        convnet (nn.Sequential): Sequential convolutional layers.
        ldim (int): The dimension of the linear layer.
        linear (nn.Linear): Linear layer for the final output.

    """

    def __init__(self, input_size, output_size, sequence_length, embedding_dim=128, fbase=25):
        super(TextEncoder2D, self).__init__()
        if sequence_length < 24 or sequence_length > 31:
            raise ValueError("TextEncoder2D expects sequence_length between 24 and 31")
        self.fbase = fbase
        self.embedding = nn.Linear(input_size, embedding_dim)
        self.convnet = nn.Sequential(
            # input size: 1 x sequence_length x embedding_dim
            nn.Conv2d(1, fbase, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fbase),
            nn.ReLU(True),
            nn.Conv2d(fbase, fbase * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fbase * 2),
            nn.ReLU(True),
            nn.Conv2d(fbase * 2, fbase * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fbase * 4),
            nn.ReLU(True),
            # size: (fbase * 4) x 3 x 16
        )
        self.ldim = fbase * 4 * 3 * 16
        self.linear = nn.Linear(self.ldim, output_size)

    def forward(self, x):
        """
        Forward pass of the encoder.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the encoder.
        """
        x = self.embedding(x).unsqueeze(1)
        x = self.convnet(x)
        x = x.view(-1, self.ldim)
        x = self.linear(x)
        return x
