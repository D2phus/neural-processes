import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Aggregator(nn.Module):
    def __init__(self, r_dim: int, num_layers: int, num_units: int, activation_cls: str = 'ReLU'):
        """The aggregator for the latent neural processes. MLP is used. 

        Args:
            num_layers: the number of layers in the aggregator.
            num_units: the number of units in each hidden layer of the aggregator.
            activation_cls: the activation function of the aggregator.
            r_dim: the dimension of the representation.
        """
        super().__init__()
        self._r_dim = r_dim
        self._num_layers = num_layers
        self._num_units = num_units

        if not hasattr(nn, activation_cls):
            raise ValueError(f"Invalid activation_cls: {activation_cls}")
        self._activation_cls = activation_cls
        self._activation = getattr(nn, activation_cls)()

        self._layer_sizes = [r_dim, *(num_units for _ in range(num_layers)), 2*r_dim]
        self._linear = nn.ModuleList(
            nn.Linear(i, o) for i, o in zip(self._layer_sizes[:-1], self._layer_sizes[1:]))


    def forward(self, representation: torch.tensor) -> torch.tensor:
        """models the latent distribution from the representation.

        Args: 
            representation of shape (batch_size, r_dim)
        Returns: 
            latent_distribution of shape (batch_size, r_dim)
            mean of shape (batch_size, r_dim)
            std of shape (batch_size, r_dim)
        """
        for idx, l in enumerate(self._linear):
            representation = l(representation)
            if idx < len(self._linear) - 1:
                representation = self._activation(representation)
            
        mean, log_std = representation.chunk(2, dim=-1)
        # bound the standard deviation to be non-negative.
        std = 0.1 + 0.9 * F.softplus(log_std)
        return Normal(loc=mean, scale=std), mean, std