import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


class DeterministicEncoder(nn.Module):
    def __init__(self,
                 x_size: int, 
                 y_size: int, 
                 r_dim: int, 
                 num_layers: int,
                 num_units: int, 
                 activation_cls: str = 'ReLU'):
        """The encoder. 
        """
        super().__init__()
        self._r_dim = r_dim
        self._x_size = x_size
        self._y_size = y_size
        self._in_dim = x_size + y_size
        
        if not hasattr(nn, activation_cls):
            raise ValueError(f"Invalid activation_cls: {activation_cls}")
        self._activation_cls = activation_cls
        self._activation = getattr(nn, activation_cls)()

        self._layer_sizes = [self._in_dim, *(num_units for _ in range(num_layers)), r_dim]
        self._linear = nn.ModuleList(
            nn.Linear(i, o) for i, o in zip(self._layer_sizes[:-1], self._layer_sizes[1:]))

    def _aggregate(self, representation: torch.tensor) -> torch.tensor:
        """permutation-invariant aggregator. 

        Args: 
            representation of shape (batch_size, num_samples, r_dim)

        Returns: 
            # global_representation of shape (r_dim)
            NOTE global_representation of shape (batch_size, r_dim)
        """
        return representation.mean(dim=1)
        # return representation.view(-1, self._r_dim).mean(dim=0)

    def forward(self, context_X: torch.Tensor, context_y: torch.Tensor)-> torch.Tensor:
        """Encodes the inputs into one representation.

        Args: 
            context_X of shape (batch_size, num_samples, x_size)
            context_y of shape (batch_size, num_samples, y_size)

        Returns: 
            representation of shape (batch_size, representation_size)
            # global_representation of shape (r_dim)
        """
        if len(context_X.shape) != 3:
            raise ValueError('Invalid `context_X` shape.')
        if len(context_y.shape) != 3:
            raise ValueError('Invalid `context_y` shape.')
        
        context = torch.cat((context_X, context_y), dim=-1)
        _, _, in_dim = context.shape
        if in_dim != self._in_dim: 
            raise ValueError('Invalid context size.')
        
        # pass the observations through MLP
        representation = context
        for idx, l in enumerate(self._linear):
            representation = l(representation)
            if idx < len(self._linear) - 1:
                representation = self._activation(representation)

        return self.aggregate(representation)
