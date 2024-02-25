import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 x_size: int,
                 y_size: int,
                 r_dim: int = 128,
                 num_units: int = 128, 
                 num_layers: int = 3, 
                 activation_cls: str = 'ReLU',
                 ):
        super().__init__()
        self._r_dim = r_dim
        self._x_size = x_size
        self._y_size = y_size
        self._in_dim = x_size + y_size
        self._num_layers = num_layers
        self._num_units = num_units

        if not hasattr(nn, activation_cls):
            raise ValueError(f"Invalid activation_cls: {activation_cls}")
        self._activation_cls = activation_cls
        self._activation = getattr(nn, activation_cls)()

        self._layer_sizes = [self._in_dim, *
                             (num_units for _ in range(num_layers)), r_dim]
        self._linear = nn.ModuleList(
            nn.Linear(i, o) for i, o in zip(self._layer_sizes[:-1], self._layer_sizes[1:]))

    def forward(self, context_X: torch.Tensor, context_y: torch.Tensor) -> torch.Tensor:
        """Encodes the inputs into one representation.

        Args: 
            context_X of shape (batch_size, num_samples, x_size)
            context_y of shape (batch_size, num_samples, y_size)

        Returns: 
            representation of shape (batch_size, representation_size)
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

        return representation