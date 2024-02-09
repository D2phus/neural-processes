import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent


class DeterministicDecoder(nn.Module):
    """The decoder.
    """

    def __init__(self,
                 x_size: int,
                 r_dim: int,
                 num_layers: int,
                 num_units: int,
                 y_size: int,
                 activation_cls: str = 'relu'):
        """The decoder. 
        """
        super().__init__()
        self._y_size = y_size
        self._r_dim = r_dim
        self._x_size = x_size
        self._in_dim = r_dim + x_size

        if not hasattr(nn, activation_cls):
            raise ValueError(f"Invalid activation_cls: {activation_cls}")
        self._activation_cls = activation_cls
        self._activation = getattr(nn, activation_cls)()

        self._layer_sizes = [
            self._in_dim, *(num_units for _ in range(num_layers)), 2*y_size]
        self._linear = nn.ModuleList(
            nn.Linear(i, o) for i, o in zip(self._layer_sizes[:-1], self._layer_sizes[1:])
        )

    def forward(self, representation: torch.tensor, target_x: torch.tensor):
        """decodes the targets.
        Args: 
            representation of shape (batch_size, r_dim): global representation learnt on observations.
            target_x of shape (batch_size, num_target, x_size)
            NOTE `representation` and `target_x` share the saem batch_size 

        Returns: 
            dist: conditional output Normal distribution for all the target location. 
            mean of shape (batch_size, num_target, y_size): the mean of conditional distribution. 
            std of shape (batch_size, num_target, y_size): the standard deviation of conditional distribution. 
            NOTE std should be bounded as non-negative.
            """
        if len(representation.shape) != 2:
            raise ValueError('Invalid `representation` shape.')
        if len(target_x.shape) != 3:
            raise ValueError('Invalid `target_x` shape.')

        _, r_dim = representation.shape
        batch_size, num_target, x_size = target_x.shape

        if x_size != self._x_size:
            raise ValueError('Invalid `target_x` size.')
        if r_dim != self._r_dim:
            raise ValueError('Invalid `representation` size.')

        # parallelism on all the context points
        representation = representation.unsqueeze(1).repeat(1, num_target, 1)

        phi = torch.cat((target_x, representation), dim=-1)
        # (batch_size*num_target, r_dim+x_size)
        phi = phi.view(batch_size*num_target, -1)
        for idx, l in enumerate(self._linear):
            phi = l(phi)
            if idx < len(self._linear) - 1:
                phi = self._activation(phi)

        mean, log_std = phi[:, :self._y_size], phi[:,
                                                   self._y_size:]  # (batch_size*num_target, y_size)
        # NOTE bound the standard deviation as non-negative
        # method 1: softplus
        std = 0.1 + 0.9 * F.softplus(log_std) # softplus can still technically return zero, so we add a small constant
        # method 2: exp. NOTE explode.
        # std = torch.exp(log_std)

        # bring back size
        mean, std = mean.view(batch_size, num_target, -
                              1), std.view(batch_size, num_target, -1)

        # NOTE equivalent to `tf.contrib.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)`
        # dist = Independent(Normal(loc=mean, scale=std), 1)
        dist = MultivariateNormal(
            loc=mean, scale_tril=torch.diag_embed(std, dim1=2, dim2=3))
        # print(dist.batch_shape,dist.event_shape)
        return dist, mean, std
