import torch
import torch.nn as nn

from .attention import Attention, MLP


class Encoder(nn.Module):
    def __init__(self,
                 x_size: int,
                 y_size: int,
                 apply_self_attention: bool = False,
                 attention: nn.Module = None,
                 mlp: nn.Module = None,
                 apply_cross_attention: bool = False,
                 cross_attention: nn.Module = None,):
        """The encoder of the neural processes, consisting of the context encoder and the representation aggregator.  

        Args:
            x_size: the dimension of the input x.
            y_size: the dimension of the input y.
            r_dim: the dimension of the representation.
            apply_self_attention: whether to apply self-attention mechanism when encoding the context pairs. MLP is used if False.
            attention: the attention mechanism.
            mlp: the mlp for the context encoding.
            apply_cross_attention: whether to apply cross-attention mechanism when encoding the individual representation. mean aggregation is used if False.
            cross_attention: the cross-attention mechanism.
        """
        super().__init__()
        self._x_size = x_size
        self._y_size = y_size
        self._in_dim = x_size + y_size

        self._apply_self_attention = apply_self_attention
        self._mlp = mlp if mlp is not None else MLP(x_size, y_size)
        self._self_attention = attention if attention is not None else Attention()

        self._apply_cross_attention = apply_cross_attention
        self._cross_attention = cross_attention if cross_attention is not None else Attention()

    def _aggregate(self, representation: torch.tensor, target_X: torch.tensor, context_X: torch.tensor) -> torch.tensor:
        """permutation-invariant aggregator. 

        Args: 
            representation of shape (batch_size, num_samples, r_dim)

        Returns: 
            global_representation of shape (batch_size, r_dim): global representation of context pairs for each task (curve).
        """
        if self._apply_cross_attention: # cross-attention mechanism
            return self._cross_attention(query=target_X, key=representation, value=context_X) # of shape (batch_size, )
        else: # mean-aggregation mechanism
            return representation.mean(dim=1) # take average along num_samples

    def forward(self, context_X: torch.Tensor, context_y: torch.Tensor, target_X: torch.Tensor) -> torch.Tensor:
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

        if self._apply_self_attention:
            representations, _ = self._self_attention(
                query=context_y, key=context_y, value=context_X) # of shape (batch_size, num_samples, x_size)
        else:
            representations = self._mlp(context_X, context_y) # of shape (batch_size, num_samples, r_dim)

        return self._aggregate(representations, target_X, context_X)
