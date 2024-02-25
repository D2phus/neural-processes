import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class Attention(nn.Module):
    def __init__(self, type: str = 'scaled_dot', num_heads: int=None) -> None:
        """The attention mechanism.
        
        Args:
            type: the type of attention mechanism, including 'uniform', 'multihead', 'scaled_dot', and 'laplace'.
                - 'uniform': the uniform attention mechanism, equivalent to the mean aggregation.
                - 'multihead': the multihead attention mechanism.
                - 'scaled_dot': the scaled dot product attention mechanism.
                - 'laplace': the laplace attention mechanism.
            num_heads: the number of heads if multihead attention is applied.
        """
        super().__init__()
        if type not in ['uniform', 'multihead', 'scaled_dot', 'laplace']:
            raise ValueError(f"Invalid attention type: {type}")
        self._type = type

        if type == 'uniform':
            self._attn = self._uniform_attn
        if type == 'multihead':
            self._num_heads = num_heads or 8
            self._attn = self._multihead_attn
        elif type == 'scaled_dot':
            self._attn = self._dot_product_attn
        elif type == 'laplace':
            self._attn = self._laplace_attn

    def _uniform_attn(self, query: torch.tensor, key: torch.tensor, value: torch.tensor) -> torch.tensor:
        """Sets up the uniform attention.
        
        Args:
            query: the query tensor of shape (batch_size, m, k)
            key: the key tensor of shape (batch_size, n, k)
            value: the value tensor of shape (batch_size, n, v)

        Returns:
            the attention output of shape (batch_size, m, v)
        """
        output = value.mean(dim=1) # (batch_size, v)
        # NOTE in np, the global representation is of shape (batch_size, r_dim), which will be broadcasted to (batch_size, num_samples, r_dim) when decoding.
        return output.unsqueeze(1).repeat(1, query.size(1), 1) # (batch_size, m, v)

    def _multihead_attn(self, query: torch.tensor, key: torch.tensor, value: torch.tensor) -> torch.tensor:
        """Sets up the multihead attention.
        Args:
            embed_dim: the dimension of the embedding.
            num_heads: the number of heads.
        """
        # TODO 

    def _laplace_attn(self, query: torch.tensor, key: torch.tensor, value: torch.tensor) -> torch.tensor:
        """Sets up the laplace attention.

        Args:
            query: the query tensor of shape (batch_size, m, k)
            key: the key tensor of shape (batch_size, n, k)
            value: the value tensor of shape (batch_size, n, v)
        
        Returns:
         # NOTE 
            the attention output of shape (batch_size, m, v)
            # the attention output of shape (m, v)
        """
        w = torch.abs(query - key).sum(dim=-1) # (batch_size, m, n)
        return torch.matmul(F.softmax(w, dim=0), value)
        # w = F.softmax(w, dim=0).transpose(0, 1) # (m, batch_size, n)
        # return torch.matmul(w.flatten(start_dim=-2, end_dim=-1), value.flatten(start_dim=0, end_dim=1)) # (m, v)

    def _dot_product_attn(self, query: torch.tensor, key: torch.tensor, value: torch.tensor) -> torch.tensor:
        """Sets up the scaled dot product attention.

        Args:
            query: the query tensor of shape (batch_size, m, k)
            key: the key tensor of shape (batch_size, n, k)
            value: the value tensor of shape (batch_size, n, v)

        Returns:
         # NOTE 
            the attention output of shape (batch_size, m, v)
            # the attention output of shape (m, v)
        """
        output = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1)) # (batch_size, m, n)
        output = F.softmax(output, dim=0) # (batch_size, m, n)
        # output = output.transpose(0, 1) # (m, batch_size, n)
        return torch.matmul(output, value) 
        # return torch.matmul(output.flatten(start_dim=-2, end_dim=-1), value.flatten(start_dim=0, end_dim=1)) # (m, v)

    def forward(self, query: torch.tensor, key: torch.tensor, value: torch.tensor) -> torch.tensor:
        """computes the attention outputs.
        
        Args:
        NOTE we sample context pairs from `batch_size` tasks, each task has `m` context pairs. 
            query: the query tensor of shape (batch_size, m, k)
            key: the key tensor of shape (batch_size, n, k)
            value: the value tensor of shape (batch_size, n, v)
        
        Returns:
         # NOTE 
            the attention output of shape (batch_size, m, v)
            # the attention output of shape (m, v)
        """
        return self._attn(query, key, value)