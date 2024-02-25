import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, 
                 apply_attention: bool,
                 mlp: nn.Sequential = None, 
                 attention: nn.Module = None):
        """The encoder. 
        """
        if apply_attention and attention is None:
            raise ValueError('Invalid attention module.')
        if not apply_attention and mlp is None:
            raise ValueError('Invalid MLP module.')
        
        super().__init__()
        self._apply_attention = apply_attention
        self._mlp = mlp
        self._attention = attention

    def forward(self, context_X: torch.Tensor, context_y: torch.Tensor, target_X: torch.Tensor) -> torch.Tensor:
        if self._apply_attention:
            attn_output, _ = self._attention(
                query=context_y, key=context_y, value=context_X)
            return attn_output
        else:
            context = torch.cat((context_X, context_y), dim=-1)
            mlp_output = self._mlp(context)
            return mlp_output
        
