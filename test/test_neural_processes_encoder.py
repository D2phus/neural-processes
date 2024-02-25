import torch
import pytest
from neural_processes.model.encoder import Encoder
from neural_processes.model.attention import Attention, MLP

batch_size, num_samples, x_size, y_size, r_dim = 100, 10, 3, 1, 128
data = [
    (torch.randn(batch_size, num_samples, x_size),
     torch.randn(batch_size, num_samples, y_size)),
]


@pytest.mark.parametrize("context_X, context_y", data)
def test_encoder_shape(context_X, context_y):
    batch_size, num_samples, x_size = context_X.shape
    y_size = context_y.shape[-1]

    apply_attention = [(False, False), (True, False),
                       (False, True), (True, True)]
    for aa in apply_attention:
        apply_self_attention, apply_cross_attention = aa
        model = Encoder(x_size=x_size, y_size=y_size,
                        apply_self_attention=apply_self_attention, apply_cross_attention=apply_cross_attention)
        assert model(context_X, context_y, context_X).shape == (
            batch_size, r_dim)
