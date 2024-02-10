from .decoder import LatentDecoder
from .encoder import MeanAggregateEncoder
from .aggregator import Aggregator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

from tqdm import tqdm


class LatentNP(nn.Module):
    def __init__(self,
                 encoder_num_layers: int,
                 encoder_num_units: int,
                 encoder_activation_cls: str,
                 decoder_num_layers: int,
                 decoder_num_units: int,
                 decoder_activation_cls: str,
                 agggreagtor_num_layers: int,
                 agggreagtor_num_units: int,
                 agggreagtor_activation_cls: str,
                 r_dim: int,
                 x_size: int,
                 y_size: int,
                 ):
        """Latent Neural Processes.

        Args:
            encoder_num_layers: the number of layers in the encoder.
            encoder_num_units: the number of units in each hidden layer of the encoder.
            encoder_activation_cls: the activation function of the encoder.
            decoder_num_layers: the number of layers in the decoder.
            decoder_num_units: the number of units in each hidden layer of the decoder.
            decoder_activation_cls: the activation function of the decoder.
            agggreagtor_num_layers: the number of layers in the aggregator.
            agggreagtor_num_units: the number of units in each hidden layer of the aggregator.
            agggreagtor_activation_cls: the activation function of the aggregator.
            r_dim: the dimension of the representation.
            x_size: the dimension of the input x.
            y_size: the dimension of the input y.
        """
        super().__init__()
        self._encoder_num_layers = encoder_num_layers
        self._encoder_num_units = encoder_num_units
        self._encoder_activation_cls = encoder_activation_cls
        self._decoder_num_layers = decoder_num_layers
        self._decoder_num_units = decoder_num_units
        self._decoder_activation_cls = decoder_activation_cls
        self._r_dim = r_dim
        self._y_size = y_size
        self._x_size = x_size

        self._encoder = MeanAggregateEncoder(x_size=self._x_size,
                                      y_size=self._y_size,
                                      r_dim=self._r_dim,
                                      num_layers=self._encoder_num_layers,
                                      num_units=self._encoder_num_units,
                                      activation_cls=self._encoder_activation_cls)
        self._aggregator = Aggregator(r_dim=self._r_dim,
                                      num_layers=agggreagtor_num_layers,
                                      num_units=agggreagtor_num_units,
                                      activation_cls=agggreagtor_activation_cls)
        self._decoder = LatentDecoder(x_size=self._x_size,
                                      y_size=self._y_size,
                                      r_dim=self._r_dim,
                                      num_layers=self._decoder_num_layers,
                                      num_units=self._decoder_num_units,
                                      activation_cls=self._decoder_activation_cls)

    def forward(self, query, num_total_points, num_contexts, target_y=None):
        (context_X, context_y), target_X = query

        representation = self._encoder(context_X, context_y)
        # aggregate the representation
        latent_dist, z_mu, z_sigma = self._aggregator(representation)
        # sample from the latent distribution
        representation = latent_dist.rsample()
        # representation = latent_dist.sample() # NOTE we cannot backpropagate since the computational graph is disconnected!
        dist, mu, sigma = self._decoder(representation, target_X)
        # when training
        if target_y is not None:
            log_p = dist.log_prob(target_y)
        # when testing
        else:
            log_p = None

        return log_p, mu, sigma
