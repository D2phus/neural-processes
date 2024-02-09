from .decoder import DeterministicDecoder
from .encoder import DeterministicEncoder

import collections
import copy
from math import exp

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

from tqdm import tqdm


class DeterministicCNP(nn.Module):
    def __init__(self,
                 encoder_num_layers: int,
                 encoder_num_units: int,
                 encoder_activation_cls: str,
                 decoder_num_layers: int,
                 decoder_num_units: int,
                 decoder_activation_cls: str,
                 r_dim: int,
                 x_size: int, 
                 y_size: int,
                 ):
        """Deterministic Conditional Neural Processes.
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

        self._encoder = DeterministicEncoder(x_size=self._x_size, 
                                            y_size=self._y_size, 
                                            r_dim=self._r_dim, 
                                            num_layers=self._encoder_num_layers,
                                            num_units=self._encoder_num_units, 
                                            activation_cls=self._encoder_activation_cls)
        self._decoder = DeterministicDecoder(x_size=self._x_size, 
                                            y_size=self._y_size, 
                                            r_dim=self._r_dim,
                                            num_layers=self._decoder_num_layers,
                                            num_units=self._decoder_num_units, 
                                            activation_cls=self._decoder_activation_cls)

    def forward(self, query, num_total_points, num_contexts, target_y=None):
        """Returns the predicted mean and variance at the target points.

        Args:
            query: A tuple of (context_X, context_y, target_X) where:
                context_x: Array of shape batch_size x num_context x 1 contains the x values of the context points.
                context_y: Array of shape batch_size x num_context x 1 contains the y values of the context points.
                target_x: Array of shape batch_size x num_target x 1 contains the x values of the target points.
                target_y: The ground truth y values of the target y. An array of shape batch_size x num_targets x 1.
                num_total_points: Number of target points.

        Returns:
            log_p of shape (batch_size, num_target): The log_probability of the target_y given the predicted distribution.
            mu of shape (batch_size, num_target, y_size): The mean of the predicted distribution.
            sigma of shape (batch_size, num_target, y_size): The variance of the predicted distribution.
        """
        (context_X, context_y), target_X = query

        representation = self._encoder(context_X, context_y)
        dist, mu, sigma = self._decoder(representation, target_X)
        # when training
        if target_y is not None:
            log_p = dist.log_prob(target_y)
        # when testing
        else:
            log_p = None

        return log_p, mu, sigma
