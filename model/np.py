from .encoder import Encoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

from tqdm import tqdm

class Neural_Processes(nn.Module):
    def __init__(self, 
                 config,): 
        """The neural processes family. 
        """
        super().__init__()
        self._x_size = config.dataset.x_size
        self._y_size = config.dataset.y_size

        np_type = config.model.type        
        # Conditional Neural Processes: deterministic MLP encoder, mean-aggregation, MLP decoder
        if np_type == 'CNP':
            r_dim = config.model.r_dim
            encoder_num_layers = config.model.encoder.num_layers 
            encoder_num_units = config.model.encoder.num_units
            encoder_activation_cls = config.model.encoder.activation_cls
            if not hasattr(nn, encoder_activation_cls):
                raise ValueError(f"Invalid activation_cls: {encoder_activation_cls}")
            encoder_activation = getattr(nn, encoder_activation_cls)

            in_dim = self._x_size + self._y_size
            layer_sizes = [in_dim, *(encoder_num_units for _ in range(encoder_num_layers)), r_dim]
            mlp = list()
            for i, o in zip(layer_sizes[:-1], layer_sizes[1:]): 
                mlp.append(nn.Linear(i, o))
                mlp.append(encoder_activation())
            self._mlp = nn.Sequential(*mlp)

            self._encoder = Encoder(apply_attention=False, mlp=self._mlp)
        elif np_type == 'NP': 
            pass
        elif np_type == 'ANP':
            pass  
