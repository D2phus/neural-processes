import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from tqdm import tqdm

import random
import collections

from .utils import plot_functions


def train(cnp, 
          data_train, 
          data_test, 
          num_epochs: int, 
          lr: float, 
          decay_rate: float, 
          log_frequency: int=1000, 
          ): 
    """train CNP.

    predicts **the whole dataset** conditioned on a randomly chosen subset. 

    Args: 
        cnp: Conditional Neural Processes model
        data_train: generate `batch_size` curves of random number of observations and targets from a given GP process. 
        data_test: linearly distributed test data.
        num_epochs: the number of training epochs
        lr: learning rate
        decay_rate: weight decay rate
    """
    optimizer = torch.optim.Adam(cnp.parameters(), lr=lr) 
    
    for epoch in tqdm(range(num_epochs), desc=f'Training'): 
        train_desc = data_train.generate_curves()
        log_prob, _, _ = cnp(train_desc.query, num_total_points=train_desc.num_total_points, num_contexts=train_desc.num_context_points, target_y=train_desc.target_y)
        loss = -log_prob.mean()
        optimizer.step()
        
        if epoch % log_frequency == 0: 
            test_desc = data_test.generate_curves()
            (context_x_test, context_y_test), target_x_test = test_desc.query
            target_y_test = test_desc.target_y
            
            _, mu_test, sigma_test = cnp(test_desc.query, num_total_points=test_desc.num_total_points, num_contexts=test_desc.num_context_points)
            plot_functions(context_x_test, context_y_test, target_x_test, target_y_test, mu_test.detach(), sigma_test.detach())
    
    return cnp