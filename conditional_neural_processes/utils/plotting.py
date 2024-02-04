import matplotlib.pyplot as plt
import torch
import numpy as np

from typing import Tuple

def plot_functions(context_x: torch.tensor, context_y: torch.tensor, target_x: torch.tensor, target_y: torch.tensor, pred_y: torch.tensor, sigma: torch.tensor):
    """Plot the first batch. 
    Args: 
        `context_x` of shape (batch_size, num_points, x_size)
        `context_y` of shape (batch_size, num_points, y_size)
        `target_x` of shape (batch_size, num_points, x_size)
        `target_y` of shape (batch_size, num_points, y_size)
        `pred_y` of shape (batch_size, num_points, y_size)
        `sigma` of shape (batch_size, num_points, y_size)
    """
    plt.plot(target_x[0], pred_y[0], 'b', linewidth=2) # blue
    plt.plot(target_x[0], target_y[0], 'k:', linewidth=2) # black, dotted
    plt.plot(context_x[0], context_y[0], 'ko', markersize=10) # black, circle
    plt.fill_between(
      target_x[0, :, 0],
      pred_y[0, :, 0] - sigma[0, :, 0],
      pred_y[0, :, 0] + sigma[0, :, 0],
      alpha=0.2,
      facecolor='#65c9f7',
      interpolate=True)

    # Make the plot pretty
    # plt.yticks([-2, 0, 2], fontsize=16)
    # plt.xticks([-2, 0, 2], fontsize=16)
    # plt.ylim([-2, 2])
    plt.grid('off')
    ax = plt.gca()
    # ax.set_axis_bgcolor('white')
    plt.show()
     
    
    