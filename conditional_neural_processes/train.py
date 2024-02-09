import torch
from tqdm import tqdm
from .utils import plot_functions
import logging

log = logging.getLogger(__name__)


def train(config, 
          model, 
          data_train,
          data_test,
          ):
    """Trains the Conditional Neural Process model.

    predicts **the whole dataset** conditioned on a randomly chosen subset. 

    Args: 
        model: Conditional Neural Processes model
        data_train: `GPCurvesReader` object to generate `batch_size` curves of random number of observations and targets from a given GP process.
        data_test: `GPCurvesReader` object to generate linearly distributed test data.
        num_epochs: number of epochs to train
        lr: learning rate
        decay_rate: weight decay
    """
    lr = config.training.lr
    decay_rate = config.training.decay_rate
    num_epochs = config.training.num_epochs
    log_frequency = config.training.log_frequency

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=decay_rate)

    for epoch in tqdm(range(1, num_epochs+1), desc=f'Training for {num_epochs} epochs'):
        train_desc = data_train.generate_curves() # generate a batch of curves
        optimizer.zero_grad()  # NOTE: zero the gradients, otherwise they will accumulate
        log_prob, _, _ = model(train_desc.query, num_total_points=train_desc.num_total_points,
                             num_contexts=train_desc.num_context_points, target_y=train_desc.target_y)
        loss = -log_prob.mean(dim=0).sum() # take mean over batch and sum over num_target and dimension of y
        loss.backward()
        optimizer.step()

        if epoch % log_frequency == 0:
            print(f'Loss: {loss.item()}')
            log.info(f'Loss: {loss.item()}')
            test_desc = data_test.generate_curves()
            (context_x_test, context_y_test), target_x_test = test_desc.query
            target_y_test = test_desc.target_y

            _, mu_test, sigma_test = model(
                test_desc.query, num_total_points=test_desc.num_total_points, num_contexts=test_desc.num_context_points)
            plot_functions(context_x_test, context_y_test, target_x_test,
                           target_y_test, mu_test.detach(), sigma_test.detach())

    return model
