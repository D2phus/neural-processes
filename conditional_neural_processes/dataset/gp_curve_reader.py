import torch
from torch.distributions.multivariate_normal import MultivariateNormal

import collections
import random
import copy


CNPRegressionDescription = collections.namedtuple(
    'CNPRegressionDescription',
    ('query',
     'target_y',
     'num_total_points',
     'num_context_points')
)


class GPCurvesReader:
    def __init__(self,
                 batch_size: int,
                 max_num_context: int,
                 x_size: int = 1,
                 y_size: int = 1,
                 length: float = 0.4,
                 sigma: float = 1.0,
                 testing: bool = False):
        """
        Initializes a GPCurveReader object.

        Args:
            batch_size (int): The batch size.
            max_num_context (int): The maximum number of context points.
            x_size (int, optional): The size of the input x. Defaults to 1.
            y_size (int, optional): The size of the output y. Defaults to 1.
            length (float, optional): The length scale of the Gaussian process kernel. Defaults to 0.4.
            sigma (float, optional): The standard deviation of the Gaussian process noise. Defaults to 1.0.
            testing (bool, optional): Whether the reader is used for testing. Defaults to False.
        """
        self._batch_size = batch_size
        self.max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._length = length
        self._sigma = sigma
        self._testing = testing

    def _gaussian_kernel(self, X: torch.tensor, length: torch.tensor, sigma: float, sigma_noise: float = 1e-2) -> torch.tensor:
        """Applies the Gaussian kernel to generate curve data.
        $k(X, X') = \sigma^2 \exp (
        \frac{-(X-X')^2}{2l}
        ) + \sigma^2_{noise}$

        Args: 
            X of shape (batch_size, num_total_points, x_size): a batch of sampled curve, which consists of multiple context and target data points.
            length of shape (batch_size, y_size, x_size): the scale parameter of the Gaussian kernel. 
            sigma of shape (batch_size, y_size): the magnitude of the std.
            sigma_noise: std of the noise that we add for stability.

        Returns: 
            The kernel of shape (batch_size, y_size, num_total_points, num_total_points)
        """
        num_total_points = X.shape[1]
        # Expand and take the difference
        # (batch_size, 1, num_total_points, x_size)
        X1 = copy.deepcopy(X)[:, None, :, :]
        # (batch_size, num_total_points, 1, x_size)
        X2 = copy.deepcopy(X)[:, :, None, :]
        # (batch_size, num_total_points, num_total_points, x_size)
        diff = X1 - X2

        # (batch_size, y_size, num_total_points, num_total_points, x_size)
        norm = torch.square(diff[:, None, :, :, :] /
                            length[:, :, None, None, :])

        # NOTE: Frobenius norm, (square root of) the sum of the absolute squares of its elements.
        # (batch_size, y_size, num_total_points, num_total_points)
        norm = norm.sum(dim=-1)

        kernel = torch.exp(-0.5 * norm) * torch.square(sigma[:, :, None, None])
        # add noise to make the matrix positive definite for cholesky decomposition
        kernel += (sigma_noise**2) * torch.eye(num_total_points)

        return kernel

    def generate_curves(self):
        """Generates a batch of curves and the corresponding context and target points between (-2, 2). 

        Returns: 
            CNPRegressionDescription: a `CNPRegressionDescription` object."""
        # set kernel parameters
        length = (
            torch.ones(self._batch_size, self._y_size, self._x_size) * self._length)
        sigma = (
            torch.ones(self._batch_size, self._y_size) * self._sigma)

        # randomly select from U[3, max_num_context]
        num_context = random.randint(3, self.max_num_context)
        # When testing, we want more targets that distribute evenly, in order to plot the function.
        if self._testing:
            num_target = 400
            num_total_points = num_target
            # (batch_size, num_total_points)
            x_values = torch.arange(-2., 2., 1./100, dtype=torch.float32).unsqueeze(dim=0).repeat([self._batch_size, 1])
            # (batch_size, num_total_points, x_size=1)
            x_values = x_values.unsqueeze(dim=-1)
        # When training, the number and position of context and target points are randomly selected.
        else:
            # randomly select from U[2, max_num_context]
            num_target = random.randint(2, self.max_num_context)
            num_total_points = num_context + num_target

            # between -2 and 2
            x_values = 4 * torch.rand(self._batch_size,
                                      num_total_points, self._x_size) - 2

        # pass the x_values through the Gaussian kernel
        kernel = self._gaussian_kernel(x_values, length, sigma)

        # when sampling a curve, transformation from `x \sim N(0, 1)` to `N(0, \Sigma)`, where `Sigma` is the kernel, is applied: `Lx, \Sigma = LL^T`
        # calculate Cholesky, using double precision for better stability.
        # (batch_size, y_size, num_total_points, num_total_points)
        cholesky = torch.linalg.cholesky(kernel.double()).float()
        y_values = torch.matmul(
            cholesky,
            torch.randn(self._batch_size, self._y_size, num_total_points, 1)
        )  # (batch_size, y_size, num_total_points, 1), column vector.
        # (batch_size, num_total_points, y_size)
        y_values = torch.transpose(torch.squeeze(y_values, -1), 1, 2)

        # select the targets
        # NOTE: all data (target and observations) are taken into consideration
        target_x = x_values
        target_y = y_values

        # select the observations
        if self._testing:  # randomly chosen observations
            idx = torch.randperm(num_target)[:num_context]
            context_x = x_values[:, idx, :]
            context_y = y_values[:, idx, :]
        else:  # already random
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        query = ((context_x, context_y), target_x)

        return CNPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=num_total_points,
            num_context_points=num_context)
