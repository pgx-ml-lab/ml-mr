from typing import Callable, Tuple

import torch
import torch.nn.functional as F

from ..estimation import MREstimator


def mse(
    estimator: MREstimator,
    true_function: Callable[[torch.Tensor], torch.Tensor],
    domain: Tuple[float, float] = (-3, 3),
    n_points: int = 5000
) -> float:
    xs = torch.linspace(domain[0], domain[1], n_points)
    y_hat = estimator.effect(xs)
    true_y = true_function(xs)

    return F.mse_loss(y_hat, true_y).item()
