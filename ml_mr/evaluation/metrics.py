from typing import Callable, Tuple, Optional

import torch
import torch.nn.functional as F

from ..estimation import MREstimator, MREstimatorWithUncertainty


def mse(
    estimator: MREstimator,
    true_function: Callable[[torch.Tensor], torch.Tensor],
    domain: Tuple[float, float] = (-3, 3),
    covars: Optional[torch.Tensor] = None,
    n_points: int = 5000
) -> float:
    xs = torch.linspace(domain[0], domain[1], n_points).reshape(-1, 1)
    y_hat = estimator.effect(xs, covars).reshape(-1, 1)
    true_y = true_function(xs)

    return F.mse_loss(y_hat, true_y.reshape(-1, 1)).item()


def mean_prediction_interval_absolute_width(
    estimator: MREstimatorWithUncertainty,
    domain: Tuple[float, float] = (-3, 3),
    covars: Optional[torch.Tensor] = None,
    alpha: float = 0.1,
    n_points: int = 5000,
) -> float:
    xs = torch.linspace(domain[0], domain[1], n_points).reshape(-1, 1)
    y_hat = estimator.effect_with_prediction_interval(
        xs, covars=covars, alpha=alpha
    )

    y_low = y_hat[:, 0]
    y_high = y_hat[:, 2]
    return torch.mean(torch.abs(y_low - y_high)).item()
