from typing import Callable, Tuple, Optional, Iterable, List, Literal, Union

import torch
import torch.nn.functional as F

from ..estimation import MREstimator, MREstimatorWithUncertainty
from ..utils import _cat
from ..logging import warn


def get_iv_reg_xy(
    estimator: MREstimator,
    domain: Tuple[float, float],
    covars: Optional[torch.Tensor] = None,
    low_memory: bool = False,
    n_points: int = 2000
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = torch.linspace(domain[0], domain[1], n_points).reshape(-1, 1)
    y_hat = estimator.avg_iv_reg_function(
        xs, covars=covars, low_memory=low_memory
    )

    if isinstance(estimator, MREstimatorWithUncertainty):
        y_hat = y_hat[:, :, 1]

    return xs, y_hat


def compute_metrics(
    metrics: Union[Literal["all"], Iterable[str]],
    estimator: MREstimator,
    domain: Tuple[float, float],
    covars: Optional[torch.Tensor] = None,
    low_memory: bool = False,
    n_points: int = 2000,
    true_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
):
    ALL_METRICS = ["mse", "correlation", "linear_slope"]

    xs, preds = get_iv_reg_xy(estimator, domain, covars, low_memory, n_points)
    if true_function is not None:
        targets = true_function(xs)
    else:
        targets = None

    if metrics == "all":
        metrics = ALL_METRICS

    results: List[Optional[float]] = []
    for metric in metrics:
        # Some metrics may crash if the true function is not reported.
        # This is meant to be handled upstream.
        if metric == "mse":
            results.append(F.mse_loss(preds, targets).item())  # type: ignore
        elif metric == "correlation":
            results.append(
                torch.corrcoef(
                    torch.hstack((preds, targets)).T  # type: ignore
                )[0, 1].item()
            )
        elif metric == "linear_slope":
            lin_approx = torch.linalg.lstsq(
                _cat(torch.ones((xs.size(0), 1)), xs),
                preds
            ).solution[1].item()
            results.append(lin_approx)
        else:
            warn(f"Unknown metric '{metric}'. Known metrics: {ALL_METRICS}")
            results.append(None)

    return dict(zip(metrics, results))


def mse(
    estimator: MREstimator,
    true_function: Callable[[torch.Tensor], torch.Tensor],
    domain: Tuple[float, float],
    covars: Optional[torch.Tensor] = None,
    low_memory: bool = False,
    n_points: int = 2000
) -> float:
    """Deprecated in favor of mse_and_correlation

       This because avg_iv_reg_function is often computationally intensive,
       it makes more sense to just compute both correlation and MSE even
       if only one metric is required.
    """
    return compute_metrics(
        ["mse"],
        estimator, domain, covars, low_memory, n_points, true_function
    )["mse"]


def mean_coverage(
    estimator: MREstimatorWithUncertainty,
    true_function: Callable[[torch.Tensor], torch.Tensor],
    domain: Tuple[float, float],
    covars: Optional[torch.Tensor] = None,
    n_points: int = 2000
) -> float:
    assert isinstance(estimator, MREstimatorWithUncertainty)
    xs = torch.linspace(domain[0], domain[1], n_points).reshape(-1, 1)
    pred = estimator.avg_iv_reg_function(xs, covars)

    true_y = true_function(xs)
    coverage = torch.mean(
        ((pred[:, 0, 0] <= true_y) &
         (true_y <= pred[:, 0, 2])).to(torch.float32)
    ).item()

    return coverage


def mean_prediction_interval_absolute_width(
    estimator: MREstimatorWithUncertainty,
    domain: Tuple[float, float],
    covars: Optional[torch.Tensor] = None,
    alpha: float = 0.1,
    n_points: int = 2000,
) -> float:
    xs = torch.linspace(domain[0], domain[1], n_points).reshape(-1, 1)

    y_hat = estimator.iv_reg_function(
        xs, covars=covars, alpha=alpha
    )

    y_low = y_hat[:, :, 0]
    y_high = y_hat[:, :, 2]
    return torch.mean(torch.abs(y_low - y_high)).item()
