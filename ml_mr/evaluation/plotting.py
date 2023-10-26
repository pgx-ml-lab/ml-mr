from typing import Callable, Tuple, Optional

import torch

from ..estimation.core import MREstimatorWithUncertainty, MREstimator

try:
    import matplotlib
    import matplotlib.pyplot as plt
    MPL_AVAIL = True
except ImportError:
    MPL_AVAIL = False


def plot_iv_reg(
    estimator: MREstimator,
    true_function: Callable[[torch.Tensor], torch.Tensor],
    domain: Tuple[float, float] = (-3, 3),
    covars: Optional[torch.Tensor] = None,
    label: str = "Predicted Y",
    plot_structural: bool = True,
    n_points: int = 2000,
    alpha: float = 0.1,
    ax: Optional["matplotlib.axes.Axes"] = None,
    multi_run: bool = False
) -> dict:

    if not MPL_AVAIL:
        raise ImportError(
            "Install matplotlib to enable plotting functionality."
        )

    xs = torch.linspace(domain[0], domain[1], n_points).reshape(-1, 1)

    uncertainty = False
    if isinstance(estimator, MREstimatorWithUncertainty):
        uncertainty = True
        y_hat_ci = estimator.avg_iv_reg_function(
            xs, covars, alpha=alpha
        )
        y_hat_l = y_hat_ci[:, :, 0]
        y_hat = y_hat_ci[:, :, 1]
        y_hat_u = y_hat_ci[:, :, 2]
    else:
        y_hat = estimator.avg_iv_reg_function(xs, covars)

    true_y = true_function(xs)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))

    if plot_structural:
        ax.plot(
            xs.numpy(),
            true_y.numpy().reshape(-1),
            ls="--",
            color="#9C0D00",
            lw=2,
            label="True Y",
            zorder=2
        )

    lines, = ax.plot(
        xs.numpy().flatten(),
        y_hat.numpy().reshape(-1),
        label=label
    )
    if uncertainty:
        if multi_run:
            color = "#BDBDBD"
            color_alpha = 0.3
        else:
            color = "#F7F7F7"
            color_alpha = 1

        ax.fill_between(
            xs.numpy().flatten(),
            y_hat_l.numpy().reshape(-1),
            y_hat_u.numpy().reshape(-1),
            zorder=-1,
            color=color,
            alpha=color_alpha
        )

    return {"ax": ax, "lines": lines}
