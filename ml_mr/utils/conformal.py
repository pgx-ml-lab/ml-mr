"""
Utilities for conformal prediction.
"""

from typing import Optional, Union, Literal
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import numpy as np
from ..estimation.core import IVDataset, FullBatchDataLoader
from .nn import MLP, OutcomeMLPBase


NONCONFORMITY_MEASURES = {
    "sqr",
    "residual-aux-nn",
    "gaussian-nn"
}
NONCONFORMITY_MEASURES_TYPE = Literal["sqr", "residual-aux-nn", "gaussian-nn"]


def estimate_q_hat(
    scores: torch.Tensor,
    alpha: float = 0.1
) -> float:
    n = scores.size(0)
    if scores.ndim == 2:
        scores = scores.reshape(-1)
    elif scores.ndim > 2:
        raise ValueError("Can't interpret tensor as 1d vector.")

    q_hat = torch.quantile(
        scores,
        math.ceil((n+1)*(1-alpha)) / n,
        interpolation="higher"
    )

    return q_hat.item()


@torch.no_grad()
def nonconformity_sqr(
    model: pl.LightningModule,
    dataset: IVDataset,
    alpha: float = 0.1
) -> torch.Tensor:
    dl = FullBatchDataLoader(dataset)
    _, y, ivs, covars = dl

    # We assume the provided model is trained with quantile regression and
    # takes taus as a input.
    y_hat_l = model.forward(
        ivs, covars,
        taus=torch.full_like(y, alpha)
    )

    y_hat_u = model.forward(
        ivs, covars,
        taus=torch.full_like(y, 1 - alpha)
    )

    return torch.maximum(y - y_hat_u, y_hat_l - y)


class OutcomeResidualPrediction(MLP):
    def __init__(
        self,
        input_size,
        wrapped_model: OutcomeMLPBase,
        hidden=[128, 64],
        alpha: float = 0.1,
        lr: float = 5e-3,
        weight_decay: float = 0,
        q_hat: Optional[Union[float, torch.Tensor]] = None
    ):
        super().__init__(
            input_size,
            hidden,
            out=1,
            lr=lr,
            weight_decay=weight_decay,
            _save_hyperparams=False
        )

        self.alpha = alpha
        self.wrapped_model = wrapped_model
        self.q_hat = q_hat
        self.save_hyperparameters(ignore=["wrapped_model", "q_hat"])

    def x_to_y(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get prediction and resid.
        with torch.no_grad():
            y_hat = self.wrapped_model.x_to_y(x, covars)
            pred_resid = self.forward(x, covars)

        return torch.hstack([
            y_hat - pred_resid * self.q_hat,
            y_hat,
            y_hat + pred_resid * self.q_hat
        ])

    def set_q_hat_from_data(self, dataset):
        """Set the conformal prediction multiplier.

        It is important to use full batches for the validation dataset,
        otherwise this won't set the correct q_hat and the x_to_y will
        not necessarily return calibrated predictions.

        """
        dl = DataLoader(dataset, batch_size=len(dataset))
        x, y, _, covars = next(iter(dl))

        n = x.size(0)
        alpha = self.hparams.alpha  # type: ignore

        with torch.no_grad():
            y_hat = self.wrapped_model.x_to_y(x, covars)
            pred_resid = self.forward(x, covars)

        actual_abs_resid = torch.abs(y - y_hat)

        scores = actual_abs_resid / pred_resid
        q_hat = torch.quantile(scores, np.ceil((n+1)*(1-alpha))/n)

        self.q_hat = q_hat

    def forward(self, x, covars):
        xs = [x]
        if covars is not None:
            xs.append(covars)

        out = super().forward(torch.hstack(xs))
        # Apply softplus to make output positive.
        return F.softplus(out)

    def _step(
        self, batch, batch_index, log_prefix: str = "train"
    ) -> torch.Tensor:
        x, y, _, covars = batch
        with torch.no_grad():
            y_hat = self.wrapped_model.x_to_y(x, covars=covars)

        residual = torch.abs(y_hat - y)
        residual_hat = self.forward(x, covars)

        mse = F.mse_loss(residual_hat, residual)

        self.log(f"{log_prefix}_resid_pred_loss", mse)
        return mse
