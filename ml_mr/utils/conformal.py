"""
Utilities for conformal prediction.
"""

from typing import Optional
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import numpy as np
from ..estimation.core import IVDataset
from .nn import MLP, OutcomeMLPBase
from ..logging import info, warn


@torch.no_grad()
def get_conformal_adjustment_sqr(
    model: pl.LightningModule,
    dataset: IVDataset,
    alpha: float = 0.1
) -> float:
    n = len(dataset)
    assert isinstance(n, int)

    dl = DataLoader(dataset, batch_size=n)
    _, y, ivs, covars = next(iter(dl))

    # We assume the provided model is trained with quantile regression and
    # takes taus as a input.
    y_hat_l = model.forward(
        ivs, covars,
        taus=torch.full_like(y, alpha / 2)
    )

    y_hat_u = model.forward(
        ivs, covars,
        taus=torch.full_like(y, 1 - alpha / 2)
    )

    scores = torch.maximum(y - y_hat_u, y_hat_l - y)
    q_hat = torch.quantile(
        scores,
        math.ceil((n+1)*(1-alpha)) / n,
        interpolation="higher"
    )

    return q_hat.item()


class OutcomeResidualPrediction(MLP):
    def __init__(
        self,
        input_size,
        wrapped_model: OutcomeMLPBase,
        hidden=[32, 32],
        alpha: float = 0.1,
        q_hat: float = 1,
        lr: float = 1e-3,
        weight_decay: float = 0,
    ):
        super().__init__(
            input_size,
            hidden,
            out=1,
            lr=lr,
            weight_decay=weight_decay,
            _save_hyperparams=False
        )

        self.register_buffer("q_hat", torch.tensor(q_hat))
        self.wrapped_model = wrapped_model
        self.save_hyperparameters(ignore=["wrapped_model"])

    def x_to_y(self, x: torch.Tensor, covars: Optional[torch.Tensor] = None):
        if self.q_hat == 1:
            warn("Observed a conformal adjustment of 1 which suggests that "
                 "it was not set. Did you use conformal_calibration?")

        # Get prediction and resid.
        with torch.no_grad():
            y_hat = self.wrapped_model.forward(x, covars)
            pred_resid = self.forward(x, covars)

        return torch.hstack([
            y_hat - pred_resid * self.q_hat,
            y_hat,
            y_hat + pred_resid * self.q_hat
        ])

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        """Set the conformal prediction multiplier.

        It is important to use full batches for the validation dataset,
        otherwise this won't set the correct q_hat and the x_to_y will
        not necessarily return calibrated predictions.

        """
        x, y, _, covars = batch

        n = x.size(0)
        alpha = self.hparams.alpha  # type: ignore

        with torch.no_grad():
            y_hat = self.wrapped_model.forward(x, covars)
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
            y_hat = self.wrapped_model.forward(x, covars=covars)

        residual = torch.abs(y_hat - y)
        residual_hat = self.forward(x, covars)

        mse = F.mse_loss(residual_hat, residual)

        self.log(f"{log_prefix}_resid_pred_loss", mse)
        return mse
