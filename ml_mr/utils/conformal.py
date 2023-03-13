"""
Utilities for conformal prediction.
"""

import math

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from ..estimation.core import IVDataset


@torch.no_grad()
def get_conformal_adjustment(
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
