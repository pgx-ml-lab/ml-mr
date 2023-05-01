"""
Implementation of Deep Feature Instrumental Variable regression.

Based on Xu L, et al. (2020):

https://arxiv.org/abs/2010.07154


This code is adapted from the author's implementation available at:

https://github.com/liyuan9988/DeepFeatureIV

Test code:

from ml_mr.estimation.core import IVDataset
from ml_mr.estimation.dfiv import DFIVModel
from torch.utils.data import DataLoader
import pandas as pd
import pytorch_lightning as pl

df = pd.read_csv("/Users/legaultm/projects/ml-mr/simulation_models/tian_biorxiv_2022/simulated_datasets/tian-scenario-A2_sim_data.csv.gz")
dataset = IVDataset.from_dataframe(df, "X", "Y", ["Z"])
# dataset = IVDataset.from_dataframe(df, "X", "Y", ["Z"], ["U"])

dl = DataLoader(dataset, num_workers=0, batch_size=10_000, shuffle=True)

model = DFIVModel(
    n_instruments=1,
    n_exposures=1,
    n_outcomes=1,
    n_covariates=0,
    n_instrument_features=2,
    n_exposure_features=2,
    n_covariate_features=1,
    instrument_net_hidden=[64, 32],
    exposure_net_hidden=[64, 32],
    covariate_net_hidden=[64, 32],
    ridge_lambda1=2,
    ridge_lambda2=2,
    n_updates_stage1=1,
    n_updates_stage2=20,
    n_updates_covariate_net=5,
    instrument_net_learning_rate=1e-2,
    exposure_net_learning_rate=1e-2,
    covariate_net_learning_rate=1e-2,
)

trainer = pl.Trainer(
    log_every_n_steps=1,
    max_epochs=200,
)

trainer.fit(model, dl)



"""

from typing import Dict, Optional, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ..utils.linear import ridge_fit_predict, ridge_regression
from ..utils.nn import build_mlp


DEFAULTS = {
    "ridge_lambda_1": 0.1,
    "ridge_lambda_2": 0.1,
    "n_instrument_features": 5,
    "n_exposure_features": 5,
    "n_covariate_features": 5,
    "n_updates_stage1": 20,
    "n_updates_covariate_net": 1,
    "n_updates_stage2": 1,
    "instrument_net_hidden": [128, 64],
    "exposure_net_hidden": [128, 64],
    "covariate_net_hidden": [128, 64],
    "instrument_net_learning_rate": 1e-3,
    "exposure_net_learning_rate": 1e-3,
    "covariate_net_learning_rate": 1e-3,
}


def outer_prod(mat1: torch.Tensor, mat2: torch.Tensor):
    """
    Parameters
    ----------
    mat1: torch.Tensor[nBatch, mat1_dim1, mat1_dim2, mat1_dim3, ...]
    mat2: torch.Tensor[nBatch, mat2_dim1, mat2_dim2, mat2_dim3, ...]
    Returns
    -------
    res : torch.Tensor[nBatch, mat1_dim1, ..., mat2_dim1, ...]
    """

    mat1_shape = tuple(mat1.size())
    mat2_shape = tuple(mat2.size())
    assert mat1_shape[0] == mat2_shape[0]
    nData = mat1_shape[0]
    aug_mat1_shape = mat1_shape + (1,) * (len(mat2_shape) - 1)
    aug_mat1 = torch.reshape(mat1, aug_mat1_shape)
    aug_mat2_shape = (nData,) + (1,) * (len(mat1_shape) - 1) + mat2_shape[1:]
    aug_mat2 = torch.reshape(mat2, aug_mat2_shape)
    return aug_mat1 * aug_mat2


def augment_with_covar_feats(
    feats: torch.Tensor,
    covar_feats: torch.Tensor
) -> torch.Tensor:
    covar_feats = add_intercept(covar_feats)
    feats = outer_prod(feats, covar_feats)
    return torch.flatten(feats, start_dim=1)


def add_intercept(x: torch.Tensor) -> torch.Tensor:
    return torch.hstack((
        x,
        torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype)
    ))


def dfiv_2sls(
    z_feats: torch.Tensor,
    x_feats: torch.Tensor,
    covar_feats: Optional[torch.Tensor],
    outcome: torch.Tensor,
    lam1: float,
    lam2: float
) -> Dict[str, torch.Tensor]:
    z_feats = add_intercept(z_feats)
    x_feats = add_intercept(x_feats)

    # Stage 1
    betas1, x_feats_pred = ridge_fit_predict(
        z_feats, x_feats, lam1
    )

    if covar_feats is not None:
        x_feats_pred = augment_with_covar_feats(x_feats_pred, covar_feats)

    # Stage 2
    betas2, y_hat = ridge_fit_predict(
        x_feats_pred, outcome, lam2
    )

    loss = (
        F.mse_loss(y_hat, outcome) +
        lam2 * torch.norm(betas2) ** 2
    )

    return {
        "betas1": betas1,
        "betas2": betas2,
        "loss": loss
    }


class DFIVModel(pl.LightningModule):
    def __init__(
        self,
        n_instruments: int,
        n_exposures: int,
        n_outcomes: int,
        n_covariates: int,  # zero if none
        n_instrument_features: int,
        n_exposure_features: int,
        n_covariate_features: Optional[int],
        instrument_net_hidden: Iterable[int],
        exposure_net_hidden: Iterable[int],
        covariate_net_hidden: Optional[Iterable[int]],
        ridge_lambda1: float,
        ridge_lambda2: float,
        n_updates_stage1: int,
        n_updates_covariate_net: int,
        n_updates_stage2: int,
        instrument_net_learning_rate: float,
        exposure_net_learning_rate: float,
        covariate_net_learning_rate: Optional[float],
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        # Instrument feature learner.
        self.z_net = nn.Sequential(*build_mlp(
            n_instruments, instrument_net_hidden, n_instrument_features
        ))

        # Exposure feature learner.
        self.x_net = nn.Sequential(*build_mlp(
            n_exposures, exposure_net_hidden, n_exposure_features
        ))

        # Covariate feature learner if needed.
        if n_covariates > 0:
            self.c_net = nn.Sequential(*build_mlp(
                n_covariates, covariate_net_hidden, n_covariate_features
            ))
        else:
            self.c_net = None

        # Variables to hold the final linear predictor coefficients.
        self.betas2: Optional[torch.Tensor] = None

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(
                self.z_net.parameters(),
                lr=self.hparams.instrument_net_learning_rate,
            ),
            torch.optim.Adam(
                self.x_net.parameters(),
                lr=self.hparams.exposure_net_learning_rate,
            ),
        ]

        if self.hparams.n_covariates > 0:
            optimizers.append(
                torch.optim.Adam(
                    self.c_net.parameters(),
                    lr=self.hparams.covariate_net_learning_rate,
                )
            )

        return optimizers

    def covariate_net_update(
        self,
        batch: torch.Tensor,
        n_updates: int,
        opt: torch.optim.Optimizer,
    ):
        x, y, ivs, covars = batch

        assert self.c_net is not None

        self.z_net.train(False)
        self.x_net.train(False)
        self.c_net.train(True)

        z_feats = add_intercept(self.z_net(ivs).detach())
        x_feats = add_intercept(self.x_net(x).detach())

        stage1_weight, x_feats_hat = ridge_fit_predict(
            x_feats,
            z_feats,
            self.hparams.ridge_lambda1
        )

        losses = torch.empty(n_updates, device=self.device)
        for i in range(n_updates):
            opt.zero_grad()
            covariate_feats = self.c_net(covars)
            combined_feats = augment_with_covar_feats(
                x_feats_hat, covariate_feats
            )

            # Stage 2 regression.
            betas2, y_hat = ridge_fit_predict(
                combined_feats, y, self.hparams.ridge_lambda2
            )

            loss = (
                F.mse_loss(y, y_hat) +
                self.hparams.ridge_lambda2 * torch.norm(betas2) ** 2
            )
            losses[i] = loss

            self.manual_backward(loss)
            opt.step()

        return torch.mean(losses)

    def stage1_update(
        self,
        batch: torch.Tensor,
        n_updates: int,
        opt_iv: torch.optim.Optimizer,
    ):
        x, _, ivs, _ = batch

        self.z_net.train(True)
        self.x_net.train(False)
        if self.c_net is not None:
            self.c_net.train(False)

        # "True" treatment features.
        x_feat = self.x_net(x).detach()

        losses = torch.empty(n_updates, device=self.device)

        for i in range(n_updates):
            opt_iv.zero_grad()

            iv_feat = add_intercept(self.z_net(ivs))
            betas1, x_feat_hat = ridge_fit_predict(
                iv_feat, x_feat, self.hparams.ridge_lambda1
            )

            loss = (
                F.mse_loss(x_feat_hat, x_feat) +
                self.hparams.ridge_lambda1 * torch.norm(betas1) ** 2
            )

            self.manual_backward(loss)
            opt_iv.step()

            losses[i] = loss

        return torch.mean(losses)

    def stage2_update(
        self,
        batch: torch.Tensor,
        n_updates: int,
        opt_exposure: torch.optim.Optimizer,
    ):
        x, y, ivs, covars = batch

        self.z_net.train(False)
        self.x_net.train(True)

        z_feats = self.z_net(ivs).detach()

        if self.c_net is not None:
            self.c_net.train(False)
            c_feats = self.c_net(covars).detach()
        else:
            c_feats = None

        losses = torch.empty(n_updates, device=self.device)
        for i in range(n_updates):
            opt_exposure.zero_grad()
            x_feats = self.x_net(x)
            results = dfiv_2sls(
                z_feats,
                x_feats,
                c_feats,
                y,
                self.hparams.ridge_lambda1,
                self.hparams.ridge_lambda2
            )
            loss = results["loss"]
            losses[i] = loss

            self.manual_backward(loss)
            opt_exposure.step()

        return torch.mean(losses)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        stage1_loss = self.stage1_update(
            batch,
            self.hparams.n_updates_stage1,
            opt[0],  # Instrument net optimizer.
        )

        if self.c_net is not None:
            covariate_loss = self.covariate_net_update(
                batch,
                self.hparams.n_updates_covariate_net,
                opt[2],  # Covariate net optimizer.
            )
            self.log("covariate_loss", covariate_loss)

        stage2_loss = self.stage2_update(
            batch,
            self.hparams.n_updates_stage2,
            opt[1],  # Exposure net optimizer.
        )

        self.log("stage1_loss", stage1_loss)
        self.log("stage2_loss", stage2_loss)
