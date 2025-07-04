"""
Implementation of Deep Feature Instrumental Variable regression.

Based on Xu L, et al. (2020):

https://arxiv.org/abs/2010.07154


This code is adapted from the author's implementation available at:

https://github.com/liyuan9988/DeepFeatureIV

TODO This implementation is not fully updated with the new covariates treatment
including saving the covariable_labels and exposing the meta dict to the
estimator.

Test code:

from ml_mr.estimation.core import IVDataset
from ml_mr.estimation.dfiv import fit_dfiv, DFIVEstimator
from torch.utils.data import DataLoader
import pandas as pd
import pytorch_lightning as pl
import numpy as np

n = 500_000
U = np.random.normal(size=n)
Z = np.random.normal(size=n)
X = 0.9*Z + 0.1*U + np.random.normal(scale=0.1, size=n)
print(np.percentile(X, [1, 99]))

y_x = lambda x: 0.8*x + 0.6*np.sin(5*x) + 0.05*x**3
Y = y_x(X) + -U + np.random.normal(scale=0.5, size=n)
df = pd.DataFrame(dict(X=X, Y=Y, Z=Z))

dataset = IVDataset.from_dataframe(df, "X", "Y", ["Z"])
# dataset = IVDataset.from_dataframe(df, "X", "Y", ["Z"], ["U"])

fit_dfiv(dataset, wandb_project="dfiv_tests")

estimator = DFIVEstimator.from_results("dfiv_estimate")

"""

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from ..log_utils import warn
from ..utils import default_validate_args
from ..utils.conformal import OutcomeResidualPrediction
from ..utils.data import IVDataset, IVDatasetWithGenotypes
from ..utils.linear import ridge_fit_predict
from ..utils.nn import build_mlp
from ..utils.training import train_model
from .core import MREstimator, MREstimatorWithUncertainty


DEFAULTS = {
    "output_dir": "dfiv_estimate",
    "validation_proportion": 0.2,
    "ridge_lambda1": 0.5,
    "ridge_lambda2": 0.9,
    "n_instrument_features": 2,
    "n_exposure_features": 2,
    "n_covariate_features": 2,
    "n_updates_stage1": 20,
    "n_updates_covariate_net": 1,
    "n_updates_stage2": 1,
    "instrument_net_hidden": [64, 32],
    "exposure_net_hidden": [64, 32],
    "covariate_net_hidden": [64, 32],
    "instrument_net_learning_rate": 0.01,
    "exposure_net_learning_rate": 0.005,
    "covariate_net_learning_rate": 0.01,
    "batch_size": 15_000,
    "max_epochs": 60,
    "accelerator": "gpu" if (
        torch.cuda.is_available() and torch.cuda.device_count() > 0
    ) else "cpu",
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
        z_feats, x_feats, lam1,
        device=z_feats.device
    )

    if covar_feats is not None:
        x_feats_pred = augment_with_covar_feats(x_feats_pred, covar_feats)

    # Stage 2
    betas2, y_hat = ridge_fit_predict(
        x_feats_pred, outcome, lam2,
        device=z_feats.device
    )

    mse = F.mse_loss(y_hat, outcome)
    loss = mse + lam2 * torch.norm(betas2) ** 2

    return {
        "betas1": betas1,
        "betas2": betas2,
        "mse": mse,
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
            n_instruments, instrument_net_hidden, n_instrument_features,
        ))

        # Exposure feature learner.
        self.x_net = nn.Sequential(*build_mlp(
            n_exposures, exposure_net_hidden, n_exposure_features,
        ))

        # Covariate feature learner if needed.
        if n_covariates > 0:
            assert covariate_net_hidden is not None
            self.c_net: Optional[nn.Sequential] = nn.Sequential(*build_mlp(
                n_covariates, covariate_net_hidden, n_covariate_features,
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
            self.hparams.ridge_lambda1,  # type: ignore
            device=self.device  # type: ignore
        )

        losses = torch.empty(n_updates, device=self.device)  # type: ignore
        for i in range(n_updates):
            opt.zero_grad()
            covariate_feats = self.c_net(covars)
            combined_feats = augment_with_covar_feats(
                x_feats_hat, covariate_feats
            )

            # Stage 2 regression.
            betas2, y_hat = ridge_fit_predict(
                combined_feats, y, self.hparams.ridge_lambda2,  # type: ignore
                device=self.device  # type: ignore
            )

            loss = (
                F.mse_loss(y, y_hat) +
                self.hparams.ridge_lambda2 * torch.norm(betas2) ** 2  # type: ignore # noqa: E501
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

        losses = torch.empty(n_updates, device=self.device)  # type: ignore

        for i in range(n_updates):
            opt_iv.zero_grad()

            iv_feat = add_intercept(self.z_net(ivs))
            betas1, x_feat_hat = ridge_fit_predict(
                iv_feat, x_feat, self.hparams.ridge_lambda1,  # type: ignore
                device=self.device  # type: ignore
            )

            loss = (
                F.mse_loss(x_feat_hat, x_feat) +
                self.hparams.ridge_lambda1 * torch.norm(betas1) ** 2  # type: ignore # noqa: E501
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

        losses = torch.empty(n_updates, device=self.device)  # type: ignore
        mses = torch.empty(n_updates, device=self.device)  # type: ignore
        for i in range(n_updates):
            opt_exposure.zero_grad()
            x_feats = self.x_net(x)
            results = dfiv_2sls(
                z_feats,
                x_feats,
                c_feats,
                y,
                self.hparams.ridge_lambda1,  # type: ignore
                self.hparams.ridge_lambda2  # type: ignore
            )
            loss = results["loss"]
            losses[i] = loss
            mses[i] = results["mse"]

            self.manual_backward(loss)
            opt_exposure.step()

        return torch.mean(losses), torch.mean(mses)

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

        stage2_loss, mse = self.stage2_update(
            batch,
            self.hparams.n_updates_stage2,
            opt[1],  # Exposure net optimizer.
        )

        self.log("stage1_loss", stage1_loss)
        self.log("stage2_loss", stage2_loss)
        self.log("mse", mse)

    def validation_step(self, batch, batch_idx):
        x, y, ivs, covars = batch
        with torch.no_grad():
            z_feats = self.z_net(ivs)
            x_feats = self.x_net(x)
            if self.c_net is not None:
                c_feats = self.c_net(covars)
            else:
                c_feats = None

        res = dfiv_2sls(z_feats, x_feats, c_feats, y,
                        self.hparams.ridge_lambda1,
                        self.hparams.ridge_lambda2)

        self.log("val_loss", res["loss"])


def get_betas(
    dataset: Dataset,
    batch_size: int,
    dfiv: DFIVModel,
    ridge_lambda1: float,
    ridge_lambda2: float
) -> Dict[str, torch.Tensor]:
    dl = DataLoader(dataset, batch_size=batch_size)
    _z_feats = []
    _x_feats = []
    _c_feats = []
    _ys = []
    for batch in iter(dl):
        x, y, ivs, covars = batch
        _z_feats.append(dfiv.z_net(ivs).detach())
        _x_feats.append(dfiv.x_net(x).detach())
        if covars.numel() > 0:
            assert dfiv.c_net is not None
            _c_feats.append(dfiv.c_net(covars).detach())

        _ys.append(y)

    z_feats = torch.vstack(_z_feats)
    x_feats = torch.vstack(_x_feats)
    if _c_feats:
        c_feats: Optional[torch.Tensor] = torch.vstack(_c_feats)
    else:
        c_feats = None
    ys = torch.vstack(_ys)

    return dfiv_2sls(z_feats, x_feats, c_feats, ys,
                     ridge_lambda1, ridge_lambda2)


def main(args: argparse.Namespace) -> None:
    default_validate_args(args)

    dataset = IVDatasetWithGenotypes.from_argparse_namespace(args)

    # Automatically add the model hyperparameters.
    kwargs = {k: v for k, v in vars(args).items() if k in DEFAULTS.keys()}

    fit_dfiv(
        dataset=dataset,
        wandb_project=args.wandb_project,
        **kwargs,
    )


def configure_argparse(parser) -> None:
    IVDatasetWithGenotypes.add_dataset_arguments(parser)

    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"])

    parser.add_argument(
        "--validation-proportion",
        type=float,
        default=DEFAULTS["validation_proportion"],
    )

    parser.add_argument(
        "--n-instrument-features",
        type=int,
        default=DEFAULTS["n_instrument_features"],
    )

    parser.add_argument(
        "--n-exposure-features",
        type=int,
        default=DEFAULTS["n_exposure_features"],
    )

    parser.add_argument(
        "--n-covariate-features",
        type=int,
        default=DEFAULTS["n_covariate_features"],
    )

    parser.add_argument(
        "--instrument-net-hidden",
        nargs="*",
        default=DEFAULTS["instrument_net_hidden"]
    )

    parser.add_argument(
        "--exposure-net-hidden",
        nargs="*",
        default=DEFAULTS["exposure_net_hidden"]
    )

    parser.add_argument(
        "--covariate-net-hidden",
        nargs="*",
        default=DEFAULTS["covariate_net_hidden"]
    )

    parser.add_argument(
        "--ridge-lambda1",
        type=float,
        default=DEFAULTS["ridge_lambda1"]
    )

    parser.add_argument(
        "--ridge-lambda2",
        type=float,
        default=DEFAULTS["ridge_lambda2"]
    )

    parser.add_argument(
        "--n-updates-stage1",
        type=int,
        default=DEFAULTS["n_updates_stage1"]
    )

    parser.add_argument(
        "--n-updates-covariate_net",
        type=int,
        default=DEFAULTS["n_updates_covariate_net"]
    )

    parser.add_argument(
        "--n-updates-stage2",
        type=int,
        default=DEFAULTS["n_updates_stage2"]
    )

    parser.add_argument(
        "--instrument-net-learning-rate",
        type=float,
        default=DEFAULTS["instrument_net_learning_rate"]
    )

    parser.add_argument(
        "--exposure-net-learning-rate",
        type=float,
        default=DEFAULTS["exposure_net_learning_rate"]
    )

    parser.add_argument(
        "--covariate-net-learning-rate",
        type=float,
        default=DEFAULTS["covariate_net_learning_rate"]
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULTS["batch_size"]
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULTS["max_epochs"]
    )

    parser.add_argument(
        "--accelerator",
        type=str,
        default=DEFAULTS["accelerator"]
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None
    )


def fit_dfiv(
    dataset: IVDataset,  # type: ignore # noqa: E501
    output_dir: str = DEFAULTS["output_dir"],  # type: ignore
    validation_proportion: float = DEFAULTS["validation_proportion"],  # type: ignore # noqa: E501
    n_instrument_features: int = DEFAULTS["n_instrument_features"],  # type: ignore # noqa: E501
    n_exposure_features: int = DEFAULTS["n_exposure_features"],  # type: ignore # noqa: E501
    n_covariate_features: Optional[int] = DEFAULTS["n_covariate_features"],  # type: ignore # noqa: E501
    instrument_net_hidden: List[int] = DEFAULTS["instrument_net_hidden"],  # type: ignore # noqa: E501
    exposure_net_hidden: List[int] = DEFAULTS["exposure_net_hidden"],  # type: ignore # noqa: E501
    covariate_net_hidden: Optional[List[int]] = DEFAULTS["covariate_net_hidden"],  # type: ignore # noqa: E501
    ridge_lambda1: float = DEFAULTS["ridge_lambda1"],  # type: ignore
    ridge_lambda2: float = DEFAULTS["ridge_lambda2"],  # type: ignore
    n_updates_stage1: int = DEFAULTS["n_updates_stage1"],  # type: ignore # noqa: E501
    n_updates_covariate_net: int = DEFAULTS["n_updates_covariate_net"],  # type: ignore # noqa: E501
    n_updates_stage2: int = DEFAULTS["n_updates_stage2"],  # type: ignore # noqa: E501
    instrument_net_learning_rate: float = DEFAULTS["instrument_net_learning_rate"],  # type: ignore # noqa: E501
    exposure_net_learning_rate: float = DEFAULTS["exposure_net_learning_rate"],  # type: ignore # noqa: E501
    covariate_net_learning_rate: Optional[float] = DEFAULTS["covariate_net_learning_rate"],  # type: ignore # noqa: E501
    batch_size: int = DEFAULTS["batch_size"],   # type: ignore
    max_epochs: int = DEFAULTS["max_epochs"],   # type: ignore
    accelerator: str = DEFAULTS["accelerator"],  # type: ignore
    wandb_project: Optional[str] = None
):
    # Create output directory if needed.
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Metadata dictionary that will be saved alongside the results.
    meta = dict(locals())
    meta["model"] = "dfiv"
    meta.update(dataset.exposure_descriptive_statistics())
    meta["covariable_labels"] = dataset.covariable_labels
    del meta["dataset"]  # We don't serialize the dataset.

    covars = dataset.save_covariables(output_dir)

    # Split here into train and val.
    train_dataset, val_dataset = random_split(
        dataset, [1 - validation_proportion, validation_proportion]
    )

    model = DFIVModel(
        n_instruments=dataset.n_instruments(),
        n_exposures=dataset.n_exposures(),
        n_outcomes=dataset.n_outcomes(),
        n_covariates=dataset.n_covars(),
        n_instrument_features=n_instrument_features,
        n_exposure_features=n_exposure_features,
        n_covariate_features=n_covariate_features,
        instrument_net_hidden=instrument_net_hidden,
        exposure_net_hidden=exposure_net_hidden,
        covariate_net_hidden=covariate_net_hidden,
        ridge_lambda1=ridge_lambda1,
        ridge_lambda2=ridge_lambda2,
        n_updates_stage1=n_updates_stage1,
        n_updates_covariate_net=n_updates_covariate_net,
        n_updates_stage2=n_updates_stage2,
        instrument_net_learning_rate=instrument_net_learning_rate,
        exposure_net_learning_rate=exposure_net_learning_rate,
        covariate_net_learning_rate=covariate_net_learning_rate,
    )

    # Cleanup weights if needed.
    try:
        os.remove(
            os.path.join(output_dir, "linear_weights.pt")
        )
    except FileNotFoundError:
        pass

    try:
        stage2_val_loss = train_model(
            train_dataset,
            val_dataset,
            model,
            "val_loss",
            output_dir,
            "dfiv_model.ckpt",
            batch_size, max_epochs, accelerator, wandb_project
        )
        meta["stage2_val_loss"] = stage2_val_loss
    except RuntimeError:
        warn("Stopping at unsolvable configuration.")
        return

    # Load the best model.
    filename = os.path.join(output_dir, "dfiv_model.ckpt")
    model = DFIVModel.load_from_checkpoint(filename)

    model.eval()

    # Find the optimal coefficients for the linear weights.
    _2sls_results = get_betas(dataset, batch_size, model,
                              ridge_lambda1, ridge_lambda2)

    torch.save(
        {"betas1": _2sls_results["betas1"],
         "betas2": _2sls_results["betas2"]},
        os.path.join(output_dir, "linear_weights.pt")
    )

    wrapped_model, resid_pred_loss = train_conformal_predictor(
        train_dataset, val_dataset,
        model, _2sls_results["betas2"],
        alpha=0.1,
        output_dir=output_dir
    )

    meta["resid_pred_loss"] = resid_pred_loss

    conformal_net: OutcomeResidualPrediction = (
        OutcomeResidualPrediction.load_from_checkpoint(
            os.path.join(output_dir, "dfiv_calibration.ckpt"),
            wrapped_model=wrapped_model
        )
    )

    conformal_net.set_q_hat_from_data(val_dataset)
    assert isinstance(conformal_net.q_hat, torch.Tensor)
    meta["q_hat"] = conformal_net.q_hat.item()

    estimator = DFIVEstimator(
        conformal_net, _2sls_results["betas1"], _2sls_results["betas2"]
    )

    with open(os.path.join(output_dir, "meta.json"), "wt") as f:
        json.dump(meta, f)

    if wandb_project is not None:
        import wandb

        # TODO log artifact.
        wandb.finish()


# Stub that has a forward function in the format expected by the conformal
# regression utility.
class _ConformalStub:
    def __init__(self, dfiv_model, betas):
        self.dfiv_model = dfiv_model
        self.betas = betas

    def x_to_y(self, x, covars):
        return dfiv_x_to_y(self.dfiv_model, self.betas, x, covars)


def train_conformal_predictor(
    train_dataset: Dataset,
    val_dataset: Dataset,
    model: DFIVModel,
    betas: torch.Tensor,
    alpha: float,
    output_dir: str
):
    # We need to have a forward method that goes from x to y.
    n_exposures = train_dataset[0][0].numel()
    n_covars = train_dataset[0][3].numel()

    wrap = _ConformalStub(model, betas)
    resid_model = OutcomeResidualPrediction(
        n_exposures + n_covars,
        wrapped_model=wrap,  # type: ignore
        alpha=alpha
    )

    resid_pred_loss = train_model(
        train_dataset,
        val_dataset,
        resid_model,
        monitored_metric="val_resid_pred_loss",
        output_dir=output_dir,
        checkpoint_filename="dfiv_calibration.ckpt",
        batch_size=DEFAULTS["batch_size"],  # type: ignore
        max_epochs=100,
    )

    return wrap, resid_pred_loss


def dfiv_x_to_y(
    model: DFIVModel,
    betas2: torch.Tensor,
    x: torch.Tensor,
    covars: Optional[torch.Tensor]
) -> torch.Tensor:
    x_feats = add_intercept(model.x_net(x))

    if covars is not None and covars.numel() > 0:
        assert model.c_net is not None
        covars_feats = model.c_net(covars)
        x_feats = augment_with_covar_feats(x_feats, covars_feats)

    return x_feats @ betas2


class DFIVEstimator(MREstimator):
    def __init__(
        self,
        dfiv_model: Union[DFIVModel, OutcomeResidualPrediction],
        betas1: torch.Tensor,
        betas2: torch.Tensor,
    ):
        self.model = dfiv_model
        self.betas1 = betas1
        self.betas2 = betas2

        # No prediction interval.
        if isinstance(self.model, DFIVModel):
            def x_to_y(x, covars=None):
                return dfiv_x_to_y(dfiv_model, betas2, x, covars)
            self.x_to_y = x_to_y
            self.alpha = None

        # With predicted interval.
        elif isinstance(self.model, OutcomeResidualPrediction):
            self.x_to_y = self.model.x_to_y
            self.alpha = self.model.alpha

        else:
            raise ValueError()

    def effect(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        res = self.average_treatment_effect(
            x, covars, self.x_to_y
        )
        if res.size(1) == 3:
            return res[:, 1]
        else:
            return res


class DFIVEstimatorWithUncertainty(MREstimatorWithUncertainty, DFIVEstimator):
    @staticmethod
    def from_results(dir_name: str) -> "DFIVEstimator":  # type: ignore
        with open(os.path.join(dir_name, "meta.json"), "rt") as f:
            meta = json.load(f)

        weights = torch.load(os.path.join(dir_name, "linear_weights.pt"))

        model = DFIVModel.load_from_checkpoint(
            os.path.join(dir_name, "dfiv_model.ckpt")
        )

        # Load conformal model if available.
        conformal_filename = os.path.join(dir_name, "dfiv_calibration.ckpt")
        if os.path.isfile(conformal_filename):
            stub = _ConformalStub(model, weights["betas2"])
            conformal = OutcomeResidualPrediction.load_from_checkpoint(
                conformal_filename,
                wrapped_model=stub
            )

            # Set the q-hat.
            conformal.q_hat = meta["q_hat"]  # type: ignore

            return DFIVEstimatorWithUncertainty(
                conformal, weights["betas1"], weights["betas2"]
            )

        return DFIVEstimator(model, weights["betas1"], weights["betas2"])

    def effect_with_prediction_interval(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        alpha: float = 0.1
    ) -> torch.Tensor:
        """Mean exposure to outcome effect at values of x."""
        if alpha != self.alpha:
            raise ValueError(
                f"Only alpha={self.alpha} was estimated for this model."
            )

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        return self.average_treatment_effect(
            x, covars, self.x_to_y
        )


estimate = fit_dfiv
load = DFIVEstimatorWithUncertainty.from_results
