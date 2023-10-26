"""
Implementation of an IV method based on estimating quantiles of the exposure
distribution.
"""

import argparse
import json
import os
from typing import Iterable, List, Optional, Tuple, Any, Union

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl

from ..logging import info
from ..utils import default_validate_args, parse_project_and_run_name
from ..utils.conformal import (
    NONCONFORMITY_MEASURES_TYPE, NONCONFORMITY_MEASURES,
    nonconformity_sqr, nonconformity_quantile_reg, nonconformity_gaussian_nn,
    estimate_q_hat
)
from ..utils.models import MLP, GaussianNet, OutcomeMLPBase
from ..utils.quantiles import QuantileLossMulti
from ..utils.training import train_model, resample_dataset
from .core import (IVDataset, IVDatasetWithGenotypes, MREstimator,
                   MREstimatorWithUncertainty)

# Default values definitions.
# fmt: off
DEFAULTS = {
    "n_quantiles": 5,
    "conformal_score": None,
    "conformal_alpha_level": 0.1,
    "exposure_hidden": [128, 64],
    "outcome_hidden": [64, 32],
    "exposure_learning_rate": 5e-4,
    "outcome_learning_rate": 5e-4,
    "exposure_batch_size": 10_000,
    "outcome_batch_size": 10_000,
    "exposure_max_epochs": 1000,
    "outcome_max_epochs": 1000,
    "nmqn_penalty_lambda": 1,
    "exposure_weight_decay": 1e-4,
    "outcome_weight_decay": 1e-4,
    "exposure_add_input_batchnorm": False,
    "outcome_add_input_batchnorm": False,
    "accelerator": "gpu" if (
        torch.cuda.is_available() and torch.cuda.device_count() > 0
    ) else "cpu",
    "validation_proportion": 0.2,
    "output_dir": "quantile_iv_estimate",
}
# fmt: on


class ExposureQuantileMLP(MLP):
    def __init__(
        self,
        n_quantiles: int,
        input_size: int,
        hidden: Iterable[int],
        lr: float,
        weight_decay: float = 0,
        add_input_layer_batchnorm: bool = False,
        add_hidden_layer_batchnorm: bool = False,
        activations: Iterable[nn.Module] = [nn.GELU()],
    ):
        """The model will predict q quantiles."""
        assert n_quantiles >= 3
        self.quantiles = torch.tensor([
            (i + 1) / (n_quantiles + 1) for i in range(n_quantiles)]
        )

        loss = QuantileLossMulti(self.quantiles)

        super().__init__(
            input_size=input_size,
            hidden=hidden,
            out=n_quantiles,
            add_input_layer_batchnorm=add_input_layer_batchnorm,
            add_hidden_layer_batchnorm=add_hidden_layer_batchnorm,
            activations=activations,
            lr=lr,
            weight_decay=weight_decay,
            loss=loss
        )

    def on_fit_start(self) -> None:
        self.loss.quantiles = self.loss.quantiles.to(  # type: ignore
            device=self.device
        )
        return super().on_fit_start()

    def _step(self, batch, batch_index, log_prefix):
        x, _, ivs, covars = batch

        x_hat = self.forward(
            torch.hstack([tens for tens in (ivs, covars) if tens.numel() > 0])
        )

        loss = self.loss(x_hat, x)
        self.log(f"exposure_{log_prefix}_loss", loss)
        return loss


class ExposureNMQN(MLP):
    def __init__(
        self,
        n_quantiles: int,
        input_size: int,
        hidden: Iterable[int],
        lr: float,
        pen_lambda: float = 1,
        weight_decay: float = 0,
        add_input_layer_batchnorm: bool = False,
        add_hidden_layer_batchnorm: bool = False,
        activations: Iterable[nn.Module] = [nn.GELU()],
    ):
        """The model will predict q quantiles."""
        assert n_quantiles >= 3
        self.quantiles = torch.tensor([
            (i + 1) / (n_quantiles + 1) for i in range(n_quantiles)]
        )

        loss = QuantileLossMulti(self.quantiles)
        hidden = list(hidden)
        assert len(hidden) >= 2

        super().__init__(
            input_size=input_size,
            hidden=hidden[:-1],
            out=hidden[-1],
            add_input_layer_batchnorm=add_input_layer_batchnorm,
            add_hidden_layer_batchnorm=add_hidden_layer_batchnorm,
            activations=activations,
            binary_output=True,
            lr=lr,
            weight_decay=weight_decay,
            loss=loss,
            _save_hyperparams=False
        )

        self.save_hyperparameters()
        self.deltas = nn.Linear(hidden[-1] + 1, n_quantiles, bias=False)

    def on_fit_start(self) -> None:
        self.loss.quantiles = self.loss.quantiles.to(  # type: ignore
            device=self.device
        )
        return super().on_fit_start()

    def penalty(self):
        return self.l1_penalty_vec(self.deltas.weight)

    @staticmethod
    def l1_penalty_vec(d):
        M = torch.sum(torch.max(torch.tensor(0), -d[1:, 1:]), dim=0)
        d0_clipped = torch.clip(d[0, 1:], min=M)
        penalty = torch.mean(torch.abs(d[0, 1:] - d0_clipped))
        return penalty

    def forward(self, x):
        mlp_out = super().forward(x)
        mlp_out = torch.hstack((
            torch.ones(mlp_out.size(0), 1, device=self.device),
            mlp_out
        ))

        betas = torch.cumsum(self.deltas.weight, dim=1)

        return mlp_out @ betas.T

    def _step(self, batch, batch_index, log_prefix):
        x, _, ivs, covars = batch

        qhat = self.forward(
            torch.hstack([tens for tens in (ivs, covars) if tens.numel() > 0])
        )

        qloss = self.loss(qhat, x)
        pen = self.penalty()
        loss = qloss + self.hparams.pen_lambda * pen

        self.log(f"exposure_{log_prefix}_qloss", qloss)
        self.log(f"exposure_{log_prefix}_pen", pen)
        self.log(f"exposure_{log_prefix}_loss", loss)

        return loss


QIVExposureNetType = Union[ExposureNMQN, ExposureQuantileMLP]


class OutcomeQuantileRegMLP(MLP):
    def __init__(
        self,
        exposure_network: QIVExposureNetType,
        input_size: int,
        hidden: Iterable[int],
        lr: float,
        weight_decay: float = 0,
        quantile_regression_alpha: float = 0.1,
        add_input_layer_batchnorm: bool = False,
        add_hidden_layer_batchnorm: bool = False,
        activations: Iterable[nn.Module] = [nn.GELU()]
    ):
        alpha = quantile_regression_alpha
        self.quantiles = torch.tensor([alpha / 2, 0.5, 1 - alpha / 2])

        super().__init__(
            input_size=input_size,
            hidden=list(hidden),
            out=3,
            add_input_layer_batchnorm=add_input_layer_batchnorm,
            add_hidden_layer_batchnorm=add_hidden_layer_batchnorm,
            activations=activations,
            lr=lr,
            weight_decay=weight_decay,
            loss=QuantileLossMulti(self.quantiles),
            _save_hyperparams=False
        )

        self.exposure_network = exposure_network
        self.save_hyperparameters(ignore=["exposure_network"])

    def on_fit_start(self) -> None:
        self.loss.quantiles = self.loss.quantiles.to(  # type: ignore
            device=self.device
        )
        return super().on_fit_start()

    def _step(self, batch, batch_index, log_prefix):
        _, y, ivs, covars = batch

        y_hat = self.forward(ivs, covars)

        loss = self.loss(y_hat, y)
        self.log(f"outcome_{log_prefix}_loss", loss)

        return loss

    def x_to_y(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if covars is not None:
            x = torch.hstack((x, covars))

        return self.mlp(x).reshape(-1, 1, 3)

    def forward(  # type: ignore
        self,
        ivs: torch.Tensor,
        covars: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Z, C to Y"""
        x_hat = _qiv_zc_to_x_hat(ivs, covars, self.exposure_network)
        x = x_hat

        if covars is not None:
            x = torch.hstack((x, covars))

        return self.mlp(x)


class OutcomeMLP(OutcomeMLPBase):
    def __init__(
        self,
        exposure_network: QIVExposureNetType,
        input_size: int,
        hidden: Iterable[int],
        lr: float,
        weight_decay: float = 0,
        binary_outcome: bool = False,
        sqr: bool = False,
        quantile_regression_alpha: Optional[float] = None,
        add_input_layer_batchnorm: bool = False,
        add_hidden_layer_batchnorm: bool = False,
        activations: Iterable[nn.Module] = [nn.GELU()]
    ):
        super().__init__(
            exposure_network=exposure_network,
            input_size=input_size,
            hidden=hidden,
            lr=lr,
            weight_decay=weight_decay,
            binary_outcome=binary_outcome,
            sqr=sqr,
            add_input_layer_batchnorm=add_input_layer_batchnorm,
            add_hidden_layer_batchnorm=add_hidden_layer_batchnorm,
            activations=activations
        )

    def forward(  # type: ignore
        self,
        ivs: torch.Tensor,
        covars: Optional[torch.Tensor],
        taus: Optional[torch.Tensor] = None
    ):
        """Forward pass throught the exposure and outcome models."""
        if self.hparams.sqr:  # type: ignore
            assert taus is not None, "Need quantile samples if SQR enabled."

        x_hat = _qiv_zc_to_x_hat(ivs, covars, self.exposure_network)
        y_hat = self.mlp(
            torch.hstack([tens for tens in (
                x_hat, covars, taus
            ) if tens is not None])
        )

        return y_hat


class OutcomeGaussianNet(GaussianNet):
    def __init__(
        self,
        exposure_network: QIVExposureNetType,
        input_size: int,
        hidden: Iterable[int],
        lr: float,
        weight_decay: float = 0,
        add_input_layer_batchnorm: bool = False,
        add_hidden_layer_batchnorm: bool = False,
        activations: Iterable[nn.Module] = [nn.GELU()]
    ):
        super().__init__(
            input_size=input_size,
            hidden=hidden,
            add_input_layer_batchnorm=add_input_layer_batchnorm,
            add_hidden_layer_batchnorm=add_hidden_layer_batchnorm,
            activations=activations,
            lr=lr,
            weight_decay=weight_decay,
            _save_hyperparams=False,
        )
        self.exposure_network = exposure_network
        self.save_hyperparameters(ignore=["exposure_network"])

    def _step(self, batch, batch_index, log_prefix="train"):
        _, y, ivs, covars = batch

        mu, var = self.forward(ivs, covars)
        loss = self.loss(mu, y, var)

        self.log(f"outcome_{log_prefix}_loss", loss)

        return loss

    def x_to_y(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if covars is not None:
            x = torch.hstack((x, covars))

        # forward parameters is the default forward pass for gaussian net.
        # Won't go through the exposure network.
        return self.forward_parameters(x)  # type: ignore

    def forward(  # type: ignore
        self, ivs: torch.Tensor, covars: Optional[torch.Tensor]
    ):
        """IV -> exposure net -> pred. quantiles -> outcome."""

        x_hat = _qiv_zc_to_x_hat(ivs, covars, self.exposure_network)
        return self.x_to_y(x_hat, covars)


def _qiv_zc_to_x_hat(
    ivs: torch.Tensor,
    covars: Optional[torch.Tensor],
    exposure_network: QIVExposureNetType
) -> torch.Tensor:
    """Utility function that gets predicted expected exposure given the IV and
    covariates.

    """
    exposure_net_xs = torch.hstack(
        [tens for tens in (ivs, covars) if tens is not None]
    )

    with torch.no_grad():
        x_hat = torch.mean(  # type: ignore
            exposure_network.forward(exposure_net_xs),
            axis=1,
            keepdim=True
        )

    return x_hat


class QuantileIVEstimator(MREstimator):
    def __init__(
        self,
        exposure_network: QIVExposureNetType,
        outcome_network: OutcomeMLP,
        covars: Optional[torch.Tensor] = None
    ):
        self.exposure_network = exposure_network
        self.outcome_network = outcome_network
        super().__init__(covars)

    def iv_reg_function(
        self, x: torch.Tensor, covars: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.outcome_network.x_to_y(x, covars)

    @classmethod
    def from_results(cls, dir_name: str) -> "QuantileIVEstimator":
        with open(os.path.join(dir_name, "meta.json"), "rt") as f:
            meta = json.load(f)

        try:
            covars = torch.load(os.path.join(dir_name, "covariables.pt"))
        except FileNotFoundError:
            covars = None

        exposure_net_cls: type[pl.LightningModule] = (
            ExposureNMQN if meta.get("nmqn", False)
            else ExposureQuantileMLP
        )

        exposure_network = exposure_net_cls.load_from_checkpoint(
            os.path.join(dir_name, "exposure_network.ckpt")
        ).to(torch.device("cpu"))

        # Get the right class for the outcome model.
        if meta.get("conformal_score") is None:
            outcome_cls = OutcomeMLP
        elif meta["conformal_score"] == "quantile-reg":
            outcome_cls = OutcomeQuantileRegMLP  # type: ignore
        elif meta["conformal_score"] == "gaussian-nn":
            outcome_cls = OutcomeGaussianNet  # type: ignore
        elif meta["conformal_score"] not in NONCONFORMITY_MEASURES:
            raise ValueError(meta["conformal_score"])

        outcome_network = outcome_cls.load_from_checkpoint(
            os.path.join(dir_name, "outcome_network.ckpt"),
            exposure_network=exposure_network
        ).to(torch.device("cpu"))

        outcome_network.eval()  # type: ignore

        if meta["conformal_score"] is not None:
            return QuantileIVEstimatorWithUncertainty(
                exposure_network,
                outcome_network,
                meta,
                covars
            )

        return cls(exposure_network, outcome_network, covars=covars)


class QuantileIVEstimatorWithUncertainty(
    QuantileIVEstimator,
    MREstimatorWithUncertainty
):
    def __init__(
        self,
        exposure_network: QIVExposureNetType,
        outcome_network: OutcomeMLP,
        meta: dict,
        covars: Optional[torch.Tensor] = None,
    ):
        self.exposure_network = exposure_network
        self.outcome_network = outcome_network
        self.covars = covars

        # Conformal prediction adjustment.
        self.conformal_score: NONCONFORMITY_MEASURES_TYPE =\
            meta["conformal_score"]
        assert self.conformal_score in NONCONFORMITY_MEASURES

        self.meta = meta

    def iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        alpha: float = 0.1
    ) -> torch.Tensor:
        if alpha != self.meta["conformal_alpha_level"]:
            raise ValueError("Conformal prediction tuned for alpha = {}"
                             "".format(self.meta["conformal_alpha_level"]))

        if self.conformal_score == "sqr":
            alpha = self.meta["conformal_alpha_level"]
            pred = []
            with torch.no_grad():
                for tau in [alpha / 2, 0.5, 1 - alpha / 2]:
                    cur_y = self.outcome_network.x_to_y(x, covars, tau)
                    pred.append(cur_y)

            # n x y dimension x 3 for the values in tau.
            pred_tens = torch.stack(pred, dim=2)

            # Conformal prediction adjustment if set.
            pred_tens[:, :, 0] -= self.meta["q_hat"]
            pred_tens[:, :, 2] += self.meta["q_hat"]

            return pred_tens

        if self.conformal_score == "quantile-reg":
            with torch.no_grad():
                ys = self.outcome_network.x_to_y(x, covars)

                ys[:, :, 0] -= self.meta["q_hat"]
                ys[:, :, 2] += self.meta["q_hat"]

                return ys

        if self.conformal_score == "gaussian-nn":
            with torch.no_grad():
                mu_y, var_y = self.outcome_network.x_to_y(x, covars)
                sigma_y = torch.sqrt(var_y)
                out = mu_y.reshape(-1, 1, 1).repeat(1, 1, 3)
                out[:, 0, [0]] -= sigma_y * self.meta["q_hat"]
                out[:, 0, [2]] += sigma_y * self.meta["q_hat"]

                return out

        raise NotImplementedError()


def main(args: argparse.Namespace) -> None:
    """Command-line interface entry-point."""
    default_validate_args(args)

    # Prepare train and validation datasets.
    # There is theoretically a little bit of leakage here because the histogram
    # or quantiles will be calculated including the validation dataset.
    # This should not have a big impact...
    dataset = IVDatasetWithGenotypes.from_argparse_namespace(args)

    # Automatically add the model hyperparameters.
    kwargs = {k: v for k, v in vars(args).items() if k in DEFAULTS.keys()}

    fit_quantile_iv(
        dataset=dataset,
        fast=args.fast,
        wandb_project=args.wandb_project,
        nmqn=args.nmqn,
        resample=args.resample,
        **kwargs,
    )


def train_exposure_model(
    n_quantiles: int,
    train_dataset: Dataset,
    val_dataset: Dataset,
    input_size: int,
    output_dir: str,
    hidden: List[int],
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    add_input_batchnorm: bool,
    max_epochs: int,
    accelerator: Optional[str] = None,
    wandb_project: Optional[str] = None,
    nmqn_penalty_lambda: Optional[float] = None
) -> Tuple[type[QIVExposureNetType], float]:
    info("Training exposure model.")
    kwargs = {
        "n_quantiles": n_quantiles,
        "input_size": input_size,
        "hidden": hidden,
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "add_input_layer_batchnorm": add_input_batchnorm,
        "add_hidden_layer_batchnorm": True,
    }

    if nmqn_penalty_lambda is None:
        model = ExposureQuantileMLP(**kwargs)  # type: ignore
    else:
        model = ExposureNMQN(
            **kwargs,  # type: ignore
            pen_lambda=nmqn_penalty_lambda
        )

    return type(model), train_model(
        train_dataset,
        val_dataset,
        model=model,
        monitored_metric="exposure_val_loss",
        output_dir=output_dir,
        checkpoint_filename="exposure_network.ckpt",
        batch_size=batch_size,
        max_epochs=max_epochs,
        accelerator=accelerator,
        wandb_project=wandb_project
    )


def train_outcome_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    exposure_network: QIVExposureNetType,
    output_dir: str,
    hidden: List[int],
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    add_input_batchnorm: bool,
    max_epochs: int,
    conformal_score: Optional[NONCONFORMITY_MEASURES_TYPE],
    conformal_alpha_level: Optional[float],
    accelerator: Optional[str] = None,
    binary_outcome: bool = False,
    wandb_project: Optional[str] = None
) -> Tuple[Any, float]:
    info("Training outcome model.")
    if binary_outcome and conformal_score is not None:
        raise NotImplementedError("Conformal prediction not implemented for "
                                  "binary outcomes.")

    n_covars = train_dataset[0][3].numel()
    if (
        conformal_score is None or
        conformal_score == "sqr" or
        conformal_score == "residual-aux-nn"
    ):
        model = OutcomeMLP(
            exposure_network=exposure_network,
            input_size=1 + n_covars,
            lr=learning_rate,
            weight_decay=weight_decay,
            hidden=hidden,
            add_input_layer_batchnorm=add_input_batchnorm,
            binary_outcome=binary_outcome,
            sqr=(conformal_score == "sqr"),
            quantile_regression_alpha=(
                conformal_alpha_level if conformal_score == "quantile-reg"
                else None
            )
        )

    elif conformal_score == "quantile-reg":
        assert conformal_alpha_level is not None
        model = OutcomeQuantileRegMLP(
            exposure_network=exposure_network,
            input_size=1 + n_covars,
            hidden=hidden,
            lr=learning_rate,
            weight_decay=weight_decay,
            quantile_regression_alpha=conformal_alpha_level,
            add_input_layer_batchnorm=add_input_batchnorm,
            add_hidden_layer_batchnorm=False,
        )

    elif conformal_score == "gaussian-nn":
        model = OutcomeGaussianNet(
            exposure_network=exposure_network,
            input_size=1 + n_covars,
            hidden=hidden,
            lr=learning_rate,
            weight_decay=weight_decay,
            add_input_layer_batchnorm=add_input_batchnorm,
            add_hidden_layer_batchnorm=False,
        )

    else:
        raise ValueError(conformal_score)

    info(f"Loss: {model.loss}")

    return type(model), train_model(
        train_dataset,
        val_dataset,
        model=model,
        monitored_metric="outcome_val_loss",
        output_dir=output_dir,
        checkpoint_filename="outcome_network.ckpt",
        batch_size=batch_size,
        max_epochs=max_epochs,
        accelerator=accelerator,
        wandb_project=wandb_project,
    )


def fit_quantile_iv(
    dataset: IVDataset,
    n_quantiles: int = DEFAULTS["n_quantiles"],  # type: ignore
    stage2_dataset: Optional[IVDataset] = None,  # type: ignore
    output_dir: str = DEFAULTS["output_dir"],  # type: ignore
    validation_proportion: float = DEFAULTS["validation_proportion"],  # type: ignore # noqa: E501
    fast: bool = False,
    binary_outcome: bool = False,
    resample: bool = False,
    nmqn: bool = False,
    nmqn_penalty_lambda: Optional[float] = DEFAULTS["nmqn_penalty_lambda"],  # type: ignore # noqa: E501
    conformal_score: Optional[NONCONFORMITY_MEASURES_TYPE] = DEFAULTS["conformal_score"],  # type: ignore # noqa: E501
    conformal_alpha_level: Optional[float] = DEFAULTS["conformal_alpha_level"],  # type: ignore # noqa: E501
    exposure_hidden: List[int] = DEFAULTS["exposure_hidden"],  # type: ignore
    exposure_learning_rate: float = DEFAULTS["exposure_learning_rate"],  # type: ignore # noqa: E501
    exposure_weight_decay: float = DEFAULTS["exposure_weight_decay"],  # type: ignore # noqa: E501
    exposure_batch_size: int = DEFAULTS["exposure_batch_size"],  # type: ignore
    exposure_max_epochs: int = DEFAULTS["exposure_max_epochs"],  # type: ignore
    exposure_add_input_batchnorm: bool = DEFAULTS["exposure_add_input_batchnorm"],  # type: ignore # noqa: E501
    outcome_hidden: List[int] = DEFAULTS["outcome_hidden"],  # type: ignore
    outcome_learning_rate: float = DEFAULTS["outcome_learning_rate"],  # type: ignore # noqa: E501
    outcome_weight_decay: float = DEFAULTS["outcome_weight_decay"],  # type: ignore # noqa: E501
    outcome_batch_size: int = DEFAULTS["outcome_batch_size"],  # type: ignore
    outcome_max_epochs: int = DEFAULTS["outcome_max_epochs"],  # type: ignore
    outcome_add_input_batchnorm: bool = DEFAULTS["outcome_add_input_batchnorm"],  # type: ignore # noqa: E501
    accelerator: str = DEFAULTS["accelerator"],  # type: ignore
    wandb_project: Optional[str] = None,
) -> QuantileIVEstimator:
    if resample:
        dataset = resample_dataset(dataset)  # type: ignore
        if stage2_dataset is not None:
            stage2_dataset = resample_dataset(stage2_dataset)  # type: ignore

    # Create output directory if needed.
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Metadata dictionary that will be saved alongside the results.
    meta = dict(locals())
    meta["model"] = "quantile_iv"
    meta.update(dataset.exposure_descriptive_statistics())
    del meta["dataset"]  # We don't serialize the dataset.
    del meta["stage2_dataset"]

    covars = dataset.save_covariables(output_dir)

    # Split here into train and val.
    train_dataset, val_dataset = random_split(
        dataset, [1 - validation_proportion, validation_proportion]
    )

    # If there is a separate dataset for stage2, we split it too, otherwise
    # we reuse the stage 1 dataset.
    if stage2_dataset is not None:
        stg2_train_dataset, stg2_val_dataset = random_split(
            stage2_dataset, [1 - validation_proportion, validation_proportion]
        )
    else:
        stg2_train_dataset, stg2_val_dataset = (
            train_dataset, val_dataset
        )

    exposure_class, exposure_val_loss = train_exposure_model(
        n_quantiles=n_quantiles,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_size=dataset.n_exog(),
        output_dir=output_dir,
        hidden=exposure_hidden,
        learning_rate=exposure_learning_rate,
        weight_decay=exposure_weight_decay,
        batch_size=exposure_batch_size,
        add_input_batchnorm=exposure_add_input_batchnorm,
        max_epochs=exposure_max_epochs,
        accelerator=accelerator,
        wandb_project=wandb_project,
        nmqn_penalty_lambda=nmqn_penalty_lambda if nmqn else None
    )

    meta["exposure_val_loss"] = exposure_val_loss

    exposure_network = exposure_class.load_from_checkpoint(
        os.path.join(output_dir, "exposure_network.ckpt"),
    ).to(torch.device("cpu")).eval()  # type: ignore

    exposure_network.freeze()

    if not fast:
        plot_exposure_model(
            exposure_network,
            val_dataset,
            output_filename=os.path.join(
                output_dir, "exposure_model_predictions.png"
            ),
        )

    outcome_class, outcome_val_loss = train_outcome_model(
        train_dataset=stg2_train_dataset,
        val_dataset=stg2_val_dataset,
        exposure_network=exposure_network,
        output_dir=output_dir,
        hidden=outcome_hidden,
        learning_rate=outcome_learning_rate,
        weight_decay=outcome_weight_decay,
        batch_size=outcome_batch_size,
        add_input_batchnorm=outcome_add_input_batchnorm,
        max_epochs=outcome_max_epochs,
        accelerator=accelerator,
        conformal_score=conformal_score,
        conformal_alpha_level=conformal_alpha_level,
        binary_outcome=binary_outcome,
        wandb_project=wandb_project
    )

    meta["outcome_val_loss"] = outcome_val_loss

    outcome_network = outcome_class.load_from_checkpoint(
        os.path.join(output_dir, "outcome_network.ckpt"),
        exposure_network=exposure_network,
    ).eval().to(torch.device("cpu"))  # type: ignore

    # Training the 2nd stage model copies the exposure net to the GPU.
    # Here, we ensure they're on the same device.
    exposure_network.to(outcome_network.device)

    if conformal_score is not None:
        assert conformal_alpha_level is not None
        fit_conformal(
            outcome_network,
            conformal_score,
            stg2_val_dataset,  # type: ignore
            meta,
            alpha=conformal_alpha_level
        )

        estimator: QuantileIVEstimator = QuantileIVEstimatorWithUncertainty(
            exposure_network, outcome_network, meta, covars
        )
    else:
        estimator = QuantileIVEstimator(
            exposure_network, outcome_network, covars
        )

    # Save the metadata, estimator statistics and log artifact to WandB if
    # required.
    with open(os.path.join(output_dir, "meta.json"), "wt") as f:
        json.dump(meta, f)

    if not fast:
        save_estimator_statistics(
            estimator,
            domain=meta["domain"],
            output_prefix=os.path.join(output_dir, "causal_estimates"),
        )

    if wandb_project is not None:
        import wandb
        _, run_name = parse_project_and_run_name(wandb_project)
        artifact = wandb.Artifact(
            "results" if run_name is None else f"{run_name}_results",
            type="results"
        )
        artifact.add_dir(output_dir)
        wandb.log_artifact(artifact)
        wandb.finish()

    return estimator


def fit_conformal(
    outcome_model: OutcomeMLPBase,
    conformal_score: NONCONFORMITY_MEASURES_TYPE,
    conformal_dataset: IVDataset,
    meta: dict,
    alpha: float = 0.1,
):
    if conformal_score == "sqr":
        # Outcome model was fitted using SQR which we use to get the conformal
        # band.
        conf_scores = nonconformity_sqr(outcome_model, conformal_dataset)
        q_hat = estimate_q_hat(conf_scores, alpha=alpha)

    elif conformal_score == "quantile-reg":
        conf_scores = nonconformity_quantile_reg(
            outcome_model, conformal_dataset
        )
        q_hat = estimate_q_hat(conf_scores, alpha=alpha)

    elif conformal_score == "gaussian-nn":
        conf_scores = nonconformity_gaussian_nn(
            outcome_model, conformal_dataset
        )
        q_hat = estimate_q_hat(conf_scores, alpha=alpha)

    elif conformal_score in NONCONFORMITY_MEASURES:
        raise NotImplementedError()

    else:
        raise ValueError(conformal_score)

    info(f"Conformal adjustment estimated at q_hat={q_hat}.")
    meta["q_hat"] = q_hat


@torch.no_grad()
def plot_exposure_model(
    exposure_network: QIVExposureNetType,
    val_dataset: Dataset,
    output_filename: str
):
    assert hasattr(val_dataset, "__len__")
    dataloader = DataLoader(val_dataset, batch_size=len(val_dataset))
    true_x, _, ivs, covariables = next(iter(dataloader))

    input = torch.hstack(
        [tens for tens in (ivs, covariables) if tens.numel() > 0]
    )

    predicted_quantiles = exposure_network(input)

    def identity_line(ax=None, ls='--', *args, **kwargs):
        # see: https://stackoverflow.com/q/22104256/3986320
        ax = ax or plt.gca()
        identity, = ax.plot([], [], ls=ls, *args, **kwargs)

        def callback(axes):
            low_x, high_x = ax.get_xlim()
            low_y, high_y = ax.get_ylim()
            low = min(low_x, low_y)
            high = max(high_x, high_y)
            identity.set_data([low, high], [low, high])

        callback(ax)
        ax.callbacks.connect('xlim_changed', callback)
        ax.callbacks.connect('ylim_changed', callback)
        return ax

    for q in range(predicted_quantiles.size(1)):
        plt.scatter(
            true_x,
            predicted_quantiles[:, q].detach().numpy(),
            label="q={:.2f}".format(exposure_network.quantiles[q].item()),
            s=1,
            alpha=0.2,
        )
    identity_line(lw=1, color="black")
    plt.xlabel("Observed X")
    plt.ylabel("Predicted X (quantiles)")
    plt.legend()

    plt.savefig(output_filename, dpi=400)
    plt.clf()
    plt.close()


def save_estimator_statistics(
    estimator: QuantileIVEstimator,
    domain: Tuple[float, float],
    output_prefix: str = "causal_estimates",
):
    # Save the causal effect at over the domain.
    xs = torch.linspace(domain[0], domain[1], 500).reshape(-1, 1)

    if isinstance(estimator, QuantileIVEstimatorWithUncertainty):
        ys = estimator.avg_iv_reg_function(xs)

        if ys.size(1) != 1:
            raise NotImplementedError(
                "Saving statistics for multidimensional outcome not "
                "implemented yet."
            )

        df = pd.DataFrame(
            torch.hstack((xs, ys[:, 0, :])).numpy(),
            columns=["x", "y_do_x_lower", "y_do_x", "y_do_x_upper"]
        )

    else:
        ys = estimator.avg_iv_reg_function(xs).reshape(-1)
        df = pd.DataFrame({"x": xs.reshape(-1), "y_do_x": ys})

    plt.figure()
    plt.scatter(df["x"], df["y_do_x"], label="Estimated Y | do(X=x)", s=3)

    if "y_do_x_lower" in df.columns:
        # Add the CI on the plot.
        plt.fill_between(
            df["x"],
            df["y_do_x_lower"],
            df["y_do_x_upper"],
            color="#dddddd",
            zorder=-1,
            label="Prediction interval"
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig(f"{output_prefix}.png", dpi=600)
    plt.clf()

    df.to_csv(f"{output_prefix}.csv", index=False)


def configure_argparse(parser) -> None:
    parser.add_argument(
        "--n-quantiles", "-q",
        type=int,
        help="Number of quantiles of the exposure distribution to estimate in "
        "the exposure model.",
        required=True,
    )

    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"])

    parser.add_argument(
        "--fast",
        help="Disable plotting and logging of causal effects.",
        action="store_true",
    )

    parser.add_argument(
        "--conformal-score",
        default=DEFAULTS["conformal_score"],
        help="Conformal prediction nonconformity measure.",
        choices=NONCONFORMITY_MEASURES,
        type=str
    )

    parser.add_argument(
        "--conformal-alpha-level",
        default=DEFAULTS["conformal_alpha_level"],
        type=float,
    )

    parser.add_argument(
        "--outcome-type",
        default="continuous",
        choices=["continuous", "binary"],
        help="Variable type for the outcome (binary vs continuous).",
    )

    parser.add_argument(
        "--nmqn",
        action="store_true"
    )

    parser.add_argument(
        "--nmqn-penalty-lambda",
        type=float,
        default=DEFAULTS["nmqn_penalty_lambda"]
    )

    parser.add_argument(
        "--validation-proportion",
        type=float,
        default=DEFAULTS["validation_proportion"],
    )

    parser.add_argument(
        "--accelerator",
        default=DEFAULTS["accelerator"],
        help="Accelerator (e.g. gpu, cpu, mps) use to train the model. This "
        "will be passed to Pytorch Lightning.",
    )

    parser.add_argument(
        "--resample",
        help="Resample with replacement to do bootstrapping.",
        action="store_true"
    )

    parser.add_argument(
        "--wandb-project",
        default=None,
        type=str,
        help="Activates the Weights and Biases logger using the provided "
             "project name. Patterns such as project:run_name are also "
             "allowed."
    )

    MLP.add_mlp_arguments(
        parser,
        "exposure-",
        "Exposure Model Parameters",
        defaults={
            "hidden": DEFAULTS["exposure_hidden"],
            "batch-size": DEFAULTS["exposure_batch_size"],
        },
    )

    MLP.add_mlp_arguments(
        parser,
        "outcome-",
        "Outcome Model Parameters",
        defaults={
            "hidden": DEFAULTS["outcome_hidden"],
            "batch-size": DEFAULTS["outcome_batch_size"],
        },
    )

    IVDatasetWithGenotypes.add_dataset_arguments(parser)


# Standard names for estimators.
estimate = fit_quantile_iv
load = QuantileIVEstimator.from_results
