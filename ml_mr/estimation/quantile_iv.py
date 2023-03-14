"""
Implementation of an IV method based on estimating quantiles of the exposure
distribution.
"""

import argparse
import json
import os
from typing import Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers.logger import Logger
from torch.utils.data import DataLoader, Dataset, random_split

from ..logging import info
from ..utils import parse_project_and_run_name, default_validate_args
from ..utils.nn import MLP, OutcomeMLPBase
from ..utils.quantiles import QuantileLossMulti
from ..utils.conformal import get_conformal_adjustment
from .core import (IVDatasetWithGenotypes, MREstimator,
                   MREstimatorWithUncertainty, IVDataset)

# Default values definitions.
# fmt: off
DEFAULTS = {
    "exposure_hidden": [128, 64],
    "outcome_hidden": [32, 16],
    "exposure_learning_rate": 5e-4,
    "outcome_learning_rate": 5e-4,
    "exposure_batch_size": 10_000,
    "outcome_batch_size": 10_000,
    "exposure_max_epochs": 1000,
    "outcome_max_epochs": 1000,
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
        q: int,
        input_size: int,
        hidden: Iterable[int],
        lr: float,
        weight_decay: float = 0,
        add_input_layer_batchnorm: bool = False,
        add_hidden_layer_batchnorm: bool = False,
        activations: Iterable[nn.Module] = [nn.GELU()],
    ):
        """The model will predict q quantiles."""
        # q = 0, 0.2, 0.4, 0.6, 0.8, 1
        self.quantiles = torch.linspace(0.01, 0.99, q)

        loss = QuantileLossMulti(self.quantiles)

        super().__init__(
            input_size=input_size,
            hidden=hidden,
            out=q,
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
        return super().on_train_start()

    def _step(self, batch, batch_index, log_prefix):
        x, _, ivs, covars = batch

        x_hat = self.forward(
            torch.hstack([tens for tens in (ivs, covars) if tens.numel() > 0])
        )

        loss = self.loss(x_hat, x)
        self.log(f"exposure_{log_prefix}_loss", loss)
        return loss


class OutcomeMLP(OutcomeMLPBase):
    def __init__(
        self,
        exposure_network: ExposureQuantileMLP,
        input_size: int,
        hidden: Iterable[int],
        lr: float,
        weight_decay: float = 0,
        sqr: bool = False,
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

        # x is the input to the exposure model.
        mb = ivs.size(0)
        exposure_net_xs = torch.hstack(
            [tens for tens in (ivs, covars) if tens is not None]
        )

        with torch.no_grad():
            exposure_qs = self.exposure_network.forward(exposure_net_xs)

        n_q = exposure_qs.size(1)

        y_hat = torch.zeros((mb, 1), device=self.device)  # type: ignore
        for q1, q2 in zip(range(n_q), range(1, n_q)):
            midpoints = torch.mean(exposure_qs[:, [q1, q2]], dim=1)
            y_hat += self.mlp(
                torch.hstack([tens for tens in (
                    midpoints.reshape(-1, 1), covars, taus
                ) if tens is not None])
            )

        return y_hat / n_q


class QuantileIVEstimator(MREstimator):
    def __init__(
        self,
        exposure_network: ExposureQuantileMLP,
        outcome_network: OutcomeMLP,
    ):
        self.exposure_network = exposure_network
        self.outcome_network = outcome_network

    @torch.no_grad()
    def _effect_no_covars(
        self,
        x: torch.Tensor,
    ):
        return self.outcome_network.x_to_y(x, None)

    @torch.no_grad()
    def _effect_covars(
        self,
        x: torch.Tensor,
        covars: torch.Tensor,
    ):
        n_cov_rows = covars.size(0)
        x_rep = torch.repeat_interleave(x, n_cov_rows, dim=0)
        covars = covars.repeat(x.size(0), 1)

        y_hats = self.outcome_network.x_to_y(x_rep, covars)

        means = torch.tensor(
            [tens.mean() for tens in torch.split(y_hats, n_cov_rows)]
        )

        return means

    def effect(
        self, x: torch.Tensor, covars: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Mean exposure to outcome effect at values of x."""
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if covars is None or covars.numel() < 1:
            return self._effect_no_covars(x)
        else:
            return self._effect_covars(x, covars)

    @classmethod
    def from_results(cls, dir_name: str) -> "QuantileIVEstimator":
        exposure_network = ExposureQuantileMLP.load_from_checkpoint(
            os.path.join(dir_name, "exposure_network.ckpt")
        )

        outcome_network = OutcomeMLP.load_from_checkpoint(
            os.path.join(dir_name, "outcome_network.ckpt"),
            exposure_network=exposure_network
        )

        if outcome_network.hparams.sqr:  # type: ignore
            with open(os.path.join(dir_name, "meta.json"), "rt") as f:
                meta = json.load(f)
                q_hat = meta.get("q_hat", 0)

            return QuantileIVEstimatorWithUncertainty(
                exposure_network,
                outcome_network,
                alpha=0.1,
                q_hat=q_hat
            )

        return cls(exposure_network, outcome_network)


class QuantileIVEstimatorWithUncertainty(
    QuantileIVEstimator,
    MREstimatorWithUncertainty
):
    def __init__(
        self,
        exposure_network: ExposureQuantileMLP,
        outcome_network: OutcomeMLP,
        alpha: float = 0.1,
        q_hat: float = 0
    ):
        self.exposure_network = exposure_network
        self.outcome_network = outcome_network
        self.alpha = alpha

        # Conformal prediction adjustment.
        self.q_hat = q_hat

    @torch.no_grad()
    def _effect_no_covars_unc(
        self,
        x: torch.Tensor,
        tau: float
    ):
        taus = torch.full((x.size(0), 1), tau)
        return self.outcome_network.x_to_y(x, None, taus)

    @torch.no_grad()
    def _effect_covars_unc(
        self,
        x: torch.Tensor,
        covars: torch.Tensor,
        tau: float
    ):
        n_cov_rows = covars.size(0)
        x_rep = torch.repeat_interleave(x, n_cov_rows, dim=0)
        covars = covars.repeat(x.size(0), 1)

        taus = torch.full((x_rep.size(0), 1), tau)
        y_hats = self.outcome_network.x_to_y(x_rep, covars, taus)

        means = torch.tensor(
            [tens.mean() for tens in torch.split(y_hats, n_cov_rows)]
        )

        return means

    def effect_with_prediction_interval(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        alpha: float = 0.05
    ) -> torch.Tensor:
        assert self.outcome_network.hparams.sqr  # type: ignore

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        q = [alpha, 0.5, 1 - alpha]

        if covars is None:
            pred = torch.hstack([
                self._effect_no_covars_unc(x, tau).reshape(-1, 1)
                for tau in q
            ])

        else:
            pred = torch.hstack([
                self._effect_covars_unc(x, covars, tau).reshape(-1, 1)
                for tau in q
            ])

        # Conformal prediction adjustment if set.
        conformal_adj = [-self.q_hat, 0, self.q_hat]
        for j in range(3):
            pred[:, j] = pred[:, j] + conformal_adj[j]

        return pred


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
        q=args.q,
        dataset=dataset,
        no_plot=args.no_plot,
        sqr=args.sqr,
        wandb_project=args.wandb_project,
        **kwargs,
    )


def train_exposure_model(
    q: int,
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
    wandb_project: Optional[str] = None
) -> None:
    info("Training exposure model.")
    model = ExposureQuantileMLP(
        q=q,
        input_size=input_size,
        hidden=hidden,
        lr=learning_rate,
        weight_decay=weight_decay,
        add_input_layer_batchnorm=add_input_batchnorm,
        add_hidden_layer_batchnorm=True,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=len(val_dataset), num_workers=0  # type: ignore
    )

    # Remove checkpoint if exists.
    full_filename = os.path.join(output_dir, "exposure_network.ckpt")
    if os.path.isfile(full_filename):
        info(f"Removing file '{full_filename}'.")
        os.remove(full_filename)

    logger: Union[bool, Iterable[Logger]] = True
    if wandb_project is not None:
        from pytorch_lightning.loggers.wandb import WandbLogger
        project, run_name = parse_project_and_run_name(wandb_project)
        logger = [
            WandbLogger(name=run_name, project=project)
        ]

    trainer = pl.Trainer(
        log_every_n_steps=1,
        max_epochs=max_epochs,
        accelerator=accelerator,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="exposure_val_loss", patience=20
            ),
            pl.callbacks.ModelCheckpoint(
                filename="exposure_network",
                dirpath=output_dir,
                save_top_k=1,
                monitor="exposure_val_loss",
            ),
        ],
        logger=logger
    )
    trainer.fit(model, train_dataloader, val_dataloader)


def train_outcome_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    exposure_network: ExposureQuantileMLP,
    output_dir: str,
    hidden: List[int],
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    add_input_batchnorm: bool,
    max_epochs: int,
    accelerator: Optional[str] = None,
    sqr: bool = False,
    wandb_project: Optional[str] = None
) -> float:
    info("Training outcome model.")
    n_covars = train_dataset[0][3].numel()
    model = OutcomeMLP(
        exposure_network=exposure_network,
        input_size=1 + n_covars,
        lr=learning_rate,
        weight_decay=weight_decay,
        hidden=hidden,
        add_input_layer_batchnorm=add_input_batchnorm,
        sqr=sqr
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=len(val_dataset), num_workers=0  # type: ignore
    )

    # Remove checkpoint if exists.
    full_filename = os.path.join(output_dir, "outcome_network.ckpt")
    if os.path.isfile(full_filename):
        info(f"Removing file '{full_filename}'.")
        os.remove(full_filename)

    logger: Union[bool, Iterable[Logger]] = True
    if wandb_project is not None:
        from pytorch_lightning.loggers.wandb import WandbLogger
        project, run_name = parse_project_and_run_name(wandb_project)
        logger = [
            WandbLogger(name=run_name, project=project)
        ]

    model_checkpoint = pl.callbacks.ModelCheckpoint(
        filename="outcome_network",
        dirpath=output_dir,
        save_top_k=1,
        monitor="outcome_val_loss",
    )

    trainer = pl.Trainer(
        log_every_n_steps=1,
        max_epochs=max_epochs,
        accelerator=accelerator,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="outcome_val_loss", patience=20
            ),
            model_checkpoint,
        ],
        logger=logger
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    # Return the val loss.
    score = model_checkpoint.best_model_score
    assert isinstance(score, torch.Tensor)
    return score.item()


def fit_quantile_iv(
    q: int,
    dataset: IVDataset,
    output_dir: str = DEFAULTS["output_dir"],  # type: ignore
    validation_proportion: float = DEFAULTS["validation_proportion"],  # type: ignore # noqa: E501
    no_plot: bool = False,
    sqr: bool = False,
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
    wandb_project: Optional[str] = None
) -> QuantileIVEstimator:
    # Create output directory if needed.
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Metadata dictionary that will be saved alongside the results.
    meta = locals()
    del meta["dataset"]  # We don't serialize the dataset.

    covars = dataset.save_covariables(output_dir)

    min_x = torch.min(dataset.exposure).item()
    max_x = torch.max(dataset.exposure).item()
    domain = (min_x, max_x)

    # Split here into train and val.
    train_dataset, val_dataset = random_split(
        dataset, [1 - validation_proportion, validation_proportion]
    )

    train_exposure_model(
        q=q,
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
        wandb_project=wandb_project
    )

    exposure_network = ExposureQuantileMLP.load_from_checkpoint(
        os.path.join(output_dir, "exposure_network.ckpt")
    ).eval()  # type: ignore

    exposure_network.freeze()

    if not no_plot:
        plot_exposure_model(
            exposure_network,
            val_dataset,
            output_filename=os.path.join(
                output_dir, "exposure_model_predictions.png"
            ),
        )

    outcome_val_loss = train_outcome_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        exposure_network=exposure_network,
        output_dir=output_dir,
        hidden=outcome_hidden,
        learning_rate=outcome_learning_rate,
        weight_decay=outcome_weight_decay,
        batch_size=outcome_batch_size,
        add_input_batchnorm=outcome_add_input_batchnorm,
        max_epochs=outcome_max_epochs,
        accelerator=accelerator,
        sqr=sqr,
        wandb_project=wandb_project
    )

    meta["outcome_val_loss"] = outcome_val_loss

    outcome_network = OutcomeMLP.load_from_checkpoint(
        os.path.join(output_dir, "outcome_network.ckpt"),
        exposure_network=exposure_network,
    ).eval()  # type: ignore

    if sqr:
        # Conformal prediction.
        q_hat = get_conformal_adjustment(
            outcome_network, val_dataset, alpha=0.1  # type: ignore
        )
        info(f"Conformal adjustment estimated at q_hat={q_hat}.")

        meta["q_hat"] = q_hat

        # TODO make the conformal adjustment opt out.
        estimator: QuantileIVEstimator = QuantileIVEstimatorWithUncertainty(
            exposure_network, outcome_network, alpha=0.1, q_hat=q_hat
        )
    else:
        estimator = QuantileIVEstimator(exposure_network, outcome_network)

    # Save the metadata, estimator statistics and log artifact to WandB if
    # required.
    with open(os.path.join(output_dir, "meta.json"), "wt") as f:
        json.dump(meta, f)

    save_estimator_statistics(
        estimator,
        covars,
        domain=domain,
        output_prefix=os.path.join(output_dir, "causal_estimates"),
        alpha=0.1 if sqr else None
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

    return estimator


@torch.no_grad()
def plot_exposure_model(
    exposure_network: ExposureQuantileMLP,
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
    covars: Optional[torch.Tensor],
    domain: Tuple[float, float],
    output_prefix: str = "causal_estimates",
    alpha: Optional[float] = None
):
    # Save the causal effect at over the domain.
    xs = torch.linspace(domain[0], domain[1], 1000)

    if estimator.outcome_network.hparams.sqr and alpha:  # type: ignore
        assert isinstance(estimator, QuantileIVEstimatorWithUncertainty)
        ys = estimator.effect_with_prediction_interval(xs, covars, alpha=alpha)
        df = pd.DataFrame(
            torch.hstack((xs.reshape(-1, 1), ys)).numpy(),
            columns=["x", "y_do_x_lower", "y_do_x", "y_do_x_upper"]
        )
    else:
        ys = estimator.effect(xs, covars).reshape(-1)
        df = pd.DataFrame({"x": xs, "y_do_x": ys})

    plt.figure()
    plt.scatter(df["x"], df["y_do_x"], label="Estimated Y | do(X=x)", s=3)

    if "y_do_x_lower" in df.columns:
        # Add the CI on the plot.
        assert alpha is not None
        plt.fill_between(
            df["x"],
            df["y_do_x_lower"],
            df["y_do_x_upper"],
            color="#dddddd",
            zorder=-1,
            label=f"{int((1 - alpha) * 100)}% Prediction interval"
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig(f"{output_prefix}.png", dpi=600)
    plt.clf()

    df.to_csv(f"{output_prefix}.csv", index=False)


def configure_argparse(parser) -> None:
    parser.add_argument(
        "--q", "-q",
        type=int,
        help="Number of quantiles of the exposure distribution to estimate in "
        "the exposure model.",
        required=True,
    )

    parser.add_argument(
        "--histogram",
        action="store_true",
        help="By default, we use quantiles for density estimation. Using this "
        "option, we will use evenly spaced bins (histogram) instead.",
    )

    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"])

    parser.add_argument(
        "--no-plot",
        help="Disable plotting of diagnostics.",
        action="store_true",
    )

    parser.add_argument(
        "--sqr",
        help="Enable simultaneous quantile regression to estimate a "
        "prediction interval.",
        action="store_true"
    )

    parser.add_argument(
        "--outcome-type",
        default="continuous",
        choices=["continuous", "binary"],
        help="Variable type for the outcome (binary vs continuous).",
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
