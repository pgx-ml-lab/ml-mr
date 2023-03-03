"""
Implementation of an IV method based on estimating quantiles of the exposure
distribution.
"""

import os
import sys
import argparse
from typing import Iterable, Optional, List, Union, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger
import matplotlib.pyplot as plt
import pandas as pd

from .core import MREstimator, IVDatasetWithGenotypes, _IVDataset
from ..logging import critical, info, warn
from ..utils import MLP, parse_project_and_run_name
from ..utils.quantiles import QuantileLossMulti, quantile_loss


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


class OutcomeMLP(MLP):
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
            input_size=input_size if not sqr else input_size + 1,
            hidden=hidden,
            out=1,
            add_input_layer_batchnorm=add_input_layer_batchnorm,
            add_hidden_layer_batchnorm=add_hidden_layer_batchnorm,
            activations=activations,
            lr=lr,
            weight_decay=weight_decay,
            loss=F.mse_loss if not sqr else quantile_loss,  # type: ignore
            _save_hyperparams=False,
        )
        self.exposure_network = exposure_network
        self.save_hyperparameters(ignore=["exposure_network"])

    def _step(self, batch, batch_index, log_prefix):
        _, y, ivs, covars = batch

        if self.hparams.sqr:
            taus = torch.rand(ivs.size(0), 1, device=self.device)
        else:
            taus = None

        y_hat = self.forward(ivs, covars, taus)

        if self.hparams.sqr:
            loss = self.loss(y_hat, y, taus)
        else:
            loss = self.loss(y_hat, y)

        self.log(f"outcome_{log_prefix}_loss", loss)

        return loss

    def x_to_y(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor],
        taus: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if taus is not None and not self.hparams.sqr:  # type: ignore
            raise ValueError("Can't provide tau if SQR not enabled.")

        stack = [x]

        if covars is not None and covars.numel() > 0:
            stack.append(covars)

        if taus is None and self.hparams.sqr:  # type: ignore
            # Predict median by default.
            taus = torch.full((x.size(0), 1), 0.5)

        if taus is not None:
            stack.append(taus)

        x = torch.hstack(stack)

        return self.mlp(x)

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
        tau: Optional[float] = None
    ):
        if tau is not None:
            tau_tens = torch.full((x.size(0), 1), tau)
            return self.outcome_network.x_to_y(x, None, tau_tens)

        return self.outcome_network.x_to_y(x, None)

    @torch.no_grad()
    def _effect_covars(
        self,
        x: torch.Tensor,
        covars: torch.Tensor,
        tau: Optional[float] = None
    ):
        n_cov_rows = covars.size(0)
        x_rep = torch.repeat_interleave(x, n_cov_rows, dim=0)
        covars = covars.repeat(x.size(0), 1)

        if tau is not None:
            y_hats = self.outcome_network.x_to_y(
                x_rep,
                covars,
                torch.full((x_rep.size(0), 1), tau)
            )
        else:
            y_hats = self.outcome_network.x_to_y(x_rep, covars)

        means = torch.tensor(
            [tens.mean() for tens in torch.split(y_hats, n_cov_rows)]
        )

        return means

    def effect_with_prediction_interval(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        alpha: Optional[float] = 0.05
    ) -> torch.Tensor:
        if (
            not getattr(self.outcome_network.hparams, "sqr", False) and
            alpha is not None
        ):
            warn(
                "Can't generate prediction interval from unsupported outcome "
                "model."
            )
            alpha = None

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if alpha is None:
            q: Optional[List[float]] = None
        else:
            q = [alpha, 0.5, 1 - alpha]

        if covars is None:
            # No covariables in the dataset or provided as arguments.
            # This will fail if covariables are necessary to go through the
            # network. The error won't be nice, so it will be better to catch
            # that. TODO
            if q is None:
                return self._effect_no_covars(x)
            else:
                return torch.hstack([
                    self._effect_no_covars(x, tau).reshape(-1, 1)
                    for tau in q
                ])

        else:
            if q is None:
                return self._effect_covars(x, covars)

            return torch.hstack([
                self._effect_covars(x, covars, tau).reshape(-1, 1)
                for tau in q
            ])

    def effect(
        self, x: torch.Tensor, covars: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Mean exposure to outcome effect at values of x."""
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        return self.effect_with_prediction_interval(x, covars, None).flatten()

    @classmethod
    def from_results(cls, dir_name: str) -> "QuantileIVEstimator":
        exposure_network = ExposureQuantileMLP.load_from_checkpoint(
            os.path.join(dir_name, "exposure_network.ckpt")
        )

        outcome_network = OutcomeMLP.load_from_checkpoint(
            os.path.join(dir_name, "outcome_network.ckpt"),
            exposure_network=exposure_network
        )

        return cls(exposure_network, outcome_network)


def main(args: argparse.Namespace) -> None:
    """Command-line interface entry-point."""
    validate_args(args)

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


def validate_args(args: argparse.Namespace) -> None:
    if args.genotypes_backend is not None and args.sample_id_col is None:
        critical(
            "When providing a genotypes dataset for the instrument, a "
            "sample id column needs to be provided using --sample-id-col "
            "so that the individuals can be matched between the genotypes "
            "and data file."
        )
        sys.exit(1)

    if args.validation_proportion < 0 or args.validation_proportion > 1:
        critical("--validation-proportion should be between 0 and 1.")
        sys.exit(1)

    if args.genotypes_backend is None and len(args.instruments) == 0:
        critical("No instruments provided.")
        sys.exit(1)


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
) -> None:
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

    trainer = pl.Trainer(
        log_every_n_steps=1,
        max_epochs=max_epochs,
        accelerator=accelerator,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="outcome_val_loss", patience=20
            ),
            pl.callbacks.ModelCheckpoint(
                filename="outcome_network",
                dirpath=output_dir,
                save_top_k=1,
                monitor="outcome_val_loss",
            ),
        ],
        logger=logger
    )
    trainer.fit(model, train_dataloader, val_dataloader)


def fit_quantile_iv(
    q: int,
    dataset: _IVDataset,
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

    if not no_plot:
        plot_exposure_model(
            exposure_network,
            val_dataset,
            output_filename=os.path.join(
                output_dir, "exposure_model_predictions.png"
            ),
        )

    train_outcome_model(
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

    outcome_network = OutcomeMLP.load_from_checkpoint(
        os.path.join(output_dir, "outcome_network.ckpt"),
        exposure_network=exposure_network,
    ).eval()  # type: ignore

    estimator = QuantileIVEstimator(exposure_network, outcome_network)

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
        ys = estimator.effect_with_prediction_interval(xs, covars, alpha=alpha)
        df = pd.DataFrame(
            torch.hstack((xs.reshape(-1, 1), ys)).numpy(),
            columns=["x", "y_do_x_lower", "y_do_x", "y_do_x_upper"]
        )
    else:
        ys = estimator.effect(xs, covars)
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
