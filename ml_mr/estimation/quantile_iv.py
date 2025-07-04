"""
Implementation of an IV method based on estimating quantiles of the exposure
distribution.
"""

import argparse
import json
import os
from typing import Iterable, List, Optional, Tuple, Any, Union, Type

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl

from ..log_utils import info
from ..utils import (
    default_validate_args,
    parse_project_and_run_name,
    initialize_meta
)
from ..utils.models import MLP, OutcomeMLPBase
from ..utils.quantiles import QuantileLossMulti
from ..utils.training import train_model, resample_dataset
from ..utils.data import IVDataset, IVDatasetWithGenotypes
from ..utils import _cat
from .core import MREstimator

# Default values definitions.
# fmt: off
DEFAULTS = {
    "n_quantiles": 5,
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
    "outcome_type": "continuous",
    "output_dir": "quantile_iv_estimate",
    "activation": "GELU",
    "outcome_dim": 3,
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
        # Previous implementation used:
        # (i + 1) / (n_quantiles + 1) for i in range(n_quantiles)]
        # However, it is more theoretically sound to use:
        self.quantiles = torch.tensor([
            (2 * k - 1) / (2 * n_quantiles) for k in range(1, n_quantiles + 1)]
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
            (2 * k - 1) / (2 * n_quantiles) for k in range(1, n_quantiles + 1)]
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
            lr=lr,
            weight_decay=weight_decay,
            loss=loss,
            _save_hyperparams=False
        )

        self.mlp.append(nn.Sigmoid())

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


class OutcomeMLP(OutcomeMLPBase):
    def __init__(
        self,
        exposure_network: QIVExposureNetType,
        input_size: int,
        hidden: Iterable[int],
        lr: float,
        weight_decay: float = 0,
        binary_outcome: bool = False,
        add_input_layer_batchnorm: bool = False,
        add_hidden_layer_batchnorm: bool = False,
        activations: Iterable[nn.Module] = [nn.GELU()],
        outcome_dim: int = 1        
    ):
        super().__init__(
            exposure_network=exposure_network,
            input_size=input_size,
            hidden=hidden,
            lr=lr,
            weight_decay=weight_decay,
            binary_outcome=binary_outcome,
            add_input_layer_batchnorm=add_input_layer_batchnorm,
            add_hidden_layer_batchnorm=add_hidden_layer_batchnorm,
            activations=activations,
        )

        self.outcome_dim = outcome_dim

        self.mlp = MLP(
            input_size=input_size,
            hidden=hidden,
            out=outcome_dim,
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

        x_hats = self.exposure_network(_cat(ivs, covars))
        n_q = x_hats.size(1)
        n = ivs.size(0)

        # Get outcome dimension from the network's output layer
        outcome_dim = self.mlp.mlp[-1].out_features

        for j in range(n_q):
            y_hat += self.mlp(_cat(x_hats[:, [j]], covars)) / n_q

        print(f"[OutcomeMLP.forward] x_hats shape: {x_hats.shape}, output y_hat shape: {y_hat.shape}")

        return y_hat


class QuantileIVEstimator(MREstimator):
    def __init__(
        self,
        exposure_network: QIVExposureNetType,
        outcome_network: OutcomeMLP,
        meta: dict,
        covars: Optional[torch.Tensor] = None,
    ):
        self.exposure_network = exposure_network
        self.outcome_network = outcome_network
        super().__init__(meta, covars)

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

        exposure_net_cls: Type[pl.LightningModule] = (
            ExposureNMQN if meta.get("nmqn", False)
            else ExposureQuantileMLP
        )

        exposure_network = exposure_net_cls.load_from_checkpoint(
            os.path.join(dir_name, "exposure_network.ckpt"),
            map_location=torch.device("cpu")
        )

        outcome_network = OutcomeMLP.load_from_checkpoint(
            os.path.join(dir_name, "outcome_network.ckpt"),
            exposure_network=exposure_network,
            map_location=torch.device("cpu")
        )

        outcome_network.eval()  # type: ignore

        return cls(exposure_network, outcome_network, meta=meta, covars=covars)


def main(args: argparse.Namespace) -> None:
    """Command-line interface entry-point."""
    default_validate_args(args)

    # Prepare train and validation datasets.
    # There is theoretically a little bit of leakage here because the histogram
    # or quantiles will be calculated including the validation dataset.
    # This should not have a big impact...
    dataset = IVDatasetWithGenotypes.from_argparse_namespace(args)
    print(f"[DEBUG] Outcome tensor shape: {dataset[0][1].shape}")
    print(f"[DEBUG] Number of outcomes parsed: {len(args.outcomes)}")


    # Automatically add the model hyperparameters.
    kwargs = {k: v for k, v in vars(args).items() if k in DEFAULTS.keys()}
    del kwargs["outcome_type"]

    fit_quantile_iv(
      dataset=dataset,
      fast=args.fast,
      wandb_project=args.wandb_project,
      nmqn=args.nmqn,
      resample=args.resample,
      binary_outcome=args.outcome_type == "binary",
      **kwargs,
  )



def train_exposure_model(
    n_quantiles: int,
    train_dataset: Dataset,
    val_dataset: Dataset,
    input_size: int,
    output_dir: str,
    hidden: List[int],
    activation: nn.Module,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    add_input_batchnorm: bool,
    max_epochs: int,
    accelerator: Optional[str] = None,
    wandb_project: Optional[str] = None,
    nmqn_penalty_lambda: Optional[float] = None
) -> Tuple[Type[QIVExposureNetType], float]:
    info("Training exposure model.")
    kwargs = {
        "n_quantiles": n_quantiles,
        "input_size": input_size,
        "hidden": hidden,
        "activations": [activation],
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
    activation: nn.Module,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    add_input_batchnorm: bool,
    max_epochs: int,
    accelerator: Optional[str] = None,
    binary_outcome: bool = False,
    wandb_project: Optional[str] = None
) -> Tuple[Any, float]:
    info("Training outcome model.")
    n_covars = train_dataset[0][3].numel()

    # Get outcome dimension from training data
    outcome_dim = train_dataset[0][1].numel()

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
    activation: str = DEFAULTS["activation"],  # type: ignore
    accelerator: str = DEFAULTS["accelerator"],  # type: ignore
    wandb_project: Optional[str] = None,
    outcome_dim: int = 1,  
) -> QuantileIVEstimator:
    if resample:
        dataset = resample_dataset(dataset)  # type: ignore
        if stage2_dataset is not None:
            stage2_dataset = resample_dataset(stage2_dataset)  # type: ignore

    activation_str = activation
    activation_cls = getattr(nn, activation_str)
    if activation_str is None:
        raise ValueError(
            f"Requested activation: '{activation_str}' is not a class in "
            f"torch.nn."
        )
    else:
        # Attempt to instantiate. We don't support parametrized activations
        # with no default values, so this may fail.
        activation_inst = activation_cls()

    # Create output directory if needed.
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Metadata dictionary that will be saved alongside the results.
    meta = dict(locals())
    meta["model"] = "quantile_iv"
    meta.update(initialize_meta())
    meta.update(dataset.exposure_descriptive_statistics())
    meta["covariable_labels"] = dataset.covariable_labels
    meta["activation"] = activation_str  # Serialize str not class.
    del meta["dataset"]  # We don't serialize the dataset.
    del meta["stage2_dataset"]
    del meta["activation_cls"]
    del meta["activation_inst"]

    covars = dataset.save_covariables(output_dir)

    # Split here into train and val.
    train_dataset, val_dataset = random_split(
        dataset, [1 - validation_proportion, validation_proportion]
    )

    # If there is a separate dataset for stage2, we split it too, otherwise
    # we reuse the stage 1 dataset.
    if stage2_dataset is not None:
        assert dataset.covariable_labels == stage2_dataset.covariable_labels
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
        activation=activation_inst,
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
        activation=activation_inst,
        learning_rate=outcome_learning_rate,
        weight_decay=outcome_weight_decay,
        batch_size=outcome_batch_size,
        add_input_batchnorm=outcome_add_input_batchnorm,
        max_epochs=outcome_max_epochs,
        accelerator=accelerator,
        binary_outcome=binary_outcome,
        wandb_project=wandb_project,
    )

    meta["outcome_val_loss"] = outcome_val_loss

    outcome_network = outcome_class.load_from_checkpoint(
        os.path.join(output_dir, "outcome_network.ckpt"),
        exposure_network=exposure_network,
    ).eval().to(torch.device("cpu"))  # type: ignore

    # Training the 2nd stage model copies the exposure net to the GPU.
    # Here, we ensure they're on the same device.
    exposure_network.to(outcome_network.device)

    estimator = QuantileIVEstimator(
        exposure_network, outcome_network, meta, covars
    )

    # Save the metadata, estimator statistics and log artifact to WandB if
    # required.
    with open(os.path.join(output_dir, "meta.json"), "wt") as f:
        try:
            json.dump(meta, f)
        except ValueError as e:
            print("Error serializing metadata.")
            print(meta)
            print(e)
            raise e

    if not fast:
        save_estimator_statistics(
        estimator,
        domain=meta["domain"],
        output_prefix=os.path.join(output_dir, "causal_estimates"),
        outcome_dim=outcome_dim,  # ← add this line
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
    outcome_dim: int = 1
):
    # Save the causal effect over the domain.
    xs = torch.linspace(domain[0], domain[1], 500).reshape(-1, 1)
    ys = estimator.avg_iv_reg_function(xs)
    print(f"[save_estimator_statistics] ys shape: {ys.shape}")

    # Prepare data dictionary for CSV
    df_data = {"x": xs.reshape(-1).numpy()}

    if ys.dim() == 1:
        # Single outcome
        df_data["y_do_x"] = ys.reshape(-1).numpy()

        # Plot single outcome
        plt.figure()
        plt.scatter(df_data["x"], df_data["y_do_x"], label="Estimated IV regression", s=3)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.savefig(f"{output_prefix}.png", dpi=600)
        plt.clf()

    else:
        # Multivariable outcome
        for i in range(ys.size(1)):
            col_name = f"y_do_x_{i+1}"
            df_data[col_name] = ys[:, i].numpy()

            # Plot each outcome separately
            plt.figure()
            plt.scatter(df_data["x"], df_data[col_name], label=f"Outcome {i+1}", s=3)
            plt.xlabel("X")
            plt.ylabel(f"Y{i+1}")
            plt.legend()
            plt.savefig(f"{output_prefix}_Y{i+1}.png", dpi=600)
            plt.clf()

    # Save all to one CSV
    df = pd.DataFrame(df_data)
    df.to_csv(f"{output_prefix}.csv", index=False)


def configure_argparse(parser) -> None:
    parser.add_argument(
        "--n-quantiles", "-q",
        type=int,
        help="Number of quantiles of the exposure distribution to estimate in "
        "the exposure model.",
        default=DEFAULTS["n_quantiles"]
    )

    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"])
    
    parser.add_argument(
        "--outcome-dim",
        type=int,
        default=DEFAULTS["outcome_dim"],
        help="Number of outcome dimensions for multivariable outcomes."
    )

    parser.add_argument(
        "--outcome-dim",
        type=int,
        default=DEFAULTS["outcome_dim"],
        help="Number of outcome dimensions for multivariable outcomes."
    )

    parser.add_argument(
        "--fast",
        help="Disable plotting and logging of causal effects.",
        action="store_true",
    )

    parser.add_argument(
        "--outcome-type",
        default=DEFAULTS["outcome_type"],
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

    # TODO add support for this for all estimators.
    parser.add_argument(
        "--activation",
        default=DEFAULTS["activation"],
        type=str,
        help="Activation function (name should be a valid class in torch.nn)",
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    configure_argparse(parser)
    args = parser.parse_args()
    main(args)

load = QuantileIVEstimator.from_results
