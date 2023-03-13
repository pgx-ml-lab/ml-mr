
import argparse
import os
from typing import Iterable, List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import Logger
from torch.utils.data import DataLoader, Dataset, random_split

from ..logging import info
from ..utils import (MixtureDensityNetwork, default_validate_args,
                     parse_project_and_run_name)
from ..utils.nn import MLP, OutcomeMLPBase
from .core import (IVDataset, IVDatasetWithGenotypes, MREstimator,
                   SupervisedLearningWrapper)

# Default values definitions.
# fmt: off
DEFAULTS = {
    "n_gaussians": 5,
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
    "output_dir": "deep_iv_estimate",
}
# fmt: on


class DeepIVEstimator(MREstimator):
    pass


class OutcomeMLP(OutcomeMLPBase):
    def __init__(
        self,
        exposure_network: MixtureDensityNetwork,
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
            # TODO FIXME
            x_hat = self.exposure_network.forward(exposure_net_xs)


def main(args: argparse.Namespace) -> None:
    default_validate_args(args)

    dataset = IVDatasetWithGenotypes.from_argparse_namespace(args)

    # Automatically add the model hyperparameters.
    kwargs = {k: v for k, v in vars(args).items() if k in DEFAULTS.keys()}

    fit_deep_iv(
        dataset=dataset,
        no_plot=args.no_plot,
        sqr=args.sqr,
        wandb_project=args.wandb_project,
        **kwargs,
    )


def train_exposure_model(
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
    n_gaussians: int = 5,
    accelerator: Optional[str] = None,
    wandb_project: Optional[str] = None
) -> None:
    model = MixtureDensityNetwork(
        input_size=input_size,
        hidden=hidden,
        n_components=n_gaussians,
        lr=learning_rate,
        weight_decay=weight_decay,
        add_input_layer_batchnorm=add_input_batchnorm,
        add_hidden_layer_batchnorm=True
    )

    # Wrap datasets for supervised learning.
    train_dataset = SupervisedLearningWrapper(train_dataset)  # type: ignore
    val_dataset = SupervisedLearningWrapper(val_dataset)  # type: ignore

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    assert hasattr(val_dataset, "__len__")
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset))

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
                monitor="mdn_val_nll", patience=20
            ),
            pl.callbacks.ModelCheckpoint(
                filename="exposure_network",
                dirpath=output_dir,
                save_top_k=1,
                monitor="mdn_val_nll",
            ),
        ],
        logger=logger
    )
    trainer.fit(model, train_dataloader, val_dataloader)


def fit_deep_iv(
    dataset: IVDataset,
    n_gaussians: int = DEFAULTS["n_gaussians"],  # type: ignore
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
) -> DeepIVEstimator:
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
        n_gaussians=n_gaussians,
        accelerator=accelerator,
        wandb_project=wandb_project
    )

    exposure_network = MixtureDensityNetwork.load_from_checkpoint(
        os.path.join(output_dir, "exposure_network.ckpt")
    ).eval()  # type: ignore

    return DeepIVEstimator()


def configure_argparse(parser) -> None:
    parser.add_argument(
        "--n-gaussians",
        type=int,
        help="Number of gaussians used for the mixture density network.",
        default=DEFAULTS["n_gaussians"]
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


estimate = fit_deep_iv
load = DeepIVEstimator.from_results
