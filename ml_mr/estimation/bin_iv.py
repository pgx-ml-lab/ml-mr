"""
Implementation of the Histogram or Quantile based deep IV model.

This model is similar to Hartford et al. ICML (2017) model, but it treats
the exposure using a classification approach based on binned from quantiles
or a histogram.

"""

import os
import sys
import argparse
import itertools
from typing import Literal, Iterator, Iterable, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_genotypes.dataset import (
    GeneticDatasetBackend,
    PhenotypeGeneticDataset,
    BACKENDS
)
from torchmetrics import ConfusionMatrix

from .core import MREstimator
from ..utils import MLP, read_data, temperature_scale
from ..logging import info, critical


BINNING_MODES = ["histogram", "qantiles"]
BinningMode = Literal["histogram", "quantiles"]


class Binning(object):
    def __init__(
        self,
        x: torch.Tensor,
        mode: BinningMode = "quantiles",
        n_bins: int = 20
    ):
        self.x = x.to(torch.float32)
        self.n_bins = n_bins

        if mode == "histogram":
            self.bin_edges = self._bins_from_histogram()

        elif mode == "quantiles":
            self.bin_edges = self._bins_from_quantiles()

        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Use one of {BINNING_MODES}"
            )

    def _bins_from_histogram(self) -> torch.Tensor:
        self._hist = torch.histogram(self.x, self.n_bins)
        return self._hist.bin_edges

    def _bins_from_quantiles(self) -> torch.Tensor:
        min = torch.min(self.x).reshape(-1)
        max = torch.max(self.x).reshape(-1)
        quantiles = torch.quantile(
            self.x,
            q=torch.tensor([i / self.n_bins for i in range(1, self.n_bins)])
        )
        return torch.cat((min, quantiles, max))

    def get_midpoints(self) -> Iterator[float]:
        for left, right in zip(self.bin_edges, self.bin_edges[1:]):
            yield (left + right) / 2

    def values_to_bin_indices(
        self,
        x_values: torch.Tensor,
        one_hot: bool = False
    ) -> torch.Tensor:
        binned_x = torch.zeros_like(x_values, dtype=torch.long)
        bin_number = 1
        for left, right in zip(self.bin_edges[1:], self.bin_edges[2:]):
            mask = (left < x_values) & (x_values <= right)
            binned_x[mask] = bin_number
            bin_number += 1

        if one_hot:
            return F.one_hot(binned_x, self.n_bins)
        else:
            return binned_x

    def bin_index_to_midpoint_value(self, bin_index: int) -> float:
        return (
            self.bin_edges[bin_index] +
            self.bin_edges[bin_index + 1]
        ).item() / 2


class ExposureCategoricalMLP(MLP):
    def __init__(
        self,
        binning: Binning,
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
            out=binning.n_bins,
            add_input_layer_batchnorm=add_input_layer_batchnorm,
            add_hidden_layer_batchnorm=add_hidden_layer_batchnorm,
            activations=activations,
            lr=lr,
            weight_decay=lr,
            loss=F.cross_entropy
        )
        self.binning = binning
        self.temperature = nn.Parameter(
            torch.tensor(1.),
            requires_grad=False
        )

    def forward(self, xs):
        return super().forward(xs) / self.temperature

    def _step(self, batch, batch_index, log_prefix):
        x, _, ivs, covars = batch
        x_hat = self.forward(
            torch.hstack([tens for tens in (ivs, covars) if tens is not None])
        )

        loss = self.loss(x_hat, x)

        self.log(f"exposure_{log_prefix}_loss", loss)

        return loss


class OutcomeWithBinsMLP(MLP):
    def __init__(
        self,
        exposure_network: pl.LightningModule,
        input_size: int,
        hidden: Iterable[int],
        lr: float,
        weight_decay: float = 0,
        add_input_layer_batchnorm: bool = False,
        add_hidden_layer_batchnorm: bool = False,
        activations: Iterable[nn.Module] = [nn.LeakyReLU()],
    ):
        super().__init__(
            input_size=input_size,
            hidden=hidden,
            out=1,
            add_input_layer_batchnorm=add_input_layer_batchnorm,
            add_hidden_layer_batchnorm=add_hidden_layer_batchnorm,
            activations=activations,
            lr=lr,
            weight_decay=weight_decay,
            loss=F.mse_loss,
            _save_hyperparams=False
        )
        self.exposure_network = exposure_network
        self.save_hyperparameters(ignore=["exposure_network"])

    def _step(self, batch, batch_index, log_prefix):
        _, y, ivs, covars = batch
        y_hat = self.forward(ivs, covars)
        loss = self.loss(y_hat, y)

        self.log(f"outcome_{log_prefix}_loss", loss)

        return loss

    def x_to_y(self, x_one_hot: torch.Tensor, covars: Optional[torch.Tensor]):
        if covars is not None:
            x = torch.hstack((x_one_hot, covars))
        else:
            x = x_one_hot

        return self.mlp(x)

    def forward(self, ivs, covars):
        # x is the input to the exposure model.
        mb = ivs.shape[0]
        exposure_net_xs = torch.hstack(
            [tens for tens in (ivs, covars) if tens is not None]
        )

        with torch.no_grad():
            exposure_probs = F.softmax(
                self.exposure_network.forward(exposure_net_xs),
                dim=1
            )

        y_hats = torch.zeros(mb, 1, device=self.device)
        for i in range(exposure_probs.shape[1]):
            weights = exposure_probs[:, [i]]

            cur_one_hot = F.one_hot(
                torch.tensor([i], device=self.device),
                self.exposure_network.binning.n_bins
            ).repeat(mb, 1)

            pred = self.x_to_y(cur_one_hot, covars)

            y_hats += weights * pred

        return y_hats


class BinIVEstimator(MREstimator):
    def __init__(
        self,
        exposure_network: ExposureCategoricalMLP,
        outcome_network: OutcomeWithBinsMLP,
        binning: Binning,
    ):
        self.exposure_network = exposure_network
        self.outcome_network = outcome_network
        self.binning = binning

    def effect(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Mean exposure to outcome effect at values of x."""
        # Get the bin for the provided xs.
        bins = self.binning.values_to_bin_indices(x, one_hot=True)\
            .to(torch.float32)

        if covars is None:
            # No covariables in the dataset or provided as arguments.
            # This will fail if covariables are necessary to go through the
            # network. The error won't be nice, so it will be better to catch
            # that. TODO
            with torch.no_grad():
                ys = self.outcome_network.x_to_y(bins, None)

            return ys.reshape(-1)

        else:
            x_one_hot = torch.repeat_interleave(
                bins, covars.size(0), dim=0
            )
            covars = covars.repeat(bins.size(0), 1)
            with torch.no_grad():
                y_hats = self.outcome_network.x_to_y(x_one_hot, covars)

            raise RuntimeError("Fixme, aggregate over right dimension.")
            return torch.mean(y_hats)


def main(args: argparse.Namespace) -> None:
    """Command-line interface entry-point."""
    validate_args(args)

    # Create output directory if needed.
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Read the data.
    expected_cols = [args.exposure, args.outcome]
    expected_cols += args.instruments
    expected_cols += args.covariables

    data = read_data(args.data, args.sep, expected_cols)

    # Read genetic data if needed.
    if args.genotypes_backend is not None:
        backend_class = BACKENDS.get(
            args.genotypes_backend_type,
            GeneticDatasetBackend
        )
        backend: Optional[GeneticDatasetBackend] = (
            backend_class.load(args.genotypes_backend)
        )

    else:
        backend = None

    # Prepare train and validation datasets.
    # There is theoretically a little bit of leakage here because the histogram
    # or quantiles will be calculated including the validation dataset.
    # This should not have a big impact...
    dataset, binning = get_dataset_and_binning(
        args,
        data,
        backend,
        exposure=args.exposure,
        outcome=args.outcome,
        covariables=args.covariables,
        instruments=args.instruments
    )

    save_covariables(
        dataset,
        os.path.join(args.output_dir, "covariables.pt")
    )

    # Split here into train and val.
    train_dataset, val_dataset = random_split(
        dataset,
        [1 - args.validation_proportion, args.validation_proportion]
    )

    train_exposure_model(
        args, train_dataset, val_dataset, binning, backend
    )

    exposure_network = ExposureCategoricalMLP\
        .load_from_checkpoint(
            os.path.join(args.output_dir, "exposure_network.ckpt")
        ).eval()  # type: ignore

    # Apply temperature scaling to the exposure model to improve calibration.
    def _batch_fwd(model, batch):
        _, _, ivs, covars = batch
        return model.forward(torch.hstack((ivs, covars)))

    temperature_scale(
        exposure_network,
        val_dataset,
        batch_forward=_batch_fwd,
        batch_target=lambda batch: batch[0]
    )

    info(f"Temperature scaling parameter after tuning: "
         f"{exposure_network.temperature.item()}")

    if not args.no_plot:
        plot_exposure_model(
            binning, exposure_network, val_dataset,
            output_filename=os.path.join(
                args.output_dir, "exposure_model_confusion_matrix.png"
            )
        )

    train_outcome_model(
        args, train_dataset, val_dataset, binning, exposure_network
    )

    outcome_network = OutcomeWithBinsMLP\
        .load_from_checkpoint(
            os.path.join(args.output_dir, "outcome_network.ckpt"),
            exposure_network=exposure_network
        ).eval()  # type: ignore

    estimator = BinIVEstimator(
        exposure_network,
        outcome_network,
        binning
    )

    save_estimator_statistics(
        estimator,
        output_prefix=os.path.join(args.output_dir, "causal_estimates")
    )


@torch.no_grad()
def plot_exposure_model(
    binning: Binning,
    exposure_network: ExposureCategoricalMLP,
    val_dataset: Dataset,
    output_filename: str = "exposure_model_confusion_matrix.png"
):
    assert hasattr(val_dataset, "__len__")
    dataloader = DataLoader(val_dataset, batch_size=len(val_dataset))
    actual_bin, _, z, covariables = next(iter(dataloader))

    input = torch.hstack(
        [tens for tens in (z, covariables) if tens is not None]
    )
    predicted_bin = torch.argmax(
        F.softmax(exposure_network.forward(input), dim=1),
        dim=1
    )

    info("Exposure model accuracy: {}".format(
        torch.mean((predicted_bin == actual_bin).to(torch.float32))
    ))

    confusion = ConfusionMatrix(
        task="multiclass",
        num_classes=binning.n_bins,
        normalize="true"
    )
    confusion_matrix = confusion(predicted_bin, actual_bin)  # type: ignore

    plt.figure(figsize=(10, 10))
    plt.matshow(confusion_matrix)
    plt.xlabel("Predicted bin")
    plt.ylabel("True bin")
    plt.colorbar()
    plt.savefig(output_filename, dpi=400)
    plt.clf()
    plt.close()


def save_estimator_statistics(estimator: BinIVEstimator,
                              output_prefix: str = "causal_estimates"):
    # Save the causal effect at every bin midpoint.
    xs = torch.tensor(list(estimator.binning.get_midpoints()))
    ys = estimator.effect(xs)
    plt.figure()
    plt.scatter(xs.numpy(), ys.numpy())
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f"{output_prefix}.png", dpi=600)
    plt.clf()

    df = pd.DataFrame({"x": xs, "y_do_x": ys})
    df.to_csv(f"{output_prefix}.csv", index=False)


def save_covariables(dataset: Dataset, output_filename: str):
    dl = DataLoader(dataset, batch_size=len(dataset))  # type: ignore
    covars = next(iter(dl))[3]
    if covars.shape[1] == 0:
        return None

    torch.save(covars, output_filename)


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
        critical(
            "--validation-proportion should be between 0 and 1."
        )
        sys.exit(1)


def get_dataset_and_binning(
    args: argparse.Namespace,
    data: pd.DataFrame,
    backend: Optional[GeneticDatasetBackend],
    exposure: str,
    outcome: str,
    covariables: Iterable[str],
    instruments: Iterable[str]
) -> Tuple[Dataset, Binning]:
    # If we have a backend, we need to add the genotypes to the exogenous
    # variable.
    if backend is not None:
        genetic_dataset = PhenotypeGeneticDataset(
            backend,
            data,
            args.sample_id_col,
            # We access columns manually later, so for now we ask for
            # everything we need through the exogenous columns.
            exogenous_columns=itertools.chain(
                instruments, covariables, [exposure, outcome]
            )
        )
        return _dataset_from_genetic_dataset(
            args,
            genetic_dataset,
            exposure,
            outcome,
            covariables,
            instruments
        )

    else:
        exposure_tens = torch.from_numpy(data[exposure].values)
        outcome_tens = torch.from_numpy(data[[outcome]].values)\
            .to(torch.float32)

        binning = Binning(
            exposure_tens,
            mode="histogram" if args.histogram else "quantiles",
            n_bins=args.n_bins
        )

        bins = binning.values_to_bin_indices(exposure_tens)
        instruments = torch.from_numpy(data[instruments].values)\
            .to(torch.float32)
        covariables = torch.from_numpy(data[covariables].values)\
            .to(torch.float32)

        class _Dataset(Dataset):
            def __getitem__(self, index):
                exposure = bins[index]
                outcome = outcome_tens[index]
                z = instruments[index]
                cur_covars = covariables[index]
                return exposure, outcome, z, cur_covars

            def __len__(self) -> int:
                return data.shape[0]

        return _Dataset(), binning


def _dataset_from_genetic_dataset(
    args: argparse.Namespace,
    genetic_dataset: PhenotypeGeneticDataset,
    exposure: str,
    outcome: str,
    covariables: Iterable[str],
    instruments: Iterable[str],
) -> Tuple[Dataset, Binning]:
    # Create binning.
    binning = Binning(
        genetic_dataset.exog[exposure],
        mode="histogram" if args.histogram else "quantiles",
        n_bins=args.n_bins
    )

    instruments_set = set(instruments)
    covariables_set = set(covariables)

    instrument_idx = []
    covariable_idx = []
    exposure_idx = None
    outcome_idx = None
    for idx, col in enumerate(genetic_dataset.exogenous_columns):
        if col in instruments_set:
            instrument_idx.append(idx)
        if col in covariables_set:
            covariable_idx.append(idx)
        if col == exposure:
            assert exposure_idx is None
            exposure_idx = idx
        if col == outcome:
            assert outcome_idx is None
            outcome_idx = idx

    instrument_idx_tens = torch.tensor(instrument_idx)
    covariable_idx_tens = torch.tensor(covariable_idx)

    class _Dataset(Dataset):
        def __getitem__(self, index: int):
            # Get the binned exposure.
            cur = genetic_dataset[index]
            bin_exposure = binning.values_to_bin_indices(
                cur.exogenous[:, exposure_idx],
                one_hot=True
            )

            outcome = cur.exogenous[:, [outcome_idx]]
            instruments = cur.dosage
            covars = None

            if covariable_idx:
                covars = cur.exogenous[:, covariable_idx_tens]

            if instrument_idx:
                instruments = torch.hstack((
                    instruments,
                    cur.exogenous[:, instrument_idx_tens]
                ))

            return bin_exposure, outcome, instruments, covars

        def __len__(self):
            return len(genetic_dataset)

    return _Dataset(), binning


def _n_exog(
    backend: Optional[GeneticDatasetBackend],
    args: argparse.Namespace
) -> int:
    n_exog = 0
    if backend is not None:
        n_exog += backend.get_n_variants()

    n_exog += len(args.covariables)
    n_exog += len(args.instruments)
    return n_exog


def train_exposure_model(
    args: argparse.Namespace,
    train_dataset: Dataset,
    val_dataset: Dataset,
    binning: Binning,
    backend: Optional[GeneticDatasetBackend],
) -> None:
    info("Training exposure model.")
    model = ExposureCategoricalMLP(
        binning=binning,
        input_size=_n_exog(backend, args),
        hidden=args.exposure_hidden,
        lr=args.exposure_learning_rate,
        weight_decay=args.exposure_weight_decay,
        add_input_layer_batchnorm=True,
        add_hidden_layer_batchnorm=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.exposure_batch_size,
        shuffle=True,
        num_workers=4
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),  # type: ignore
        num_workers=4
    )

    # Remove checkpoint if exists.
    full_filename = os.path.join(
        args.output_dir, "exposure_network.ckpt"
    )
    if os.path.isfile(full_filename):
        info(f"Removing file '{full_filename}'.")
        os.remove(full_filename)

    trainer = pl.Trainer(
        log_every_n_steps=1,
        max_epochs=args.exposure_max_epochs,
        accelerator=args.accelerator,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="exposure_val_loss", patience=20
            ),
            pl.callbacks.ModelCheckpoint(
                filename="exposure_network",
                dirpath=args.output_dir,
                save_top_k=1,
                monitor="exposure_val_loss"
            )
        ]
    )
    trainer.fit(model, train_dataloader, val_dataloader)


def train_outcome_model(
    args: argparse.Namespace,
    train_dataset: Dataset,
    val_dataset: Dataset,
    binning: Binning,
    exposure_network: ExposureCategoricalMLP
) -> None:
    info("Training outcome model.")
    model = OutcomeWithBinsMLP(
        exposure_network=exposure_network,
        input_size=binning.n_bins + len(args.covariables),
        lr=args.outcome_learning_rate,
        weight_decay=args.outcome_weight_decay,
        hidden=args.outcome_hidden
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.outcome_batch_size,
        shuffle=True,
        num_workers=4
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),  # type: ignore
        num_workers=4
    )

    # Remove checkpoint if exists.
    full_filename = os.path.join(args.output_dir, "outcome_network.ckpt")
    if os.path.isfile(full_filename):
        info(f"Removing file '{full_filename}'.")
        os.remove(full_filename)

    trainer = pl.Trainer(
        log_every_n_steps=1,
        max_epochs=args.outcome_max_epochs,
        accelerator=args.accelerator,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="outcome_val_loss"),
            pl.callbacks.ModelCheckpoint(
                filename="outcome_network",
                dirpath=args.output_dir,
                save_top_k=1,
                monitor="outcome_val_loss"
            )
        ]
    )
    trainer.fit(model, train_dataloader, val_dataloader)


def configure_argparse(parser) -> None:
    parser.add_argument(
        "--n-bins",
        type=int,
        help="Number of bins used for density estimation in the "
             "exposure model.",
        required=True
    )

    parser.add_argument(
        "--histogram",
        action="store_true",
        help="By default, we use quantiles for density estimation. Using this "
             "option, we will use evenly spaced bins (histogram) instead."
    )

    parser.add_argument(
        "--output-dir", default="binning_iv_estimate"
    )

    parser.add_argument(
        "--genotypes-backend",
        help=(
            "Pickle containing a pytorch-genotypes backend. This can be "
            "created from various genetic data formats using the "
            "'pt-geno-create-backend' command line utility provided by "
            "pytorch genotypes."
        ),
        type=str
    )

    parser.add_argument(
        "--genotypes-backend-type",
        help=(
            "Pickle containing a pytorch-genotypes backend. This can be "
            "created from various genetic data formats using the "
            "'pt-geno-create-backend' command line utility provided by "
            "pytorch genotypes."
        ),
        type=str
    )

    parser.add_argument(
        "--no-plot",
        help="Disable plotting of diagnostics.",
        action="store_true"
    )

    parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to a data file."
    )

    parser.add_argument(
        "--sep",
        default="\t",
        help="Separator (column delimiter) for the data file."
    )

    parser.add_argument(
        "--instruments", "-z",
        nargs="*",
        default=[],
        help="The instrument (Z or G) in the case where we're not using "
             "genotypes provided through --genotypes. Multiple values can "
             "be provided for multiple instruments.\n"
             "This should be column(s) in the data file."
    )

    parser.add_argument(
        "--covariables",
        nargs="*",
        default=[],
        help="Variables which will be included in both stages."
             "This should be column(s) in the data file."
    )

    parser.add_argument(
        "--exposure", "-x",
        help="The exposure (X). This should be a column name in the data "
             "file.",
        required=True,
        type=str
    )

    parser.add_argument(
        "--outcome", "-y",
        help="The outcome (Y). This should be a column name in the data "
             "file.",
        required=True,
        type=str
    )

    parser.add_argument(
        "--outcome-type",
        default="continuous",
        choices=["continuous", "binary"],
        help="Variable type for the outcome (binary vs continuous)."
    )

    parser.add_argument(
        "--sample-id-col",
        default="sample_id",
        help="Column that contains the sample id. This is mandatory if "
             "genotypes are provided to enable joining."
    )

    parser.add_argument(
        "--validation-proportion",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--accelerator",
        default=(
            "gpu" if (
                torch.cuda.is_available() and
                torch.cuda.device_count() > 0
             ) else "cpu"
        ),
        help="Accelerator (e.g. gpu, cpu, mps) use to train the model. This "
             "will be passed to Pytorch Lightning."
    )

    MLP.add_argparse_parameters(
        parser, "exposure-", "Exposure Model Parameters",
        defaults={"hidden": [128, 64], "batch-size": 5000}
    )

    MLP.add_argparse_parameters(
        parser, "outcome-", "Outcome Model Parameters",
        defaults={"hidden": [16, 8]}
    )
