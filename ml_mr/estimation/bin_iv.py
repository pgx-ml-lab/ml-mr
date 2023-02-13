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
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_genotypes.dataset import (
    GeneticDatasetBackend,
    PhenotypeGeneticDataset,
    BACKENDS
)

from ..utils import MLP, build_mlp, read_data
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
        self.x = x
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
        activations: Iterable[nn.Module] = [nn.LeakyReLU()]
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
            loss=F.mse_loss,  # FIXME
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

            pred = self.mlp(
                torch.hstack(
                    [tens for tens in (cur_one_hot, covars)
                     if tens is not None]
                )
            )

            y_hats += weights * pred

        return y_hats


def main(args: argparse.Namespace) -> None:
    """Command-line interface entry-point.

    We should provide either --genotype or --instrument and a column name from
    the data file.

    ml-mr estimation --algorithm bin_iv \
        --n-bins 20 \
        --output-prefix my_bin_iv \
        --genotypes-backend my_geno.pkl \
        --genotypes-backend-type zarr \
        --data my_pheno.csv \
        --exposure ldl_c_std \
        --outcome crp \
        --outcome-type continuous \
        --accelerator gpu \
        --exposure-max-epochs 200 \
        --exposure-batch-size 2000 \
        --exposure-optimizer adam \
        --exposure-learning-rate 9e-4 \
        --exposure-weight-decay 0 \
        --outcome-max-epochs 200 \
        --outcome-batch-size 2000 \
        --outcome-optimizer adam \
        --outcome-learning-rate 9e-4 \
        --outcome-weight-decay 0 \

    """
    validate_args(args)

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

    # Split here into train and val.
    train_dataset, val_dataset = random_split(
        dataset,
        [1 - args.validation_proportion, args.validation_proportion]
    )

    train_exposure_model(
        args, train_dataset, val_dataset, binning, backend
    )

    exposure_network = ExposureCategoricalMLP\
        .load_from_checkpoint("exposure_network.ckpt")

    train_outcome_model(
        args, train_dataset, val_dataset, binning, backend, exposure_network
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
        exposure_tens = torch.from_numpy(data[exposure].values)\
            .to(torch.float32)

        outcome_tens = torch.from_numpy(data[[outcome]].values)\
            .to(torch.float32)

        binning = Binning(
            exposure_tens,
            mode="histogram" if args.histogram else "quantiles",
            n_bins=args.n_bins
        )

        bins = binning.values_to_bin_indices(exposure_tens, one_hot=True)
        instruments = torch.from_numpy(data[instruments].values)\
            .to(torch.float32)
        covariables = torch.from_numpy(data[covariables].values)\
            .to(torch.float32)

        class _Dataset(Dataset):
            def __getitem__(self, index):
                exposure = bins[index].to(torch.float32)
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
        hidden=[32, 16],
        lr=args.exposure_learning_rate,
        weight_decay=args.exposure_weight_decay,
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
    )

    # Remove checkpoint if exists.
    if os.path.isfile("exposure_network.ckpt"):
        os.remove("exposure_network.ckpt")

    trainer = pl.Trainer(
        max_epochs=args.exposure_max_epochs,
        accelerator=args.accelerator,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="exposure_val_loss"),
            pl.callbacks.ModelCheckpoint(
                filename="exposure_network",
                dirpath=".",
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
    backend: GeneticDatasetBackend,
    exposure_network: ExposureCategoricalMLP
) -> None:
    info("Training outcome model.")
    model = OutcomeWithBinsMLP(
        exposure_network=exposure_network,
        input_size=binning.n_bins + len(args.covariables),
        lr=args.outcome_learning_rate,
        weight_decay=args.outcome_weight_decay,
        hidden=[16, 8],  # FIXME Parametrize.
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
    )

    # Remove checkpoint if exists.
    if os.path.isfile("outcome_network.ckpt"):
        os.remove("outcome_network.ckpt")

    trainer = pl.Trainer(
        max_epochs=args.outcome_max_epochs,
        accelerator=args.accelerator,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="outcome_val_loss"),
            pl.callbacks.ModelCheckpoint(
                filename="outcome_network",
                dirpath=".",
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
        "--output-prefix", default="binning_iv_estimate"
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
        parser, "exposure-", "Exposure Model Parameters"
    )

    MLP.add_argparse_parameters(
        parser, "outcome-", "Outcome Model Parameters"
    )
