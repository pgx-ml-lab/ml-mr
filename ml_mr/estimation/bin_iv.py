"""
Implementation of the Histogram or Quantile based deep IV model.

This model is similar to Hartford et al. ICML (2017) model, but it treats
the exposure using a classification approach based on binned from quantiles
or a histogram.

"""

import sys
import argparse
from typing import Literal, Iterator, Iterable, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_genotypes.dataset import (
    GeneticDatasetBackend,
    PhenotypeGeneticDataset
)

from ..utils import MLP, read_data
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

    def _step(self, batch, batch_index, log_prefix):
        x, covars, y = batch

        # X is in the original scale, we'll convert to one hot bin vectors.
        x = self.binning.values_to_bin_indices(x, one_hot=True)
        if covars is not None:
            x = torch.hstack((x, covars))

        y_hat = self.mlp(x)
        loss = self.loss(y_hat, y)

        self.log(f"{log_prefix}_loss", loss)

        return loss


def main(args):
    """Command-line interface entry-point.

    We should provide either --genotype or --instrument and a column name from
    the data file.

    ml-mr estimation --algorithm bin_iv \
        --n-bins 20 \
        --output-prefix my_bin_iv \
        --genotypes my_geno \
        --genotypes-format plink \
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
    expected_cols += args.instrument
    expected_cols += args.covariables

    data = read_data(args.data, args.sep, expected_cols)

    # TODO serialize and maybe do basic model evaluation.
    train_exposure_model(args, data)


def validate_args(args: argparse.Namespace):
    if args.genotypes is not None and args.sample_id_col is None:
        critical(
            "When providing a genotypes dataset for the instrument, a "
            "sample id column needs to be provided using --sample-id-col "
            "so that the individuals can be matched between the genotypes "
            "and data file."
        )
        sys.exit(1)


def get_dataset_and_binning(
    args: argparse.Namespace,
    data: pd.DataFrame,
    backend: Optional[GeneticDatasetBackend],
    endogenous_column: str,
    exogenous_columns: Iterable[str]
) -> Tuple[Dataset, Binning]:
    # If we have a backend, we need to add the genotypes to the exogenous
    # variable.
    if backend is not None:
        genetic_dataset = PhenotypeGeneticDataset(
            backend,
            data,
            args.sample_id_col,
            endogenous_columns=[endogenous_column],
            exogenous_columns=exogenous_columns
        )
        return _dataset_from_genetic_dataset(
            args,
            genetic_dataset,
            endogenous_column,
            exogenous_columns
        )

    else:
        exposure_tens = torch.from_numpy(data[endogenous_column].values)
        binning = Binning(
            exposure_tens,
            mode="histogram" if args.histogram else "quantiles",
            n_bins=args.n_bins
        )

        bins = binning.values_to_bin_indices(exposure_tens, one_hot=True)
        exogenous = torch.from_numpy(data[exogenous_columns].values)

        class _Dataset(Dataset):
            def __getitem__(self, index):
                return bins[index], exogenous[index]

            def __len__(self) -> int:
                return data.shape[0]

        return _Dataset(), binning


def _dataset_from_genetic_dataset(
    args: argparse.Namespace,
    genetic_dataset: PhenotypeGeneticDataset,
    endogenous_column: str,
    exogenous_columns: Iterable[str]
) -> Tuple[Dataset, Binning]:
    # Create binning.
    binning = Binning(
        genetic_dataset.exog[endogenous_column],
        mode="histogram" if args.histogram else "quantiles",
        n_bins=args.n_bins
    )

    class _Dataset(Dataset):
        def __getitem__(self, index: int):
            # Get the binned exposure.
            cur = genetic_dataset[index]
            y = binning.values_to_bin_indices(cur.endogenous, one_hot=True)

            if hasattr(cur, "exogenous"):
                x = torch.hstack((cur.dosage, cur.exogenous))
            else:
                x = cur.dosage

            return y, x

        def __len__(self):
            return len(genetic_dataset)

    return _Dataset(), binning


def train_exposure_model(
    args: argparse.Namespace,
    data: pd.DataFrame,
    backend: Optional[GeneticDatasetBackend],
) -> ExposureCategoricalMLP:
    info("Training exposure model.")

    dataset, binning = get_dataset_and_binning(
        args,
        data,
        backend,
        endogenous_column=args.exposure,
        exogenous_columns=args.covariables + args.instrument
    )

    print(dataset)
    print(dataset[0])

    model = ExposureCategoricalMLP(
        binning,
        input_size=dataset[0][1].shape[1],
        hidden=[32, 16],
        lr=args.exposure_learning_rate,
        weight_decay=args.exposure_weight_decay,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.exposure_batch_size,
        shuffle=True
    )

    trainer = pl.Trainer()
    trainer.fit(model, dataloader)

    return model


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
        "--genotypes",
        help="Filename containing the genotypes (or plink prefix).",
        type=str
    )

    parser.add_argument(
        "--genotypes-format",
        help="File format for the genotypes file.",
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
        "--instrument", "-z",
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
