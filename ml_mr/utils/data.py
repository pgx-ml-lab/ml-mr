from typing import Tuple, Dict, Any, Optional, Iterable, List
import argparse
import itertools
import collections
import os

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

from ..log_utils import warn

try:
    from pytorch_genotypes.dataset import (BACKENDS, GeneticDatasetBackend,
                                           PhenotypeGeneticDataset)
    PT_GENO_AVAIL = True
except ImportError:
    PT_GENO_AVAIL = False


IVDatasetBatch = Tuple[
    torch.Tensor,  # exposure
    torch.Tensor,  # outcome
    torch.Tensor,  # IVs
    torch.Tensor   # Covariables
]


class IVDataset(Dataset):
    """Dataset class for IV analysis.

    The batches contain exposure, outcome, IVs and covariables.

    """
    def __init__(
        self,
        exposure: torch.Tensor,
        outcome: torch.Tensor,
        ivs: torch.Tensor,
        covariables: torch.Tensor = torch.Tensor(),
        covariable_labels: Optional[Iterable[str]] = None,
        sampling_weights: Optional[torch.Tensor] = None,
        outcome_dim: Optional[int] = None
    ):
        self.exposure = exposure.reshape(-1, 1)
        self.outcome = outcome

        self.ivs = ivs
        self.covariables = covariables
        self.sampling_weights = sampling_weights

        if covariable_labels is None:
            self.covariable_labels = None
        else:
            self.covariable_labels = list(covariable_labels)

        n = self.ivs.size(0)
        assert n != 0
        for tens in (exposure, outcome, covariables):
            assert tens.numel() == 0 or tens.size(0) == n

    def __getitem__(self, index: int) -> IVDatasetBatch:
        if self.exposure.numel() > 0:
            exposure = self.exposure[index]
        else:
            exposure = torch.Tensor()

        if self.outcome.numel() > 0:
            outcome = self.outcome[index]
        else:
            outcome = torch.Tensor()

        ivs = self.ivs[index]
        covars = self.covariables[index]

        return exposure, outcome, ivs, covars

    def __len__(self) -> int:
        return self.ivs.size(0)

    def to_dataframe(self) -> Tuple[pd.DataFrame, dict]:
        cols = collections.OrderedDict()
        stack = []

        contents = [
            (self.exposure, "exposure"),
            (self.outcome, "outcome"),
            (self.ivs, "ivs"),
            (self.covariables, "covariables")
        ]

        if self.sampling_weights is not None:
            contents.append((self.sampling_weights, "sampling_weights"))

        def add(variable, name):
            mat = variable.numpy()
            assert mat.ndim == 2
            stack.append(mat)
            cols[name] = []
            for j in range(mat.shape[1]):
                cols[name].append(
                    name + (f"_{j+1}" if mat.shape[1] > 1 else "")
                )

        for variable, name in contents:
            if variable.shape[0] > 0:
                add(variable, name)

        if self.covariable_labels is not None:
            cols["covariables"] = self.covariable_labels

        df = pd.DataFrame(
            np.hstack(stack),
            columns=itertools.chain(*cols.values())
        )

        return df, cols

    def exposure_descriptive_statistics(self) -> Dict[str, Any]:
        x = self.exposure.numpy()

        min = np.min(x).item()
        max = np.max(x).item()

        return {
            "domain": [min, max],
            "exposure_95_percentile": np.percentile(x, [2.5, 97.5]).tolist(),
            "exposure_99_percentile": np.percentile(x, [0.5, 99.5]).tolist()
        }

    def n_outcomes(self) -> int:
        """Counts the number of outcomes."""
        _, y, _, _ = self[0]
        return y.numel()

    def n_exposures(self) -> int:
        """Counts the number of exposures."""
        x, _, _, _ = self[0]
        return x.numel()

    def n_instruments(self) -> int:
        """Counts the number of IVs."""
        _, _, ivs, _ = self[0]
        return ivs.numel()

    def n_covars(self) -> int:
        """Counts the number of covariables."""
        _, _, _, covars = self[0]
        return covars.numel()

    def n_exog(self) -> int:
        """Counts the number of exogenous variables (IVs + covariables)."""
        return self.n_instruments() + self.n_covars()

    def save_covariables(
        self,
        output_directory: str
    ) -> Optional[torch.Tensor]:
        """Saves the covars to disk and returns them."""
        output_filename = os.path.join(output_directory, "covariables.pt")

        if (
            isinstance(self.covariables, torch.Tensor) and
            self.covariables.numel() > 0
        ):
            torch.save(self.covariables, output_filename)
            return self.covariables

        return None

    @staticmethod
    def from_dataframe(
        dataframe: pd.DataFrame,
        exposure_col: Optional[str],
        outcome_col: Optional[List[str]],
        iv_cols: Iterable[str],
        covariable_cols: Iterable[str] = [],
        sampling_weights_col: Optional[str] = None,
    ) -> "IVDataset":
        # We'll do complete case analysis if the user provides a df with NAs.

        # Convert single string exposure to list of one element
        exposure_cols = [exposure_col] if isinstance(exposure_col, str) else exposure_col
        outcome_cols = outcome_col if isinstance(outcome_col, list) else [outcome_col]
        covariable_cols = covariable_cols or []
        iv_cols = iv_cols or []
        sampling_weights_col = [sampling_weights_col] if sampling_weights_col else []

        keep_cols = list(itertools.chain(
            exposure_cols, outcome_cols, iv_cols, covariable_cols, sampling_weights_col
        ))

        dataframe = dataframe[keep_cols]
        if dataframe.isna().values.any():
            n_before = dataframe.shape[0]
            dataframe = dataframe.dropna()
            n_after = dataframe.shape[0]
            n = n_before - n_after
            warn(
                f"Doing complete case analysis. Dropped {n} rows with missing "
                f"values from the input data."
            )

        # Keep 2D shape even if single column
        if exposure_cols:
            exposure = torch.from_numpy(dataframe[exposure_cols].values).float()
        else:
            exposure = torch.Tensor()

        if outcome_cols:
            outcome = torch.from_numpy(dataframe[outcome_cols].values).float()
        else:
            outcome = torch.Tensor()

        if sampling_weights_col:
            sampling_weights = torch.from_numpy(
                dataframe[sampling_weights_col].values
            ).float()
        else:
            sampling_weights = None

        ivs = torch.from_numpy(dataframe[iv_cols].values).float()
        covars = torch.from_numpy(dataframe[covariable_cols].values).float()

        return IVDataset(
            exposure, outcome, ivs, covars, covariable_labels=covariable_cols,
            sampling_weights=sampling_weights
        )

    @staticmethod
    def from_json_configuration(configuration) -> "IVDataset":
        allowed_keys = {
            "filename", "sep", "exposure", "outcome", "instruments",
            "covariables", "sampling_weights"
        }

        bad_keys = set(configuration.keys()) - allowed_keys
        if bad_keys:
            raise ValueError(
                f"Invalid dataset configuration parameter(s): {bad_keys}"
            )

        data = pd.read_csv(
            configuration["filename"],
            sep=configuration.get("sep", "\t")
        )

        return IVDataset.from_dataframe(
            data,
            exposure_col=configuration["exposure"],
            outcome_col=configuration["outcome"],
            iv_cols=configuration["instruments"],
            covariable_cols=configuration.get("covariables", []),
            sampling_weights_col=configuration.get("sampling_weights", None)
        )

    @staticmethod
    def from_argparse_namespace(args: argparse.Namespace) -> "IVDataset":
        data = pd.read_csv(
            args.data, sep=args.sep
        )
        return IVDataset.from_dataframe(
            data,
            exposure_col=args.exposure,
            outcome_col=args.outcomes,
            iv_cols=args.instruments,
            covariable_cols=args.covariables,
            sampling_weights_col=args.resample_weights_col
        )

    @classmethod
    def add_dataset_arguments(cls, parser: argparse.ArgumentParser):
        """Adds commonly used arguments to load a dataset to an argument
        parser.

        """
        parser.add_argument(
            "--data", "-d", required=True, help="Path to a data file."
        )

        parser.add_argument(
            "--sep",
            default="\t",
            help="Separator (column delimiter) for the data file.",
        )

        parser.add_argument(
            "--instruments",
            "-z",
            nargs="*",
            default=[],
            help="The instrument (Z or G) in the case where we're not using "
            "genotypes provided through --genotypes. Multiple values can "
            "be provided for multiple instruments.\n"
            "This should be column(s) in the data file.",
        )

        parser.add_argument(
            "--covariables",
            nargs="*",
            default=[],
            help="Variables which will be included in both stages."
            "This should be column(s) in the data file.",
        )

        parser.add_argument(
            "--exposure", "-x",
            nargs="+",
            required=True,
            help="List of exposure column(s), e.g. T1 T2"
        )

        parser.add_argument(
            "--outcomes", "-y",
            nargs="+",
            help="List of outcome columns (e.g. Y1 Y2 Y3).",
            required=True,
            type=str
        )


        parser.add_argument(
            "--resample-weights-col",
            help="Sampling weights column if used when bootstrapping.",
            default=None,
            type=str
        )


class FullBatchDataLoader(DataLoader):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset, batch_size=len(dataset))  # type: ignore

        # Cache the whole dataset.
        dl = DataLoader(dataset, batch_size=len(dataset))  # type: ignore
        self.payload = next(iter(dl))

    def __iter__(self):
        yield self.payload


class IVDatasetWithGenotypes(IVDataset):
    def __init__(
        self,
        genetic_dataset: "PhenotypeGeneticDataset",
        exposure_col: str,
        outcome_col: str,
        iv_cols: Iterable[str],
        covariable_cols: Iterable[str]
    ):
        """Dataset that also includes genotypes read using pytorch genotypes.

        TODO: This is not tested yet.
        Weighted resampling is not supported.

        """
        if not PT_GENO_AVAIL:
            raise ImportError("pytorch_genotypes needed to create "
                              "IVDatasetWithGenotypes")

        self.genetic_dataset = genetic_dataset

        instruments_set = set(iv_cols)
        covariables_set = set(covariable_cols)

        iv_indices: List[int] = []
        covariable_indices: List[int] = []
        self.exposure_index = None
        self.outcome_index = None
        for idx, col in enumerate(genetic_dataset.exogenous_columns):
            if col in instruments_set:
                iv_indices.append(idx)
            elif col in covariables_set:
                covariable_indices.append(idx)
            elif col == exposure_col:
                assert self.exposure_index is None
                self.exposure_index = idx
            elif col == outcome_col:
                assert self.outcome_index is None
                self.outcome_index = idx

        self.iv_idx_tens = torch.tensor(iv_indices)
        self.covariable_idx_tens = torch.tensor(covariable_indices)

        if self.exposure_index is None:
            warn(f"Exposure '{exposure_col}' not found in genetic dataset "
                 f"(will not be accessible).")

        if self.outcome_index is None:
            warn(f"Outcome '{outcome_col}' not found in genetic dataset "
                 f"(will not be accessible).")

    def __getitem__(self, index: int) -> IVDatasetBatch:
        cur = self.genetic_dataset[index]
        if self.exposure_index is not None:
            exposure = cur.exogenous[[self.exposure_index]]
        else:
            exposure = torch.Tensor()

        if self.outcome_index is not None:
            outcome = cur.exogenous[[self.outcome_index]]
        else:
            outcome = torch.Tensor()

        instruments = cur.dosage
        covars = torch.Tensor()

        if self.covariable_idx_tens.numel() > 0:
            covars = cur.exogenous[self.covariable_idx_tens]

        if self.iv_idx_tens.numel() > 0:
            instruments = torch.hstack(
                (instruments, cur.exogenous[:, self.iv_idx_tens])
            )

        return exposure, outcome, instruments, covars

    def __len__(self) -> int:
        return len(self.genetic_dataset)

    @property
    def covariables(self):
        covars = self.genetic_dataset.exog[:, self.covariable_idx_tens]\
            .to(torch.float32)
        return covars

    @property
    def exposure(self):
        return self.genetic_dataset.exog[:, [self.exposure_index]]

    @staticmethod
    def from_argparse_namespace(args: argparse.Namespace) -> IVDataset:
        # Defer to parent if no genetic data provided.
        if args.genotypes_backend is None:
            return IVDataset.from_argparse_namespace(args)

        if args.resample_weights_col is not None:
            raise NotImplementedError(
                "Resampling weights not implemented for genetic datasets."
            )

        # Read genotype data.
        backend_class = BACKENDS.get(
            args.genotypes_backend_type, GeneticDatasetBackend
        )

        backend = backend_class.load(args.genotypes_backend)

        # Read phenotype data.
        data = pd.read_csv(
            args.data, sep=args.sep
        )

        dataset = PhenotypeGeneticDataset(
            backend,
            data,
            phenotypes_sample_id_column=args.sample_id_col,
            exogenous_columns=itertools.chain(
                args.instruments, args.covariables,
                [args.exposure, args.outcomes]
            ),
        )

        return IVDatasetWithGenotypes(
            genetic_dataset=dataset,
            exposure_col=args.exposure,
            outcome_col=args.outcomes,
            iv_cols=args.instruments,
            covariable_cols=args.covariables
        )

    @classmethod
    def add_dataset_arguments(cls, parser: argparse.ArgumentParser):
        super().add_dataset_arguments(parser)

        parser.add_argument(
            "--genotypes-backend",
            help=(
                "Pickle containing a pytorch-genotypes backend. This can be "
                "created from various genetic data formats using the "
                "'pt-geno-create-backend' command line utility provided by "
                "pytorch genotypes."
            ),
            type=str,
        )

        parser.add_argument(
            "--genotypes-backend-type",
            help=(
                "Pickle containing a pytorch-genotypes backend. This can be "
                "created from various genetic data formats using the "
                "'pt-geno-create-backend' command line utility provided by "
                "pytorch genotypes."
            ),
            type=str,
        )

        parser.add_argument(
            "--sample-id-col",
            default="sample_id",
            help="Column that contains the sample id. This is mandatory if "
            "genotypes are provided to enable joining.",
        )


class SupervisedLearningWrapper(Dataset):
    """Wraps an IVDataset for supervised learning.

    This will yield pairs (IVs + covariables, exposure).

    """
    def __init__(self, dataset: IVDataset):
        self.dataset = dataset

    def n_exog(self):
        return self.dataset.n_exog()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, _, ivs, covars = self.dataset[idx]
        return (torch.hstack([ivs, covars]), x)

    def __len__(self):
        return len(self.dataset)
