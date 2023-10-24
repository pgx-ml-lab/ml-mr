import argparse
import itertools
import os
from typing import (Callable, Iterable, List, Literal, Optional, Tuple,
                    TypeVar, Union, Dict, Any)

import numpy as np
import pandas as pd
import torch
from pytorch_genotypes.dataset import (BACKENDS, GeneticDatasetBackend,
                                       PhenotypeGeneticDataset)
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, DataLoader

from ..logging import warn

INTERPOLATION = ["linear", "quadratic", "cubic"]
Interpolation = Literal["linear", "quadratic", "cubic"]
InterpolationCallable = Callable[[torch.Tensor], torch.Tensor]


MREstimatorType = TypeVar("MREstimatorType", bound="MREstimator")
IVDatasetBatch = Tuple[
    torch.Tensor,  # exposure
    torch.Tensor,  # outcome
    torch.Tensor,  # IVs
    torch.Tensor   # Covariables
]


class MREstimator(object):
    def __init__(
        self,
        covars: Optional[torch.Tensor],
        num_samples: int = 10_000
    ):
        if covars is None:
            self.covars = None
            return

        # Sample covariates if needed.
        if num_samples <= covars.shape[0]:
            idx = torch.multinomial(
                torch.ones((covars.shape[0])),
                num_samples=num_samples,
                replacement=False,
            )
            covars = covars[idx]

        self.covars = covars

    def set_covars(self, covars: torch.Tensor) -> None:
        self.covars = covars

    def iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError()

    def avg_iv_reg_function(
        self,
        x: torch.Tensor,
        low_memory: bool = False,
    ) -> torch.Tensor:
        if self.covars is None:
            return self.iv_reg_function(x, None)

        if low_memory:
            return self._low_mem_avg_iv_reg_function(x)

        n_covars = self.covars.shape[0]
        x_rep = torch.repeat_interleave(x, n_covars, dim=0)
        covars = self.covars.repeat(x.shape[0], 1)

        y_hats = self.iv_reg_function(x_rep, covars)

        return torch.vstack([
            tens.mean(dim=0) for tens in torch.split(y_hats, n_covars)
        ])

    def _low_mem_avg_iv_reg_function(self, x: torch.Tensor) -> torch.Tensor:
        avgs = []
        assert self.covars is not None
        num_covars = self.covars.shape[0]
        for cur_x in x:
            cur_cf = torch.mean(self.iv_reg_function(
                cur_x.repeat(num_covars).reshape(num_covars, -1),
                self.covars
            ))
            avgs.append(cur_cf)

        return torch.vstack(avgs)

    def ate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """Average treatment effect."""
        y1 = self.avg_iv_reg_function(x1)
        y0 = self.avg_iv_reg_function(x0)

        return y1 - y0

    def cate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        covars: torch.Tensor
    ) -> torch.Tensor:
        """Conditional average treatment effect."""
        y1 = self.iv_reg_function(x1, covars)
        y0 = self.iv_reg_function(x0, covars)

        return y1 - y0

    @staticmethod
    def interpolate(
        xs: Union[torch.Tensor, np.ndarray],
        ys: Union[torch.Tensor, np.ndarray],
        mode: Interpolation = "cubic",
        bounds_error: bool = True
    ) -> InterpolationCallable:
        if mode not in INTERPOLATION:
            raise ValueError(f"Unknown interpolation type {mode}.")

        if isinstance(xs, torch.Tensor):
            xs = xs.numpy()

        if isinstance(ys, torch.Tensor):
            ys = ys.numpy()

        interpolator = interp1d(xs, ys, kind=mode, bounds_error=bounds_error)

        def interpolate_torch(x):
            return torch.from_numpy(interpolator(x))

        return interpolate_torch

    @classmethod
    def from_results(
        cls: type[MREstimatorType],
        filename: str
    ) -> MREstimatorType:
        """Initialize an estimator from the results.

        The results can vary by estimator, but typically should be a results
        file or directory generated by the estimation module.

        """
        raise NotImplementedError()


class MREstimatorWithUncertainty(MREstimator):
    """Estimator that quantifies uncertainty on the IV regression.

    This is only a semantic class, but we use the convention that uncertainty
    is reflected by providing the alpha / 2, median and 1 - alpha / 2 quantiles
    in the last dimensions of the tensor.

    For example, the counterfactual y tensor could have shape (N, 1, 3) for a
    univariable outcome and when the number of samples is N.

    """
    def iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        alpha: float = 0.1,
    ) -> torch.Tensor:
        return super().iv_reg_function(x, covars)


class IVDataset(Dataset):
    """Dataset class for IV analysis.

    The batches contain exposure, outcome, IVs and covariables.

    """
    def __init__(
        self,
        exposure: torch.Tensor,
        outcome: torch.Tensor,
        ivs: torch.Tensor,
        covariables: torch.Tensor = torch.Tensor()
    ):
        self.exposure = exposure.reshape(-1, 1)
        self.outcome = outcome.reshape(-1, 1)
        self.ivs = ivs
        self.covariables = covariables

        assert (
            self.exposure.size(0) == self.outcome.size(0) == self.ivs.size(0)
        ), (
            f"exposure={self.exposure.shape}, outcome={self.outcome.shape},"
            f"ivs={self.ivs.shape}"
        )

    def __getitem__(self, index: int) -> IVDatasetBatch:
        exposure = self.exposure[index]
        outcome = self.outcome[index]
        ivs = self.ivs[index]
        covars = self.covariables[index]

        return exposure, outcome, ivs, covars

    def __len__(self) -> int:
        return self.ivs.size(0)

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
        exposure_col: str,
        outcome_col: str,
        iv_cols: Iterable[str],
        covariable_cols: Iterable[str] = []
    ) -> "IVDataset":
        # We'll do complete case analysis if the user provides a df with NAs.
        keep_cols = list(itertools.chain(
            [exposure_col, outcome_col], iv_cols, covariable_cols
        ))
        dataframe = dataframe[keep_cols]
        if dataframe.isna().values.any():
            n_before = dataframe.shape[0]
            dataframe = dataframe.dropna()
            n_after = dataframe.shape[0]
            n = n_before - n_after
            warn(
                f"Doing complete case analysis. Dropped {n} rows with missing"
                f"values from the input data."
            )

        exposure = torch.from_numpy(dataframe[exposure_col].values).float()
        outcome = torch.from_numpy(dataframe[outcome_col].values).float()
        ivs = torch.from_numpy(dataframe[iv_cols].values).float()
        covars = torch.from_numpy(dataframe[covariable_cols].values).float()

        return IVDataset(exposure, outcome, ivs, covars)

    @staticmethod
    def from_json_configuration(configuration) -> "IVDataset":
        allowed_keys = {
            "filename", "sep", "exposure", "outcome", "instruments",
            "covariables"
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
        )

    @staticmethod
    def from_argparse_namespace(args: argparse.Namespace) -> "IVDataset":
        data = pd.read_csv(
            args.data, sep=args.sep
        )
        return IVDataset.from_dataframe(
            data,
            exposure_col=args.exposure,
            outcome_col=args.outcome,
            iv_cols=args.instruments,
            covariable_cols=args.covariables
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
            "--exposure",
            "-x",
            help="The exposure (X). This should be a column name in the data "
            "file.",
            required=True,
            type=str,
        )

        parser.add_argument(
            "--outcome",
            "-y",
            help="The outcome (Y). This should be a column name in the data "
            "file.",
            required=True,
            type=str,
        )


class BootstrapIVDataset(IVDataset):
    def __init__(
        self,
        exposure: torch.Tensor,
        outcome: torch.Tensor,
        ivs: torch.Tensor,
        covariables: torch.Tensor = torch.Tensor()
    ):
        super().__init__(exposure, outcome, ivs, covariables)
        # Resample with replacement.
        n = len(self)
        self.bootstrap_idx = torch.multinomial(
            torch.ones(n),
            n,
            replacement=True
        )

    def __getitem__(self, idx: int) -> IVDatasetBatch:
        bs_idx = int(self.bootstrap_idx[idx].item())
        return super().__getitem__(bs_idx)


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
        genetic_dataset: PhenotypeGeneticDataset,
        exposure_col: str,
        outcome_col: str,
        iv_cols: Iterable[str],
        covariable_cols: Iterable[str]
    ):
        """Dataset that also includes genotypes read using pytorch genotypes.

        TODO: This is not tested yet.

        """
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

    def __getitem__(self, index: int) -> IVDatasetBatch:
        cur = self.genetic_dataset[index]
        exposure = cur.exogenous[:, [self.exposure_index]]
        outcome = cur.exogenous[:, [self.outcome_index]]

        instruments = cur.dosage
        covars = torch.Tensor()

        if self.covariable_idx_tens.numel() > 0:
            covars = cur.exogenous[:, self.covariable_idx_tens]

        if self.iv_idx_tens.numel() > 0:
            instruments = torch.hstack(
                (instruments, cur.exogenous[:, self.iv_idx_tens])
            )

        return exposure, outcome, instruments, covars

    def __len__(self) -> int:
        return len(self.genetic_dataset)

    @property
    def covariables(self):
        self.genetic_dataset.exog[:, self.covariable_idx_tens]

    @property
    def exposure(self):
        return self.genetic_dataset.exog[:, [self.exposure_index]]

    @staticmethod
    def from_argparse_namespace(args: argparse.Namespace) -> IVDataset:
        # Defer to parent if no genetic data provided.
        if args.genotypes_backend is None:
            return IVDataset.from_argparse_namespace(args)

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
                [args.exposure, args.outcome]
            ),
        )

        return IVDatasetWithGenotypes(
            genetic_dataset=dataset,
            exposure_col=args.exposure,
            outcome_col=args.outcome,
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
