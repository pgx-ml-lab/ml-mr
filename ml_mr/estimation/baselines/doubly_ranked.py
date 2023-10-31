"""
Implementation of the doubly ranked method.

Tian, H., Mason, A. M., Liu, C. & Burgess, S. Relaxing parametric assumptions
for non-linear Mendelian randomization using a doubly-ranked stratification
method. bioRxiv 2022.06.28.497930 (2022) doi:10.1101/2022.06.28.497930

from ml_mr.estimation.core import _IVDataset
from ml_mr.estimation.baselines.doubly_ranked import DoublyRankedEstimator

---

import pandas as pd
import scipy
from ml_mr.estimation.core import _IVDataset
from ml_mr.estimation.baselines.doubly_ranked import fit_doubly_ranked
n = 100_000
df = pd.DataFrame({
    "z": scipy.stats.norm.rvs(size=n)
})
df["u"] = scipy.stats.norm.rvs(size=n)
df["x"] = 0.21 * df.z + 0.2 * df.u + scipy.stats.norm.rvs(size=n)
df["y"] = (
    0.314159 * df.x +
    0.05 * df.x ** 2 +
    0.15 * df.u + scipy.stats.norm.rvs(size=n)
)

dataset = _IVDataset.from_dataframe(
    df, "x", "y", ["z"]
)

estimator = fit_doubly_ranked(dataset)

"""

from typing import Optional
import os

import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ...logging import warn
from ..core import MREstimator
from ...utils.data import IVDataset
from .linear_two_stage import twosls


class DoublyRankedEstimator(MREstimator):
    def __init__(self, results: pd.DataFrame):
        # mean_x, lace, lace_se
        self.results = results

        self._interpolator = self.interpolate(
            results["mean_x"], results["lace"]
        )

    def iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if covars is not None:
            warn(
                "The doubly ranked method does not estimate conditional "
                "treatment effects. The provided covariates will be ignored."
            )

        interpolated_effects = self._interpolator(x)
        return interpolated_effects * x

    @classmethod
    def from_results(cls, results_dir: str) -> "DoublyRankedEstimator":
        df = pd.read_csv(os.path.join(results_dir, "lace.csv"))
        return cls(df)


def create_strata(
    df: pd.DataFrame,
    q: int,
    z_col: str,
    x_col: str,
):
    # Drop samples if we can't make an event split.
    n = df.shape[0]
    n_keep = n - (df.shape[0] % q)

    if n_keep < n:
        warn(f"Dropping {n - n_keep} individual(s) at random.")

    df = df.sample(n=n_keep).sort_values([z_col])

    # Sort by blocks of the pre-strata.
    n_strata = n_keep // q
    for pre_strata_indices in np.split(np.arange(n_keep), n_strata):
        df.iloc[pre_strata_indices] = df.iloc[pre_strata_indices]\
            .sort_values(x_col)

    # Create the strata by matching ranks from pre-strata.
    strata = []
    for i in range(0, q):
        cur_indices = np.fromiter((
            i + k * q for k in range(n_strata)
        ), dtype=int)
        strata.append(cur_indices)

    return df, strata


def fit_doubly_ranked(
    dataset: IVDataset,
    q: int = 10,
    output_dir: str = "doubly_ranked_results",
    no_plot: bool = False
) -> DoublyRankedEstimator:
    # Create output directory if needed.
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Load the dataset and prepare the data.
    dl = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(dl))

    x, y, iv, covars = [tens.numpy() for tens in data]

    del data, dl

    if iv.shape[1] > 1:
        warn("Collapsing IVs into an unweighted score because the doubly "
             "ranked methods needs a single continuous IV.")
        iv = np.sum(iv, axis=1, keepdims=True)

    data_mat = np.hstack((x, y, iv, covars))
    names = ["x", "y", "z"]

    covar_cols = []
    for i in range(covars.shape[1]):
        covar_cols.append(f"covars{i+1}")

    names += covar_cols

    df = pd.DataFrame(data_mat, columns=names)

    # Create strata.
    df, strata = create_strata(df, q, "z", "x")

    # Compute the LACE estimates.
    results = []

    for mask in strata:
        cur = df.iloc[mask, :]
        mean_x = cur["x"].mean()
        res = twosls(cur, "y", "x", ["z"], covar_cols)
        assert isinstance(res, tuple)
        cur_beta, cur_se = res

        results.append((mean_x, cur_beta, cur_se))

    results_df = pd.DataFrame(results, columns=["mean_x", "lace", "lace_se"])
    results_df.index.name = "strata"

    results_df.to_csv(os.path.join(output_dir, "lace.csv"))

    if not no_plot:
        # Plot the LACE estimates.
        plt.errorbar(
            results_df.index.values,
            results_df["lace"],
            yerr=1.96*results_df["lace_se"],
            fmt="o",
            markersize=5
        )
        plt.axhline(y=0, ls="--", lw=1, color="black")
        plt.xlabel("Strata rank")
        plt.ylabel("LACE estimate (95% CI)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "lace.png"), dpi=400)
        plt.clf()
        plt.close()

        # Plot the E[Y|do(X)] estimates.
        idx = np.argsort(results_df["mean_x"])
        plt.plot(
            results_df.iloc[idx]["mean_x"],
            results_df.iloc[idx]["lace"] * results_df.iloc[idx]["mean_x"],
            marker="o"
        )
        plt.xlabel("Mean of exposure (x) in strata")
        plt.ylabel("Predicted Y | do(X=x)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "causal_effect.png"), dpi=400)
        plt.clf()
        plt.close()

    # Return an estimator instance.
    return DoublyRankedEstimator(results_df)


estimate = fit_doubly_ranked
load = DoublyRankedEstimator.from_results
