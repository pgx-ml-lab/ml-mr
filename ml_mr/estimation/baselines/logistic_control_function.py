"""
import scipy
import numpy as np
import pandas as pd
from ml_mr.estimation.baselines.logistic_control_function import *
from ml_mr.utils.data import IVDataset
n = 50_000
z = scipy.stats.norm.rvs(size=n)
u = scipy.stats.norm.rvs(size=n)
x = z + u + scipy.stats.norm.rvs(size=n)
logistic = lambda x: 1 / (1 + np.exp(-x))
y = (
    logistic(0.3 * x - 2 * u + scipy.stats.norm.rvs(size=n) - 1.5) > 0.5
).astype(int)

df = pd.DataFrame({
    "z": z,
    "x": x,
    "y": y
})
dataset = IVDataset.from_dataframe(df, "x", "y", ["z"])
fit_logistic_control_function(dataset)
"""

from typing import Iterable, Optional
import os
import json

import numpy as np
import torch


try:
    import statsmodels.api as sm
    STATSMODELS_AVAIL = True
except ImportError:
    STATSMODELS_AVAIL = False


from ..core import MREstimator
from ...utils.data import IVDataset


class LogisticControlFunctionEstimator(MREstimator):
    def __init__(self, intercept, log_or, meta):
        super().__init__(meta, None)
        self.intercept = intercept
        self.log_or = log_or

    def iv_reg_function(self, x: torch.Tensor, covars=None) -> torch.Tensor:
        logits = x * self.log_or + self.intercept
        return logits

    @classmethod
    def from_results(cls, filename: str) -> "LogisticControlFunctionEstimator":
        with open(
            os.path.join(filename, "logistic_control_function_fit.json")
        ) as f:
            params = json.load(f)

        with open(os.path.join(filename, "meta.json")) as f:
            meta = json.load(f)

        return cls(**params, meta=meta)


def fit_logistic_control_function(
    dataset: IVDataset,
    output_dir: str = "logistic_control_function_results"
) -> LogisticControlFunctionEstimator:
    meta = {"model": "logistic_control_function"}
    meta.update(dataset.exposure_descriptive_statistics())

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    df, cols = dataset.to_dataframe()

    iv_results = logistic_control_func(
        df,
        y_col=cols["outcome"],
        x_col=cols["exposure"],
        z_cols=cols["ivs"],
        covar_cols=cols["covariables"],
    )

    # Serialize results in json format.
    with open(
        os.path.join(output_dir, "logistic_control_function_fit.json"), "wt"
    ) as f:
        json.dump(iv_results, f)

    with open(os.path.join(output_dir, "meta.json"), "wt") as f:
        json.dump(meta, f)

    return LogisticControlFunctionEstimator(**iv_results, meta=meta)


def logistic_control_func(
    df,
    y_col: str,
    x_col: str,
    z_cols: Iterable[str],
    covar_cols: Optional[Iterable[str]] = None,
):
    # Stage 1.
    df["_ones"] = 1
    stg1_exog_cols = ["_ones"] + list(z_cols)
    if covar_cols is not None:
        stg1_exog_cols += list(covar_cols)

    stg1_exog = df[stg1_exog_cols].values
    x = df[x_col].values.reshape(-1, 1)

    betas_1 = np.linalg.lstsq(stg1_exog, x, rcond=None)[0]
    x_hat = stg1_exog @ betas_1
    df["x_hat"] = x_hat
    df["resids_1"] = x - x_hat

    # Stage 2.
    stg2_exog_cols = ["_ones", "x_hat", "resids_1"]
    if covar_cols is not None:
        stg2_exog_cols += list(covar_cols)
    stg2 = sm.Logit(df[y_col], df[stg2_exog_cols])

    fit = stg2.fit()
    # Intercept, log_OR
    return {
        "intercept": fit.params[0],
        "log_or": fit.params[1]
    }


estimate = fit_logistic_control_function
load = LogisticControlFunctionEstimator.from_results
