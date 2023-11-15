from typing import Iterable, Optional, Tuple, Union
import os
import json

import pandas as pd
from linearmodels.iv.model import IV2SLS
from linearmodels.iv.results import IVResults
import torch

from ..core import MREstimator
from ...utils.data import IVDataset


class TwoSLSEstimator(MREstimator):
    def __init__(self, const_beta, exposure_beta, exposure_se, meta):
        super().__init__(meta, None)
        self.const_beta = torch.tensor(const_beta).reshape(-1, 1)
        self.exposure_beta = torch.tensor(exposure_beta).reshape(-1, 1)
        self.exposure_se = exposure_se

    def iv_reg_function(self, x: torch.Tensor, covars=None) -> torch.Tensor:
        return x @ self.exposure_beta + self.const_beta

    @classmethod
    def from_results(cls, filename: str) -> "TwoSLSEstimator":
        with open(os.path.join(filename, "2sls_fit.json")) as f:
            params = json.load(f)

        return cls(**params)


def fit_2sls(
    dataset: IVDataset,
    output_dir: str = "2sls_results"
) -> TwoSLSEstimator:
    meta = {"model": "2sls"}
    meta.update(dataset.exposure_descriptive_statistics())

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    df, cols = dataset.to_dataframe()

    iv_results = twosls(
        df,
        y_col=cols["outcome"],
        x_col=cols["exposure"],
        z_cols=cols["ivs"],
        covar_cols=cols["covariables"],
        full=True
    )

    assert isinstance(iv_results, IVResults)
    betas = iv_results.params

    estimates = {
        "const_beta": betas["const"],
        "exposure_beta": betas[cols["exposure"]].to_list(),
        "exposure_se": iv_results.std_errors[cols["exposure"]].to_list()
    }

    with open(os.path.join(output_dir, "2sls_fit.json"), "wt") as f:
        json.dump(estimates, f)

    with open(os.path.join(output_dir, "meta.json"), "wt") as f:
        json.dump(meta, f)

    return TwoSLSEstimator(**estimates, meta=meta)


def twosls(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    z_cols: Iterable[str],
    covar_cols: Optional[Iterable[str]] = None,
    full: bool = False
) -> Union[IVResults, Tuple[float, float]]:
    df = df.copy()
    df["const"] = 1

    exog = ["const"]
    if covar_cols is not None:
        exog += list(covar_cols)

    model = IV2SLS(
        df[y_col],
        df[exog],
        df[x_col],
        df[z_cols]
    ).fit(cov_type="robust")

    assert isinstance(model, IVResults)

    if full:
        return model
    else:
        return model.params, model.std_errors


estimate = fit_2sls
load = TwoSLSEstimator.from_results
