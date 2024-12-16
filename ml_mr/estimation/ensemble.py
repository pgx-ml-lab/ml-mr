from typing import Optional, Callable, Union

from .core import MREstimator, MREstimatorWithUncertainty, load_estimator
from ..logging import info, warn

import torch

import glob
from typing import Iterable


def bs_quantile_reduce(mat: torch.Tensor, alpha: float):
    """Get quantile CIs for bootstrapped statistics."""
    return torch.quantile(
        mat,
        torch.tensor([alpha / 2, 0.5, 1 - alpha / 2]),
        dim=1
    ).T.reshape(-1, 1, 3)


def bs_parametric_reduce(mat: torch.Tensor, alpha: float):
    import scipy.stats

    means = torch.mean(mat, dim=1)
    ses = torch.std(mat, dim=1)
    z = scipy.stats.norm.ppf(1 - alpha / 2)

    return torch.stack(
        (
            means - z * ses,
            means,
            means + z * ses
        )
    ).T.reshape(-1, 1, 3)


# Type alias for functions used to summarize bootstrap results as CIs.
REDUCE_TYPE = Union[bool, Callable[[torch.Tensor, float], torch.Tensor]]


def _apply_reduce(
    mat: torch.Tensor, alpha: float, reduce: REDUCE_TYPE
) -> torch.Tensor:
    if reduce is False:
        return mat
    elif reduce is True:
        # Default reducer is quantile CI.
        return bs_quantile_reduce(mat, alpha)
    else:
        return reduce(mat, alpha)


class EnsembleMREstimator(MREstimatorWithUncertainty):
    def __init__(self, *estimators: MREstimator):
        if len(estimators) < 2:
            raise ValueError("Need at least two estimators to ensemble.")

        self.estimators = estimators
        self.covars = getattr(estimators[0], "covars")
        if self.covars is None:
            info("EnsembleMREstimator has no bound covariables.")

    @classmethod
    def from_glob(cls, a_glob):
        estimators = []
        for path in glob.glob(a_glob):
            try:
                estimator = load_estimator(path)
            except Exception as e:
                warn(f"Could not load estimator in '{path}': "
                     f"{type(e).__name__} (ignoring).")
                continue

            estimators.append(estimator)

        return cls(*estimators)

    def ate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        alpha: float = 0.05,
        reduce: REDUCE_TYPE = True
    ) -> torch.Tensor:
        ates = []
        for estimator in self.estimators:
            if isinstance(estimator, MREstimatorWithUncertainty):
                mu0 = estimator.avg_iv_reg_function(x0)[:, 0, [1]]
                mu1 = estimator.avg_iv_reg_function(x1)[:, 0, [1]]
            else:
                mu0 = estimator.avg_iv_reg_function(x0)
                mu1 = estimator.avg_iv_reg_function(x1)

            ates.append(mu1 - mu0)

        combined = torch.concat(ates, dim=1)
        return _apply_reduce(combined, alpha, reduce)

    def cate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        covars: torch.Tensor,
        alpha: float = 0.05,
        reduce: REDUCE_TYPE = True
    ) -> torch.Tensor:
        cates = []
        for estimator in self.estimators:
            if isinstance(estimator, MREstimatorWithUncertainty):
                cates.append(estimator.cate(x0, x1, covars)[:, 0, [1]])
            else:
                cates.append(estimator.cate(x0, x1, covars))

        combined = torch.concat(cates, dim=1)
        return _apply_reduce(combined, alpha, reduce)

    @staticmethod
    def _call_estimators(
        estimators: Iterable[MREstimator],
        func_name: str,
        x: torch.Tensor,
        covars: Optional[torch.Tensor],
        alpha: float = 0.05,
        reduce: REDUCE_TYPE = True,
    ):
        estimates = []
        for estimator in estimators:
            # Get the method to call.
            func = getattr(estimator, func_name)
            if func is None:
                raise ValueError(func_name)

            cur = func(x, covars)

            if isinstance(estimator, MREstimatorWithUncertainty):
                cur = cur[:, 0, [1]]

            estimates.append(cur)

        combined = torch.concat(estimates, dim=1)
        return _apply_reduce(combined, alpha, reduce)

    def iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        alpha: float = 0.05,
        reduce: REDUCE_TYPE = True
    ) -> torch.Tensor:
        return self._call_estimators(
            self.estimators, "iv_reg_function", x, covars, alpha, reduce
        )

    def avg_iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        low_memory: bool = False,
        alpha: float = 0.05,
        reduce: REDUCE_TYPE = True
    ) -> torch.Tensor:
        return self._call_estimators(
            self.estimators, "avg_iv_reg_function", x, covars, alpha, reduce
        )
