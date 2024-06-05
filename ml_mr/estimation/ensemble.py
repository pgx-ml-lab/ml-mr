from typing import Optional

from .core import MREstimator, MREstimatorWithUncertainty, load_estimator
from ..logging import info, warn

import torch

import glob
from typing import Iterable


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
        alpha: float = 0.1,
        reduce: bool = True
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

        if not reduce:
            return combined

        return torch.quantile(
            combined,
            torch.tensor([alpha / 2, 0.5, 1 - alpha / 2]),
            dim=1
        ).T.reshape(-1, 1, 3)

    def cate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        covars: torch.Tensor,
        alpha: float = 0.1,
        reduce: bool = True
    ) -> torch.Tensor:
        cates = []
        for estimator in self.estimators:
            if isinstance(estimator, MREstimatorWithUncertainty):
                cates.append(estimator.cate(x0, x1, covars)[:, 0, [1]])
            else:
                cates.append(estimator.cate(x0, x1, covars))

        combined = torch.concat(cates, dim=1)

        if not reduce:
            return combined

        return torch.quantile(
            combined,
            torch.tensor([alpha / 2, 0.5, 1 - alpha / 2]),
            dim=1
        ).T.reshape(-1, 1, 3)

    @staticmethod
    def _call_estimators(
        estimators: Iterable[MREstimator],
        func_name: str,
        x: torch.Tensor,
        covars: Optional[torch.Tensor],
        alpha: float = 0.1,
        reduce: bool = True,
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

        if not reduce:
            return combined

        return torch.quantile(
            combined,
            torch.tensor([alpha / 2, 0.5, 1 - alpha / 2]),
            dim=1
        ).T.reshape(-1, 1, 3)

    def iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        alpha: float = 0.1,
        reduce: bool = True
    ) -> torch.Tensor:
        return self._call_estimators(
            self.estimators, "iv_reg_function", x, covars, alpha, reduce
        )

    def avg_iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        low_memory: bool = False,
        alpha: float = 0.1,
        reduce: bool = True
    ) -> torch.Tensor:
        return self._call_estimators(
            self.estimators, "avg_iv_reg_function", x, covars, alpha, reduce
        )
