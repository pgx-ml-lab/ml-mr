from typing import Literal, Callable, TypeVar, Optional, Union, Type, Iterable
import numpy as np
import torch

from scipy.interpolate import interp1d

from ..logging import info


INTERPOLATION = ["linear", "quadratic", "cubic"]
Interpolation = Literal["linear", "quadratic", "cubic"]
InterpolationCallable = Callable[[torch.Tensor], torch.Tensor]


MREstimatorType = TypeVar("MREstimatorType", bound="MREstimator")


class MREstimator(object):
    def __init__(
        self,
        meta: dict,
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
        self.meta = meta

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
        covars: Optional[torch.Tensor] = None,
        low_memory: bool = False,
    ) -> torch.Tensor:

        if covars is None:
            if self.covars is None:
                return self.iv_reg_function(x, None)
            else:
                covars = self.covars

        if low_memory:
            return self._low_mem_avg_iv_reg_function(x)

        n_covars = covars.shape[0]
        x_rep = torch.repeat_interleave(x, n_covars, dim=0)
        covars = covars.repeat(x.shape[0], 1)

        y_hats = self.iv_reg_function(x_rep, covars)

        return torch.vstack([
            tens.mean(dim=0, keepdim=True)
            for tens in torch.split(y_hats, n_covars)
        ])

    def _low_mem_avg_iv_reg_function(self, x: torch.Tensor) -> torch.Tensor:
        avgs = []
        assert self.covars is not None
        num_covars = self.covars.shape[0]
        for cur_x in x:
            cur_cf = torch.mean(self.iv_reg_function(
                cur_x.repeat(num_covars).reshape(num_covars, -1),
                self.covars
            ), dim=0, keepdim=True)
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
        y1 = self.avg_iv_reg_function(x1, covars)
        y0 = self.avg_iv_reg_function(x0, covars)

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
        cls: Type[MREstimatorType],
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

    def avg_iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        low_memory: bool = False,
        alpha: float = 0.1,
    ) -> torch.Tensor:
        return super().avg_iv_reg_function(x, covars, low_memory=low_memory)


class EnsembleMREstimator(MREstimatorWithUncertainty):
    def __init__(self, *estimators: MREstimator):
        if len(estimators) < 2:
            raise ValueError("Need at least two estimators to ensemble.")

        self.estimators = estimators
        self.covars = getattr(estimators[0], "covars")
        if self.covars is None:
            info("EnsembleMREstimator has no bound covariables.")

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
