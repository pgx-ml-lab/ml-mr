from typing import Optional, Callable

import pytest
import torch

from ...estimation.core import MREstimator, MREstimatorWithUncertainty


class MREstimatorFromFunction(MREstimator):
    def __init__(
        self,
        f: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        covars: Optional[torch.Tensor] = None
    ):
        super().__init__(covars)
        self.f = f

    def iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.f(x, covars)


class MREstimatorWithUncertaintyFromFunction(
    MREstimatorWithUncertainty
):
    def __init__(
        self,
        f: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        covars: Optional[torch.Tensor] = None
    ):
        super().__init__(covars)
        self.f = f

    def iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        alpha: float = 0.1,
    ) -> torch.Tensor:
        out = self.f(x, covars)
        return torch.cat((out - 0.1, out, out + 0.1), dim=1).reshape(-1, 1, 3)


@pytest.fixture
def mr_estimator_identity_ignore_covars():
    def f(x, covars=None):
        return x

    return MREstimatorFromFunction(f)


@pytest.fixture
def mr_estimator_identity_elem_product_covars():
    """Implements Y = X * C where C are the covariates.

    This is for Hadamard (elementwise) product.

    """
    def f(x, covars):
        assert covars is not None
        assert x.shape == covars.shape
        return x * covars

    return MREstimatorFromFunction(f)


@pytest.fixture
def mr_estimator_covar_itx_with_unc():
    """Implements Y = X * C where C are the covariates and simulates
    uncertainty.

    This is to simulate the output of a MREstimatorWithUncertainty.

    """
    def f(x, covars):
        assert covars is not None
        assert x.shape == covars.shape
        return x * covars

    return MREstimatorWithUncertaintyFromFunction(f)
