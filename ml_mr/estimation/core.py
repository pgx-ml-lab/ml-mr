
from typing import Optional
import torch


class MREstimator(object):
    def effect(self, x: torch.Tensor, covars: Optional[torch.Tensor] = None):
        """Return the expected effect on the outcome when the exposure is set
        to X.

        E[Y | do(X=x)]

        In many cases, the ml-mr estimators condition on variables for
        estimation. If a sample of covariables is provided, the causal effect
        will be empirically averaged over the covariable values.

        i.e. Sum P(Y | C, do(X=x)) P(C) as empirically observed in the provided
        data.

        """
        raise NotImplementedError()
