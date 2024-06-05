from .ensemble import EnsembleMREstimator

from .core import MREstimator, MREstimatorWithUncertainty, load_estimator

from . import deep_iv
from . import quantile_iv
from . import baselines
from . import dfiv
from . import delivr

MODELS = {
    "quantile_iv": {
        "estimate": quantile_iv.estimate,
        "load": quantile_iv.load
    },
    "doubly_ranked": {
        "estimate": baselines.doubly_ranked.estimate,
        "load": baselines.doubly_ranked.load
    },
    "2sls": {
        "estimate": baselines.linear_two_stage.estimate,
        "load": baselines.linear_two_stage.load
    },
    "logistic_control_function": {
        "estimate": baselines.logistic_control_function.estimate,
        "load": baselines.logistic_control_function.load
    },
    "deep_iv": {
        "estimate": deep_iv.estimate,
        "load": deep_iv.load
    },
    "dfiv": {
        "estimate": dfiv.estimate,
        "load": dfiv.load
    },
    "delivr": {
        "estimate": delivr.estimate,
        "load": delivr.load
    },
}
