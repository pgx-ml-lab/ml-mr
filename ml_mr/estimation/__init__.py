from . import baselines, deep_iv, delivr, dfiv, quantile_iv
from .core import (MREstimator, MREstimatorWithUncertainty,
                   RescaledMREstimator, load_estimator)
from .ensemble import EnsembleMREstimator

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
