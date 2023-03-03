from .core import MREstimator

from . import bin_iv
from . import quantile_iv
from . import baselines

MODELS = {
    "bin_iv": {
        "estimate": bin_iv.estimate,
        "load": bin_iv.load
    },
    "quantile_iv": {
        "estimate": quantile_iv.estimate,
        "load": quantile_iv.load
    },
    "doubly_ranked": {
        "estimate": baselines.doubly_ranked.estimate,
        "load": baselines.doubly_ranked.load,
    }
}
