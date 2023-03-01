from .core import MREstimator

from . import bin_iv

MODELS = {
    "bin_iv": {
        "estimate": bin_iv.estimate,
        "load": bin_iv.load
    }
}
