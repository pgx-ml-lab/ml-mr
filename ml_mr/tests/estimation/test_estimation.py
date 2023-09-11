import torch
from numpy.testing import assert_array_equal

from .fixtures import *


def test_estimator_instantiation(mr_estimator_identity_ignore_covars):
    x = torch.arange(5)
    y = mr_estimator_identity_ignore_covars.effect(x)
    assert_array_equal(x.numpy(), y.numpy())

def test_linear_interpolation(mr_estimator_identity_ignore_covars):
    x = torch.arange(5)
    y = mr_estimator_identity_ignore_covars.effect(x)

    interpolator = mr_estimator_identity_ignore_covars.interpolate(
        x, y,
        mode="linear"
    )
    y_2 = interpolator(torch.tensor([2]))
    assert torch.all(y_2 == torch.tensor([2]))


