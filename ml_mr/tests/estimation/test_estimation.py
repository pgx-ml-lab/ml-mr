import torch
from numpy.testing import assert_array_equal

from .fixtures import *  # noqa: F401, F403


def test_estimator_instantiation(mr_estimator_identity_ignore_covars):
    x = torch.arange(5)
    y = mr_estimator_identity_ignore_covars.iv_reg_function(x)
    assert_array_equal(x.numpy(), y.numpy())


def test_linear_interpolation(mr_estimator_identity_ignore_covars):
    x = torch.arange(5)
    y = mr_estimator_identity_ignore_covars.iv_reg_function(x)

    interpolator = mr_estimator_identity_ignore_covars.interpolate(
        x, y,
        mode="linear"
    )
    y_2 = interpolator(torch.tensor([2]))
    assert torch.all(y_2 == torch.tensor([2]))


def test_iv_reg_fct(mr_estimator_identity_elem_product_covars):
    xs = torch.arange(10).reshape(-1, 1)
    covars = torch.ones((10, 1)) * 2
    y_cf = mr_estimator_identity_elem_product_covars.iv_reg_function(
        xs, covars
    )

    expected = xs * covars

    assert torch.all(y_cf == expected)


def test_avg_iv_reg_fct(mr_estimator_identity_elem_product_covars):
    xs = torch.arange(10).reshape(-1, 1).to(torch.float32)
    covars = torch.arange(5).reshape(-1, 1).to(torch.float32) + 5  # 5 to 9
    mr_estimator_identity_elem_product_covars.set_covars(covars)

    # 7 is the mean of covars (range 5 to 9)
    expected = torch.tensor([7.0 * x for x in range(10)]).reshape(-1, 1)

    y_cv = mr_estimator_identity_elem_product_covars.avg_iv_reg_function(xs)

    assert torch.all(y_cv == expected)


def test_avg_iv_reg_fct_lowmem(mr_estimator_identity_elem_product_covars):
    xs = torch.arange(10).reshape(-1, 1).to(torch.float32)
    covars = torch.arange(5).reshape(-1, 1).to(torch.float32) + 5
    mr_estimator_identity_elem_product_covars.set_covars(covars)

    expected = torch.tensor([7.0 * x for x in range(10)]).reshape(-1, 1)

    y_cv = mr_estimator_identity_elem_product_covars.avg_iv_reg_function(
        xs, low_memory=True
    )

    assert torch.all(y_cv == expected)
