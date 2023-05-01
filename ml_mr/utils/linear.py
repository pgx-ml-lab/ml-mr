"""
Linear models and other utilities.
"""

from typing import Tuple

import torch


def ridge_regression(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    n_samples, n_features = x.shape
    assert y.shape[0] == n_samples

    L = x.T @ x + alpha * torch.eye(n_features)
    try:
        L_chol = torch.linalg.cholesky(L)
    except torch.linalg.LinAlgError as e:
        print(L)
        raise e

    betas = torch.cholesky_solve(x.T @ y, L_chol)

    return betas


def ridge_fit_predict(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    betas = ridge_regression(x, y, alpha)
    return betas, x @ betas
