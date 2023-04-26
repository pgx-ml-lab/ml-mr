"""
Linear models and other utilities.
"""


import torch


def ridge_regression(x, y, alpha):
    n_samples, n_features = x.shape
    assert y.shape[0] == n_samples

    L = x.T @ x + alpha * torch.eye(n_features)
    L_chol = torch.linalg.cholesky(L)
    betas = torch.cholesky_solve(x.T @ y, L_chol)

    return betas
