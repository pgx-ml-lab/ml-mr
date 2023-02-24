"""Utilities for methods that work with quantiles.

This include the pinball loss and tools to measure uncertainty such as
simultaneous quantile regression (see Tagasovska & Lopez-Paz NeurIPS 2019).


"""


import torch


def quantile_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    tau: torch.Tensor
) -> torch.Tensor:
    """Implementation of the quantile loss.

    Adapted from https://github.com/facebookresearch/SingleModelUncertainty

    """
    diff = target - input
    mask = (diff.ge(0).float() - tau).detach()
    return (mask * diff).mean()


class QuantileLossMulti(object):
    """Fits multiple (but discrete) quantile losses simultaneously."""
    def __init__(self, quantiles: torch.Tensor):
        self.quantiles = quantiles

    def __call__(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        y_1 = torch.maximum(target - input, torch.zeros_like(target))
        y_2 = torch.maximum(input - target, torch.zeros_like(target))
        loss = (y_1 @ self.quantiles + y_2 @ (1 - self.quantiles))

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise ValueError("Unknown reduction: '{reduction}'.")
