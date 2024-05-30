"""
Internal utilities to construct neural networks.
"""

import argparse
from typing import Optional, Iterable, List, Callable, Dict, Any, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .quantiles import quantile_loss
from .linear import ridge_regression
from ..utils.data import IVDataset, SupervisedLearningWrapper


def build_mlp(
    input_size: int,
    hidden: Iterable[int],
    out: Optional[int] = None,
    add_input_layer_batchnorm: bool = False,
    add_hidden_layer_batchnorm: bool = False,
    activations: Iterable[nn.Module] = [nn.LeakyReLU()]
):
    layers: List[nn.Module] = []

    if add_input_layer_batchnorm:
        layers.append(nn.BatchNorm1d(input_size))

    h_prev = input_size
    for h in hidden:
        layers.extend([
            nn.Linear(in_features=h_prev, out_features=h),
        ])
        layers.extend(activations)
        if add_hidden_layer_batchnorm:
            layers.append(nn.BatchNorm1d(h))

        h_prev = h

    if out is not None:
        layers.append(nn.Linear(in_features=h_prev, out_features=out))

    return layers


class DensityModel(pl.LightningModule):
    """Mixin for models that allow sampling.

    For example, the Mixture Density Network and the Gaussian networks enable
    sampling from conditional densities.

    """
    def sample(
        self,
        x: torch.Tensor,
        n_samples: int,
        device: Optional[torch.device] = None
    ):
        raise NotImplementedError()


class RidgeDensity(DensityModel):
    def fit(self, dataset: IVDataset, alpha: float = 1.0):
        dl = DataLoader(
            SupervisedLearningWrapper(dataset), batch_size=len(dataset)
        )

        x, y = next(iter(dl))

        # Get ridge solution.
        self.betas = ridge_regression(x, y, alpha)

        # Predicted values.
        y_hat = x @ self.betas
        self.sigma = torch.std(y - y_hat)

    def sample(
        self,
        x: torch.Tensor,
        n_samples: int,
        device: Optional[torch.device] = None
    ):
        if device is not None:
            self.betas = self.betas.to(device)

        y_pred = x @ self.betas
        eps = torch.randn((x.size(0), n_samples), device=device) * self.sigma
        return y_pred + eps


class MLP(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden: Iterable[int],
        out: Optional[int] = 1,
        add_input_layer_batchnorm: bool = False,
        add_hidden_layer_batchnorm: bool = False,
        activations: Iterable[nn.Module] = [nn.LeakyReLU()],

        # Hyperparameters and training parameters.
        lr: float = 1e-3,
        weight_decay: float = 0,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,  # noqa: E501
        _save_hyperparams: bool = True
    ):
        super().__init__()
        if _save_hyperparams:
            self.save_hyperparameters()

        layers = build_mlp(input_size, hidden, out, add_input_layer_batchnorm,
                           add_hidden_layer_batchnorm, activations)

        if out is not None:
            # We split the linear representation from the rest of the model
            # for convenience in some model formulations.
            self.representation: Optional[nn.Module] = nn.Sequential(
                *layers[:-1]
            )
            self.mlp = nn.Sequential(self.representation, layers[-1])
        else:
            self.representation = None
            self.mlp = nn.Sequential(*layers)

        self.loss = loss
        self.mlp = nn.Sequential(*layers)

    def get_repr(self, x: torch.Tensor) -> torch.Tensor:
        if self.representation is None:
            raise ValueError(
                "No output layer specified. Can't interpret MLP as learning a "
                "representation prior to prediction."
            )

        return self.representation(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def _step(
        self,
        batch,
        batch_index,
        log_prefix: str = "train"
    ) -> torch.Tensor:
        x, y = batch

        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        self.log(f"{log_prefix}_loss", loss)

        return loss

    def training_step(self, batch, batch_index):
        return self._step(batch, batch_index, "train")

    def validation_step(self, batch, batch_index):
        return self._step(batch, batch_index, "val")

    def test_step(self, batch, batch_index):
        return self._step(batch, batch_index, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,  # type: ignore
            weight_decay=self.hparams.weight_decay  # type: ignore
        )

    @staticmethod
    def add_mlp_arguments(
        parser: argparse.ArgumentParser,
        argument_prefix: Optional[str] = None,
        group_name: str = "MLP Parameters",
        defaults: Optional[Dict[str, Any]] = None
    ) -> None:
        if argument_prefix is None:
            prefix = ""
        else:
            prefix = argument_prefix

        group = parser.add_argument_group(group_name)

        if defaults is None:
            defaults = {}

        group.add_argument(
            f"--{prefix}hidden",
            nargs="*",
            default=defaults.get("hidden", []),
            type=int
        )

        group.add_argument(
            f"--{prefix}max-epochs",
            type=int,
            default=defaults.get("max-epochs", 1000)
        )

        group.add_argument(
            f"--{prefix}batch-size",
            type=int,
            default=defaults.get("batch-size", 1024)
        )

        group.add_argument(
            f"--{prefix}optimizer",
            type=str,
            default=defaults.get("optimizer", "adam")
        )

        group.add_argument(
            f"--{prefix}learning-rate",
            type=float,
            default=defaults.get("learning-rate", 1e-4)
        )

        group.add_argument(
            f"--{prefix}weight-decay",
            type=float,
            default=defaults.get("weight-decay", 0)
        )

        group.add_argument(
            f"--{prefix}add-input-batchnorm",
            action="store_true"
        )


class GaussianNet(MLP, DensityModel):
    """MLP that outputs a mu and sigma and trains using the Gaussian NLL."""
    def __init__(
        self,
        input_size: int,
        hidden: Iterable[int],
        add_input_layer_batchnorm: bool = False,
        add_hidden_layer_batchnorm: bool = False,
        activations: Iterable[nn.Module] = [nn.LeakyReLU()],

        # Hyperparameters and training parameters.
        lr: float = 1e-3,
        weight_decay: float = 0,
        _save_hyperparams: bool = True
    ):
        hidden = list(hidden)
        super().__init__(
            input_size,
            hidden=hidden,
            out=None,
            add_input_layer_batchnorm=add_input_layer_batchnorm,
            add_hidden_layer_batchnorm=add_hidden_layer_batchnorm,
            activations=activations,
            lr=lr,
            weight_decay=weight_decay,
            loss=F.gaussian_nll_loss,  # type: ignore
            _save_hyperparams=_save_hyperparams
        )

        self.mu_head = nn.Linear(hidden[-1], 1)
        self.sigma2_head = nn.Linear(hidden[-1], 1)

    def forward(self, x: torch.Tensor):
        raise NotImplementedError()

    def forward_parameters(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.mlp(x)
        mu = self.mu_head(z)
        sigma2 = F.softplus(self.sigma2_head(z))
        return mu, sigma2

    def sample(self, x, n_samples, device=None):
        mu, sigma2 = self.forward_parameters(x)
        return sigma2 * torch.randn(n_samples, device=device) + mu

    def _step(
        self,
        batch,
        batch_index,
        log_prefix: str = "train"
    ) -> torch.Tensor:
        x, y = batch

        mu, sigma2 = self.forward_parameters(x)
        loss = self.loss(mu, y, sigma2)  # type: ignore

        self.log(f"{log_prefix}_loss", loss)

        return loss


class OutcomeMLPBase(MLP):
    def __init__(
        self,
        exposure_network: pl.LightningModule,
        input_size: int,
        hidden: Iterable[int],
        lr: float,
        weight_decay: float = 0,
        binary_outcome: bool = False,
        sqr: bool = False,
        add_input_layer_batchnorm: bool = False,
        add_hidden_layer_batchnorm: bool = False,
        activations: Iterable[nn.Module] = [nn.GELU()]
    ):
        if sqr:
            loss: Callable = quantile_loss
        elif binary_outcome:
            loss = F.binary_cross_entropy_with_logits
        else:
            loss = F.mse_loss

        super().__init__(
            input_size=input_size if not sqr else input_size + 1,
            hidden=hidden,
            out=1,
            add_input_layer_batchnorm=add_input_layer_batchnorm,
            add_hidden_layer_batchnorm=add_hidden_layer_batchnorm,
            activations=activations,
            lr=lr,
            weight_decay=weight_decay,
            loss=loss,  # type: ignore
            _save_hyperparams=False,
        )
        self.exposure_network = exposure_network
        self.save_hyperparameters(ignore=["exposure_network"])

    def _step(self, batch, batch_index, log_prefix):
        _, y, ivs, covars = batch

        if self.hparams.sqr:
            taus = torch.rand(ivs.size(0), 1, device=self.device)
        else:
            taus = None

        y_hat = self.forward(ivs, covars, taus)

        if self.hparams.sqr:
            loss = self.loss(y_hat, y, taus)
        else:
            loss = self.loss(y_hat, y)

        self.log(f"outcome_{log_prefix}_loss", loss)

        return loss

    def x_to_y(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor],
        taus: Optional[Union[torch.Tensor, float]] = None
    ) -> torch.Tensor:
        if taus is not None and not self.hparams.sqr:  # type: ignore
            raise ValueError("Can't provide tau if SQR not enabled.")

        stack = [x]

        if covars is not None and covars.numel() > 0:
            stack.append(covars)

        if taus is None and self.hparams.sqr:  # type: ignore
            # Predict median by default.
            taus = torch.full((x.size(0), 1), 0.5)

        if taus is not None:
            if isinstance(taus, torch.Tensor):
                stack.append(taus)
            elif isinstance(taus, float):
                stack.append(torch.full((x.size(0), 1), taus))
            else:
                raise ValueError("Provide vector of taus or float.")

        x = torch.hstack(stack)

        return self.mlp(x)

    def forward(  # type: ignore
        self,
        ivs: torch.Tensor,
        covars: Optional[torch.Tensor],
        taus: Optional[torch.Tensor] = None
    ):
        """Forward pass throught the exposure and outcome models."""
        raise NotImplementedError()
