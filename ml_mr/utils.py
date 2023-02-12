"""
General utilities to construct neural networks. Used across the project for
causal effect estimation or for simulation, for example.
"""

import argparse
from typing import Optional, Iterable, List, Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MLP(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden: Iterable[int],
        out: int = 1,
        add_input_layer_batchnorm: bool = False,
        add_hidden_layer_batchnorm: bool = False,
        activations: Iterable[nn.Module] = [nn.LeakyReLU()],

        # Hyperparameters and training parameters.
        lr: float = 1e-3,
        weight_decay: float = 0,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss
    ):
        super().__init__()
        self.save_hyperparameters()

        layers = build_mlp(input_size, hidden, out, add_input_layer_batchnorm,
                           add_hidden_layer_batchnorm, activations)

        self.loss = loss
        self.mlp = nn.Sequential(*layers)

    def _step(
        self,
        batch,
        batch_index,
        log_prefix: str = "train"
    ) -> torch.Tensor:
        x, y = batch

        y_hat = self.mlp(x)
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
    def add_argparse_parameters(
        parser: argparse.ArgumentParser,
        argument_prefix: Optional[str] = None,
        group_name: str = "MLP Parameters"
    ) -> None:
        if argument_prefix is None:
            prefix = ""
        else:
            prefix = argument_prefix

        group = parser.add_argument_group(group_name)

        group.add_argument(
            f"--{prefix}max-epochs",
            type=int,
            default=100
        )

        group.add_argument(
            f"--{prefix}batch-size",
            type=int,
            default=1024
        )

        group.add_argument(
            f"--{prefix}optimizer",
            type=str,
            default="adam"
        )

        group.add_argument(
            f"--{prefix}learning-rate",
            type=float,
            default=1e-4
        )

        group.add_argument(
            f"--{prefix}weight-decay",
            type=float,
            default=0
        )
