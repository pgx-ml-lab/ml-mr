"""
Mixture density network implementation.

Inspired heavily from the implementation from pytorch-tabular:

https://github.com/manujosephv/pytorch_tabular

"""

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .nn import build_mlp, DensityModel


EPS = 1e-30
LOG2PI = math.log(2 * math.pi)


class MixtureDensityNetwork(DensityModel):
    def __init__(
        self,
        input_size: int,
        hidden: Iterable[int],
        n_components: int,
        lr: float = 1e-3,
        weight_decay: float = 0,
        add_input_layer_batchnorm: bool = False,
        add_hidden_layer_batchnorm: bool = False,
        activations: Iterable[nn.Module] = [nn.LeakyReLU()],
        softmax_temperature: float = 1.0
    ):
        super().__init__()
        self.save_hyperparameters()

        hidden = list(hidden)

        self.mlp = nn.Sequential(*build_mlp(
            input_size=input_size,
            hidden=hidden,
            add_input_layer_batchnorm=add_input_layer_batchnorm,
            add_hidden_layer_batchnorm=add_hidden_layer_batchnorm,
            activations=activations
        ))

        rep_size = hidden[-1]

        # Add a head for the Gaussian mixture.
        self.pi = nn.Linear(rep_size, n_components)
        self.log_sigma = nn.Linear(rep_size, n_components)
        self.mu = nn.Linear(rep_size, n_components)

    def forward_parameters(self, x):
        """Forward pass up to the parameters of the mixture model."""
        rep = self.mlp(x)
        pi = self.pi(rep)
        mu = self.mu(rep)
        log_sigma = self.log_sigma(rep)
        sigma = F.softplus(log_sigma) + EPS

        return pi, mu, sigma

    @staticmethod
    def gaussian_log_likelihood(x, mu, sigma):
        """Simultaneously compute k gaussian log likelihoods for input vector.

        sigma is BxK: Batch x K-components.
        mu is BxK
        x is B

        returns log densities (BxK)

        """
        n = x.size(0)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        sigma2 = sigma ** 2
        return (
            -1 / (2 * sigma2) * torch.sum((x - mu) ** 2)
            - n / 2 * (LOG2PI + torch.log(sigma2 + EPS))
        )

    def model_nll(self, x, y):
        # Get the predicted parameters.
        pi, mu, sigma = self.forward_parameters(x)

        log_component_prob = self.gaussian_log_likelihood(y, mu, sigma)
        log_mix_prob = torch.log(
            F.gumbel_softmax(
                pi,
                tau=self.hparams.softmax_temperature,
                dim=-1
            ) + EPS
        )

        ll = torch.logsumexp(log_component_prob + log_mix_prob, dim=-1)
        return -torch.mean(ll)

    @staticmethod
    def sample_given_params(n_samples, pi, mu, sigma, device=None):
        cat = torch.distributions.Categorical(logits=pi)
        components = cat.sample().unsqueeze(1)

        noise = torch.randn((mu.size(0), n_samples), device=device)
        return noise * sigma.gather(1, components) + mu.gather(1, components)

    def sample(self, x, n_samples, device=None):
        pis, mus, sigmas = self.forward_parameters(x)
        return self.sample_given_params(n_samples, pis, mus, sigmas, device)

    def training_step(self, batch, batch_index):
        x, y = batch
        loss = self.model_nll(x, y)
        self.log("mdn_train_nll", loss)
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        loss = self.model_nll(x, y)
        self.log("mdn_val_nll", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
