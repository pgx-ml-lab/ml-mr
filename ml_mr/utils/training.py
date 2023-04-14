"""
Utilities to simplify fitting neural networks.
"""

import os
from typing import Union, Optional, Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger


from ..estimation.core import Dataset
from ..logging import info
from . import parse_project_and_run_name


def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    model: Union[pl.LightningModule, nn.Module],
    monitored_metric: str,
    output_dir: str,
    checkpoint_filename: str,
    batch_size: int,
    max_epochs: int,
    accelerator: Optional[str] = None,
    wandb_project: Optional[str] = None,
    early_stopping_patience: int = 20,
) -> float:
    if not checkpoint_filename.endswith(".ckpt"):
        checkpoint_filename += ".ckpt"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=len(val_dataset), num_workers=0  # type: ignore
    )

    # Remove checkpoint if exists.
    full_filename = os.path.join(output_dir, checkpoint_filename)
    if os.path.isfile(full_filename):
        info(f"Removing file '{full_filename}'.")
        os.remove(full_filename)

    logger: Union[bool, Iterable[Logger]] = True
    if wandb_project is not None:
        from pytorch_lightning.loggers.wandb import WandbLogger
        project, run_name = parse_project_and_run_name(wandb_project)
        logger = [
            WandbLogger(name=run_name, project=project)
        ]

    # .ckpt added by pl, so we remove it.
    checkpoint_filename = checkpoint_filename[:-5]

    model_checkpoint = pl.callbacks.ModelCheckpoint(
        filename=checkpoint_filename,
        dirpath=output_dir,
        save_top_k=1,
        monitor=monitored_metric,
    )

    trainer = pl.Trainer(
        log_every_n_steps=1,
        max_epochs=max_epochs,
        accelerator=accelerator,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor=monitored_metric, patience=early_stopping_patience
            ),
            model_checkpoint,
        ],
        logger=logger
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    # Return the best score on the tracked metric.
    score = model_checkpoint.best_model_score
    assert isinstance(score, torch.Tensor)
    return score.item()