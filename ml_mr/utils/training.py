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
from ..logging import info
from . import parse_project_and_run_name
from .data import Dataset, FullBatchDataLoader


def resample_dataset(dataset: Dataset) -> Dataset:
    n = len(dataset)  # type: ignore

    if getattr(dataset, "sampling_weights") is None:
        weights = torch.ones(n)
    else:
        info("Using attached sampling weights.")
        weights = dataset.sampling_weights

    bootstrap_idx = torch.multinomial(
        weights, n, replacement=True
    ).tolist()

    class ResampledDataset(Dataset):
        def __getattr__(self, k):
            # Defer to dataset if unknown method, only change the behaviour of
            # indexing.
            return getattr(dataset, k)

        def to_dataframe(self):
            df, cols = dataset.to_dataframe()
            df = df.iloc[bootstrap_idx, :].reset_index(drop=True)
            return df, cols

        def __len__(self):
            return n

        def __getitem__(self, idx):
            bs_idx = bootstrap_idx[idx]
            return dataset[bs_idx]

    return ResampledDataset()


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
    use_full_batch_validation: bool = True
) -> float:
    if not checkpoint_filename.endswith(".ckpt"):
        checkpoint_filename += ".ckpt"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    if use_full_batch_validation:
        val_dataloader = FullBatchDataLoader(val_dataset)
    else:
        val_dataloader = DataLoader(
            val_dataset, batch_size=len(val_dataset),  # type: ignore
            num_workers=0
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
        accelerator=accelerator,  # type: ignore
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor=monitored_metric, patience=early_stopping_patience
            ),
            model_checkpoint,
        ],
        logger=logger,
        enable_progress_bar=os.environ.get("ML_MR_QUIET", "0") != "1"
    )
    trainer.fit(model, train_dataloader, val_dataloader)  # type: ignore

    # Return the best score on the tracked metric.
    score = model_checkpoint.best_model_score
    assert isinstance(score, torch.Tensor)
    return score.item()
