import torch
from torch.utils.data import DataLoader

from ...utils.training import resample_dataset

from .fixtures import *  # noqa: F401, F403


def test_resample(iv_dataset_range):
    bs = resample_dataset(iv_dataset_range)

    n = len(bs)
    dl = DataLoader(bs, batch_size=n)

    x, _, _, _ = next(iter(dl))

    assert not torch.all(x == torch.arange(n))
