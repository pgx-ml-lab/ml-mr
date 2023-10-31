import torch
from torch.utils.data import DataLoader
import pandas as pd

from ...utils.training import resample_dataset

from .fixtures import *  # noqa: F401, F403


def test_resample(iv_dataset_range):
    bs = resample_dataset(iv_dataset_range)

    n = len(bs)
    dl = DataLoader(bs, batch_size=n)

    x, _, _, _ = next(iter(dl))

    assert not torch.all(x == torch.arange(n))


def test_resample_df(iv_dataset_range):
    df_0, _ = iv_dataset_range.to_dataframe()

    bs = resample_dataset(iv_dataset_range)
    df_1, _ = bs.to_dataframe()

    try:
        pd.testing.assert_frame_equal(df_0, df_1)    
        FRAMES_EQ = True
    except AssertionError:
        FRAMES_EQ = False

    assert not FRAMES_EQ
