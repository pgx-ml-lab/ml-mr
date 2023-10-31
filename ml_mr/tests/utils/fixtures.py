import pytest
import pandas as pd

from ...utils.data import IVDataset


@pytest.fixture
def iv_dataset_range():
    df = pd.DataFrame({
        "x": range(1000),
        "y": range(1000),
        "iv1": range(1000),
        "iv2": range(1000),
        "covar1": range(1000),
        "covar2": range(1000)
    })

    return IVDataset.from_dataframe(
        df, "x", "y", ["iv1", "iv2"], ["covar1", "covar2"]
    )
