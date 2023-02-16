from typing import Iterable

import pandas as pd


def read_data(
    filename: str, sep: str, expect_columns: Iterable[str]
) -> pd.DataFrame:
    """Read a delimited data file and do minimal QC."""
    return pd.read_csv(filename, sep=sep, usecols=expect_columns)
