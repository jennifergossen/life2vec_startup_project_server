from typing import List, Optional, Sequence

import dask.dataframe as dd
import numpy as np
import pandas as pd


def _sort_using_index(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return data.set_index(columns, append=True).sort_index().reset_index(columns)


def sort_partitions(data: dd.core.DataFrame, columns: Sequence[str]) -> dd.core.DataFrame:
    result = data.map_partitions(_sort_using_index, columns=columns)
    assert isinstance(result, dd.core.DataFrame)
    return result
