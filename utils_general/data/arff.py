from pathlib import Path
from typing import Tuple, Callable, Any, Union

import pandas as pd
from scipy.io import arff

__all__ = ['read_arff_as_df', 'process_df_according_to_dtypes']


def read_arff_as_df(arff_file_path: Union[Path, str]) -> Tuple[pd.DataFrame, Any]:
    """
    Read arff file as pandas dataframe.
    :param arff_file_path: path to arff file
    :return: pandas dataframe with arff data
    """
    data, meta = arff.loadarff(arff_file_path)
    df_data = pd.DataFrame(data)
    return df_data, meta


def process_df_according_to_dtypes(df_data: pd.DataFrame, meta: Any, dtypes_functions_map: dict[str, Callable]) \
        -> pd.DataFrame:
    """
    Process dataframe according to dtypes functions map.
    :param df_data: dataframe to process
    :param meta: arff metadata extracted from arff file
    :param dtypes_functions_map: map of dtypes to functions
    :return: processed dataframe
    """
    for col_name in meta.names():
        col_type = meta[col_name][0]
        df_data[col_name] = dtypes_functions_map[col_type](df_data[col_name])
    return df_data
