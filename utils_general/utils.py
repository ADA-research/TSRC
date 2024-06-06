__all__ = ['get_num_samples_from_ts', 'load_json', 'compose', 'flatten_list', 'flatten_list_of_np_arrays',
           'get_data_stats_single', 'get_data_stats_list', 'separate_target_feature_from_df', 'prepare_dropout_values',
           'convert_tuple_list_to_2d_array', 'closest_power_of_2', 'add_prefix_to_string',
           'find_project_root']

import json
import math
from collections import namedtuple
from pathlib import Path
from typing import List, Any, Tuple, Union, Optional, Callable

import numpy as np
import pandas as pd


def load_json(json_file: str) -> dict:
    """
    Load json file.
    :param json_file: json file to load
    :return: json file
    """
    with open(json_file, 'r') as f:
        return json.load(f)


def get_num_samples_from_ts(ts: np.ndarray) -> int:
    """
    Get number of samples from time series.
    :param ts: time series
    :return: length of time series
    """
    return len(ts)


class FunctionComposer:
    def __init__(self, functions: List[Callable]):
        self.functions = [f for f in functions if f is not None]

    def __call__(self, data):
        result = data
        for f in reversed(self.functions):
            result = f(result)
        return result


def compose(*functions: Callable) -> Callable:
    """
    Compose functions.
    :param functions: the functions to compose
    :return: a callable that is the composition of the given functions
    """
    return FunctionComposer(list(functions))


def flatten_list(list_of_lists: List[List[Any]]) -> List[Any]:
    """
    Flatten a list of lists.
    :param list_of_lists: list of lists
    :return: flattened list
    """
    if not list_of_lists:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten_list(list_of_lists[0]) + flatten_list(list_of_lists[1:])
    return list_of_lists[:1] + flatten_list(list_of_lists[1:])


def flatten_list_of_np_arrays(list_of_np_arrays: List[np.ndarray]) -> np.ndarray:
    """
    Flatten a list of numpy arrays.
    :param list_of_np_arrays: list of numpy arrays
    :return: flattened np array
    """
    return np.concatenate(list_of_np_arrays).ravel()


def get_data_stats_single(data: np.array) -> namedtuple:
    """
    Get statistics for data.
    :param data: data to get statistics for
    :return: namedtuple with statistics
    """
    data_stats = namedtuple('data_stats', ['median', 'mean', 'std', 'min', 'max'])
    return data_stats(np.median(data), np.mean(data), np.std(data), np.min(data), np.max(data))


def get_data_stats_list(data: List[np.array]) -> namedtuple:
    """
    Get statistics for data.
    :param data: data to get statistics for
    :return: namedtuple with statistics
    """
    f_data = flatten_list_of_np_arrays(data)
    return get_data_stats_single(f_data)


def separate_target_feature_from_df(df: pd.DataFrame, target_feature_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate target feature from dataframe.
    :param df: dataframe to separate target feature from
    :param target_feature_name: name of target feature
    :return: tuple of dataframe without target feature and target feature
    """
    target_feature = df[target_feature_name]
    df = df.drop(target_feature_name, axis=1)
    return df, target_feature


def prepare_dropout_values(layers: List[int], dropout: Union[float, List[float]]) -> List[float]:
    """
    Prepare dropout values for layers. If dropout is a float, then the same dropout value is used for all layers.
    :param layers: the layers to prepare dropout values for
    :param dropout: the dropout values to use for the layers (either a float or a list of floats)
    :return:
    """
    if isinstance(dropout, float):
        return [dropout] * len(layers)
    elif len(dropout) == len(layers):
        return dropout
    else:
        raise ValueError("The number of dropout values must be equal to the number of layers")


def convert_tuple_list_to_2d_array(tuple_list: List[Tuple]) -> np.ndarray:
    """
    Function to convert a list of tuples to a 2D numpy array.
    This function is specifically designed to be used with the pandas apply function.

    :param tuple_list: A list of tuples to be converted.
                       Each tuple is considered as a sequence of values.
    :type tuple_list: List[Tuple]

    :return: A 2D numpy array where each tuple is transformed into an individual list (or sub-array).
    :rtype: np.ndarray
    """

    # Initialize an empty list to store the converted tuples
    array_data = []

    # Iterate through each tuple in the list
    for tup in tuple_list:
        # Convert the tuple to a list and append it to array_data
        array_data.append(list(tup))

    # Convert the list of lists to a numpy array and return it
    return np.array(array_data)


def closest_power_of_2(m: Union[int, float], v: Union[int, float]) -> int:
    """
    Calculate the closest power of 2 for the division result of two numbers.

    This function takes two numbers, `m` and `v`, divides `m` by `v`, and finds the nearest
    power of 2 to the division result, whether it's lower or higher.

    @param m: The dividend. It can be an integer or a floating-point number.
    @param v: The divisor. It can be an integer or a floating-point number.
    @return: The closest power of 2 to the division result.

    @raise ZeroDivisionError: If `v` is zero.

    @example
    ```
    m = 100
    v = 3
    result = closest_power_of_2(m, v)  # Output will be 32 or 64 depending on which is closer
    ```
    """
    # Check for zero division
    if v == 0:
        raise ZeroDivisionError("The divisor v should not be zero.")

    # Calculate the division result
    div_result = m / v

    # Find the logarithm base 2 of the division result
    log_result = math.log2(div_result)

    # Determine the lower and upper power of 2
    lower_power = 2 ** math.floor(log_result)
    upper_power = 2 ** math.ceil(log_result)

    # Determine which is closer to the division result
    if abs(div_result - lower_power) < abs(upper_power - div_result):
        return lower_power
    else:
        return upper_power


def add_prefix_to_string(base: str, prefix: Optional[str] = None) -> str:
    """
    Add prefix to string if prefix is not None.
    :param base: string to add prefix to
    :param prefix: prefix to add
    :return: string with prefix
    """
    return f'{prefix}_{base}' if prefix else base


def find_project_root(current_path: Path, marker: str = 'README.md') -> Path:
    """
    Recursively look for a marker file to find the project root.
    """
    # Check if the marker exists in the current path
    if (current_path / marker).exists():
        return current_path
    # If the marker is not found and this is not the root path, check the parent directory
    elif current_path.parent != current_path:
        return find_project_root(current_path.parent, marker)
    else:
        # If the root is reached without finding the marker, return None or raise an error
        raise FileNotFoundError(f"Project root marker '{marker}' not found.")
