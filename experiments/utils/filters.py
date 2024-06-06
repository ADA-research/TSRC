__all__ = ['get_data_list_filter']

from pathlib import Path
from typing import List, Callable

from experiments.utils.parsers import file_name_raw_data_parser


def filter_raw_data_list(data_list: List[Path]) -> List[Path]:
    """Keep only the files that are the test data."""
    return [file for file in data_list if file_name_raw_data_parser(file).get('data_part') == 'test']


def identity_filter(data_list: List[Path]) -> List[Path]:
    """An identity filter that returns the data list as is."""
    return data_list


def get_data_list_filter(task_type: str, data_type: str) -> Callable:
    """
    Returns the appropriate filter function based on task type and data type.

    :param task_type: The type of task ('classification' or 'clustering').
    :param data_type: The type of data ('raw' or 'representation').
    :return: A callable filter function.
    """
    if task_type == 'clustering' and data_type == 'raw':
        return filter_raw_data_list
    elif data_type == 'representation' or task_type == 'classification':
        return identity_filter
    else:
        raise ValueError(f"Invalid task_type '{task_type}' or data_type '{data_type}' combination")
