__all__ = ['convert_numpy_to_tensor', 'expand_data_dimensionality']

from typing import List, Tuple, Union

import numpy as np
import torch


def convert_numpy_to_tensor(data: np.ndarray, dtype='float') -> torch.Tensor:
    """
    Convert numpy array to tensor.
    :param data: numpy array
    :param dtype: data type
    :return: tensor
    """

    dtype_map = {
        'float': torch.float,
        'long': torch.long,
        'int': torch.int,
        'double': torch.double,
    }
    tensor = torch.from_numpy(data)

    return tensor.to(dtype=dtype_map[dtype])


def expand_data_dimensionality(data: Union[np.ndarray, torch.Tensor, List, Tuple],
                               expand_dims_axis: int) -> np.ndarray:
    """
    Expand data dimensionality.
    :param data: data to expand
    :param expand_dims_axis: axis to expand
    :return: expanded data
    """
    return np.expand_dims(data, axis=expand_dims_axis)
