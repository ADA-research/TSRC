__all__ = ['FixedTimeSeriesDatasetUnivariate']

from abc import ABC, abstractmethod
from functools import partial
from typing import List, Union, Tuple, Optional, Callable

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from project_datasets.classes.transformations import expand_data_dimensionality
from enums import TimeSeriesDatasetMode
from utils_general import compose


class TimeSeriesDataset(Dataset, ABC):
    _get_sample_fun_map = {
        TimeSeriesDatasetMode.WITHOUT_LABELS: '_get_sample_1',
        TimeSeriesDatasetMode.WITH_LABELS: '_get_sample_2'
    }

    def __init__(self, data,
                 labels,
                 mode: TimeSeriesDatasetMode,
                 expand_dims_axis: Optional[int],
                 transformations_sequence: Optional[Union[List[Callable], Tuple[Callable]]]):
        super().__init__()
        self._data = data
        self._labels = labels
        self._mode = mode
        self._get_sample = getattr(self, self._get_sample_fun_map[self._mode])
        self._initiate_transformation_functionality(transformations_sequence, expand_dims_axis)

    @abstractmethod
    def _go_to_idx(self, idx: int):
        pass

    @abstractmethod
    def _get_current_data(self) -> np.ndarray:
        pass

    @abstractmethod
    def _get_current_label(self) -> Optional[Union[np.ndarray, int]]:
        pass

    def _initiate_transformation_functionality(self, transformations_sequence, expand_dims_axis):
        transformations_sequence = [*transformations_sequence]
        if expand_dims_axis:
            transformations_sequence.append(partial(expand_data_dimensionality, expand_dims_axis=expand_dims_axis))
        self._transform = compose(*transformations_sequence)

    def _get_sample_1(self) -> np.ndarray:
        sample = self._get_current_data()
        return self._transform(sample)

    def _get_sample_2(self) -> Tuple[np.ndarray, Optional[Union[np.ndarray, int]]]:
        sample = self._get_current_data()
        label = self._get_current_label()
        return self._transform(sample), label

    def __getitem__(self, item):
        self._go_to_idx(item)
        sample = self._get_sample()
        return sample


class FixedTimeSeriesDataset(TimeSeriesDataset, ABC):
    def __init__(self,
                 data: pd.DataFrame,
                 labels: Optional[Union[pd.Series, pd.DataFrame]],
                 mode,
                 expand_dims_axis: Optional[int],
                 transformations_sequence: Optional[Union[List[Callable], Tuple[Callable]]]):
        super().__init__(
            data=data,
            labels=labels,
            mode=mode,
            expand_dims_axis=expand_dims_axis,
            transformations_sequence=transformations_sequence
        )
        self._n = 0

    def __len__(self):
        return len(self._data)

    def _go_to_idx(self, idx: int):
        self._n = idx

    @abstractmethod
    def _get_current_data(self) -> np.ndarray:
        pass

    def _get_current_label(self) -> Optional[int]:
        if self._labels is None:
            return None
        return self._labels.iloc[self._n]


class FixedTimeSeriesDatasetUnivariate(FixedTimeSeriesDataset, ABC):
    def __init__(self,
                 data: pd.DataFrame,
                 labels: Optional[Union[pd.Series, pd.DataFrame]],
                 mode,
                 expand_dims_axis: Optional[int],
                 transformations_sequence: Optional[Union[List[Callable], Tuple[Callable]]]):
        super().__init__(
            data=data,
            labels=labels,
            mode=mode,
            expand_dims_axis=expand_dims_axis,
            transformations_sequence=transformations_sequence
        )
        self._n = 0

    def _get_current_data(self) -> np.ndarray:
        return self._data.iloc[self._n].values
