__all__ = ['TimeSeriesDatasetMode', 'TimeSeriesClassificationDatasetSplittingStrategy']

from enum import Enum


class TimeSeriesDatasetMode(Enum):
    WITH_LABELS = 'with_labels'
    WITHOUT_LABELS = 'without_labels'


class TimeSeriesClassificationDatasetSplittingStrategy(Enum):
    AS_DEFINED = 'as_defined'
    MANUAL = 'manual'
