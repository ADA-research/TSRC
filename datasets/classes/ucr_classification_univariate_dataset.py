__all__ = ['UCRClassificationUnivariateDataset']

from datasets.classes.abstract import FixedTimeSeriesDatasetUnivariate

from datasets.classes.transformations import convert_numpy_to_tensor


class UCRClassificationUnivariateDataset(FixedTimeSeriesDatasetUnivariate):
    def __init__(self, data,
                 labels,
                 mode,
                 expand_dims_axis=1,
                 transformations_sequence=(convert_numpy_to_tensor,)):
        super().__init__(data=data,
                         labels=labels,
                         mode=mode,
                         expand_dims_axis=expand_dims_axis,
                         transformations_sequence=transformations_sequence)
