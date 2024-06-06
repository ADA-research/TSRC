__all__ = ['UCRTimeSeriesClassificationUnivariateDataModule']

from collections import defaultdict
from typing import Tuple

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from datasets.classes import UCRClassificationUnivariateDataset
from datasets.modules.abstract import TimeSeriesClassificationDataModule
from enums import TimeSeriesDatasetMode, \
    TimeSeriesClassificationDatasetSplittingStrategy
from utils_general.data.strategies.normalization import create_data_normalizer


class UCRTimeSeriesClassificationUnivariateDataModule(TimeSeriesClassificationDataModule):
    def __init__(self,
                 dataset_folder_path,
                 dataset_config_path,
                 batch_size: int = 32,
                 seq_len: int = 128,
                 valid_size: float = 0.1,  # percentage of the training set to use as validation set
                 shuffle: bool = False,
                 normalize: bool = True,
                 normalize_range: Tuple[float, float] = (0, 1),
                 splitting_strategy: TimeSeriesClassificationDatasetSplittingStrategy =
                 TimeSeriesClassificationDatasetSplittingStrategy.AS_DEFINED,
                 test_size: float = 0.5,
                 # percentage of the dataset to use as test set; only valid for MANUAL splitting
                 num_workers: int = 1):
        super().__init__(
            dataset_folder_path=dataset_folder_path,
            dataset_config_path=dataset_config_path,
            batch_size=batch_size,
            seq_len=seq_len,
            valid_size=valid_size,
            shuffle=shuffle,
            normalize=normalize,
            normalize_range=normalize_range,
            splitting_strategy=splitting_strategy,
            test_size=test_size,
            num_workers=num_workers)

    def _initiate_datatypes_handling_functions_map(self):
        datatype_handling_functions_map = {
            'nominal': lambda x: x.str.decode('utf-8').astype('category').astype('int64'),
            'numeric': lambda x: x.astype('float64')
        }

        self.datatype_handling_functions_map = defaultdict(lambda: lambda x: x,
                                                           datatype_handling_functions_map)

    @property
    def sequence_len(self):
        return self.seq_len

    @sequence_len.setter
    def sequence_len(self, value):
        self.seq_len = value

    def setup(self, stage: str = None) -> None:
        normalizer = create_data_normalizer(normalize=self.normalize, normalize_range=self.normalize_range)
        self._train_data_samples, self._valid_data_samples, self._test_data_samples = \
            normalizer(self._train_data_samples, self._valid_data_samples, self._test_data_samples)

    def _get_custom_collate_fn(self, desired_batch_size: int = None):
        if desired_batch_size is None:
            desired_batch_size = self.batch_size

        def _custom_collate_fn(batch):
            current_batch_size = len(batch)
            if current_batch_size < desired_batch_size:
                # Calculate how many additional samples are needed
                additional_needed = desired_batch_size - current_batch_size
                # Duplicate last items in the batch
                additional_samples = [batch[-1] for _ in range(additional_needed)]
                # Add these to the original batch
                batch.extend(additional_samples)
            return default_collate(batch)

        return _custom_collate_fn

    def train_dataloader(self,
                         mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.WITHOUT_LABELS,
                         shuffle: bool = None,
                         strict_batch_size: bool = False,
                         extra_args: dict = None) -> DataLoader:
        dataset = UCRClassificationUnivariateDataset(data=self._train_data_samples,
                                                     labels=self._train_data_labels,
                                                     mode=mode)
        # the shuffle param should override the shuffle attribute if set to a value
        if shuffle is None:
            shuffle = self.shuffle
        dataloader_args = {
            'dataset': dataset,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'shuffle': shuffle,
            **(extra_args or {})
        }
        if self.num_workers > 0:
            dataloader_args['persistent_workers'] = True

        if strict_batch_size:
            dataloader_args['collate_fn'] = self._get_custom_collate_fn()

        return DataLoader(**dataloader_args)

    def val_dataloader(self, mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.WITHOUT_LABELS,
                       strict_batch_size: bool = False,
                       extra_args: dict = None) -> DataLoader:

        if self.valid_size == 0.0:
            return None

        dataset = UCRClassificationUnivariateDataset(data=self._valid_data_samples,
                                                     labels=self._valid_data_labels,
                                                     mode=mode)
        dataloader_args = {
            'dataset': dataset,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'shuffle': False,
            **(extra_args or {})
        }

        if self.num_workers > 0:
            dataloader_args['persistent_workers'] = True

        if strict_batch_size:
            dataloader_args['collate_fn'] = self._get_custom_collate_fn()

        return DataLoader(**dataloader_args)

    def test_dataloader(self, mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.WITHOUT_LABELS,
                        strict_batch_size: bool = False,
                        extra_args: dict = None) -> DataLoader:

        dataset = UCRClassificationUnivariateDataset(data=self._test_data_samples,
                                                     labels=self._test_data_labels,
                                                     mode=mode)
        dataloader_args = {
            'dataset': dataset,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'shuffle': False,
            **(extra_args or {})
        }

        if self.num_workers > 0:
            dataloader_args['persistent_workers'] = True

        if strict_batch_size:
            dataloader_args['collate_fn'] = self._get_custom_collate_fn()

        return DataLoader(**dataloader_args)
