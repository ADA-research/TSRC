__all__ = ['TimeSeriesClassificationDataModule']

import os
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Tuple, Union

import lightning.pytorch as pl
import pandas as pd
from sklearn.model_selection import train_test_split

from enums import TimeSeriesClassificationDatasetSplittingStrategy
from utils_general import load_json, separate_target_feature_from_df
from utils_general.data.arff import read_arff_as_df, process_df_according_to_dtypes


class TimeSeriesClassificationDataModule(pl.LightningDataModule, ABC):
    def __init__(self,
                 dataset_folder_path: Union[str, Path],
                 dataset_config_path: Union[str, Path],
                 batch_size: int,
                 seq_len: int,
                 valid_size: float,  # percentage of the training set to use as validation set
                 shuffle: bool,
                 normalize: bool,
                 normalize_range: Tuple[float, float],
                 splitting_strategy: TimeSeriesClassificationDatasetSplittingStrategy =
                 TimeSeriesClassificationDatasetSplittingStrategy.AS_DEFINED,
                 test_size: float = 0.5,
                 # percentage of the dataset to use as test set; only valid for MANUAL splitting
                 num_workers: int = 1):
        super().__init__()
        self.dataset_folder_path = dataset_folder_path
        self.dataset_config = load_json(dataset_config_path)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.valid_size = valid_size
        self.shuffle = shuffle
        self.normalize = normalize
        self.normalize_range = normalize_range
        self.splitting_strategy = splitting_strategy
        self.test_size = test_size
        self.num_workers = num_workers
        self._separate_target_feature = partial(separate_target_feature_from_df,
                                                target_feature_name=self.dataset_config['main_config'][
                                                    'target_col_name'])
        self.target_column_name = self.dataset_config['main_config']['target_col_name']
        self.data_column_name = None
        self.datatype_handling_functions_map = None
        self._initiate_datatypes_handling_functions_map()
        self.dataset_name = None
        self.num_classes = None
        self.num_features = None
        self._train_data_samples = None
        self._train_data_labels = None
        self._test_data_samples = None
        self._test_data_labels = None
        self._valid_data_samples = None
        self._valid_data_labels = None

    @property
    def name(self):
        return self.dataset_name

    @property
    def n_classes(self):
        return self.num_classes

    @property
    def n_features(self):
        return self.num_features

    @property
    def train_data_samples(self):
        return self._train_data_samples

    @property
    def train_data_labels(self):
        return self._train_data_labels

    @property
    def test_data_samples(self):
        return self._test_data_samples

    @property
    def test_data_labels(self):
        return self._test_data_labels

    @property
    def valid_data_samples(self):
        return self._valid_data_samples

    @property
    def valid_data_labels(self):
        return self._valid_data_labels

    # define property to get all of the data combined
    @property
    def all_data_samples(self):
        return pd.concat([self._train_data_samples, self._test_data_samples, self._valid_data_samples], axis=0)

    @property
    def all_data_labels(self):
        return pd.concat([self._train_data_labels, self._test_data_labels, self._valid_data_labels], axis=0)

    @abstractmethod
    def _initiate_datatypes_handling_functions_map(self):
        pass

    def _extract_data_column_names(self):
        # get the name of the data column containing the time series (the column that is not the 'target' column)
        self.data_column_name = [column_name for column_name in self._train_data_samples.columns
                                 if column_name != self.target_column_name][0]

    def _read_arff_file_as_df(self, file_path: Path):
        df, meta = read_arff_as_df(file_path)
        df = process_df_according_to_dtypes(df, meta, self.datatype_handling_functions_map)
        return df

    def prepare_data(self) -> None:
        self.dataset_name = os.path.basename(self.dataset_folder_path)
        arff_train_file_name = self.dataset_config['main_config']['file_name_patterns']['train']['arff'] \
            .replace('{dataset_name}', self.dataset_name)
        arff_test_file_name = self.dataset_config['main_config']['file_name_patterns']['test']['arff'] \
            .replace('{dataset_name}', self.dataset_name)
        arff_train_file_path = Path(self.dataset_folder_path, arff_train_file_name)
        arff_test_file_path = Path(self.dataset_folder_path, arff_test_file_name)

        train_data = self._read_arff_file_as_df(arff_train_file_path)
        test_data = self._read_arff_file_as_df(arff_test_file_path)

        if self.splitting_strategy == TimeSeriesClassificationDatasetSplittingStrategy.MANUAL:
            # merge the train and test data
            combined_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
            train_data, test_data = train_test_split(combined_data, test_size=self.test_size,
                                                     stratify=combined_data[self.target_column_name], random_state=42)

        self._train_data_samples, self._train_data_labels = self._separate_target_feature(train_data)
        self._test_data_samples, self._test_data_labels = self._separate_target_feature(test_data)

        self.num_classes = len(self._train_data_labels.unique())
        self.num_features = len(self._train_data_samples.columns)

        self._extract_data_column_names()

        if self.valid_size > 0.0:
            # Convert to DataFrame for easier manipulation
            data_df = self._train_data_samples.copy(deep=True)
            data_df['label'] = self._train_data_labels.copy(deep=True)

            # Filter out classes with only one instance
            filtered_data = data_df.groupby('label').filter(lambda x: len(x) > 1)

            # Split the filtered data
            X_filtered = filtered_data.drop('label', axis=1)
            y_filtered = filtered_data['label']
            self._train_data_samples, self._valid_data_samples, self._train_data_labels, self._valid_data_labels = \
                train_test_split(X_filtered, y_filtered, test_size=self.valid_size,
                                 stratify=y_filtered, random_state=42)
