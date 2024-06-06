__all__ = ['get_ucr_univariate_classification_datamodule_constants', 'get_list_of_datasets_to_process',
           'process_data_dict', 'get_data_dict_preparation_function', 'get_datasets_to_skip',
           'get_early_stopping_params']

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from enums import TimeSeriesClassificationDatasetSplittingStrategy
from utils_general import find_project_root


def get_ucr_univariate_classification_datamodule_constants():
    current_file_path = Path(__file__).resolve()
    project_root = find_project_root(current_file_path)
    ucr_classification_univariate_dataset_config_path = \
        project_root / 'configs' / 'datasets' / 'ucr_classification_univariate_config.json'
    batch_size = 8
    validation_size = 0.35
    test_size = 0.50
    normalize_range = (0, 1)
    return {
        'dataset_config_path': ucr_classification_univariate_dataset_config_path,
        'batch_size': batch_size,
        'normalize_range': normalize_range,
        'valid_size': validation_size,
        'test_size': test_size,
        'splitting_strategy': TimeSeriesClassificationDatasetSplittingStrategy.MANUAL,
        'shuffle': True,
    }


def get_list_of_datasets_to_process(datasets_to_process_file_path: Path) -> list[str]:
    return pd.read_csv(datasets_to_process_file_path, header=None)[0].tolist()


def prepare_raw_data_dict(data_folder_path: Path):
    train_files = list(data_folder_path.glob('*_train.csv'))
    test_files = list(data_folder_path.glob('*_test.csv'))
    valid_files = list(data_folder_path.glob('*_valid.csv'))

    data_files = {
        'train': train_files,
        'test': test_files,
        'valid': valid_files
    }

    return data_files


def prepare_representations_data_dict(representations_folder_path: Path):
    train_data_folder = representations_folder_path / 'train'
    train_data_files = list(train_data_folder.glob('*.csv'))

    valid_data_folder = representations_folder_path / 'valid'
    valid_data_files = list(valid_data_folder.glob('*.csv'))

    test_data_folder = representations_folder_path / 'test'
    test_data_files = list(test_data_folder.glob('*.csv'))

    data_files = {
        'train': train_data_files,
        'valid': valid_data_files,
        'test': test_data_files
    }


def get_data_dict_preparation_function(data_type: str):
    if data_type == 'raw':
        return prepare_raw_data_dict
    elif data_type == 'representations':
        return prepare_representations_data_dict
    else:
        raise ValueError(f'Invalid data type: {data_type}')


def process_data_dict(data_dict: dict[str, list[Path]],
                      datasets_to_skip: tuple[str, ...],
                      file_name_parser: callable):
    processed_data = {}
    for train_data_file in data_dict['train']:
        data_file_metadata = file_name_parser(train_data_file)
        dataset_name = data_file_metadata['dataset_name']
        if dataset_name in datasets_to_skip:
            logging.info(f'Skipping file {train_data_file} because it is in the datasets to skip list.')
            continue

        df_train_data = pd.read_csv(train_data_file)

        valid_data_file = next(data_file for data_file in data_dict['valid'] if dataset_name in data_file.name)
        test_data_file = next(data_file for data_file in data_dict['test'] if dataset_name in data_file.name)

        df_valid_data = pd.read_csv(valid_data_file)
        df_test_data = pd.read_csv(test_data_file)

        df_train_features = df_train_data.drop(columns=['label']).reset_index(drop=True)

        df_valid_features = df_valid_data.drop(columns=['label']).reset_index(drop=True)

        df_test_features = df_test_data.drop(columns=['label']).reset_index(drop=True)

        # drop any columns that start with 'Unnamed'
        df_train_features = df_train_features.loc[:, ~df_train_features.columns.str.contains('^Unnamed')]
        df_valid_features = df_valid_features.loc[:, ~df_valid_features.columns.str.contains('^Unnamed')]
        df_test_features = df_test_features.loc[:, ~df_test_features.columns.str.contains('^Unnamed')]

        processed_data[dataset_name] = {
            'train': {'features': df_train_features, 'labels': df_train_data['label']},
            'valid': {'features': df_valid_features, 'labels': df_valid_data['label']},
            'test': {'features': df_test_features, 'labels': df_test_data['label']}
        }

    return processed_data


def get_datasets_to_skip() -> Tuple[str, ...]:
    return ('DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'GestureMidAirD1',
            'GestureMidAirD2', 'GestureMidAirD3', 'MelbournePedestrian')


def get_early_stopping_params(max_epochs: int) -> dict:
    return {
        'monitor': 'val_loss_epoch',
        'min_delta': 0.0005,
        'patience': max_epochs // 10,
        'verbose': True,
        'mode': 'min'
    }
