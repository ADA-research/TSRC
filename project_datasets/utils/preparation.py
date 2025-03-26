__all__ = ['get_ucr_classification_univariate_datasets_lightning_modules']

import logging
from functools import partial
from pathlib import Path
from typing import List, Tuple, Union, TypeVar, Type

from project_datasets.modules.abstract import TimeSeriesClassificationDataModule
from project_datasets.modules.ucr_classification_univariate_module import UCRTimeSeriesClassificationUnivariateDataModule

DataModuleType = TypeVar('DataModuleType', bound=TimeSeriesClassificationDataModule)


def get_datasets_lightning_modules(datasets_to_get: Union[str, List[str], Tuple[str]],
                                   datasets_folder_path: Path,
                                   data_module_class: Type[DataModuleType],
                                   data_module_params: dict,
                                   execute_setup: bool = True) -> List[Type[DataModuleType]]:
    """
    Get initialized LightningDataModules for datasets.
    Args:
        datasets_to_get: the datasets to get LightningDataModules for
        datasets_folder_path: the folder path where the datasets are located
        data_module_class: the LightningDataModule class to use
        data_module_params: dictionary with parameters for the LightningDataModules
        execute_setup: whether to execute the setup method of the LightningDataModules

    Returns:
        list of LightningDataModules
    """
    lightning_data_modules = list()

    if isinstance(datasets_to_get, str):
        datasets_to_get = [datasets_to_get]

    for dataset_name in datasets_to_get:
        logging.debug(f'Getting LightningDataModule for dataset {dataset_name}')
        dataset_folder_path = datasets_folder_path / dataset_name
        dataset_module = data_module_class(dataset_folder_path=dataset_folder_path, **data_module_params)

        dataset_module.prepare_data()
        dataset_module.sequence_len = dataset_module.n_features

        if execute_setup:
            dataset_module.setup()

        lightning_data_modules.append(dataset_module)

    return lightning_data_modules


# ------- Partial functions ------- #

get_ucr_classification_univariate_datasets_lightning_modules \
    = partial(get_datasets_lightning_modules,
              data_module_class=UCRTimeSeriesClassificationUnivariateDataModule)
