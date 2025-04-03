from argparse import ArgumentParser
from functools import partial
import logging
from pathlib import Path

from experiments.utils import (
    get_list_of_datasets_to_process,
    get_ucr_univariate_classification_datamodule_constants,
)
from project_datasets.utils import get_ucr_classification_univariate_datasets_lightning_modules


def process_data_module(data_module, dataset_output_folder: Path):
    dataset_name = data_module.dataset_name

    logging.info(f'Processing dataset {dataset_name}')

    train_data_samples = data_module.train_data_samples
    train_data_labels = data_module.train_data_labels

    valid_data_samples = data_module.valid_data_samples
    valid_data_labels = data_module.valid_data_labels

    test_data_samples = data_module.test_data_samples
    test_data_labels = data_module.test_data_labels

    # Save the data (include the labels in the same file and name the column with the labels as 'label')
    train_data = train_data_samples.copy(deep=True)
    train_data['label'] = train_data_labels.values

    valid_data = valid_data_samples.copy(deep=True)
    valid_data['label'] = valid_data_labels.values

    test_data = test_data_samples.copy(deep=True)
    test_data['label'] = test_data_labels.values

    train_data.to_csv(dataset_output_folder / f'{dataset_name}_train.csv', index=False)
    valid_data.to_csv(dataset_output_folder / f'{dataset_name}_valid.csv', index=False)
    test_data.to_csv(dataset_output_folder / f'{dataset_name}_test.csv', index=False)

    logging.info(f'Dataset {dataset_name} has been processed and saved to {dataset_output_folder}')


def main_logic(
    datasets_folder_path: Path,
    processed_data_output_folder_path: Path,
    datasets_to_process: list[str],
):
    data_module_params = get_ucr_univariate_classification_datamodule_constants()

    data_modules_getter_fn = partial(
        get_ucr_classification_univariate_datasets_lightning_modules,
        datasets_to_get=datasets_to_process,
        datasets_folder_path=datasets_folder_path,
        data_module_params=data_module_params,
    )
    data_modules = data_modules_getter_fn(execute_setup=False)

    for data_module in data_modules:
        process_data_module(data_module, processed_data_output_folder_path)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--datasets_folder_path', type=str, help='Path to the folder where the raw data is saved'
    )
    parser.add_argument(
        '--datasets_to_process_file_path',
        type=str,
        help='A path to a file containing the datasets to process,'
        ' they should be separated by a new line',
    )
    parser.add_argument(
        '--processed_data_output_folder_path',
        type=str,
        help='Path to where the processed datasets will be saved',
    )
    parser.add_argument('--loging_level', type=str, default='INFO')
    args = parser.parse_args()

    # Set logging level
    logging.basicConfig(level=args.loging_level)

    datasets_folder_path = Path(args.datasets_folder_path)
    datasets_to_process_file_path = Path(args.datasets_to_process_file_path)
    processed_data_output_folder_path = Path(args.processed_data_output_folder_path)

    # make the output folder if it does not exist
    processed_data_output_folder_path.mkdir(parents=True, exist_ok=True)

    datasets_to_process = get_list_of_datasets_to_process(datasets_to_process_file_path)

    main_logic(
        datasets_folder_path=datasets_folder_path,
        processed_data_output_folder_path=processed_data_output_folder_path,
        datasets_to_process=datasets_to_process,
    )


if __name__ == '__main__':
    main()
