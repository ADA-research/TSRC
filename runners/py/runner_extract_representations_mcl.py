import argparse
import logging
from pathlib import Path

import lightning.pytorch as pl

from datasets.utils import get_ucr_classification_univariate_datasets_lightning_modules
from experiments.baselines.mcl import extract_representations_mcl
from experiments.utils import get_list_of_datasets_to_process, get_ucr_univariate_classification_datamodule_constants
from utils_general import closest_power_of_2


def main_logic(datasets_folder_path: Path,
               dataset_name: str,
               compression_level: int,
               outputs_folder_path: Path,
               max_epochs: int,
               device: str = 'cpu',
               seed: int = 42):
    pl.seed_everything(seed=seed, workers=True)

    logging.debug(f'seeded everything with seed {seed}')

    data_module_params = get_ucr_univariate_classification_datamodule_constants()

    dataset_module = \
        get_ucr_classification_univariate_datasets_lightning_modules(datasets_to_get=dataset_name,
                                                                     datasets_folder_path=datasets_folder_path,
                                                                     data_module_params=data_module_params)[0]
    dataset_seq_length = dataset_module.sequence_len

    output_dims = closest_power_of_2(m=dataset_seq_length, v=compression_level)

    mcl_model_params = {
        'n_in': dataset_seq_length,
        'output_dims': output_dims,
        'batch_size': data_module_params['batch_size'],
        'device': device,
        'alpha': 1.0,
        'learning_rate': 1e-3
    }

    extract_representations_mcl(dataset_module=dataset_module,
                                model_params=mcl_model_params,
                                max_epochs=max_epochs,
                                outputs_folder_path=outputs_folder_path,
                                device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_index', type=int, help='Index of the dataset to process')
    parser.add_argument('--datasets_folder_path', type=str, help='Path to the folder where the raw data is saved')
    parser.add_argument('--datasets_to_process_file_path', type=str,
                        help='A path to a file containing the datasets to process,'
                             ' they should be separated by a new line')
    parser.add_argument('--outputs_folder_path', type=str, help='Path to the folder where the outputs will be saved')
    parser.add_argument('--compression_level', type=int, help='Compression level to use to construct the architecture, '
                                                              'the number represents dividing the length by this number.'
                                                              ' For example, a compression level of 2 means dividing'
                                                              ' the length by 2 ', default=2)
    parser.add_argument('--max_epochs', type=int, help='Maximum number of epochs to train the model', default=200)
    parser.add_argument('--device', type=str, help='Device to use for training', default='cpu')
    parser.add_argument('--seed', type=int, help='Seed to use for reproducibility', default=42)

    parser.add_argument('--loging_level', type=str, default='INFO')
    args = parser.parse_args()
    datasets_folder_path = Path(args.datasets_folder_path)
    datasets_to_process_file_path = Path(args.datasets_to_process_file_path)
    outputs_folder_path = Path(args.outputs_folder_path)

    datasets_names_list = get_list_of_datasets_to_process(datasets_to_process_file_path)

    dataset_name = datasets_names_list[args.dataset_index]

    logging.basicConfig(level=args.loging_level)

    logging.info(f'Running with seed {args.seed}')

    main_logic(datasets_folder_path=datasets_folder_path,
               dataset_name=dataset_name,
               compression_level=args.compression_level,
               outputs_folder_path=outputs_folder_path,
               max_epochs=args.max_epochs,
               device=args.device,
               seed=args.seed)


if __name__ == '__main__':
    main()
