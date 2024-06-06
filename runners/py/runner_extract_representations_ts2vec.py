import argparse
import logging
from pathlib import Path

from experiments.baselines.ts2vec import extract_representations_ts2vec
from experiments.utils import get_data_dict_preparation_function, process_data_dict
from experiments.utils import get_datasets_to_skip
from experiments.utils import get_list_of_datasets_to_process
from experiments.utils.parsers import get_data_file_name_parser


def main_logic(datasets_folder_path: Path,
               dataset_name: str,
               compression_level: int,
               outputs_folder_path: Path,
               device: str = 'cpu',
               seed: int = 42):
    data_dict_preparation_fn = get_data_dict_preparation_function(data_type='raw')
    data_dict = data_dict_preparation_fn(datasets_folder_path)

    file_name_parser = get_data_file_name_parser('raw')

    processed_data = process_data_dict(data_dict=data_dict,
                                       datasets_to_skip=get_datasets_to_skip(),
                                       file_name_parser=file_name_parser)

    dataset_dict = processed_data[dataset_name]

    representations_dict, model = extract_representations_ts2vec(dataset_dict=dataset_dict,
                                                                 compression_level=compression_level,
                                                                 device=device,
                                                                 seed_number=seed)

    output_file_name = f'{dataset_name}_ts2vec_ts2vec.csv'
    representations_output_folder_path = outputs_folder_path / 'representations'

    train_output_file_path = representations_output_folder_path / 'train' / output_file_name
    valid_output_file_path = representations_output_folder_path / 'valid' / output_file_name
    test_output_file_path = representations_output_folder_path / 'test' / output_file_name

    train_output_file_path.parent.mkdir(parents=True, exist_ok=True)
    valid_output_file_path.parent.mkdir(parents=True, exist_ok=True)
    test_output_file_path.parent.mkdir(parents=True, exist_ok=True)

    representations_dict['train'].to_csv(train_output_file_path, index=False)
    representations_dict['valid'].to_csv(valid_output_file_path, index=False)
    representations_dict['test'].to_csv(test_output_file_path, index=False)

    trained_model_output_folder_path = outputs_folder_path / 'trained_models'
    trained_model_output_folder_path.mkdir(parents=True, exist_ok=True)

    model_output_file_path = trained_model_output_folder_path / f'{dataset_name}_ts2vec_model.pt'

    model.save(fn=model_output_file_path)


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
    parser.add_argument('--seed', type=int, help='Seed to use for reproducibility', default=42)
    parser.add_argument('--device', type=str, help='Device to use for training', default='cpu')
    parser.add_argument('--logging_level', type=str, help='Logging level', default='INFO')
    args = parser.parse_args()

    datasets_folder_path = Path(args.datasets_folder_path)
    datasets_to_process_file_path = Path(args.datasets_to_process_file_path)
    outputs_folder_path = Path(args.outputs_folder_path)
    compression_level = args.compression_level

    datasets_names_list = get_list_of_datasets_to_process(datasets_to_process_file_path)

    dataset_name = datasets_names_list[args.dataset_index]

    logging.basicConfig(level=args.logging_level)

    logging.info(f'Running with seed {args.seed}')

    main_logic(datasets_folder_path=datasets_folder_path,
               dataset_name=dataset_name,
               compression_level=compression_level,
               outputs_folder_path=outputs_folder_path,
               device=args.device,
               seed=args.seed)


if __name__ == '__main__':
    main()
