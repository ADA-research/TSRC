import argparse
import logging
import re
from functools import partial
from pathlib import Path
from typing import Optional

from experiments.utils import get_datasets_to_skip
from experiments.classification.classification_utils import classify_and_evaluate_representations


def main_logic(representations_folder_path: Path,
               run_folder_name_pattern: re.Pattern,
               classification_results_output_folder_path: Path,
               seeds_to_consider: list[int],
               representations_sub_folder: Optional[str] = None):
    datasets_to_skip = get_datasets_to_skip()

    classify_and_evaluate_fn = partial(classify_and_evaluate_representations,
                                       classification_results_output_folder_path=classification_results_output_folder_path,
                                       datasets_to_skip=datasets_to_skip)

    total_processed_folders_count = 0
    for experiment_folder in representations_folder_path.iterdir():
        if experiment_folder.is_dir():
            match = run_folder_name_pattern.match(experiment_folder.name)
            if match:
                seed_number = int(match.group('seed_number'))
                # Process only if the seed number is in the seeds_to_consider list
                if seeds_to_consider is None or seed_number in seeds_to_consider:
                    logging.info(f'Processing folder {experiment_folder}, seed number {seed_number}')

                    representations_folder_path = experiment_folder / 'representations'

                    if representations_sub_folder is not None:
                        representations_folder_path = representations_folder_path / representations_sub_folder

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

                    classify_and_evaluate_fn(data_files=data_files,
                                             output_files_prefix=f'representations_seed_{seed_number}')

                    logging.info(f'Finished processing folder {experiment_folder}, seed number {seed_number}')
                    total_processed_folders_count += 1

    logging.info(f'Finished processing all folders. Total processed folders count: {total_processed_folders_count}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--representations_results_folder', type=str,
                        help='Path to the folder where the experiments results are saved')
    parser.add_argument('--run_folder_name_pattern', type=str,
                        help='Pattern to match the folder names for the runs,'
                             ' should be something like "run_{{seed_number}}"')
    parser.add_argument('--classification_results_output_folder', type=str,
                        help='Path to the folder where the classification results will be saved')
    parser.add_argument('--representations_sub_folder', type=str,
                        help='In case the results are divided based on the final models and the best models,'
                             ' you need to define which ones you want to process.'
                             ' The options are: final_models, best_models,'
                             ' or None in case they are not divided in the representations folder', default=None)
    parser.add_argument('--seeds_to_consider', nargs='+', type=int, default=None,
                        help='Seeds and their corresponding runs to be considered,'
                             ' if None then all seeds will be considered')
    parser.add_argument('--loging_level', type=str, default='INFO')
    args = parser.parse_args()

    # Set logging level
    logging.basicConfig(level=args.loging_level)

    # Replace {{seed_number}} with a regex pattern to capture digits
    pattern = args.run_folder_name_pattern.replace('{{seed_number}}', '(?P<seed_number>\\d+)')
    run_folder_name_pattern = re.compile(pattern)

    representations_folder_path = Path(args.representations_results_folder)
    classification_results_output_folder_path = Path(args.classification_results_output_folder)
    seeds_to_consider = args.seeds_to_consider
    representations_sub_folder = args.representations_sub_folder

    main_logic(representations_folder_path=representations_folder_path,
               run_folder_name_pattern=run_folder_name_pattern,
               classification_results_output_folder_path=classification_results_output_folder_path,
               seeds_to_consider=seeds_to_consider,
               representations_sub_folder=representations_sub_folder)


if __name__ == '__main__':
    main()
