import argparse
from functools import partial
import logging
from pathlib import Path
import re

from experiments.clustering.clustering_utils import cluster_and_evaluate_representations
from experiments.utils import get_datasets_to_skip


def main_logic(
    representations_folder_path: Path,
    run_folder_name_pattern: re.Pattern,
    clustering_results_output_folder_path: Path,
    seeds_to_consider: list[int],
    representations_sub_folder: str | None = None,
):
    datasets_to_skip = get_datasets_to_skip()
    cluster_and_evaluate_fn = partial(
        cluster_and_evaluate_representations,
        clustering_results_output_folder_path=clustering_results_output_folder_path,
        datasets_to_skip=datasets_to_skip,
    )

    total_processed_folders_count = 0
    for experiment_folder in representations_folder_path.iterdir():
        if experiment_folder.is_dir():
            match = run_folder_name_pattern.match(experiment_folder.name)
            if match:
                seed_number = int(match.group('seed_number'))
                # Process only if the seed number is in the seeds_to_consider list
                if seeds_to_consider is None or seed_number in seeds_to_consider:
                    logging.info(
                        f'Processing folder {experiment_folder}, seed number {seed_number}'
                    )

                    representations_folder_path = experiment_folder / 'representations'

                    if representations_sub_folder is not None:
                        representations_folder_path = (
                            representations_folder_path / representations_sub_folder
                        )

                    representations_files_paths = representations_folder_path.glob('*.csv')
                    cluster_and_evaluate_fn(
                        data_files=representations_files_paths,
                        output_files_prefix=f'representations_seed_{seed_number}',
                    )

                    logging.info(
                        f'Finished processing folder {experiment_folder}, seed number {seed_number}'
                    )
                    total_processed_folders_count += 1

    logging.info(
        f'Finished processing all folders. Total processed folders count: {total_processed_folders_count}'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--representations_results_folder',
        type=str,
        help='Path to the folder where the experiments results are saved',
    )
    parser.add_argument(
        '--run_folder_name_pattern',
        type=str,
        help='Pattern to match the folder names for the runs,'
        ' should be something like "run_{{seed_number}}"',
    )
    parser.add_argument(
        '--clustering_results_output_folder',
        type=str,
        help='Path to the folder where the clustering results will be saved',
    )
    parser.add_argument(
        '--representations_sub_folder',
        type=str,
        help='In case the results are divided based on the final models and the best models,'
        ' you need to define which ones you want to process.'
        ' The options are: final_models, best_models,'
        ' or None in case they are not divided in the representations folder',
        default=None,
    )
    parser.add_argument(
        '--seeds_to_consider',
        nargs='+',
        type=int,
        default=None,
        help='Seeds and their corresponding runs to be considered,'
        ' if None then all seeds will be considered',
    )
    parser.add_argument('--loging_level', type=str, default='INFO')
    args = parser.parse_args()

    # Set logging level
    logging.basicConfig(level=args.loging_level)

    # Replace {{seed_number}} with a regex pattern to capture digits
    pattern = args.run_folder_name_pattern.replace('{{seed_number}}', '(?P<seed_number>\\d+)')
    run_folder_name_pattern = re.compile(pattern)

    representations_folder_path = Path(args.representations_results_folder)
    clustering_results_output_folder_path = Path(args.clustering_results_output_folder)
    seeds_to_consider = args.seeds_to_consider
    representations_sub_folder = args.representations_sub_folder

    main_logic(
        representations_folder_path=representations_folder_path,
        run_folder_name_pattern=run_folder_name_pattern,
        clustering_results_output_folder_path=clustering_results_output_folder_path,
        seeds_to_consider=seeds_to_consider,
        representations_sub_folder=representations_sub_folder,
    )


if __name__ == '__main__':
    main()
