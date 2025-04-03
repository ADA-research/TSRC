import argparse
from functools import partial
import logging
from pathlib import Path

from experiments.clustering.clustering_utils import cluster_and_evaluate_raw_data
from experiments.utils import get_datasets_to_skip


def main_logic(data_folder: Path, clustering_results_output_folder: Path):
    for seed_number in [1, 3, 5, 7, 9]:
        decomposer_seed = seed_number
        clustering_algorithm_seed = 42

        cluster_and_evaluate_fn = partial(
            cluster_and_evaluate_raw_data,
            clustering_results_output_folder_path=clustering_results_output_folder,
            datasets_to_skip=get_datasets_to_skip(),
            decomposer_seed=decomposer_seed,
            clustering_algorithm_seed=clustering_algorithm_seed,
        )

        cluster_and_evaluate_fn(data_files=data_folder.glob('*.csv'), output_files_prefix='raw_clu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_folder', type=str, help='Path to the folder where the raw data is saved'
    )
    parser.add_argument(
        '--clustering_results_output_folder',
        type=str,
        help='Path to the folder where the clustering results will be saved',
    )
    parser.add_argument('--loging_level', type=str, default='INFO')
    args = parser.parse_args()

    # Set logging level
    logging.basicConfig(level=args.loging_level)

    data_folder = Path(args.data_folder)
    clustering_results_output_folder = Path(args.clustering_results_output_folder)

    main_logic(
        data_folder=data_folder, clustering_results_output_folder=clustering_results_output_folder
    )


if __name__ == '__main__':
    main()
