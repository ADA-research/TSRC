__all__ = ['cluster_and_evaluate_representations', 'cluster_and_evaluate_raw_data']

import logging
import warnings
from functools import partial
from pathlib import Path
from typing import Tuple, List

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score

from experiments.utils.filters import get_data_list_filter
from experiments.utils.parsers import get_data_file_name_parser
from utils_general import add_prefix_to_string


def get_clustering_algorithm(num_classes: int, seed_number: int):
    kmeans = KMeans(n_clusters=num_classes,
                    init='k-means++',
                    n_init=100,
                    max_iter=500,
                    tol=1e-04,
                    verbose=0,
                    random_state=seed_number,
                    copy_x=True,
                    algorithm='lloyd')

    return kmeans


def cluster_and_evaluate(data_files: List[Path],
                         data_type: str,
                         clustering_results_output_folder_path: Path,
                         datasets_to_skip: Tuple[str, ...] = None,
                         output_files_prefix: str = None,
                         seed_number: int = 42):
    rows_for_df = list()

    file_name_parser = get_data_file_name_parser(data_type)
    data_list_filter = get_data_list_filter(task_type='clustering', data_type=data_type)

    data_files = data_list_filter(data_files)
    for data_file in data_files:
        data_file_metadata = file_name_parser(data_file)
        dataset_name = data_file_metadata['dataset_name']
        if dataset_name in datasets_to_skip:
            logging.info(f'Skipping file {data_file} because it is in the datasets to skip list.')
            continue

        df = pd.read_csv(data_file)

        num_classes = len(df['label'].unique())

        # get the features without the labels
        features = df.drop('label', axis=1)

        features.reset_index(drop=True, inplace=True)

        features = features.loc[:, ~features.columns.str.contains('^Unnamed')]

        n_features = features.shape[1]

        if features.isnull().values.any():
            logging.warning(f'The features of file {data_file} contain NaN values.')
            features.dropna(inplace=True, axis=0)
            logging.warning(f'The number of removed rows is {len(df) - len(features)}')
            logging.warning(f'The indices of the removed rows are {df.index.difference(features.index)}')
            continue

        logging.info(f'Clustering file {data_file} with {num_classes} clusters and seed number {seed_number}')

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)

            kmeans = get_clustering_algorithm(num_classes, seed_number)

            kmeans.fit(features)

            # get the clusters
            clusters = kmeans.labels_

        # get the labels
        labels = df['label'].values

        # external evaluation metrics
        ari = adjusted_rand_score(labels, clusters)

        # internal evaluation metrics
        chs = calinski_harabasz_score(features, clusters)

        row_dict = {
            **data_file_metadata,
            'n_features': n_features,
            'n_classes': num_classes,
            'ari': ari,
            'chs': chs,
        }

        rows_for_df.append(row_dict)

    df_results = pd.DataFrame(rows_for_df)

    results_file_name = add_prefix_to_string(base='clustering_results.csv', prefix=output_files_prefix)

    df_results.to_csv(clustering_results_output_folder_path / results_file_name, index=False)


cluster_and_evaluate_representations = partial(cluster_and_evaluate, data_type='representation')
cluster_and_evaluate_raw_data = partial(cluster_and_evaluate, data_type='raw')
