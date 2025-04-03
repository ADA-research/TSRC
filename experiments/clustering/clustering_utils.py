__all__ = ['cluster_and_evaluate_raw_data', 'cluster_and_evaluate_representations']

from collections.abc import Callable
from concurrent.futures import as_completed, ProcessPoolExecutor
from functools import partial
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor

from experiments.utils.filters import get_data_list_filter
from experiments.utils.parsers import get_data_file_name_parser
from utils_general import add_prefix_to_string, closest_power_of_2


def prepare_panel_data(X):
    """Convert input data to sktime-compatible 3D numpy array format."""
    # If X is a simple DataFrame (rows=instances, columns=timepoints)
    if isinstance(X, pd.DataFrame):
        # Convert to 3D numpy array (n_instances, n_variables, n_timepoints)
        return X.values.reshape(X.shape[0], 1, X.shape[1])

    # If X is a numpy array
    if isinstance(X, np.ndarray):
        # Check dimensionality
        if X.ndim == 2:  # (n_instances, n_timepoints)
            # Reshape to (n_instances, n_variables=1, n_timepoints)
            return X.reshape(X.shape[0], 1, X.shape[1])
        if X.ndim == 3:  # Already in (n_instances, n_variables, n_timepoints)
            return X

    # If we get here, format is not supported
    raise ValueError('Unsupported data format')


def get_clustering_algorithm(
    num_classes: int,
    clustering_algorithm_seed: int,
    decomposer_seed: int,
    data_type: str,
    n_features: int,
) -> Any:
    """
    Get the appropriate clustering algorithm based on data type.

    Args:
        num_classes: Number of clusters to form
        clustering_algorithm_seed: Random seed for clustering algorithm
        decomposer_seed: Random seed for decomposer
        data_type: Type of data ('representation' or 'raw')
        n_features: Number of features of the data

    Returns:
        Clustering algorithm instance or function

    """
    kmeans = KMeans(
        n_clusters=num_classes,
        init='k-means++',
        n_init=100,
        max_iter=500,
        tol=1e-04,
        verbose=0,
        random_state=clustering_algorithm_seed,
        copy_x=True,
        algorithm='lloyd',
    )

    prepare_transformer = FunctionTransformer(prepare_panel_data)

    min_length = max(2, int(n_features * 0.05))

    target_n_components = closest_power_of_2(m=n_features, v=2) + 1

    decomposer = RandomIntervalFeatureExtractor(
        n_intervals=target_n_components, random_state=decomposer_seed, min_length=min_length
    )

    if data_type == 'representation':
        clustering_algorithm = Pipeline([('kmeans', kmeans)])
    elif data_type == 'raw':
        clustering_algorithm = Pipeline(
            [('prepare', prepare_transformer), ('decomposer', decomposer), ('kmeans', kmeans)]
        )
    else:
        raise ValueError(f'Unknown data type: {data_type}')

    return clustering_algorithm


def process_single_file(
    data_file: Path,
    data_type: str,
    clustering_algorithm_seed: int,
    decomposer_seed: int,
    file_name_parser: Callable,
) -> dict[str, Any]:
    """
    Process a single data file for clustering and evaluation.

    Args:
        data_file: Path to the data file
        data_type: Type of data ('representation' or 'raw')
        clustering_algorithm_seed: Random seed for clustering algorithm
        decomposer_seed: Random seed for decomposer
        file_name_parser: Function to parse file name metadata

    Returns:
        Dictionary containing clustering results

    """
    data_file_metadata = file_name_parser(data_file)
    dataset_name = data_file_metadata['dataset_name']

    try:
        df = pd.read_csv(data_file)
        num_classes = len(df['label'].unique())

        # get the features without the labels
        features = df.drop('label', axis=1)
        features.reset_index(drop=True, inplace=True)
        features = features.loc[:, ~features.columns.str.contains('^Unnamed')]
        n_features = features.shape[1]

        logging.info(
            f'Processing dataset {dataset_name} with {n_features} features and {num_classes} classes and shape {features.shape} [file: {data_file}]'
        )

        if features.isnull().values.any():
            logging.warning(f'The features of file {data_file} contain NaN values.')
            features.dropna(inplace=True, axis=0)
            logging.warning(f'The number of removed rows is {len(df) - len(features)}')
            logging.warning(
                f'The indices of the removed rows are {df.index.difference(features.index)}'
            )
            return None

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)

            clustering_algorithm = get_clustering_algorithm(
                num_classes=num_classes,
                clustering_algorithm_seed=clustering_algorithm_seed,
                decomposer_seed=decomposer_seed,
                data_type=data_type,
                n_features=n_features,
            )

            clustering_algorithm.fit(features)

            logging.info(
                f'Processed dataset {dataset_name} with {n_features} features and {num_classes} classes [file: {data_file}]'
            )

        kmeans = clustering_algorithm.named_steps['kmeans']
        clusters = kmeans.labels_
        # get the labels
        labels = df['label'].to_numpy()

        # external evaluation metrics
        ari = adjusted_rand_score(labels, clusters)
        nmi = normalized_mutual_info_score(labels, clusters)
        fmi = fowlkes_mallows_score(labels, clusters)

        # internal evaluation metrics
        chs = calinski_harabasz_score(features, clusters)
        dbs = davies_bouldin_score(features, clusters)
        ss = silhouette_score(features, clusters)

        return {
            **data_file_metadata,
            'n_features': n_features,
            'n_classes': num_classes,
            'ari': ari,
            'nmi': nmi,
            'fmi': fmi,
            'chs': chs,
            'dbs': dbs,
            'ss': ss,
        }
    except Exception as e:
        logging.exception(f'Error processing file {data_file}: {e!s}')
        return None


def cluster_and_evaluate(
    data_files: list[Path],
    data_type: str,
    clustering_results_output_folder_path: Path,
    datasets_to_skip: tuple[str, ...] = None,
    output_files_prefix: str = None,
    clustering_algorithm_seed: int = 42,
    decomposer_seed: int = 42,
    n_workers: int = 4,
):
    """
    Cluster and evaluate data files in parallel.

    Args:
        data_files: List of data file paths
        data_type: Type of data ('representation' or 'raw')
        clustering_results_output_folder_path: Path to save results
        datasets_to_skip: Tuple of dataset names to skip
        output_files_prefix: Prefix for output files
        clustering_algorithm_seed: Random seed for clustering algorithm
        decomposer_seed: Random seed for decomposer
        n_workers: Number of worker processes (defaults to number of CPU cores)

    """
    if datasets_to_skip is None:
        datasets_to_skip = tuple()

    # Determine number of workers (defaults to CPU count)
    if n_workers is None:
        n_workers = mp.cpu_count()

    logging.info(f'Parallel processing with {n_workers} workers')

    file_name_parser: Callable = get_data_file_name_parser(data_type)
    data_list_filter = get_data_list_filter(task_type='clustering', data_type=data_type)

    # Filter files
    data_files = data_list_filter(data_files)

    # Filter out datasets to skip
    filtered_data_files = []
    for data_file in data_files:
        data_file_metadata = file_name_parser(data_file)
        dataset_name = data_file_metadata['dataset_name']
        if dataset_name in datasets_to_skip:
            logging.info(f'Skipping file {data_file} because it is in the datasets to skip list.')
            continue
        filtered_data_files.append(data_file)

    rows_for_df = []

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                process_single_file,
                data_file,
                data_type,
                clustering_algorithm_seed,
                decomposer_seed,
                file_name_parser,
            ): data_file
            for data_file in filtered_data_files
        }

        # Process results as they complete
        for future in as_completed(future_to_file):
            data_file = future_to_file[future]
            try:
                result = future.result()
                if result is not None:
                    rows_for_df.append(result)
                    logging.info(f'Successfully processed {data_file}')
            except Exception as exc:
                logging.exception(f'Processing {data_file} generated an exception: {exc}')

    # Create and save results dataframe
    df_results = pd.DataFrame(rows_for_df)

    seed_to_use_in_file_name = decomposer_seed

    file_name = f'clustering_results_seed_{seed_to_use_in_file_name}.csv'
    results_file_name = add_prefix_to_string(base=file_name, prefix=output_files_prefix)

    df_results.to_csv(clustering_results_output_folder_path / results_file_name, index=False)
    logging.info(f'Results saved to {clustering_results_output_folder_path / results_file_name}')


cluster_and_evaluate_representations = partial(cluster_and_evaluate, data_type='representation')
cluster_and_evaluate_raw_data = partial(cluster_and_evaluate, data_type='raw')
