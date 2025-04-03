__all__ = ['classify_and_evaluate_representations']

from functools import partial
import logging
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.utils.filters import get_data_list_filter
from experiments.utils.parsers import get_data_file_name_parser
from utils_general import add_prefix_to_string


def get_classification_algorithm(data_type: str, classification_algorithm_seed: int = 42):
    imputer = SimpleImputer(strategy='constant', fill_value=None)
    scaler = StandardScaler()
    classifier_object = SVC(
        kernel='rbf',
        coef0=0.0,
        gamma='scale',
        shrinking=True,
        probability=False,
        tol=0.001,
        cache_size=200,
        class_weight=None,
        max_iter=10_000_000,
        decision_function_shape='ovr',
        random_state=classification_algorithm_seed,
        verbose=False,
    )

    if data_type == 'representation':
        pipeline = Pipeline([('imputer', imputer), ('scaler', scaler), ('svc', classifier_object)])
    else:
        raise ValueError(f'No defined classification for data type: {data_type}')

    return pipeline


def classify_and_evaluate(
    data_files: dict[str, list[Path]],
    data_type: str,
    classification_results_output_folder_path: Path,
    datasets_to_skip: tuple[str, ...] = None,
    output_files_prefix: str = None,
    classification_algorithm_seed: int = 42,
):
    rows_for_df = list()

    file_name_parser = get_data_file_name_parser(data_type)
    data_list_filter = get_data_list_filter(task_type='classification', data_type=data_type)

    data_files = data_list_filter(data_files)

    for train_data_file in data_files['train']:
        data_file_metadata = file_name_parser(train_data_file)
        dataset_name = data_file_metadata['dataset_name']
        if dataset_name in datasets_to_skip:
            logging.info(
                f'Skipping file {train_data_file} because it is in the datasets to skip list.'
            )
            continue

        df_train_data = pd.read_csv(train_data_file)

        valid_data_file = next(
            data_file for data_file in data_files['valid'] if dataset_name in data_file.name
        )
        test_data_file = next(
            data_file for data_file in data_files['test'] if dataset_name in data_file.name
        )

        df_valid_data = pd.read_csv(valid_data_file)
        df_test_data = pd.read_csv(test_data_file)

        num_classes = len(df_train_data['label'].unique())

        df_train_features = df_train_data.drop(columns=['label']).reset_index(drop=True)
        y_train_true = df_train_data['label']

        df_valid_features = df_valid_data.drop(columns=['label']).reset_index(drop=True)
        y_valid_true = df_valid_data['label']

        df_test_features = df_test_data.drop(columns=['label']).reset_index(drop=True)
        y_test_true = df_test_data['label']

        # drop any columns that start with 'Unnamed'
        df_train_features = df_train_features.loc[
            :, ~df_train_features.columns.str.contains('^Unnamed')
        ]
        df_valid_features = df_valid_features.loc[
            :, ~df_valid_features.columns.str.contains('^Unnamed')
        ]
        df_test_features = df_test_features.loc[
            :, ~df_test_features.columns.str.contains('^Unnamed')
        ]

        train_features = df_train_features.to_numpy()
        valid_features = df_valid_features.to_numpy()
        test_features = df_test_features.to_numpy()

        train_labels = y_train_true.to_numpy()
        valid_labels = y_valid_true.to_numpy()
        test_labels = y_test_true.to_numpy()

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)

            combined_features = np.concatenate([train_features, valid_features], axis=0)
            combined_labels = np.concatenate([train_labels, valid_labels], axis=0)

            test_fold = [-1] * len(train_features) + [0] * len(valid_features)

            predefined_split = PredefinedSplit(test_fold)

            pipeline = get_classification_algorithm(
                data_type=data_type, classification_algorithm_seed=classification_algorithm_seed
            )

            param_grid = {'svc__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}

            grid_search = GridSearchCV(pipeline, param_grid, cv=predefined_split, n_jobs=-1)
            grid_search.fit(combined_features, combined_labels)

            best_classifier = grid_search.best_estimator_

            logging.info(f'Best parameters: {grid_search.best_params_}')

        y_test_pred = best_classifier.predict(test_features)

        accuracy = accuracy_score(test_labels, y_test_pred)
        precision = precision_score(test_labels, y_test_pred, average='macro')
        recall = recall_score(test_labels, y_test_pred, average='macro')
        f1 = f1_score(test_labels, y_test_pred, average='macro')

        row_dict = {
            **data_file_metadata,
            'data_part': 'test',
            'n_features': df_train_features.shape[1],
            'n_classes': num_classes,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

        rows_for_df.append(row_dict)

    df_results = pd.DataFrame(rows_for_df)

    file_name = f'classifications_results_seed_{classification_algorithm_seed}.csv'

    results_file_name = add_prefix_to_string(base=file_name, prefix=output_files_prefix)

    df_results.to_csv(classification_results_output_folder_path / results_file_name, index=False)


classify_and_evaluate_representations = partial(classify_and_evaluate, data_type='representation')
