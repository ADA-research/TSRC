import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional, Union

__all__ = ['create_data_normalizer']


def create_data_normalizer(normalize: bool,
                           normalize_range: Tuple[float, float],
                           nested: bool = False) -> callable:
    """
    Create a data normalizer function.
    :param normalize: if True, normalize data
    :param normalize_range: the range to normalize to
    :param nested: if True, the data is nested (multivariate) and we need to normalize each dimension independently
    :return: a data normalizer function
    """

    def normalize_data(train_data: Union[np.ndarray, pd.DataFrame],
                       valid_data: Optional[Union[np.ndarray, pd.DataFrame]],
                       test_data: Union[np.ndarray, pd.DataFrame]
                       ):
        """
        Normalize data.
        :param train_data: the training data
        :param valid_data: the validation data
        :param test_data: the test data
        :return: a function that normalizes data
        """

        if normalize:
            if nested:
                return _normalize_nested_data_independent_dimensions(train_data=train_data,
                                                                     valid_data=valid_data,
                                                                     test_data=test_data,
                                                                     normalize_range=normalize_range)
            else:
                return _normalize_regular_data_and_return_the_same_input_type(train_data=train_data,
                                                                              valid_data=valid_data,
                                                                              test_data=test_data,
                                                                              normalize_range=normalize_range)
        else:
            return train_data, valid_data, test_data

    return normalize_data


def _normalize_regular_data(train_data: Union[np.ndarray, pd.DataFrame],
                            valid_data: Optional[Union[np.ndarray, pd.DataFrame]],
                            test_data: Union[np.ndarray, pd.DataFrame],
                            normalize_range: Tuple[float, float]) \
        -> Tuple[Union[np.ndarray, pd.DataFrame], Optional[Union[np.ndarray, pd.DataFrame]],
        Union[np.ndarray, pd.DataFrame]]:
    """
    Normalize data.
    :param train_data: the training data
    :param valid_data: the validation data
    :param test_data: the test data
    :return: normalized data
    """
    scaler = MinMaxScaler(feature_range=normalize_range)
    train_data = scaler.fit_transform(train_data)
    if valid_data is not None:
        valid_data = scaler.transform(valid_data)
    test_data = scaler.transform(test_data)
    return train_data, valid_data, test_data


def _normalize_regular_data_and_return_the_same_input_type(
        train_data: Union[np.ndarray, pd.DataFrame],
        valid_data: Optional[Union[np.ndarray, pd.DataFrame]],
        test_data: Union[np.ndarray, pd.DataFrame],
        normalize_range: Tuple[float, float]) \
        -> Tuple[Union[np.ndarray, pd.DataFrame], Optional[Union[np.ndarray, pd.DataFrame]],
        Union[np.ndarray, pd.DataFrame]]:
    """
    Normalize data.
    :param train_data: the training data
    :param valid_data: the validation data
    :param test_data: the test data
    :return: normalized data
    """
    normalized_train_data, normalized_valid_data, normalized_test_data = _normalize_regular_data(train_data=train_data,
                                                                                                 valid_data=valid_data,
                                                                                                 test_data=test_data,
                                                                                                 normalize_range=normalize_range)
    if isinstance(train_data, pd.DataFrame):
        normalized_train_data = pd.DataFrame(normalized_train_data, columns=train_data.columns)
        if valid_data is not None:
            normalized_valid_data = pd.DataFrame(normalized_valid_data, columns=valid_data.columns)
        normalized_test_data = pd.DataFrame(normalized_test_data, columns=test_data.columns)

    return normalized_train_data, normalized_valid_data, normalized_test_data


def _normalize_nested_data_independent_dimensions(train_data: pd.DataFrame,
                                                  valid_data: Optional[pd.DataFrame],
                                                  test_data: pd.DataFrame,
                                                  normalize_range: Tuple[float, float]) -> \
        Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    if len(train_data.iloc[0, 0].shape) == 1:
        n_features = 1
    else:
        n_features = train_data.iloc[0, 0].shape[1]

    # Prepare the scalers for each dimension
    scalers = [MinMaxScaler(feature_range=normalize_range) for _ in range(n_features)]

    # Function to handle 1D and 2D arrays while partial fitting
    def handle_array(array):
        if len(array.shape) == 1:
            scalers[0].partial_fit(array.reshape(-1, 1))
        else:
            for i, scaler in enumerate(scalers):
                scaler.partial_fit(array[:, i].reshape(-1, 1))

    # Fit each scaler using the corresponding dimension from all arrays in the training data
    for array in train_data.applymap(lambda x: x[0]).values.flatten():
        handle_array(array)

    # Function to normalize a single 2D array
    def normalize_array(array):
        if len(array.shape) == 1:
            return scalers[0].transform(array.reshape(-1, 1)).ravel()
        else:
            return np.column_stack(
                [scaler.transform(array[:, i].reshape(-1, 1)).ravel() for i, scaler in enumerate(scalers)])

    # Normalize all data using the fitted scalers
    normalized_train_data = train_data.applymap(lambda x: np.stack([normalize_array(array) for array in x]))
    normalized_valid_data = valid_data.applymap(
        lambda x: np.stack([normalize_array(array) for array in x])) if valid_data is not None else None
    normalized_test_data = test_data.applymap(lambda x: np.stack([normalize_array(array) for array in x]))

    return normalized_train_data, normalized_valid_data, normalized_test_data
