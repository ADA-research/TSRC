__all__ = ['extract_representations_ts2vec', 'encode_data_ts2vec', 'train_ts2vec_model', 'get_ts2vec_model']

from functools import partial

import numpy as np
import pandas as pd
from lightning.pytorch import seed_everything

from dependencies.ts2vec.ts2vec import TS2Vec
from utils_general import closest_power_of_2


def get_ts2vec_static_params():
    return {
        'input_dims': 1,
        'batch_size': 8,
        'depth': 10,
    }


def get_ts2vec_model(model_params: dict):
    return TS2Vec(**model_params)


def train_ts2vec_model(model: TS2Vec,
                       train_data: np.array,
                       return_loss_log: bool = True):
    loss_log = model.fit(train_data=train_data,
                         verbose=True)

    if return_loss_log:
        return model, loss_log
    else:
        return model


def encode_data_ts2vec(data: np.array, model: TS2Vec, encoding_window: str):
    return model.encode(data=data, encoding_window=encoding_window)


def extract_representations_ts2vec(dataset_dict: dict,
                                   compression_level: int,
                                   device: str = 'cpu',
                                   seed_number: int = 42):
    df_train_features = dataset_dict['train']['features']
    df_valid_features = dataset_dict['valid']['features']
    df_test_features = dataset_dict['test']['features']

    df_train_labels = dataset_dict['train']['labels']
    df_valid_labels = dataset_dict['valid']['labels']
    df_test_labels = dataset_dict['test']['labels']

    n_features = df_train_features.shape[1]

    output_dims = closest_power_of_2(m=n_features, v=compression_level)

    model_params = get_ts2vec_static_params()
    model_params['output_dims'] = output_dims
    model_params['device'] = device

    ts2vec = get_ts2vec_model(model_params=model_params)
    seed_everything(seed=seed_number)
    train_data = df_train_features.to_numpy().astype(np.float64)[..., np.newaxis]
    valid_data = df_valid_features.to_numpy().astype(np.float64)[..., np.newaxis]
    test_data = df_test_features.to_numpy().astype(np.float64)[..., np.newaxis]

    ts2vec, loss_log = train_ts2vec_model(model=ts2vec, train_data=train_data)

    encoding_fn = partial(encode_data_ts2vec, model=ts2vec, encoding_window='full_series')

    representations_train = encoding_fn(data=train_data)
    representations_valid = encoding_fn(data=valid_data)
    representations_test = encoding_fn(data=test_data)

    # append the labels to the representations and convert to pandas DataFrame
    df_representations_train = pd.DataFrame(representations_train)
    df_representations_train['label'] = df_train_labels

    df_representations_valid = pd.DataFrame(representations_valid)
    df_representations_valid['label'] = df_valid_labels

    df_representations_test = pd.DataFrame(representations_test)
    df_representations_test['label'] = df_test_labels

    return {'train': df_representations_train, 'valid': df_representations_valid,
            'test': df_representations_test}, ts2vec
