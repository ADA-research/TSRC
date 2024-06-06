__all__ = ['teacher_model_factory', 'student_model_factory', 'get_teacher_model_freezing_params']

from functools import partial

import numpy as np
import torch

from experiments.baselines.mcl import train_mcl_model, encode_data_mcl
from experiments.baselines.mcl.model import get_mcl_model
from experiments.baselines.ts2vec import get_ts2vec_model, train_ts2vec_model, encode_data_ts2vec
from lt_models.recurrent_autoencoder import RecurrentAutoEncoder


# ---------- TS2Vec specific functions ---------- #
def pre_process_ts2vec(x):
    """Preprocessing for ts2vec model: Detach and move to CPU, convert to NumPy."""
    device = x.device
    return x.cpu().detach().numpy().astype(np.float64), device


def post_process_ts2vec(r, device):
    """Postprocessing for ts2vec model: Convert NumPy array to tensor and move to original device."""
    return torch.from_numpy(r).to(device)


def ts2vec_encode_fn(data, model):
    data, device = pre_process_ts2vec(data)
    r = encode_data_ts2vec(data=data, model=model, encoding_window='full_series')
    r = post_process_ts2vec(r, device)
    return r


# ---------- END TS2Vec specific functions ---------- #


def get_teacher_model_encode_fn(teacher_model: str):
    teacher_model_encode_fn_map = {
        'ts2vec': ts2vec_encode_fn,
        'mcl': encode_data_mcl
    }

    teacher_model_encode_fn = teacher_model_encode_fn_map[teacher_model]

    return teacher_model_encode_fn


def get_teacher_model_trainer_params(teacher_model: str,
                                     teacher_model_params: dict,
                                     epochs: int):
    ts2vec_trainer_params = {
        'return_loss_log': False
    }

    mcl_trainer_params = {'max_epochs': epochs,
                          'outputs_folder_path': None,
                          'best_models_folder_path': None,
                          'final_models_folder_path': None,
                          'device': teacher_model_params['device']}

    teacher_model_trainer_params_map = {
        'ts2vec': ts2vec_trainer_params,
        'mcl': mcl_trainer_params
    }

    return teacher_model_trainer_params_map[teacher_model]


def get_teacher_model_freezing_params(teacher_model: str):
    teacher_model_freezing_params_map = {
        'ts2vec': False,
        'mcl': True
    }

    return teacher_model_freezing_params_map[teacher_model]


def teacher_model_factory(teacher_model: str,
                          teacher_model_params: dict,
                          epochs: int):
    teacher_models_map = {
        'ts2vec': get_ts2vec_model,
        'mcl': get_mcl_model,
    }

    teacher_models_trainers_fn_map = {
        'ts2vec': train_ts2vec_model,
        'mcl': train_mcl_model
    }

    teacher_model_obj = teacher_models_map[teacher_model](teacher_model_params)

    teacher_trainer = teacher_models_trainers_fn_map[teacher_model]
    teacher_trainer_params = get_teacher_model_trainer_params(teacher_model=teacher_model,
                                                              teacher_model_params=teacher_model_params,
                                                              epochs=epochs)
    teacher_trainer = partial(teacher_trainer, **teacher_trainer_params)

    teacher_encode_fn = get_teacher_model_encode_fn(teacher_model=teacher_model)

    return {'model': teacher_model_obj, 'trainer': teacher_trainer, 'encode_fn': teacher_encode_fn}


def student_model_factory(student_model: str,
                          student_model_params: dict):
    student_models_map = {
        'timenet': RecurrentAutoEncoder(**student_model_params),
        'lstmae': RecurrentAutoEncoder(**student_model_params)
    }

    student_model = student_models_map[student_model]

    student_model_encoder = student_model.encoder
    student_model_decoder = student_model.decoder

    return student_model_encoder, student_model_decoder
