__all__ = ['create_recurrent_layers_units_list',
           'do_extract_representations',
           'construct_and_make_representations_folders']

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from utils_general import closest_power_of_2


def create_recurrent_layers_units_list(seq_len: int, compression_level: int):
    recurrent_layers_units = list()
    recurrent_layers_units.append(closest_power_of_2(m=seq_len, v=compression_level))

    return recurrent_layers_units


def extract_representations(encoder: torch.nn.Module, dataloader: Any):
    representations = list()
    labels = list()
    for batch in dataloader():
        x, y = batch
        x = x.to(encoder.device)
        encoder.eval()
        with torch.inference_mode():
            r = encoder(x)
            r = r[:, -1, :]  # get the last hidden state
            representations.append(r)
            labels.append(y)
    representations = torch.cat(representations, dim=0)
    labels = torch.cat(labels, dim=0)
    representations_df = pd.DataFrame(representations.cpu().numpy())  # move data back to CPU for saving
    representations_df['label'] = labels.cpu().numpy()  # move data back to CPU for saving

    return representations_df


def do_extract_representations(encoder: torch.nn.Module,
                               model_name: str,
                               encoder_name: str,
                               dataset_name: str,
                               data_loader_map: dict,
                               output_folder_path: Path):
    for dataset_part, data_loader_fn in data_loader_map.items():
        representations_df = extract_representations(encoder=encoder,
                                                     dataloader=data_loader_fn)
        output_sub_folder_path = output_folder_path / dataset_part
        output_sub_folder_path.mkdir(parents=True, exist_ok=True)

        output_file_path = output_sub_folder_path / f'{dataset_name}_{model_name}_{encoder_name}.csv'
        representations_df.to_csv(output_file_path, index=False)


def construct_and_make_representations_folders(outputs_folder_path: Path):
    best_models_folder_path = outputs_folder_path / 'best_models'
    final_models_folder_path = outputs_folder_path / 'final_models'
    representations_folder_path = outputs_folder_path / 'representations'

    # create the folders if they don't exist
    best_models_folder_path.mkdir(parents=True, exist_ok=True)
    representations_folder_path.mkdir(parents=True, exist_ok=True)
    final_models_folder_path.mkdir(parents=True, exist_ok=True)

    best_model_representations_folder_path = representations_folder_path / 'best_models'
    best_model_representations_folder_path.mkdir(parents=True, exist_ok=True)

    final_model_representations_folder_path = representations_folder_path / 'final_models'
    final_model_representations_folder_path.mkdir(parents=True, exist_ok=True)

    return {
        'best_models_folder_path': best_models_folder_path,
        'final_models_folder_path': final_models_folder_path,
        'representations_folder_path': representations_folder_path,
        'best_model_representations_folder_path': best_model_representations_folder_path,
        'final_model_representations_folder_path': final_model_representations_folder_path
    }
