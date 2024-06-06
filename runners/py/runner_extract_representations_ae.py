import argparse
import logging
from collections import namedtuple
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from pytorch_lightning.loggers import CSVLogger

from datasets.utils import get_ucr_classification_univariate_datasets_lightning_modules
from enums import TimeSeriesDatasetMode
from experiments.representaions import (do_extract_representations,
                                        create_recurrent_layers_units_list,
                                        construct_and_make_representations_folders)
from experiments.utils import get_ucr_univariate_classification_datamodule_constants, \
    get_list_of_datasets_to_process, get_early_stopping_params
from lt_models import RecurrentAutoEncoder

# -------- CONSTANTS -------- #

NUM_FEATURES = 1

# -------- END CONSTANTS -------- #

# -------- TYPES -------- #
Model = namedtuple('Model', ['name', 'model', 'encoders'])


# -------- END TYPES -------- #

def log_early_stopping_details(callback: pl.callbacks.EarlyStopping):
    details = f"monitor='{callback.monitor}', patience={callback.patience}, min_delta={callback.min_delta}, mode='{callback.mode}'"
    return details


def log_model_checkpoint_details(callback: pl.callbacks.ModelCheckpoint):
    details = f"dirpath='{callback.dirpath}', filename='{callback.filename}', monitor='{callback.monitor}', save_top_k={callback.save_top_k}, mode='{callback.mode}', save_weights_only={callback.save_weights_only}"
    return details


def log_lr_monitor_details(callback: pl.callbacks.LearningRateMonitor):
    details = f"logging_interval='{callback.logging_interval}'"
    return details


def main_logic(dataset_index: int,
               datasets_folder_path: Path,
               datasets_names_list: list[str],
               outputs_folder_path: Path,
               cell_type: str,
               total_epochs: int,
               compression_level: int = 2,
               seed: int = 42):
    pl.seed_everything(seed=seed, workers=True)
    logging.debug(f'seeded everything with seed {seed}')

    representations_folders_dict = construct_and_make_representations_folders(outputs_folder_path=outputs_folder_path)
    best_models_folder_path = representations_folders_dict['best_models_folder_path']
    final_models_folder_path = representations_folders_dict['final_models_folder_path']
    best_model_representations_folder_path = representations_folders_dict['best_model_representations_folder_path']
    final_model_representations_folder_path = representations_folders_dict['final_model_representations_folder_path']

    logging.basicConfig(level=logging.INFO)

    dataset_name = datasets_names_list[dataset_index]

    data_module_params = get_ucr_univariate_classification_datamodule_constants()

    dataset_module = \
        get_ucr_classification_univariate_datasets_lightning_modules(datasets_to_get=dataset_name,
                                                                     datasets_folder_path=datasets_folder_path,
                                                                     data_module_params=data_module_params)[0]
    dataset_seq_length = dataset_module.sequence_len

    input_size = (data_module_params['batch_size'], dataset_seq_length, NUM_FEATURES)

    recurrent_layers = create_recurrent_layers_units_list(seq_len=dataset_seq_length,
                                                          compression_level=compression_level)

    logging.info(f"Running {dataset_name} with sequence length {dataset_module.sequence_len};"
                 f" input size {input_size}; recurrent layers {recurrent_layers};"
                 f" validation size {data_module_params['valid_size']};")

    ae_model_params = {
        'input_shape': input_size,
        'layers': recurrent_layers,
        'dropout': 0.0,
        'optimizer': 'Adam',
        'loss_fn': 'mse_loss',
        'lr': 1e-3,
        'cell_type': cell_type.upper()
    }

    ae_model_name = f'{cell_type}-auto-encoder'
    ae_model_class = RecurrentAutoEncoder(**ae_model_params)

    model = Model(name=ae_model_name, model=ae_model_class, encoders=['encoder'])

    logging.info(f'Running {model.name} on {dataset_name}')

    logging.info(f'Model params: {str(ae_model_params)}')

    logging.info(f'Model description: {str(model.model)}')

    early_stopping_constants = get_early_stopping_params(max_epochs=total_epochs)

    # set the callbacks for the trainer: early stopping, model checkpoint, and learning rate monitor
    early_stopping_callback = pl.callbacks.EarlyStopping(**early_stopping_constants)

    # make the model checkpoint save best model based on validation loss only
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss_epoch',
                                                             dirpath=best_models_folder_path,
                                                             filename=f'{dataset_name}_{model.name}_best',
                                                             save_top_k=1,
                                                             mode='min',
                                                             save_weights_only=True,
                                                             verbose=True)

    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

    callbacks = [early_stopping_callback, model_checkpoint_callback, lr_monitor_callback]

    # display info about the callbacks in logs with details

    # Log details about each callback
    logging.info(f'Running with early stopping: {log_early_stopping_details(early_stopping_callback)}')
    logging.info(f'Running with model checkpoint: {log_model_checkpoint_details(model_checkpoint_callback)}')
    logging.info(f'Running with learning rate monitor: {log_lr_monitor_details(lr_monitor_callback)}')

    # make the lightning logs folder
    lightning_logs_folder_path = outputs_folder_path / 'lightning_logs'
    lightning_logs_folder_path.mkdir(parents=True, exist_ok=True)
    # Define a CSV logger
    logger = CSVLogger(save_dir=lightning_logs_folder_path, name=f"{dataset_name}_{model.name}",
                       version=None)

    trainer = Trainer(devices='auto',
                      accelerator='auto',
                      callbacks=callbacks,
                      max_epochs=total_epochs,
                      logger=logger,
                      deterministic=True,
                      log_every_n_steps=5)

    trainer.fit(model=model.model,
                train_dataloaders=dataset_module.train_dataloader(),
                val_dataloaders=dataset_module.val_dataloader())

    # save the final model
    final_model_path = final_models_folder_path / f'{dataset_name}_{model.name}_final.ckpt'
    trainer.save_checkpoint(filepath=final_model_path,
                            weights_only=True)

    logging.info(f'Finished running {model.name} on {dataset_name}')

    model_class = type(model.model)

    final_model = model_class.load_from_checkpoint(checkpoint_path=final_model_path)

    # load the best model using the class instead of the instance
    model_class = type(model.model)
    best_model_path = model_checkpoint_callback.best_model_path
    if best_model_path is not None and Path(best_model_path).exists():
        best_model = model_class.load_from_checkpoint(checkpoint_path=best_model_path)
    else:
        logging.error(f'No best model found for {model.name} on {dataset_name}')
        best_model = final_model

    # extract the representations for the test set by passing them to the model for each encoder
    # then add the labels and save them in a dataframe then save the dataframe as a csv file.

    logging.info(f'Extracting representations for {model.name} on {dataset_name}')

    for encoder in model.encoders:
        best_model_encoder = getattr(best_model, encoder)
        best_model_encoder.eval()
        final_model_encoder = getattr(final_model, encoder)
        final_model_encoder.eval()
        data_loader_map = {
            'train': partial(dataset_module.train_dataloader, mode=TimeSeriesDatasetMode.WITH_LABELS,
                             shuffle=False),
            'valid': partial(dataset_module.val_dataloader, mode=TimeSeriesDatasetMode.WITH_LABELS),
            'test': partial(dataset_module.test_dataloader, mode=TimeSeriesDatasetMode.WITH_LABELS)
        }

        encoder_name = encoder.replace('_', '-')
        do_extract_representations(encoder=best_model_encoder,
                                   model_name=model.name,
                                   encoder_name=encoder_name,
                                   dataset_name=dataset_name,
                                   data_loader_map=data_loader_map,
                                   output_folder_path=best_model_representations_folder_path)

        do_extract_representations(encoder=final_model_encoder,
                                   model_name=model.name,
                                   encoder_name=encoder_name,
                                   dataset_name=dataset_name,
                                   data_loader_map=data_loader_map,
                                   output_folder_path=final_model_representations_folder_path)

    logging.info(f'Finished extracting representations for {model.name} on {dataset_name}')
    logging.info(f'Finished running {model.name} on {dataset_name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_index',
                        type=int,
                        help='Index of the dataset to process')
    parser.add_argument('--datasets_folder_path',
                        type=str,
                        help='Path to the folder where the raw data is saved')
    parser.add_argument('--datasets_to_process_file_path',
                        type=str,
                        help='A path to a file containing the datasets to process,'
                             ' they should be separated by a new line')
    parser.add_argument('--outputs_folder_path',
                        type=str,
                        help='Path to the folder where the outputs will be saved')
    parser.add_argument('--seed',
                        type=int,
                        help='Seed to use for reproducibility',
                        default=42)
    parser.add_argument('--compression_level',
                        type=int,
                        help='Compression level to use to construct the architecture, '
                             'the number represents dividing the length by this number.'
                             ' For example, a compression level of 2 means dividing'
                             ' the length by 2 '
                        , default=2)
    parser.add_argument('--cell_type',
                        type=str,
                        help='Cell types to use for the models comma separated',
                        default='lstm')
    parser.add_argument('--total_epochs',
                        type=int,
                        help='The total number of epochs to train the model (psi_t)',
                        default=500)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--logging_level', type=str, default='INFO')
    args = parser.parse_args()
    datasets_folder_path = Path(args.datasets_folder_path)
    datasets_to_process_file_path = Path(args.datasets_to_process_file_path)
    outputs_folder_path = Path(args.outputs_folder_path)

    datasets_names_list = get_list_of_datasets_to_process(datasets_to_process_file_path)

    logging.basicConfig(level=args.logging_level)

    logging.info(f'Running with seed {args.seed}')

    main_logic(dataset_index=args.dataset_index,
               datasets_folder_path=datasets_folder_path,
               datasets_names_list=datasets_names_list,
               outputs_folder_path=outputs_folder_path,
               seed=args.seed,
               compression_level=args.compression_level,
               cell_type=args.cell_type,
               total_epochs=args.total_epochs)


if __name__ == '__main__':
    main()
