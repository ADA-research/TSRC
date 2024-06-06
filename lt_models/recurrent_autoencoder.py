__all__ = ['RecurrentEncoder', 'RecurrentDecoder', 'RecurrentAutoEncoder']

import logging
from typing import Tuple, List, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn

from utils_general import prepare_dropout_values
from utils_general.maps import get_loss_functions_map, get_optimizers_map


class RecurrentCellFactory:
    def __init__(self):
        self.recurrent_cells_map = {
            'lstm': nn.LSTM,
            'gru': nn.GRU,
            'rnn': nn.RNN
        }

    def get_cell(self, cell_type, **kwargs):
        return self.recurrent_cells_map.get(cell_type, nn.RNN)(**kwargs)


class RecurrentCellWrapper(nn.Module):
    def __init__(self, cell_type, input_size, hidden_size, num_layers=1, batch_first=True, dropout=0):
        super(RecurrentCellWrapper, self).__init__()
        self.cell_type = cell_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.recurrent_cell_factory = RecurrentCellFactory()

        self.cell = self.recurrent_cell_factory.get_cell(cell_type=self.cell_type,
                                                         input_size=self.input_size,
                                                         hidden_size=self.hidden_size,
                                                         num_layers=self.num_layers,
                                                         batch_first=self.batch_first,
                                                         dropout=self.dropout)

    def forward(self, x):
        output, _ = self.cell(x)
        return output


class RecurrentEncoder(pl.LightningModule):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 layers: List[int],
                 dropout: Union[float, List[float]],
                 cell_type: str = 'LSTM'):
        super().__init__()

        self.cell_type = cell_type.lower()

        # Prepare dropout layers
        self.dropout_values = prepare_dropout_values(layers, dropout)

        self.layers = layers

        self.input_shape = input_shape

        self.recurrent_layers = self._build_recurrent_layers()

    def _build_recurrent_layers(self):
        model_layers = [RecurrentCellWrapper(cell_type=self.cell_type,
                                             input_size=self.input_shape[2],
                                             hidden_size=self.layers[0],
                                             num_layers=1,
                                             batch_first=True)]
        for i in range(1, len(self.layers)):
            model_layers.append(RecurrentCellWrapper(cell_type=self.cell_type,
                                                     input_size=self.layers[i - 1],
                                                     hidden_size=self.layers[i],
                                                     num_layers=1,
                                                     batch_first=True))
            if self.dropout_values[i] > 0:
                model_layers.append(nn.Dropout(self.dropout_values[i]))

        return nn.Sequential(*model_layers)

    def forward(self, x):
        return self.recurrent_layers(x)


class RecurrentDecoder(pl.LightningModule):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 layers: List[int],
                 dropout: Union[float, List[float]],
                 cell_type: str = 'LSTM'):
        super().__init__()

        self.cell_type = cell_type.lower()

        # Prepare dropout layers
        self.dropout_values = prepare_dropout_values(layers, dropout)

        self.layers = layers

        self.input_shape = input_shape

        # Prepare the layers
        self.recurrent_layers = self._build_recurrent_layers()

    def _build_recurrent_layers(self):
        model_layers = [RecurrentCellWrapper(cell_type=self.cell_type,
                                             input_size=self.layers[0],
                                             hidden_size=self.layers[0],
                                             num_layers=1,
                                             batch_first=True)]
        for i in range(1, len(self.layers)):
            if i > 1 and self.dropout_values[i] > 0:
                model_layers.append(nn.Dropout(self.dropout_values[i]))
            model_layers.append(RecurrentCellWrapper(cell_type=self.cell_type,
                                                     input_size=self.layers[i - 1],
                                                     hidden_size=self.layers[i],
                                                     num_layers=1,
                                                     batch_first=True))

        model_layers.append(nn.Linear(self.layers[-1], self.input_shape[2]))

        return nn.Sequential(*model_layers)

    def forward(self, x):
        return self.recurrent_layers(x)


class RecurrentAutoEncoder(pl.LightningModule):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 layers: List[int],
                 cell_type: str = 'LSTM',
                 dropout: Union[float, List[float]] = 0.2,
                 loss_fn: str = "mse_loss",
                 optimizer: str = "Adam",
                 lr: float = 1e-3,
                 sync_dist: bool = False):
        super().__init__()

        self.save_hyperparameters()

        self.loss_fn = get_loss_functions_map()[loss_fn]
        self.optimizer_name = optimizer.lower()
        self.lr = lr
        self.sync_dist = sync_dist
        self.cell_type = cell_type.lower()

        # inverse layers to use in the decoder
        inverse_layers = layers[::-1]

        # inverse dropout in case it is a list
        if isinstance(dropout, list):
            inverse_dropout = dropout[::-1]
        else:
            inverse_dropout = dropout

        # prepare the layers
        self.encoder = RecurrentEncoder(input_shape=input_shape,
                                        layers=layers,
                                        dropout=dropout,
                                        cell_type=self.cell_type)
        self.decoder = RecurrentDecoder(input_shape=input_shape,
                                        layers=inverse_layers,
                                        dropout=inverse_dropout,
                                        cell_type=self.cell_type)

    def configure_optimizers(self):
        optimizer = get_optimizers_map()[self.optimizer_name](self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        # Encoder
        encode = self.encoder(x)
        logging.debug(f"Encoder output shape: {x.shape}")

        # Decoder
        flipped_encode = torch.flip(encode, dims=[1])
        decode = self.decoder(flipped_encode)  # Reverse the encoded sequence
        logging.debug(f"Decoder output shape: {x.shape}")

        return decode

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = self.loss_fn(x, x_hat)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        logging.debug(f"Validation batch shape: {x.shape}")
        x_hat = self(x)
        logging.debug(f"Validation batch output shape: {x_hat.shape}")
        valid_loss = self.loss_fn(x, x_hat)
        self.log('val_loss', valid_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        return valid_loss

    def test_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        valid_loss = self.loss_fn(x, x_hat)
        self.log('test_loss', valid_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        return valid_loss
