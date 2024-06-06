__all__ = ['get_timenet_model']

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn


class GRUWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, dropout=0):
        super(GRUWrapper, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout)

    def forward(self, x):
        output, _ = self.gru(x)
        return output


class TimeNet(LightningModule):

    def __init__(self,
                 hidden_dims: int,
                 num_layers: int,
                 dropout: float = 0.1,
                 learning_rate: float = 1e-3):
        super(TimeNet, self).__init__()
        self.save_hyperparameters()
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()

        self.output_representation = False  # Control flag

    def switch_to_representation_mode(self):
        self.output_representation = True  # Switch to representation mode

    def switch_to_training_mode(self):
        self.output_representation = False  # Switch back to training mode

    def _build_encoder(self):
        encoder_layers = [GRUWrapper(1, self.hidden_dims, num_layers=1, batch_first=True)]
        for _ in range(1, self.num_layers):
            encoder_layers.append(GRUWrapper(self.hidden_dims, self.hidden_dims, num_layers=1, batch_first=True))
            if self.dropout > 0:
                encoder_layers.append(nn.Dropout(self.dropout))
        return nn.Sequential(*encoder_layers)

    def _build_decoder(self):
        decoder_layers = [GRUWrapper(self.hidden_dims, self.hidden_dims, num_layers=1, batch_first=True)]
        for i in range(1, self.num_layers):
            if i > 1 and self.dropout > 0:
                decoder_layers.append(nn.Dropout(self.dropout))
            decoder_layers.append(GRUWrapper(self.hidden_dims, self.hidden_dims, num_layers=1, batch_first=True))
        decoder_layers.append(nn.Linear(self.hidden_dims, 1))
        return nn.Sequential(*decoder_layers)

    def forward(self, x):
        # Encoder
        encode = self.encoder(x)

        # Decoder
        decode = self.decoder(torch.flip(encode, dims=[1]))  # Reverse the encoded sequence

        if self.output_representation:
            return encode
        else:
            return decode

    def training_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        loss = self.loss_fn(output, x)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        loss = self.loss_fn(output, x)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def get_timenet_model(model_params: dict):
    return TimeNet(**model_params)
