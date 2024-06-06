__all__ = ['TSRC']

import logging
from functools import partial
from typing import Any

import lightning.pytorch as pl
import torch

from losses.recontruction_loss_functions import ReconstructionWithHintLoss
from tsrc.models_factory import student_model_factory, teacher_model_factory, get_teacher_model_freezing_params
from utils_general.maps import get_optimizers_map


class TSRC(pl.LightningModule):
    def __init__(self,
                 student_model: str,
                 student_model_params: dict,
                 teacher_model: str,
                 teacher_model_params: dict,
                 optimizer: str = 'Adam',
                 lr: float = 1e-3,
                 reg_lambda: float = 0.2,
                 reg_lambda_start_value: float = 0.0,
                 reg_lambda_increase_rate: float = 1.5,
                 val_loss_coeff: float = 0.0,
                 loss_start_epoch_threshold: int = 0,
                 total_epochs: int = 500):
        super().__init__()
        self.save_hyperparameters()

        self.train_loss_fn = ReconstructionWithHintLoss(repr_sim_coeff=reg_lambda,
                                                        coeff_start_value=reg_lambda_start_value,
                                                        coeff_increase_rate=reg_lambda_increase_rate,
                                                        val_loss_coeff=val_loss_coeff,
                                                        coeff_start_epoch_threshold=loss_start_epoch_threshold,
                                                        total_epochs=total_epochs,
                                                        mode='train')

        self.valid_loss_fn = ReconstructionWithHintLoss(repr_sim_coeff=reg_lambda,
                                                        coeff_start_value=reg_lambda_start_value,
                                                        coeff_increase_rate=reg_lambda_increase_rate,
                                                        val_loss_coeff=val_loss_coeff,
                                                        coeff_start_epoch_threshold=loss_start_epoch_threshold,
                                                        total_epochs=total_epochs,
                                                        mode='valid')

        teacher_model_items = teacher_model_factory(teacher_model=teacher_model,
                                                    teacher_model_params=teacher_model_params,
                                                    epochs=total_epochs)
        self.teacher_model = teacher_model_items['model']
        self.teacher_model_trainer = teacher_model_items['trainer']
        self.teacher_encode_fn = partial(teacher_model_items['encode_fn'], model=self.teacher_model)

        self.student_model_encoder, self.student_model_decoder = \
            student_model_factory(student_model=student_model, student_model_params=student_model_params)

        self.optimizer_name = optimizer.lower()
        self.lr = lr

    def configure_optimizers(self):
        optimizer = get_optimizers_map()[self.optimizer_name](self.parameters(), lr=self.lr)
        return optimizer

    def train_teacher_model(self, train_data: Any):
        logging.info('Training teacher model')
        self.teacher_model = self.teacher_model_trainer(model=self.teacher_model,
                                                        train_data=train_data)

        if get_teacher_model_freezing_params(teacher_model=self.hparams.teacher_model):
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def _compute_loss(self, output, x, mode='train'):
        loss_args = {
            'x': x,
            'x_hat': output['x_hat'],
            'r1': output.get('r1'),
            'r2': output.get('r2'),
            'current_epoch': self.current_epoch
        }

        if mode == 'train':
            loss = self.train_loss_fn(**loss_args)
        elif mode == 'valid':
            loss = self.valid_loss_fn(**loss_args)
        else:
            raise ValueError(f'Unknown mode: {mode}')
        return loss

    def forward(self, x):
        r1 = self.teacher_encode_fn(x)

        r2 = self.student_model_encoder(x)

        r2_flipped = torch.flip(r2, dims=[1])

        x_hat = self.student_model_decoder(r2_flipped)

        return {'r1': r1, 'r2': r2, 'x_hat': x_hat}

    def training_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        loss = self._compute_loss(output=output, x=x, mode='train')
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        loss = self._compute_loss(output, x, mode='valid')
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        output = self(x)
        loss = self._compute_loss(output, x, mode='valid')
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
