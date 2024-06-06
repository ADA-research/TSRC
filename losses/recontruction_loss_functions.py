__all__ = ['ReconstructionWithHintLoss']

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionWithHintLoss(nn.Module):
    """
    This class implements the reconstruction loss with regularization. The reconstruction loss is computed as the
    mean squared error between the original and reconstructed sequences, and the regularization term is computed
     as the distance between the representations of the original sequence from two different encoders.
    """

    def __init__(self,
                 total_epochs: int,
                 repr_sim_coeff: float = 0.5,
                 coeff_start_value: float = 0.0,
                 coeff_increase_rate: float = 1.0,
                 coeff_start_epoch_threshold: int = 0,
                 val_loss_coeff: float = 0.0,
                 mode: str = 'train'):
        """
        Constructor for the ReconstructionLossWithRegularization.

        :param total_epochs: Total number of epochs for training.
        :param repr_sim_coeff: Coefficient for the similarity measure of representations.
        :param coeff_start_value: Initial value for the coefficient.
        :param coeff_increase_rate: Rate of increase for the coefficient.
        :param val_loss_coeff: Coefficient for the validation loss.
        :param mode: Mode of the model, either 'train' or 'valid'.
        """
        super().__init__()
        self.current_epoch = 0
        self.total_epochs = total_epochs
        self.mode = mode
        self.initial_repr_sim_coeff = coeff_start_value
        self.coeff_increase_rate = coeff_increase_rate
        self.final_repr_sim_coeff = repr_sim_coeff
        self.coeff_start_epoch_threshold = coeff_start_epoch_threshold
        self.val_loss_coeff = val_loss_coeff
        self.total_epochs_shifted = self.total_epochs - self.coeff_start_epoch_threshold

    def update_repr_sim_coeff(self):
        if self.mode == 'valid':
            return self.val_loss_coeff
        elif self.mode == 'train':
            if self.current_epoch < self.coeff_start_epoch_threshold:
                return 0.0
            else:
                curr_epoch_shifted = self.current_epoch - self.coeff_start_epoch_threshold
                normalized_epoch = (curr_epoch_shifted / self.total_epochs_shifted) ** self.coeff_increase_rate
                coeff_progress = (self.final_repr_sim_coeff - self.initial_repr_sim_coeff) * normalized_epoch
                current_coeff = self.initial_repr_sim_coeff + coeff_progress
                return current_coeff

    def update_current_epoch(self, current_epoch):
        self.current_epoch = current_epoch

    def check_for_nan_and_inf(self, tensor: torch.Tensor, tensor_name: str = "Tensor"):
        if torch.isnan(tensor).any():
            logging.warning(f'{tensor_name} contains NaN values.')
        # check positive infinity
        if torch.isposinf(tensor).any():
            logging.warning(f'{tensor_name} contains inf values.')
        # check negative infinity
        if torch.isneginf(tensor).any():
            logging.warning(f'{tensor_name} contains -inf values.')

    def forward(self, **kwargs) -> torch.Tensor:
        """
        Forward pass for the ReconstructionLossWithRegularization.

        This method expects keyword arguments that should include:
        - 'x': the original sequence.
        - 'x_hat': the second reconstructed sequence.
        - 'r1': the embedding of the original sequence from the first encoder.
        - 'r2': the embedding of the original sequence from the second encoder.
        - 'current_epoch': the current epoch.
        """
        x = kwargs['x']
        x_hat = kwargs['x_hat']
        r1 = kwargs['r1']
        r2 = kwargs['r2']
        r2 = r2[:, -1, :]  # get the last hidden state
        current_epoch = kwargs['current_epoch']

        r1_clone = r1.clone()

        if len(r1_clone.shape) == 3:
            r1_clone = r1_clone[:, -1, :]

        logging.debug(f'current_epoch: {current_epoch}')
        self.update_current_epoch(current_epoch)

        self.check_for_nan_and_inf(r1_clone, 'r1 (r_teacher)')
        self.check_for_nan_and_inf(r2, 'r2 (r_student)')

        hint_loss = F.mse_loss(r2, r1_clone)

        logging.debug(f'hint_loss: {hint_loss}')

        self.check_for_nan_and_inf(x, 'x')
        self.check_for_nan_and_inf(x_hat, 'x_hat')
        reconstruction_loss = F.mse_loss(x_hat, x)

        logging.debug(f'reconstruction_loss: {reconstruction_loss}')

        self.check_for_nan_and_inf(hint_loss, 'hint_loss')
        self.check_for_nan_and_inf(reconstruction_loss, 'reconstruction_loss')

        current_coeff = self.update_repr_sim_coeff()
        logging.debug(f'current_coeff: {current_coeff}')

        total_loss = (1 - current_coeff) * reconstruction_loss + current_coeff * hint_loss
        logging.debug(f'total_loss: {total_loss}')

        return total_loss
