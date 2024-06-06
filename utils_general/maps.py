__all__ = ["get_loss_functions_map", "get_optimizers_map"]


def get_loss_functions_map():
    import torch.nn.functional as F

    return {
        "mse_loss": F.mse_loss,
        "l1_loss": F.l1_loss
    }


def get_optimizers_map():
    import torch

    return {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD
    }
