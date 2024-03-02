import numpy as np
import logging

logger = logging.getLogger(__name__)


class EarlyStopper:
    """
    Code from: https://stackoverflow.com/a/73704579
    """
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class StopTrainingWhenTrainLossIsNearZero:
    @staticmethod
    def training_loss_is_near_zero(train_loss):
        if train_loss < 0.01:
            return True
        return False
