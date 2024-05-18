import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.epsilon = None
        self.label_tensor = None
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor

        self.epsilon = np.finfo(float).eps

        loss = -np.sum(self.label_tensor * np.log(prediction_tensor + self.epsilon))
        return loss

    def backward(self, label_tensor):
        self.label_tensor = label_tensor

        error_tensor = self.label_tensor / - (self.prediction_tensor + self.epsilon)
        return error_tensor
