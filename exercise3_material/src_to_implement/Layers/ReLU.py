import numpy as np
from . import Base


class ReLU (Base.BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        self.outward_tensor = None
        self.error_tensor = None
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.outward_tensor = np.maximum(input_tensor, 0)
        return self.outward_tensor

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        gradient = np.where(self.input_tensor > 0, 1, 0)
        return gradient * error_tensor
