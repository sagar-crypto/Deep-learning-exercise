import numpy as np
from . import Base


class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.outwards_tensor = None
        self.error_tensor = None
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        normalised_input_tensor = self.input_tensor - self.input_tensor.max()  # prevent value from getting too large
        exponential_tensor = np.exp(normalised_input_tensor)
        summed_exponential_tensor = np.sum(exponential_tensor, axis=1, keepdims=True)
        self.outwards_tensor = exponential_tensor / summed_exponential_tensor

        return self.outwards_tensor 

    def backward(self, error_tensor):
        self.error_tensor = error_tensor

        sum_error_tensor = np.sum((self.error_tensor * self.outwards_tensor), axis=1, keepdims=True)
        error_gradient = self.outwards_tensor * (self.error_tensor - sum_error_tensor)

        return error_gradient
