import numpy as np
from . Base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input_tensor):
        self.activations = 1 / (1+ np.exp(-input_tensor))
        return self.activations
    
    def backward(self, error_tensor):
        gradient = self.activations * (1 - self.activations)
        
        return gradient * error_tensor