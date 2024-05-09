import Base
import numpy as np

class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        #TODO Correct shape: Goal: Rows * columns
        self.weights = np.random.uniform(0,1,(input_size,output_size))
        self._optimizer = None
        
        
    def forward(self, input_tensor):
        
        # Shape input_tensor: (rows: batch_size, columns: input_size)
        # Shape weights: (rows: inputs, columns: outputs)
        # To get output shape (rows: batch_size, columns: input_size*weights), we have to transpose the input tensor
        return self.weights @ input_tensor.T
    
    def get_optimizer(self):
        return self._optimizer
    
    def set_optimizer(self, optimizer):
        self._optimizer = optimizer
        
        
    def backward(error_tensor):
        pass