from . import Base
import numpy as np

class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        # Shape: (according to task) input_size + 1 bias rows, output_size columns
        # input_size + 1 to add another row for the bias, such that each batch has an additional parameter "bias"   
        self.weights = np.random.uniform(0,1,(self.input_size +1, self.output_size))
        self._optimizer = None
        self.current_input = None
        
    def forward(self, input_tensor):
        # Shape input_tensor: (rows: batch_size, columns: input_size)
        # Shape weights: (rows: inputs, columns: outputs)
        # Add an additional column with ones to the input tensor to pass the bias through
        # Then: Save the input tensor to be used in the backward pass
        ones = np.ones(shape=(input_tensor.shape[0],1))
        self.current_input = np.hstack((input_tensor, ones))
        # To get output shape (rows: batch_size, columns: input_size*weights), we have to transpose the input tensor
        #return self.weights @ input_tensor.T
        #return self.weights @ self.current_input
        return self.current_input @ self.weights
    
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        
        
    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, weights):
        self._gradient_weights = weights
        
        
    def backward(self, error_tensor):
        # Gradient w.r.t. input X 
        # E_n-1 = W.T @ E_n
        # E_n = error tensor = Gradient of loss function wrt output of layer
        grad_X = error_tensor @ self.weights.T
        
        # Compute gradient w.r.t. weights
        grad_W = self.current_input.T @ error_tensor
        
        # Set the gradient weights 
        self._gradient_weights = grad_W
        
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        
        # Removing the additional colummn used for the bias
        return grad_X[:,:-1]
    
    
    def initialize(self, weights_initializer, bias_initializer):
        init_weights = weights_initializer.initialize([self.input_size, self.output_size], self.input_size,
                                                      self.output_size)  # initialise with shape, fan_in, fan_out
        init_bias = bias_initializer.initialize([1, self.output_size], 1, 1)  # TODO Understand it
        self.weights = np.concatenate((init_weights, init_bias), axis=0)