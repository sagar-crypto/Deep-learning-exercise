import numpy as np
from . Base import BaseLayer

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
                
        self._memorize = False
        self.prev_hidden_state = None
        
        self.weights_current_hidden_state = np.random.uniform(0,1,(self.input_size))
        # Not sure if shape should be input or output size
        self.bias_output = np.random.uniform(0,1,(self.output_size))
        self.weights_prev_hidden_state = np.random.uniform(0,1,(self.hidden_size))
        self.weights_current_input = np.random.uniform(0,1,(self.input_size))
        self.bias_update = np.random.uniform(0,1,(self.output_size))
        

    
    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        self.batch_size = input_tensor.shape[0]
        
        if self._memorize:
            self.hidden_state = self.prev_hidden_state
        else:
            self.hidden_state = np.zeros(shape=(self.batch_size, self.hidden_size))
                                         
                                         
                                         
                                         

    @property
    def memorize(self):
        return self._memorize
    
    @memorize.setter
    def memorize(self,bool_value):
        self._memorize = bool_value