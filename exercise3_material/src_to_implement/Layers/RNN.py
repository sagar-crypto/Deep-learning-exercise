import numpy as np
from . Base import BaseLayer
from . FullyConnected import FullyConnected
from . TanH import TanH
from . Sigmoid import Sigmoid
from copy import deepcopy

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
                
        self.memorize = False
        self.prev_hidden_state = None
        
        # Initialize the hidden state and the output state  
        # Input size of hidden state is the size of the input and of the previous hidden state (same size as current hidden state)
        self.hidden_layer = FullyConnected((self.input_size+self.hidden_size),self.hidden_size)
        # Initialze the hidden states with all zeros (+1 as the bias is included in the weights)
        self.hidden_layer.weights = np.zeros((self.input_size+self.hidden_size +1, self.hidden_size))
        
        self.output_layer = FullyConnected(self.hidden_size, self.output_size)


        # Initialize the tanh and Sigmoid laywers
        self.tanh_layer = TanH()
        self.sigmoid_layer = Sigmoid()
        
    
    def forward(self,input_tensor):
        self.first_iteration = True
        self.input_tensor = input_tensor
        self.batch_size = input_tensor.shape[0]

        if self.first_iteration:
            if self.memorize:
                self.hidden_state = self.prev_hidden_state
                # temp = self.prev_hidden_state
                # self.hidden_state = np.zeros((self.batch_size, self.hidden_size))
                # self.hidden_state[0, :] = temp[-1, :]
            else:
                self.hidden_state = np.zeros(shape=(self.batch_size, self.hidden_size))
            self.first_iteration = False
            
        # The states of a single sequence
        self.forward_output = np.zeros((self.batch_size, self.output_size))
        
        
        for i, sequence in enumerate(input_tensor):
            
            if i !=0 :
                # Pass hidden state from previous TIME STEP (=sequence) to current hidden state
                self.hidden_state_prev_call = self.hidden_state[i-1].reshape((1, -1))
            else:
                self.hidden_state_prev_call = self.hidden_state
                
            # Concatenate hidden state from prev TS and the sequence input 
            #TODO Here the concatenation might cause the error!
            conc_hidden_input = np.concatenate((sequence.reshape((1,-1)),self.hidden_state_prev_call.reshape((1,-1))), axis=1)
            self.hidden_state[i] = self.tanh_layer.forward(self.hidden_layer.forward(conc_hidden_input))
      
            self.forward_output[i] = self.output_layer.forward(self.sigmoid_layer.forward(self.hidden_state[i].reshape(1,-1) ))
  
            
        # Storing the hidden state for the whole current iteration (not just one sequence) for
        # possible use in next forward iteration
        self.prev_hidden_state = self.hidden_state
        
        return self.forward_output                             
                                         
                  
    def backward(self, error_tensor):
        """ RNN backward pass

        :param error_tensor: Gradient tensor from the upper layer
        :return: Gradient w.r.t. input tensor that serves as input to the lower layer during backpropagation
        """
        self.gradient_output = np.zeros_like(self.output_layer.weights)
        self._gradient_weights = np.zeros((self.conc_size + 1, self.hidden_size))

        self.input_gradient = np.zeros((error_tensor.shape[0], self.input_size))
        hidden_state_gradient = np.zeros((1, self.hidden_size))

        for i, sequence in enumerate(np.flip(error_tensor, axis=0)):
            it = error_tensor.shape[0] - i - 1
            current_hidden_seq = self.hidden_state[it].reshape(1, -1)

            self.output_layer.recent_input = self.output_layer_inputs[it]

            self.sigmoid_layer.prev_tensor = self.sigmoids[it].reshape(1, -1)

            # Gradient of the second fully connected layer + sigmoid
            gradient_of_output_layer = self.output_layer.backward(self.sigmoid_layer.backward(sequence))
            self.gradient_output += self.output_layer.gradient_weights

            # Copy operation is just a sum
            gradient_after_hidden_layer = gradient_of_output_layer + hidden_state_gradient

            # Gradient of the second fully connected layer + tanH
            # Concatenated gradient is split into input and hidden
            self.tanH_layer.prev_tensor = self.tanHs[it].reshape(1, -1)
            self.hidden_layer.recent_input = self.hidden_layer_inputs[it]

            gradient_of_hidden_layer = self.hidden_layer.backward(self.tanH_layer.backward(gradient_after_hidden_layer))
            self._gradient_weights += self.hidden_layer.gradient_weights

            self.input_gradient[it], hidden_state_gradient, _ = np.split(gradient_of_hidden_layer,
                                                                            [self.input_size,
                                                                            self.input_size + self.hidden_size], axis=1)

        self.hidden_layer.gradient_weights = self._gradient_weights
        self.output_layer.gradient_weights = self.gradient_output

        # Update each layer with one of the two optimizers
        if self.optimizer_hidden and self.optimizer_output:
            self.output_layer.weights = self.optimizer_output.calculate_update(self.output_layer.weights, self.gradient_output)
            self.hidden_layer.weights = self.optimizer_hidden.calculate_update(self.hidden_layer.weights, self._gradient_weights)
        else:
            print("No optimizer specified")

        self._weights = self.hidden_layer.weights

        return self.input_gradient

    def initialize(self, weights_initializer, bias_initializer):
        """Initializes weights and bias for RNN

        :param weights_initializer: A weight initializer from Initializers.py
        :param bias_initializer: A bias initializer from Initializers.py
        :return: Weights and biases
        """
        self.hidden_layer.initialize(weights_initializer, bias_initializer)
        self.output_layer.initialize(weights_initializer, bias_initializer)

                                         
                                         

    @property
    def memorize(self):
        return self._memorize
    
    @memorize.setter
    def memorize(self,bool_value):
        self._memorize = bool_value
        
        
    @property
    def gradient_weights(self):
        return self.hidden_layer.gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def weights(self):
        return self.hidden_layer.weights

    @weights.setter
    def weights(self, value):
        self.hidden_layer.weights = value

    @property
    def optimizer(self):
        return self.optimizer_hidden

    @optimizer.setter
    def optimizer(self, value):
        self.optimizer_hidden = deepcopy(value)
        self.optimizer_output = deepcopy(value)