import numpy as np


from Layers import FullyConnected
from Layers import Sigmoid
from Layers import TanH

from copy import deepcopy

class RNN:

    def __init__(self, input_size, hidden_size, output_size):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.memorize = False
        self.trainable = True
        self._optimizer = None

        self.hidden_state = np.zeros((1, self.hidden_size))
        self.output_forward = []
        self.FCNN_hxh_gradient_weights = []
        self.FCNN_hy_gradient_weights = []


        # First FCNN: from ht-1 & xt to h (after tanh)
            # Taking them as separate FCNNs is illegal - whysoever
        self.FCNN_hxh = FullyConnected.FullyConnected(input_size + hidden_size, hidden_size)

        # Second FCNN: from ht to yt (after sigmoid
        self.FCNN_hy = FullyConnected.FullyConnected(hidden_size, output_size)

        self.TanH = TanH.TanH()
        self.Sigmoid = Sigmoid.Sigmoid()




    def initialize(self, weights_initializer, bias_initializer):
        self.FCNN_hxh.initialize(weights_initializer, bias_initializer)
        self.FCNN_hy.initialize(weights_initializer, bias_initializer)




    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        # Dummies for memory
        self.mem_FCNN_hxh_out_forward = []
        self.mem_TanH = []
        self.mem_FCNN_hy_out_forward = []
        self.mem_Sig = []
        
        self.output_forward = np.zeros((self.input_tensor.shape[0], self.output_size))

        if self.memorize == False:  # Don't use hidden state --> see what influnece xt has on ht&yt
            self.hidden_state = np.zeros(shape=(1, self.hidden_size))



        for t in range(self.input_tensor.shape[0]):


            # Concatenate x and ht first - whysoever :-\ - lost 2h thanks 4 nothing
            self.xt_htprev_concatenated = np.concatenate((self.input_tensor[t].reshape(1, -1), self.hidden_state), axis=1)
            # calculate NN with Whh&ht-1 & Wxh&xt
            self.FCNN_hxh_out_forward = self.FCNN_hxh.forward(self.xt_htprev_concatenated)

            # Calculate ht
            self.hidden_state = self.TanH.forward(self.FCNN_hxh_out_forward)

            # Calculate yt
            self.FCNN_hy_out_forward = self.FCNN_hy.forward(self.hidden_state)
            new_output_forward = self.Sigmoid.forward(self.FCNN_hy_out_forward)


            #Concatenate solutions
            self.output_forward[t] = new_output_forward

            # Memory for backward
            self.mem_FCNN_hxh_out_forward.append(self.FCNN_hxh.optimized_input_tensor)
            self.mem_TanH.append(self.TanH.output_tensor_forward)
            self.mem_FCNN_hy_out_forward.append(self.FCNN_hy.optimized_input_tensor)
            self.mem_Sig.append(self.Sigmoid.output_tensor_forward)


        return self.output_forward






    def backward(self, error_tensor):
        self.error_tensor = error_tensor

        self.FCNN_hxh_gradient_weights = []
        self.FCNN_hy_gradient_weights = []

        self.hidden_state = np.zeros((1, self.hidden_size))  # initialization for first iteration


        #Create some dummies
        self.output_backward = np.zeros((self.error_tensor.shape[0], self.input_size))

        for t in reversed(range(self.error_tensor.shape[0])):
            # Regain the memory
            self.FCNN_hxh.optimized_input_tensor = self.mem_FCNN_hxh_out_forward[t]
            self.TanH.output_tensor_forward = self.mem_TanH[t]
            self.FCNN_hy.optimized_input_tensor = self.mem_FCNN_hy_out_forward[t]
            self.Sigmoid.output_tensor_forward = self.mem_Sig[t]


            #Go backwards through sigmoid
            self.error_tensor_after_sigmoid = self.Sigmoid.backward(error_tensor[t].reshape(1, -1))

            #Go backwards through Why
            self.error_tensor_after_Why = self.FCNN_hy.backward(self.error_tensor_after_sigmoid)

            # Sum of previous hidden and FCNN_hy_backward
            self.error_tensor_after_Why_plus_prev_hidden_state =  self.error_tensor_after_Why + self.hidden_state

            # Go backwards through tanH
            self.error_tensor_after_TanH = self.TanH.backward(self.error_tensor_after_Why_plus_prev_hidden_state)

            # Go backwards through Whxh
            self.error_tensor_after_Whxh = self.FCNN_hxh.backward(self.error_tensor_after_TanH)


            # Split ht-1 and xt again
            self.output_backward[t] = self.error_tensor_after_Whxh[0, :self.input_size]
            self.hidden_state = self.error_tensor_after_Whxh[0, self.input_size:]


            # Memorize all the gradients in each time step
            self.FCNN_hy_gradient_weights.append(self.FCNN_hy.gradient_weights)
            self.FCNN_hxh_gradient_weights.append(self.FCNN_hxh.gradient_weights)


        # Sum up the gradients
        self.FCNN_hxh_gradient = np.sum(self.FCNN_hxh_gradient_weights, axis=0)
        self.FCNN_hy_gradient = np.sum(self.FCNN_hy_gradient_weights, axis=0)

        # OPTIMIZERS
        if self._optimizer:
            self.FCNN_hxh.weights = self.FCNN_hxh_optimizer.calculate_update(self.FCNN_hxh.weights, self.FCNN_hxh_gradient)
            self.FCNN_hy.weights = self.FCNN_hy_optimizer.calculate_update(self.FCNN_hy.weights, self.FCNN_hy_gradient)


        return self.output_backward






    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize


    @property
    def weights(self):
        return self.FCNN_hxh.weights

    @weights.setter
    def weights(self, weights):
        self.FCNN_hxh.weights = weights

    @property
    def gradient_weights(self):
        return self.FCNN_hxh_gradient

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.FCNN_hxh_gradient = gradient_weights


    @property
    def optimizer (self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self.FCNN_hxh_optimizer = deepcopy(optimizer)   # Paranoia strikes again...
        self.FCNN_hy_optimizer = deepcopy(optimizer)

