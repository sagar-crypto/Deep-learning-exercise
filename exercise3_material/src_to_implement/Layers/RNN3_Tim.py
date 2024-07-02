import enum
from queue import Full
import numpy as np
from .TanH import TanH
from .Sigmoid import Sigmoid
from .Base import BaseLayer
from .FullyConnected import FullyConnected
from copy import deepcopy

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        """Recurrent neural network based on elman cell

        :param input_size:
        :param hidden_size:
        :param output_size:
        """
        super().__init__()
        self.trainable = True

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.forward_first_time = True
        self._memorize = False

        # Concatenation size for layer initialization
        # We don't need to add one because that's done in fully connected layers?
        self.conc_size = input_size + hidden_size

        # Initialize the layers
        # Two fully connected layers, one for hidden state and another for output
        self.hidden_layer = FullyConnected(self.conc_size, hidden_size)
        self.output_layer = FullyConnected(hidden_size, output_size)
        self.tanH_layer = TanH()
        self.sigmoid_layer = Sigmoid()

        # Init optimizers
        self.optimizer_hidden = None
        self.optimizer_output = None

        # Initialize the weights (This weight is for the hidden layer with tanh function)
        # self._weights = np.zeros((self.conc_size + 1, self.hidden_size))
        self._gradient_weights = np.zeros((self.conc_size + 1, self.hidden_size))
        self.gradient_output = np.zeros_like(self.output_layer.weights)


    def forward(self, input_tensor):
        """RNN forward pass.

        :param input_tensor: Output tensor from the lower layer
        :return: The input tensor for the next layer
        """
        self.hidden_layer_inputs = []
        self.output_layer_inputs = []
        self.input_tensor = input_tensor

        self.batch_size = input_tensor.shape[0]

        self.sigmoids = np.zeros((self.batch_size, self.output_size))
        self.tanHs = np.zeros((self.batch_size, self.hidden_size))
        self.prev_input = input_tensor

        # States of a single sequence
        self.forward_output = np.zeros((self.batch_size, self.output_size))

        # If this is the first time calling forward, initiate previous hidden state as zeros
        if self.forward_first_time:
            self.prev_hidden_state = np.zeros((self.batch_size, self.hidden_size))
            self.forward_first_time = False

        # Initialize the hidden layer as zeros if memorize is false, and restore previous if true
        if self._memorize:
            temp = self.prev_hidden_state
            self.hidden_state = np.zeros((self.batch_size, self.hidden_size))
            self.hidden_state[0, :] = temp[-1, :]
        else:
            self.hidden_state = np.zeros((self.batch_size, self.hidden_size))

        # Hidden state is calculated using fully connected layer (hidden_layer)
        for i, sequence in enumerate(input_tensor):

            if i != 0:
                self.hidden_state[i] = hidden_state_minus

            intermediate = np.concatenate((sequence.reshape((1, -1)), self.hidden_state[i].reshape((1, -1))), axis=1)
            self.hidden_state[i] = self.tanH_layer.forward(self.hidden_layer.forward(intermediate))
            self.hidden_layer_inputs.append(self.hidden_layer.recent_input)
            hidden_state_minus = self.hidden_state[i].reshape((1, -1))

            # Output is calculated using fully connected layer (output_layer)
            self.forward_output[i] = self.sigmoid_layer.forward(
                self.output_layer.forward(self.hidden_state[i].reshape(1, -1)))

            self.output_layer_inputs.append(self.output_layer.recent_input)

            self.sigmoids[i] = self.sigmoid_layer.prev_tensor
            self.tanHs[i] = self.tanH_layer.prev_tensor

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
    def memorize(self, value):
        self._memorize = value

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