import numpy as np
from . import FullyConnected
from . import TanH
from . import Sigmoid
from copy import deepcopy
import logging
logging.basicConfig(level=logging.DEBUG)


class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.tanh_error_tensor = None
        self.previous_h_why_tensor = None
        self.why_error_tensor = None
        self.sigmoid_error_tensor = None
        self.output_backward_tensor = None
        self.hy_gradient_weights = None
        self.hx_gradient_weights = None
        self.error_tensor = None
        self.hy_outward_forward_tensor = None
        self.hx_outward_forward_tensor = None
        self.prev_xt_ht = None
        self.memorize_sigmoid = None
        self.memorize_tanh = None
        self.memorize_hx = None
        self.memorize_hy = None
        self.input_tensor = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.memorize = False
        self.trainable = True
        self.optimizer_set = None
        self.hidden_state = np.zeros((1, self.hidden_size))
        self.forward_output_tensor = []
        self.gradient_weight_tensor_hx = []
        self.gradient_weight_tensor_hy = []
        self.hx_fullyconnected_tensor = FullyConnected.FullyConnected(input_size + hidden_size, hidden_size)
        self.hy_fullyconnected_tensor = FullyConnected.FullyConnected(hidden_size, output_size)

        self.tanH = TanH.TanH()
        self.sigmoid = Sigmoid.Sigmoid()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        self.memorize_hy = []
        self.memorize_hx = []
        self.memorize_tanh = []
        self.memorize_sigmoid = []

        self.forward_output_tensor = np.zeros((self.input_tensor.shape[0], self.output_size))

        if not self.memorize:
            self.hidden_state = np.zeros(shape=(1, self.hidden_size))

        for i in range(self.input_tensor.shape[0]):
            self.prev_xt_ht = np.concatenate((self.input_tensor[i].reshape(1, -1), self.hidden_state), axis=1)
            self.hx_outward_forward_tensor = self.hx_fullyconnected_tensor.forward(self.prev_xt_ht)
            self.hidden_state = self.tanH.forward(self.hx_outward_forward_tensor)

            self.hy_outward_forward_tensor = self.hy_fullyconnected_tensor.forward(self.hidden_state)
            forward_output_tensor_iteration = self.sigmoid.forward(self.hy_outward_forward_tensor)

            self.forward_output_tensor[i] = forward_output_tensor_iteration

            # memorizing for backward
            self.memorize_hx.append(self.hx_fullyconnected_tensor.current_input)
            self.memorize_hy.append(self.hy_fullyconnected_tensor.current_input)
            self.memorize_tanh.append(self.tanH.activations)
            self.memorize_sigmoid.append(self.sigmoid.activations)

        return self.forward_output_tensor

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        self.hidden_state = np.zeros(shape=(1, self.hidden_size))

        self.output_backward_tensor = np.zeros((self.error_tensor.shape[0], self.input_size))

        for i in range(self.error_tensor.shape[0]):
            # get the values back
            self.hx_fullyconnected_tensor.current_input = self.memorize_hx[i]
            self.hy_fullyconnected_tensor.current_input = self.memorize_hy[i]
            self.tanH.activations = self.memorize_tanh[i]
            self.sigmoid.activations = self.memorize_sigmoid[i]

            self.sigmoid_error_tensor = self.sigmoid.backward(error_tensor[i].reshape(1, -1))

            self.why_error_tensor = self.hy_fullyconnected_tensor.backward(self.sigmoid_error_tensor)

            self.previous_h_why_tensor = self.why_error_tensor + self.hidden_state

            self.tanh_error_tensor = self.tanH.backward(self.previous_h_why_tensor)

            self.whxh_error_tensor = self.hx_fullyconnected_tensor.backward(self.tanh_error_tensor)

            self.output_backward_tensor[i] = self.whxh_error_tensor[0, :self.input_size]
            self.hidden_state = self.whxh_error_tensor[0, self.input_size:]

            np.append(self.hx_gradient_weights, self.hx_fullyconnected_tensor.gradient_weights)
            np.append(self.hy_gradient_weights, self.hy_fullyconnected_tensor.gradient_weights)

            # Sum up the gradients
        self.gradient_weight_tensor_hx = np.sum(self.hx_gradient_weights, axis=0)
        self.gradient_weight_tensor_hy = np.sum(self.hy_gradient_weights, axis=0)

        # OPTIMIZERS
        if self.optimizer_set:
            self.hx_fullyconnected_tensor.weights = self.gradient_hx_optimizer.calculate_update(
                self.hx_fullyconnected_tensor.weights,
                self.gradient_weight_tensor_hx)
            self.hy_fullyconnected_tensor.weights = self.gradient_hy_optimizer.calculate_update(
                self.hy_fullyconnected_tensor.weights, self.gradient_weight_tensor_hy)

        return self.output_backward_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.hx_fullyconnected_tensor.initialize(weights_initializer, bias_initializer)
        self.hy_fullyconnected_tensor.initialize(weights_initializer, bias_initializer)

    @property
    def memorize(self):
        return self.set_memorize

    @memorize.setter
    def memorize(self, memorize):
        self.set_memorize = memorize

    @property
    def weights(self):
        return self.hx_fullyconnected_tensor.weights

    @weights.setter
    def weights(self, weights):
        self.hy_fullyconnected_tensor.weights = weights

    @property
    def gradient_weights(self):
        return self.hx_gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.hx_gradient_weights = gradient_weights

    @property
    def optimizer(self):
        return self.optimizer_set

    @optimizer.setter
    def optimizer(self, optimizer):
        self.optimizer_set = optimizer
        self.gradient_hx_optimizer = deepcopy(optimizer)
        self.gradient_hy_optimizer = deepcopy(optimizer)


