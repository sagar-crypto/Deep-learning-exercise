import numpy as np

from . import Helpers


class BatchNormalization:

    def __init__(self, channels):

        self.tensor = None
        self.output_reformated_tensor = None
        self.original_shape = None
        self.gradient_wrt_bias = None
        self.gradient_wrt_weights = None
        self.output_tensor_backward = None
        self.error_tensor = None
        self.output_tensor_forward = None
        self.input_tensor_normalised = None
        self.reformated_input_tensor = None
        self.input_tensor = None
        self.bias = None
        self.weights = None
        self.trainable = True
        self.epsilon = np.finfo(float).eps
        self.testing_phase = False
        self.mean = None
        self.var = None
        self.channels = channels
        self.initialize(0, 0)
        self.reformated_flag = False
        self.mu_0 = 0
        self.sig_0 = 0
        self.optimzer_set = None
        self.test_iteration = 0

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones([self.channels])
        self.bias = np.zeros([self.channels])

    def reformat(self, tensor):
        self.tensor = tensor

        """
            Transformation steps:
            1. shape = batch , channel , n*m
            2. shape = batch , n*m , channel
            3. shape = batch * n * m , channel

            Initialize shape = batch ,channel , n , m --> batch * n * m , channel
        """

        if len(self.tensor.shape) == 4:

            self.original_shape = self.tensor.shape

            self.output_reformated_tensor = np.reshape(self.tensor, [self.tensor.shape[0], self.tensor.shape[1],
                                                          np.prod(self.tensor.shape[2:])])

            self.output_reformated_tensor = np.moveaxis(self.output_reformated_tensor, [1, 2], [2, 1]) 

            self.output_reformated_tensor = np.reshape(self.output_reformated_tensor,
                                            (np.prod(self.output_reformated_tensor.shape[:-1]), self.output_reformated_tensor.shape[2]), )

            return self.output_reformated_tensor

        else:
            self.output_reformated_tensor = np.reshape(self.tensor, [self.original_shape[0], np.prod(self.original_shape[2:]),
                                                          self.original_shape[1]])

            self.output_reformated_tensor = np.moveaxis(self.output_reformated_tensor, [1, 2], [2, 1])
            self.output_reformated_tensor = np.reshape(self.output_reformated_tensor, self.original_shape)

            return self.output_reformated_tensor

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        if len(self.input_tensor.shape) == 4:
            self.reformated_flag = True
            self.reformated_input_tensor = self.reformat(self.input_tensor)
        else:
            self.reformated_input_tensor = self.input_tensor

        if not self.testing_phase:
            self.mean = np.mean(self.reformated_input_tensor, axis=0)
            self.var = np.var(self.reformated_input_tensor, axis=0)

        self.input_tensor_normalised = np.divide((self.reformated_input_tensor - self.mean), np.sqrt(self.var + self.epsilon))
        self.output_tensor_forward = self.weights * self.input_tensor_normalised + self.bias

        if self.reformated_flag:
            self.output_tensor_forward = self.reformat(self.output_tensor_forward)
            self.reformated_flag = False

        return self.output_tensor_forward

    def backward(self, error_tensor):
        self.error_tensor = error_tensor

        if len(self.error_tensor.shape) == 4:
            self.reformated_flag = True
            self.error_tensor = self.reformat(self.error_tensor)
            self.reformated_input_tensor = self.reformat(self.input_tensor)
        else:
            self.reformated_input_tensor = self.input_tensor

        self.gradient_wrt_weights = np.sum(self.error_tensor * self.input_tensor_normalised, axis=0)

        self.gradient_wrt_bias = np.sum(self.error_tensor, axis=0)

        if self.optimzer_set is not None:
            self.weights = self.optimzer_set.calculate_update(self.weights, self.gradient_wrt_weights)
            self.bias = self.optimzer_set.calculate_update(self.bias, self.gradient_wrt_bias)

        self.output_tensor_backward = Helpers.compute_bn_gradients(self.error_tensor, self.reformated_input_tensor,
                                                            self.weights, self.mean, self.var)

        # Reformat
        if self.reformated_flag:
            self.output_tensor_backward = self.reformat(self.output_tensor_backward)

        return self.output_tensor_backward



    @property
    def optimizer(self):
        return self.optimzer_set

    @optimizer.setter
    def optimizer(self, optimizer):
        self.optimzer_set = optimizer

    @property
    def gradient_weights(self):
        return self.gradient_wrt_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.gradient_wrt_weights = gradient_weights

    @property
    def gradient_bias(self):
        return self.gradient_wrt_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self.gradient_wrt_bias = gradient_bias
