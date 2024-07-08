import numpy as np
import random as rd


class Dropout:
    def __init__(self, probability):
        self.error_tensor = np.array([])
        self.back_output_tensor = np.array([])
        self.neuron_survival_probability = probability
        self.survived_neurons = np.array([])
        self.trainable = False
        self.testing_phase = False
        self.output_tensor_dropout = np.array([])

    def forward(self, input_tensor):
        self.output_tensor_dropout = np.copy(input_tensor)  # does not work if not using copy
        self.survived_neurons = np.zeros(input_tensor.shape[0])

        if not self.testing_phase:
            for i in range(input_tensor.shape[0]):
                if rd.random() > self.neuron_survival_probability:
                    self.output_tensor_dropout[i] = np.zeros(input_tensor[i].shape)
                else:
                    self.survived_neurons[i] = 1  # updating the index with 1 which survived
                    self.output_tensor_dropout[i] = self.output_tensor_dropout[i] / self.neuron_survival_probability

        return self.output_tensor_dropout

    def backward(self, error_tensor):
        self.error_tensor = np.copy(error_tensor)
        if not self.testing_phase:
            output_tensor = np.zeros(self.error_tensor.shape)
            for i in range(error_tensor.shape[0]):
                if int(self.survived_neurons[i]) == 1:
                    output_tensor[i] = self.error_tensor[i]/ self.neuron_survival_probability
        else:
            output_tensor = error_tensor

        return output_tensor


