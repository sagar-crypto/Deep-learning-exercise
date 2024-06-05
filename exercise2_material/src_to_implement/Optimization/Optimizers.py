import numpy as np


class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = []  # the current velocity
        self.previous_velocity = []  # the previous velocity to calculate the current velocity

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.previous_velocity = self.velocity
        if not self.previous_velocity:  # if the previous velocity is not there
            self.velocity = -self.learning_rate * gradient_tensor
        else:
            self.velocity = self.previous_velocity * self.momentum_rate - self.learning_rate * gradient_tensor  # just implementing the formula

        continuation_weight_tensor = weight_tensor + self.velocity
        return continuation_weight_tensor


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.rk_currected = []
        self.velocity_currected = []
        self.k = 0
        self.prev_rk = []
        self.prev_velocity = []
        self.velocity = []
        self.g = []
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.rk = []
        self.epsilon = np.finfo(float).eps

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.g = gradient_tensor
        self.prev_rk = self.rk
        self.prev_velocity = self.velocity
        self.k += 1

        if not self.prev_velocity:
            self.velocity = (1 - self.mu) * self.g
        else:
            self.velocity = self.prev_velocity * self.mu + (1 - self.mu) * self.g

        if not self.prev_rk:
            self.rk = (1 - self.rho) * np.multiply(self.g, self.g)
        else:
            self.rk = self.rho * self.prev_rk + (1 - self.rho) * np.multiply(self.g, self.g)

        self.velocity_currected = self.velocity / (1 - pow(self.mu, self.k))
        self.rk_currected = self.rk / (1 - pow(self.rho, self.k))
        return weight_tensor - self.learning_rate * self.velocity_currected/(np.sqrt(self.rk_currected) + self.epsilon)
