import numpy as np

class Optimizer:
    def __init__(self):
        self.regularizer = None
    
    def add_regularizer(self, regularizer):
        self.regularizer = regularizer
        

class Sgd(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # If a regularizer is set, add it to the loss term with a negative sign 
        if self.regularizer:
            gradient_tensor = (self.regularizer.calculate_gradient(weight_tensor) + gradient_tensor)
            
        return weight_tensor - self.learning_rate * gradient_tensor

# Only tried to fix the errors with SGDwithmomentum for the start
# Because its the same issue for Adam

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = np.array([])   # the current velocity
        self.previous_velocity = np.array([])   # the previous velocity to calculate the current velocity

    def calculate_update(self, weight_tensor, gradient_tensor):

        self.previous_velocity = self.velocity
        if not self.previous_velocity.any():  # if the previous velocity is not there
            self.velocity = -self.learning_rate * gradient_tensor
        else:
            self.velocity = self.previous_velocity * self.momentum_rate - self.learning_rate * gradient_tensor  # just implementing the formula

        # If a regularizer is set, add it to the loss term with a negative sign 
        if self.regularizer:
            continuation_weight_tensor = weight_tensor + self.velocity - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        else:
            continuation_weight_tensor = weight_tensor + self.velocity
            
        return continuation_weight_tensor


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.rk_currected = np.array([])
        self.velocity_currected = np.array([])
        self.k = 0
        self.prev_rk = np.array([])
        self.prev_velocity = np.array([])
        self.velocity = np.array([])
        self.g = np.array([])
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.rk = np.array([])
        self.epsilon = np.finfo(float).eps

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.g = gradient_tensor
        self.prev_rk = self.rk
        self.prev_velocity = self.velocity
        self.k += 1

        if not self.prev_velocity.any():
            self.velocity = (1 - self.mu) * self.g
        else:
            self.velocity = self.prev_velocity * self.mu + (1 - self.mu) * self.g

        if not self.prev_rk.any():
            self.rk = (1 - self.rho) * np.multiply(self.g, self.g)
        else:
            self.rk = self.rho * self.prev_rk + (1 - self.rho) * np.multiply(self.g, self.g)

        self.velocity_currected = self.velocity / (1 - pow(self.mu, self.k))
        self.rk_currected = self.rk / (1 - pow(self.rho, self.k))
       
        # If a regularizer is set, add it to the loss term with a negative sign 
        if self.regularizer:
            return weight_tensor - self.learning_rate * self.velocity_currected / (
                    np.sqrt(self.rk_currected) + self.epsilon) - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        else:
            return weight_tensor - self.learning_rate * self.velocity_currected / (
                    np.sqrt(self.rk_currected) + self.epsilon) 

