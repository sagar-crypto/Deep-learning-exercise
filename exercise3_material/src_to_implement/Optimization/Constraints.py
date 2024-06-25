import numpy as np

class L2_Regularizer:
    def __init__(self,alpha) -> None:
        self.alpha = alpha
    
    def calculate_gradient(self, weights):
        return self.alpha * weights
    
    def norm(self, weights):
        # Following the math. implementation from the pdf
        # Using frobenius norm on a matrix is equivalent to applying the L2 norm to each vector and sum them later
        return self.alpha * np.square(np.linalg.norm(weights, ord="fro"))
    
    
class L1_Regularizer:
    def __init__(self,alpha) -> None:
        self.alpha = alpha
    
    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)
    
    def norm(self,weights):
        # ord = 1 is equal to max(sum(abs(x), axis=0)) which is the L1 norm
        # Adding axis = 1 to sum along an axis, as its a matrix
        return self.alpha * np.linalg.norm(weights, ord = 1, axis = 1) 