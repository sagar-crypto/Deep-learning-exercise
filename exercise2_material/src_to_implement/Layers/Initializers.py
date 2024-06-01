import numpy as np

class Constant:
    def __init__(self,initialization_constant = 0.1) -> None:
        self.initialization_constant = initialization_constant
        
    
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape,self.initialization_constant)
        

class UniformRandom:
    def __init__(self) -> None:
        pass
    
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(0,1,weights_shape)

class Xavier:
    def __init__(self) -> None:
        pass
    
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/(fan_in+fan_out))
        
        return np.full(weights_shape,sigma)
            

class He:
    def __init__(self) -> None:
        pass
    
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/fan_in)
        
        return np.full(weights_shape,sigma)