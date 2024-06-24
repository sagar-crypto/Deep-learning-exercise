from . import Base
import numpy as np
from scipy import ndimage

class Conv(Base.BaseLayer):
    def __init__(self, stride_shape,convolution_shape, num_kernels):
        super.__init__()
        self.trainable = True
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.one_dimentional_conv = False    
        # Notes:
        # 1D-Input: Shape : [b,c,y] = batch, channel, spatial dimension
        # 2D-Input: Shape : [b, c, y, x] = batch, channel, spatial dimension y, spatial dimension x
        
    
        if len(self.convolution_shape) == 2:
            self.one_dimentional_conv = True
            #output_height = np.floor(input_tensor.shape[])
            
        # Making sure stride shape is a tuple
        if isinstance(stride_shape,int):
            self.stride_shape = (stride_shape,stride_shape)
        else:
            self.stride_shape = stride_shape
        
        
        # Uniform initialization for 2D Weights - Separate weights for each Kernel
        self.weights = np.random.uniform(size=(self.num_kernels, self.conv_shape))
        
        # Uniform initialization for 2D Bias - Single bias for each kernel
        self.bias = np.random.uniform(size=self.num_kernels)    
        
    
    def forward(self, input_tensor):
        
        if self.one_dimentional_conv:
            batch_size, channel, y = input_tensor.shape
            # Y shape after subsampling
            y = np.ceil(y / self.stride_shape[0])  # Example: Y=15, S=2 -> y=8
            # Zero padding to have same spatial shape for stride = 1
            self.output_tensor = np.zeros((batch_size, self.K, y))
            # Go through each batch
            for b, batch in enumerate(input_tensor):
                # Apply multiple Kernel on current batch
                for k, Kernel in enumerate(self.weights):
                    r = ndimage.correlate(batch, Kernel, mode='constant', cval=0.0)
                    # Subsampling aka taking every stride_shape[0]-th element from r 
                    # Example: if stride_shape[0] is 2, it takes every second element from
                    # Adding a bias to every element of the array
                    self.output_tensor[b][k] = r[::self.stride_shape[0]] + self.bias[k]

            return self.output_tensor     
        
        
        output_shape = (output_height)