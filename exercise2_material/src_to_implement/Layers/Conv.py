from . import Base
import numpy as np

class Conv(Base.BaseLayer):
    def __init__(self, stride_shape,convolution_shape, num_kernels):
        super.__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
            
    # Notes:
    # 1D-Input: Shape : [b,c,y] = batch, channel, spatial dimension
    # 2D-Input: Shape : [b, c, y, x] = batch, channel, spatial dimension y, spatial dimension x
    
    def forward(self, input_tensor):
        if len(input_tensor.shape) <= 3:
            one_dimentional_conv = True
            #output_height = np.floor(input_tensor.shape[])

        
        
        output_shape = (output_height)