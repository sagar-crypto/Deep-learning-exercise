
import numpy as np

from scipy import signal
import copy



class Conv:

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.trainable = True

        # Safe input in self
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        # Initialize weights
        self.weight_size = list(self.convolution_shape)
        self.weight_size.insert(0, self.num_kernels)
        self.weights = np.random.uniform(low=0.0, high=1.0, size=self.weight_size)
        self.bias = np.random.uniform(size=self.num_kernels)

        # Initializers
        self.optimizer = None
        self.optimizer_weights = None
        self.optimizer_bias = None
        


    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        self.shape_input_tensor = input_tensor.shape


        #Create a new template for Output_tensor_forward
        self.shape_output_tensor = np.array(self.shape_input_tensor)    # Batch-size for 1D/2D is correct - the rest needs to be adjusted 
        self.shape_output_tensor[1] = self.num_kernels  # New channel number = number of used kernel filters
        self.shape_output_tensor[2] = np.ceil(self.shape_output_tensor[2] / self.stride_shape[0]) # Image shrinkage depends on stride; ceil because of padding
        if len(self.shape_input_tensor) == 4:
            self.shape_output_tensor[3] = np.ceil(self.shape_output_tensor[3] / self.stride_shape[1]) # Image shrinkage depends on stride; ceil becuase of padding
        self.output_tensor_forward = np.zeros(self.shape_output_tensor) # Create the dummy



        # Apply padding to the input tensor
        if len(self.shape_input_tensor) == 3:

            self.x_pad_left = int(self.convolution_shape[1] / 2)
            self.x_pad_right = int((self.convolution_shape[1] - 1) / 2) # -1 to have symmetric padding also for even kernels

            input_tensor_padded = np.pad(self.input_tensor, ((0, 0), (0, 0), (self.x_pad_left, self.x_pad_right)))

        else:
            # using int ensures a symmetric padding, even for even-sized kernels
            self.x_pad_left = int(self.convolution_shape[1] / 2)
            self.x_pad_right = int((self.convolution_shape[1] - 1) / 2) # -1 to have symmetric padding also for even kernels
            self.y_pad_top = int(self.convolution_shape[2]/2)
            self.y_pad_bottom = int((self.convolution_shape[2]-1) / 2)   # -1 to have symmetric padding also for even kernels

            input_tensor_padded = np.pad(self.input_tensor, ((0, 0), (0, 0), (self.x_pad_left, self.x_pad_right), (self.y_pad_top, self.y_pad_bottom)))


        # Use padded input-Tensor as the new input_tensor
        self.input_tensor = input_tensor_padded
        self.shape_input_tensor_padded = self.input_tensor.shape


        # Perform correlation
        for b in range(self.shape_input_tensor_padded[0]):
            for k in range(self.num_kernels):
                if len(self.input_tensor[b].shape) == 2:
                    # Correlate all pixels from all channels using the available kernels
                    r = signal.correlate(self.input_tensor[b], self.weights[k], mode='valid')
                    self.output_tensor_forward[b][k] = r[:,::self.stride_shape[0]] + self.bias[k]
                else:
                    # Correlate all pixels from all channels using the available kernels
                    # For the 2D case, apply separate striding for the second dimension 
                    r = signal.correlate(self.input_tensor[b], self.weights[k], mode='valid')
                    self.output_tensor_forward[b][k] = r[:,::self.stride_shape[0],::self.stride_shape[1]] + self.bias[k]

        return self.output_tensor_forward

    def backward(self, error_tensor):

        # Calculate dummy for strided error_tensor
        # Error Tensor describes error for the kernels in every batch
        # Same shape as input tensor except the channel dimension, which is the number of kernels
        error_tensor_strided = np.array(self.shape_input_tensor)
        error_tensor_strided[1] = self.num_kernels
        # fill with zeros
        self.error_tensor_strided = np.zeros(shape=error_tensor_strided)

        # Masking: place the values of the error tensor only where we actually correlated
        # --> Using stride  - the rest stays zero
        # Check if 1D or 2D
        if len(error_tensor.shape) == 3:
            self.error_tensor_strided[:, :, ::self.stride_shape[0]] = error_tensor
        else:
            self.error_tensor_strided[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor


        # Apply padding
        if len(error_tensor.shape) == 3:
            
            # using int ensures a symmetric padding, even for even-sized kernels
            self.x_pad_l = int(self.convolution_shape[1] / 2)   
            self.x_pad_r = int((self.convolution_shape[1] - 1) / 2) # -1 to have symmetric padding also for even kernels

            self.error_tensor_padded = np.pad(self.error_tensor_strided, ((0, 0), (0, 0), (self.x_pad_l, self.x_pad_r)))

        else:

            self.x_pad_left = int(self.convolution_shape[1] / 2)
            self.x_pad_right = int((self.convolution_shape[1] - 1) / 2) # -1 to have symmetric padding also for even kernels
            self.y_pad_top = int(self.convolution_shape[2] / 2)
            self.y_pad_bottom = int((self.convolution_shape[2] - 1) / 2) # -1 to have symmetric padding also for even kernels

            self.error_tensor_padded = np.pad(self.error_tensor_strided,((0, 0), (0, 0), (self.x_pad_left, self.x_pad_right), (self.y_pad_top, self.y_pad_bottom)))

        # Caltulating the output
        self.output_tensor_backward = np.zeros(self.shape_input_tensor) # Size of original, unpadded Input-Tensor
        # compute the gradient of the loss with respect to input tensor using convolution
        # Weights are flipped / reversed
        for b in range(self.shape_input_tensor[0]):
            for k in range(self.shape_input_tensor[1]):
                # Rearranging the weights
                self.output_tensor_backward[b,k] = signal.convolve(self.error_tensor_padded[b], self.weights.swapaxes(0, 1)[:, ::-1, ...][k], mode='valid') 



        

        # Create templates for weight gradient and bias gradient tensors, initialize with zeros
        self.gradient_weights = np.zeros(self.weights.shape)
        self.gradient_bias = np.zeros(self.num_kernels)


        # Calculate the gradients with respect to bias and weights using correlation
        for b in range(self.shape_input_tensor[0]):
            for k in range(self.num_kernels):
                # For each kernel: Add up gradients from each batch
                self.gradient_weights[k] = self.gradient_weights[k] + signal.correlate(self.input_tensor[b],self.error_tensor_strided[b,k,None],mode='valid')
                self.gradient_bias[k] = self.gradient_bias[k] + np.sum(error_tensor[b, k])


        # UPDATE OPTIMIZERS using the functions from previous exercises
        if self.optimizer_weights != None:
            self.weights = self.optimizer_weights.calculate_update(self.weights, self.gradient_weights)
        else:
            print("Warning: No optimizer specified!")

        if self.optimizer_bias != None:
            self.bias = self.optimizer_bias.calculate_update(self.bias, self.gradient_bias)



        return self.output_tensor_backward




    def initialize(self, weights_initializer, bias_initializer):
        # Initialize the weights & bias using the Initializers.py file
        # Number of total weights per channel are needed to calculate fan_in & fan_out
        total_weights = np.prod(self.weight_size[2::])

        # Initializing the weights
        # shape (weight_shape, fan_in, fan_out)
        self.weights = weights_initializer.initialize(self.weight_size, total_weights  * self.weight_size[1], total_weights  * self.num_kernels)  # shape (weight_shape, fan_in, fan_out)

        #  Initializing the bias
        # shape (weight_shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.num_kernels, total_weights * self.weight_size[1], total_weights * self.num_kernels)        



    # Some properties

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self.optimizer_weights = copy.deepcopy(optimizer)
        self.optimizer_bias = copy.deepcopy(optimizer)

    @property
    def optimizer_weights(self):
        return self._optimizer_weights

    @optimizer_weights.setter
    def optimizer_weights(self, optimizer_weights):
        self._optimizer_weights = optimizer_weights

    @property
    def optimizer_bias(self):
        return self._optimizer_bias

    @optimizer_bias.setter
    def optimizer_bias(self, optimizer_bias):
        self._optimizer_bias = optimizer_bias



    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights



    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias



