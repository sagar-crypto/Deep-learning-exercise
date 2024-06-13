import numpy as np
from Layers import Base
from scipy import signal
from scipy import ndimage
import math


class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        assert len(convolution_shape) == 2 or len(convolution_shape) == 3

        self.trainable = True
        self._optimizer = None

        # 1D: [c, m]; 2D:[c, m, n];  c -> channels in, and m, n -> spatial extent of the filter kernel
        self.conv_shape = convolution_shape
        self.K = num_kernels

        self._oneD = True if len(self.conv_shape) == 2 else False

        self._gradient_weights = None
        self._gradient_bias = None

        # Ensure stride_shape is tuple. Either same values for X and Y or different values
        self.stride_shape = (stride_shape, stride_shape) if isinstance(stride_shape, int) else stride_shape
        self.shape_in = None

        # Weights for 2D - Seperate weights for each Kernel
        self.weights = np.random.uniform(size=(self.K, *self.conv_shape))
        # Bias for 2D - Single bias for each kernel
        self.bias = np.random.uniform(size=self.K)

        self.output_tensor = None
        self.input_tensor = None
        self.gradient_X = None

        self.up = False

    def forward(self, input_tensor):
        self.shape_in = input_tensor.shape
        self.input_tensor = input_tensor
        # 1D
        if self._oneD:
            batch_size, channel, y = input_tensor.shape
            # Y shape after subsampling
            y = math.ceil(y / self.stride_shape[0])  # Example: Y=15, S=2 -> y=8
            self.output_tensor = np.zeros((batch_size, self.K, y))
            # Go through each batch
            for b, batch in enumerate(input_tensor):
                # Apply multiple Kernel on current batch
                for k, Kernel in enumerate(self.weights):
                    r = ndimage.correlate(batch, Kernel, mode='constant', cval=0.0)[1]
                    # Subsampling
                    self.output_tensor[b][k] = r[::self.stride_shape[0]] + self.bias[k]

            return self.output_tensor

        else:
            # Parameters
            batch_size, channel, y, x = input_tensor.shape
            # Calculate the shape the convolution would end with without padding
            _, m, n = signal.correlate(np.pad(input_tensor[0], ((0, 0), (0, 0), (0, 0)), mode="constant"),
                                      self.weights[0],
                                      mode='valid').shape
            # Calculate the padding values
            m = y - m
            n = x - n
            # Y,X after subsampling. Example: (X,Y)=(11,15), S=(3,2) -> (X,Y) = (4,8)
            y = math.ceil(y / self.stride_shape[0])
            x = math.ceil(x / self.stride_shape[1])
            # Init output shape
            self.output_tensor = np.zeros((batch_size, self.K, y, x))

            # Go through each batch
            for b, batch in enumerate(input_tensor):

                # Apply multiple Kernel on current batch
                for k, Kernel in enumerate(self.weights):
                    # Correlate the batch(es) with the kernel(s)
                    r = signal.correlate(
                        np.pad(batch, ((0, 0), (int(m / 2), math.ceil(m / 2)), (int(n / 2), math.ceil(n / 2))),
                               mode="constant"), Kernel, mode='valid')
                    # Subsampling
                    self.output_tensor[b][k] = r[:, ::self.stride_shape[0], ::self.stride_shape[1]] + self.bias[k]

            return self.output_tensor

    def backward(self, error_tensor):
        # Arrange gradient w.r.t. to lower layers shape
        self.gradient_X = np.zeros(self.shape_in)
        self._gradient_bias = np.zeros((error_tensor.shape[1]))
        self._gradient_weights = np.zeros_like(self.weights)

        # Rearranging Kernels
        new_K = np.swapaxes(self.weights, 0, 1)[:, ::-1]

        # 1D CASE
        if len(error_tensor.shape) == 3:
            result_weights = np.zeros_like(self._gradient_weights)
            result_lower_layers = np.zeros_like(self.gradient_X)
            # Iterate thorugh batch
            for b, batch in enumerate(error_tensor):
                # Apply multiple Kernel on current batch
                for k, kernel in enumerate(new_K):
                    # Upsampling if necessary
                    if batch.shape[-2:] != self.shape_in[-2:]:
                        upsampled = np.zeros((batch.shape[0], self.gradient_X.shape[-1]))
                        upsampled[::, ::self.stride_shape[0]] = batch
                        # Cross correlation - Since convolution in forward pass
                        r = signal.correlate(upsampled, kernel[::,::-1], mode='same')[1]
                    # Cross correlation without upsampling
                    else:
                        r = signal.correlate(batch, kernel[::,::-1], mode='same')[1]
                    # Insert accordingly
                    self.gradient_X[b][k] = r

            # Bias gradient
            self._gradient_bias = np.sum(np.sum(error_tensor, axis=0), axis=-1)

            # Return for lower layers
            return self.gradient_X


        # 2D CASE
        else:
            # Padding input tensor for gradient w.r.t. weights
            pad_X = [math.ceil((self.conv_shape[-2:-1][0] - 1) / 2),
                     np.floor_divide(self.conv_shape[-2:-1][0] - 1, 2).item()]
            pad_Y = [math.ceil((self.conv_shape[-1:][0] - 1) / 2),
                     np.floor_divide(self.conv_shape[-1:][0] - 1, 2).item()]

            X = np.pad(self.input_tensor, ((0, 0), (0, 0), pad_X, pad_Y), mode="constant")

            result_weights = np.zeros_like(self._gradient_weights)
            result_lower_layers = np.zeros_like(self.gradient_X)

            for b, batch in enumerate(error_tensor):
                # Upsample if necessary
                if batch.shape[-2:] != self.shape_in[-2:]:
                    upsampled = np.zeros((self.K,) + self.shape_in[2:])
                    upsampled[::1, ::self.stride_shape[0], ::self.stride_shape[1]] = batch

                # LOWER LAYERS GRADIENT
                for k, kernel in enumerate(new_K):
                    # Check whether using upsampled tensor or normal one -> Convolve new arranged kernel with upsampled batch
                    if batch.shape[-2:] != self.shape_in[-2:]:
                        result_lower_layers[b, k] = signal.convolve(upsampled, kernel, mode='same')[self.K // 2]

                    # Case if no upsampling needed -> Convolve new arranged kernel with batch
                    else:
                        result_lower_layers[b, k] = signal.convolve(batch, kernel, mode='same')[self.K // 2]

                self.gradient_X = result_lower_layers

                # WEIGHTS GRADIENT
                for c, channel in enumerate(batch):
                    # If necessary use the upsampled error tensor
                    if batch.shape[-2:] != self.shape_in[-2:]:
                        # Correlate the input_tensor batch with an upsampled channel of the error tensor
                        result_weights[c] += signal.correlate(X[b], upsampled[c].reshape(1, *upsampled[0].shape), mode='valid')

                    # If no upsampling needed use the normal error tensor
                    else:
                        # Correlate the input_tensor batch with a channel of the error tensor
                        result_weights[c] += signal.correlate(X[b], channel.reshape(1, *channel.shape), mode='valid')

                self._gradient_weights = result_weights

            # Optimize weights
            if self._optimizer:
                self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)

            else:
                print("No optimizer specified")


            # BIAS GRADIENT
            self._gradient_bias = np.sum(np.sum(np.sum(error_tensor, axis=0), axis=-1), axis=-1)

            return self.gradient_X

    def initialize(self, weights_initializer, bias_initializer):
        # Tuple: (fan_in, fan_out)
        if self._oneD:
            fan_in_out = (self.conv_shape[0] * self.conv_shape[1],
                          self.K * self.conv_shape[1])
        else:
            fan_in_out = (self.conv_shape[0] * self.conv_shape[1] * self.conv_shape[2],
                          self.K * self.conv_shape[1] * self.conv_shape[2])

        # Bias
        self.bias = bias_initializer.initialize(self.bias.shape, *fan_in_out)
        # Weights
        self.weights = weights_initializer.initialize(self.weights.shape, *fan_in_out)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias