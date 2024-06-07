import numpy as np


class Pooling:

    def __init__(self, stride_shape, pooling_shape):
        self.trainable = False
        self.backward_output_tensor = []
        self.forward_output_tensor = []
        self.max_pos_tracker = []
        self.output_tensor_forward = []
        self.input_dimensions = []
        self.input_tensor = []
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        self.input_dimensions = input_tensor.shape

        batch_size, num_channels, height, width = self.input_dimensions
        batch_size, num_channels, height, width = int(batch_size), int(num_channels), int(height), int(width)
        pool_height, pool_width = self.pooling_shape
        stride_height, stride_width = self.stride_shape

        output_width = int(((width - pool_width) / stride_width) + 1)
        output_height = int(((height - pool_height) / stride_height) + 1)
        self.forward_output_tensor = np.zeros((batch_size, num_channels, output_height, output_width))
        self.max_pos_tracker = np.zeros((batch_size, num_channels, output_height, output_width, 2), dtype=int)

        for i in range(batch_size):
            for j in range(num_channels):
                for k in range(output_height):
                    for z in range(output_width):
                        start_x = k * stride_height
                        end_x = start_x + pool_height
                        start_y = z * stride_width
                        end_y = start_y + pool_width

                        temp_arr = self.input_tensor[i, j, start_x:end_x, start_y:end_y]

                        # Check if the pooling window is non-empty
                        if temp_arr.size == 0:
                            raise ValueError("Pooling window is empty. Check the slicing indices.")

                        # Perform max pooling
                        self.forward_output_tensor[i, j, k, z] = np.max(temp_arr)

                        # Get the position of the maximum value
                        relative_max_x, relative_max_y = np.unravel_index(np.argmax(temp_arr), temp_arr.shape)
                        self.max_pos_tracker[i, j, k, z] = (start_x + relative_max_x, start_y + relative_max_y)

        return self.forward_output_tensor

    def backward(self, error_tensor):

        self.backward_output_tensor = np.zeros(self.input_dimensions)
        batch_size, num_channels, height, width = error_tensor.shape
        for i in range(batch_size):
            for j in range(num_channels):
                for k in range(height):
                    for z in range(width):
                        x_pos, y_pos = self.max_pos_tracker[i][j][k][z]
                        x_pos, y_pos = int(x_pos), int(y_pos)
                        self.backward_output_tensor[i][j][x_pos][y_pos] += error_tensor[i][j][k][z]
        return self.backward_output_tensor
