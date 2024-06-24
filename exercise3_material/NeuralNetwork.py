import copy


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer=None, bias_initializer=None):
        self.iterations = None
        self.error_tensor = None
        self.label_tensor = None
        self.cross_entropy_loss = None
        self.predicted_input = None
        self.input_tensor = None
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.phase_var = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        self.predicted_input = self.input_tensor

        for layer in self.layers:
            self.predicted_input = layer.forward(self.predicted_input)

        self.cross_entropy_loss = self.loss_layer.forward(self.predicted_input, self.label_tensor)

        return self.cross_entropy_loss

    def backward(self):
        self.error_tensor = self.loss_layer.backward(self.label_tensor)

        for layer in reversed(self.layers):
            self.error_tensor = layer.backward(self.error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.initialize(self.weights_initializer, self.bias_initializer)
            layer.weights_initializer = (copy.deepcopy(self.weights_initializer))
            layer.bias_initializer = (copy.deepcopy(self.bias_initializer))
            layer.initialize(self.weights_initializer, self.bias_initializer)
            layer.optimizer = (copy.deepcopy(self.optimizer))
        self.layers.append(layer)

    def train(self, iterations):
        self.iterations = iterations

        for iteration in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        predicted_values = input_tensor
        for layer in self.layers:
            predicted_values = layer.forward(predicted_values)
        return predicted_values

    @property
    def phase(self):
        return self.layers[0].testing_phase
    @phase.setter
    def phase(self, phase):
        self.phase_var = phase
        for layer in self.layers:
            layer.testing_phase = phase

    @property
    def test_phase(self):
        return self.layers[0].testing_phase
    @test_phase.setter
    def test_phase(self, test_phase):
        for layer in self.layers:
            layer.testing_phase = test_phase

