from Base import BaseLayer


class SoftMax:
    def __init__(self):
        super().__init__()
        self.error_tensor = None
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
