import numpy as np
from functions import *


class Conv2D:
    def __init__(self, input_shape=None, filters=1, kernel_size=(3, 3), isbias=True, activation=None, stride=(1, 1), padding=None, bias=None):
        self.input_shape = input_shape
        self.filters = filters
        self.isbias = isbias
        self.activation = activation
        self.stride = stride
        self.padding = padding
        sefl.p = 1 if padding == 'same' else 0
        self.bias = bias

        if input_shape != None:
            self.kernel_size = (
                kernel_size[0], kernel_size[1], input_shape[2], filters)
            self.output_shape = ((int((input_shape[0] - kernel_size[0] + 2 * self.p) / stride[0]) + 1,
                                  int((input_shape[1] - kernel_size[1] + 2 * self.p) / stride[1]) + 1, filters))
            self.set_variables()
            self.out = np.zeros(self.output_shape)
        else:
            self.kernel_size = (kernel_size[0], kernel_size[1])

    def init_param(self, size):
        stddev = 1/np.sqrt(np.prod(size))
        return np.random.normal(loc=0, scale=stddev, size=size)

    def set_variables(self):
        self.weights = self.init_param(self.kernel_size)
        self.biases = self.init_param((self.filters, 1))
        self.parameters = np.multiply.reduce(
            self.kernel_size) + self.filters if self.isbias else 1
        self.delta_weights = np.zeros(self.kernel_size)
        self.delta_biases = np.zeros(self.biases.shape)

    def forward_propagation(self, image):
        pass
