from ast import Str
from json import detect_encoding
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
        self.p = 1 if padding == 'same' else 0
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
        self.input = image                      # keep track of last input for backpropagation
        kshape = self.kernel_size

        # Check kernel size and stride values
        if kshape[0] % 2 != 1 or kshape[1] % 2 != 1:
            raise ValueError('Please provide odd length of 2D kernel.')
        if type(self.stride) == int:
            stride = (stride, stride)
        else:
            stride = self.stride

        # Zero padding
        if self.padding == None:
            pass
        elif self.padding == 'valid':
            pass
        elif self.padding == 'same':
            h = np.zeros((image.shape[1], image.shape[2])
                         ).reshape(-1, image.shape[1], image.shape[2])
            v = np.zeros((image.shape[0] + 2, image.shape[2])
                         ).reshape(image.shape[0] + 2, -1, image.shape[2])
            padded_image = np.vstack(h, image, h)           # add rows
            padded_image = np.vstack(v, padded_image, v)    # add columns
            image = padded_image

        # Convolve each filter over padded image
        feature_maps = []
        for f in range(self.filters):
            rv = 0
            for r in range(kshape[0], image.shape[0] + 1, stride[0]):
                cv = 0
                for c in range(kshape[1], image.shape[1] + 1, stride[1]):
                    convolved_region = image[rv:r, cv:c]
                    detection = (np.multiply(
                        detection, self.weights[:, :, :, f]))
                    result = detection.sum() + self.biases[f]
                    feature_maps.append(result)
                    cv += stride[1]
                rv += stride[0]
            feature_maps = np.array(feature_maps).reshape(
                int(rv/stride[0]), int(cv/stride[1]))
            self.out[:, :, f] = feature_maps
        self.out = relu(self.out)
        return self.out

    def backward_propagation(self, next_layer):
        current_layer = self
        current_layer.delta = np.zeros(
            (current_layer.input_shape[0], current_layer.input_shape[1], current_layer.input_shape[2]))
        image = current_layer.input
        kshape = current_layer.kernel_size
        shape = current_layer.input_shape
        stride = current_layer.stride

        for f in range(current_layer.filters):
            rv = 0
            i = 0
            for r in range(kshape[0], shape[0]+1, stride[0]):
                cv = 0
                j = 0
                for c in range(kshape[1], shape[1]+1, stride[1]):
                    convolved_region = image[rv:r, cv:c]
                    current_layer.delta_weights[:, :, :,
                                                f] += convolved_region * next_layer.delta[i, j, f]
                    current_layer.delta[rv:r, cv:c, :] += next_layer.delta[i,
                                                                           j, f] * current_layer.weights[:, :, :, f]
                    j += 1
                    cv += stride[1]
                rv += stride[0]
                i += 1
            current_layer.delta_biases[f] = np.sum(next_layer.delta[:, :, f])
        current_layer.delta = relu_backward(current_layer.delta)


class MaxPooling2D:
    def __init__(self, kernel_size=(2, 2), stride=None, padding=None):
        self.input_shape = None
        self.output_shape = None
        self.output = None
        self.isbias = False
        self.activation = None
        self.parameters = 0
        self.delta = 0
        self.weights = 0
        self.bias = 0
        self.delta_weights = 0
        self.delta_biases = 0
        self.padding = padding
        self.p = 1 if padding == 'same' else 0
        self.kernel_size = kernel_size
        if type(stride) == int:
            stride = (stride, stride)
        self.stride = stride
        if self.stride == None:
            self.stride = self.kernel_size
        self.output_shape = (int((self.input_shape[0] - self.kernel_size[0] + 2 * self.p) / self.stride[0] + 1),
                             int((
                                 self.input_shape[1] - self.kernel_size[1] + 2 * self.p) / self.stride[1] + 1),
                             self.input_shape[2])
        self.out = np.zeros(self.output_shape)

    def forward_propagation(self, image):
        self.input = image
        kshape = self.kernel_size
        if type(self.stride) == int:
            stride = (stride, stride)
        else:
            stride = self.stride

        for f in range(image.shape[2]):
            pooled_feature_maps = []
            rv = 0
            for r in range(kshape[0], image.shape[0]+1, stride[0]):
                cv = 0
                for c in range(kshape[1], image.shape[1]+1, stride[1]):
                    pooled_region = image[rv:r, cv:c, f]
                    pooled_region = np.max(pooled_region)
                    cv += stride[1]
                rv += stride[0]
            pooled_feature_maps = np.array(pooled_feature_maps).reshape(
                int(rv/stride[0]), int(cv/stride[1]))
            self.out[:, :, f] = pooled_feature_maps
        return self.out

    def backward_propagation(self, next_layer):
        current_layer = self
        current_layer.delta = np.zeros(
            (current_layer.input_shape[0], current_layer.input_shape[1], current_layer.input_shape[2]))
        image = current_layer.input
        kshape = current_layer.kernel_size
        shape = current_layer.input_shape
        stride = current_layer.stride

        for f in range(shape[2]):
            i = 0
            rv = 0
            for r in range(kshape[0], shape[0]+1, stride[0]):
                cv = 0
                j = 0
                for c in range(kshape[1], shape[1]+1, stride[1]):
                    pooled_region = image[rv:r, cv:c, f]
                    dout = next_layer.delta[i, j, f]
                    p = np.max(pooled_region)
                    index = np.argwhere(pooled_region == p)[0]
                    current_layer.delta[rv+index[0], cv+index[1], f] = dout
                    j += 1
                    cv += stride[1]
                rv += stride[0]
                i += 1


class Flatten:
    def __init__(self, input_shape=None):
        self.input_shape = None
        self.out = None
        self.isbias = False
        self.parameters = 0
        self.delta = 0
        self.weights = 0
        self.bias = 0
        self.delta_weights = 0
        self.delta_biases = 0

        self.output_shape = (
            self.input_shape[0] * self.input_shape[1] * self.input_shape[2])

    def forward_propagation(self, image):
        self.input = image
        self.out = np.array(image).flatten()
        return self.out

    def back_propagation(self, next_layer):
        self.error = np.dot(next_layer.weights, next_layer.delta)
        self.delta = self.error * self.out
        self.delta = self.delta.reshape(self.input_shape)
