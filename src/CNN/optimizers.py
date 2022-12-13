import numpy as np


class Optimizer:
    def __init__(self, layers, name=None, learning_rate=0.01, mr=0.001):
        self.name = name
        self.learning_rate = learning_rate
        self.mr = mr
        keys = ["sgd", "iterative", "momentum", "rmsprop",
                "adagrad", "adam", "adamax", "adadelta"]
        values = [self.sgd, self.iterative, self.momentum, self.rmsprop,
                  self.adagrad, self.adam, self.adamax, self.adadelta]
        self.opt_dict = {keys[i]: values[i] for i in range(len(keys))}
        if name != None and name in keys:
            self.opt_dict[name](layers=layers, training=False)

    def adam(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0, training=True):
        for l in layers:
            if l.parameters != 0:
                if training:
                    l.t += 1
                    if l.t == 1:
                        l.pdelta_biases = 0
                        l.pdelta_weights = 0
                    l.weights_adam1 = beta1 * l.weights_adam1 + \
                        (1-beta1)*l.delta_weights
                    l.weights_adam2 = beta2 * l.weights_adam2 + \
                        (1-beta2)*(l.delta_weights**2)
                    mcap = l.weights_adam1/(1-beta1**l.t)
                    vcap = l.weights_adam2/(1-beta2**l.t)
                    l.delta_weights = mcap/(np.sqrt(vcap) + epsilon)
                    l.weights += l.pdelta_weights * self.mr + learning_rate * l.delta_weights
                    l.pdelta_weights = l.delta_weights * 0
                    l.biases_adam1 = beta1 * l.biases_adam1 + \
                        (1-beta1)*l.delta_biases
                    l.biases_adam2 = beta2 * l.biases_adam2 + \
                        (1-beta2)*(l.delta_biases**2)
                    mcap = l.biases_adam1/(1-beta1**l.t)
                    vcap = l.biases_adam2/(1-beta2**l.t)
                    l.delta_biases = mcap/(np.sqrt(vcap) + epsilon)
                    l.biases += l.pdelta_biases * self.mr + learning_rate * l.delta_biases
                    l.pdelta_biases = l.delta_biases * 0
                else:
                    l.t = 0
                    l.weights_adam1 = 0
                    l.weights_adam2 = 0
                    l.biases_adam1 = 0
                    l.biases_adam2 = 0
