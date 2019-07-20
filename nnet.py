from typing import List
import numpy as np
from numpy.random import randn

class NeuralNetwork:
    dlayers: List[int] # Layer description
    weight: List # List of (n + 1) * m matrices (+1 for bias)
    run: bool # Has the network been run with the current weights?

    def __init__(self, dlayers: List[int], params: List[int] = None):
        """
        dlayers: description of the network layers,
        params: Parameters to create_layers, if None, then randoms will be used
        """
        self.dlayers = dlayers
        self.run = False
        params = params if params is not None else self.get_random_params()
        self.create_layers(params)

    # TODO: Understand and try other initializations as xavier and kaiming
    # see https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
    # TODO: Decide on a return type, generator or list? in either case, remove
    # the list() conversions from the code
    def get_random_params(self):
        # CUSTOM initialization heuristic
        # i / n to scale the sum result based on number of input weights
        # seems to make outputs stable
        # TODO: Should biases be initialized differently?
        return np.array([
            i / (n + 1) for n, m in zip(self.dlayers, self.dlayers[1:])
            for i in randn((n + 1) * m)
        ])

    def create_layers(self, params):
        """
        Maps the flat params iterable to a proper weight matrix

        params: Flat list of weights, will be modified
        params: [
            Weights                     | Biases
            w_L1_n1_m1, ..., w_L1_nN_mM, w_L1_nB_m1, ..., w_L1_nB_mM,
                        ...
            w_LL_n1_m1, ..., w_LL_nN,mM
        ]
        """
        self.run = False
        l = len(self.dlayers)
        self.weight = [None] * (l - 1)
        wsizes = [(n + 1) * m for n, m in zip(self.dlayers, self.dlayers[1:])]
        offset_layer = np.concatenate([[0], np.cumsum(wsizes)])
        for k in range(l - 1):
            n = self.dlayers[k] + 1 # Neurons in current layer
            m = self.dlayers[k + 1] # Neurons in next layer
            self.weight[k] = (
                params[offset_layer[k]:offset_layer[k + 1]].reshape((n, m))
            )

    def feedforward(self, ilayer):
        "Forward propagation of the network, returns last layer activations"
        # TODO: Try more activation functions as ReLU and others
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        inout = ilayer
        for layer in range(0, len(self.dlayers) - 1):
            inout = sigmoid(np.concatenate((inout, [1])) @ self.weight[layer])
        return inout

    def get_error(self, expected: List[float]):
        "Returns the mean squared error, expected has an output layer structure"
        pass

    def __call__(self, params, ilayer):
        """
        params: List of params, same format as in get_random_params
        ilayer: Normalized input layer scaled in range [0, 1]
        returns: last layer activations (the guess)
        """
        self.create_layers(params)
        return self.feedforward(ilayer)
