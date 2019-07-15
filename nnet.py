from typing import List
import itertools as it
import numpy as np
from numpy.random import rand, randn

# TODO: Try more activation functions as ReLU and others
sigmoid = lambda x: 1 / (1 + np.exp(-x))

class NeuralNetwork:
    """
    Let n be "amount of neurons in current layer"
    Let m be "amount of neurons in next layer"
    """

    dlayers: List[int] # Layer description
    weight: List # List of n*m matrices
    bias: List # List of 1*n arrays

    def set_layers_description(self, dlayers):
        self.dlayers = dlayers

    # TODO: Understand and try other initializations as xavier and kaiming
    # see https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
    def get_random_params(self):
        return it.chain(
            # CUSTOM initialization heuristic
            # i / n to scale the sum result based on number of input weights
            # seems to work decently
            (
                i / (n + 1) for n, m in zip(self.dlayers, self.dlayers[1:])
                for i in randn(n * m)
            ), # Weights
            (i for n in self.dlayers[1:] for i in randn(n)), # Biases
        )

    def create_layers(self, params):
        """
        Maps the flat params iterator to proper weights and biases matrices

        params: Flat list of weights ++ biases:
            [w[li][mj][nk] for i layers in j neurons in k connections] ++
            [b[li][mj] for i layers in j neurons]
        """
        # TODO: don't make this kind of conversions explicitly
        # use the params generator directly for performance
        params = list(params)

        wsizes = [n * m for n, m in zip(self.dlayers, self.dlayers[1:])]
        bsizes = self.dlayers.copy()[1:]
        assert len(params) == sum(wsizes) + sum(bsizes), (
            "len(params) doesn't match layers description"
        )

        # Fill weights

        """
        l: number of layers
        n: number of previous layer connections
        m: number of neurons in current layer

        i: layer index
        j: neuron index
        k: connection index

        weights[i:1..l][k:0..n][j:0..m] = params[offset_layer_i + n * j + k]
        """

        # Offset of layer based on previous layer weights matrix size
        # First two layers start at 0 (input layer, and first layer)
        offset_layer = np.concatenate(([0, 0], np.cumsum(wsizes[:-1])))
        l = len(self.dlayers)
        self.weight = [np.empty(0) for i in range(l)]
        for i in range(1, l): # Weights Matrix
            n = self.dlayers[i - 1]
            m = self.dlayers[i]
            self.weight[i] = np.empty((n, m))
            for j in range(m): # Column
                row_start = offset_layer[i] + n * j
                self.weight[i][0:n, j] = params[row_start:row_start + n]

        # Fill biases

        # First index that is not a weight
        bias_start = offset_layer[-1] + self.weight[-1].size
        # Same idea as offset_layer but for biases
        offset_bias = np.concatenate(([0, 0], np.cumsum(bsizes[:-1])))
        self.bias = [np.empty(0) for i in range(l)]
        for i, n in enumerate(self.dlayers[1:], 1):
            self.bias[i] = np.empty(n)
            start = bias_start + offset_bias[i]
            self.bias[i][0:n] = params[start:start + n]

    def __call__(self, params, ilayer):
        """
        params: List of params, same format as in get_random_params
        ilayer: Normalized input layer scaled in range [0, 1]
        """
        self.create_layers(params)
        inout = ilayer
        for layer in range(1, len(self.dlayers)):
            inout = sigmoid(inout @ self.weight[layer] - self.bias[layer])
            print(f"inout={inout}")
        return inout


nnet = NeuralNetwork()
