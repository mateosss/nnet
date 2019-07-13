from typing import List
import itertools as it
import numpy as np
from numpy.random import rand

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

    def get_random_params(self):
        return it.chain(
            (
                i for n, m in zip(self.dlayers, self.dlayers[1:])
                for i in rand(n * m)
            ), # Weights
            (i for n in self.dlayers[1:] for i in rand(n)), # Biases
        )

    def create_layers(self, params):
        """
        Maps the flat params iterator to proper weights and biases matrices

        params: Flat list of weights ++ biases:
            [w[li][mj][nk] for i layers in j neurons in k connections] ++
            [b[li][mj] for i layers in j neurons]
        """
        # TODO: never make this conversion explicit
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
                for k in range(n): # Row
                    # TODO: Use direct copy of range 0..n instead of for k if this works
                    # print(f"""
                    #     wsizes={wsizes},
                    #     offset_layer={offset_layer},
                    #     len(self.weight)={len(self.weight)},
                    #     len(self.weight[i])={len(self.weight[i])},
                    #     len(self.weight[i][k])={len(self.weight[i][k])},
                    #     len(params)={len(params)},
                    #     len(offset_layer)={len(offset_layer)},
                    #     i={i},
                    #     k={k},
                    #     j={j},
                    #     offset_layer[i] + n * j + k={offset_layer[i] + n * j + k},
                    # """)
                    self.weight[i][k][j] = params[offset_layer[i] + n * j + k]

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

    def __call__(self, params, values): # TODO
        """
        params: List of
        """
        self.create_layers(params)
        # values = np.array(values)
        # for layer in len(layers):
        #     values = values @ weight[layer] + bias[layer]
        # return values


nnet = NeuralNetwork()
