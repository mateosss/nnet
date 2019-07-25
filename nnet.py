from typing import List
import numpy as np
from numpy.random import randn

np.random.seed(1)

def star(self, l, r, k, i, j):
    Alr = self.activation[l][r]
    return Alr * (1 - Alr) * four(self, l, r, k, i, j)

# HACK: for memoizing just one instance
# TODO: put inside class and keep the memoization
def four(self, l, q, k, i, j):
    if not hasattr(four, "cache"): four.cache = {}
    args = (l, q, k, i, j)

    if args in four.cache:
        return four.cache[args]
    # TODO: It is called four because of my own note names, in which this
    # was the fourth formula, find a more descriptive name
    # print(f"(l, q, k, i, j)={(l, q, k, i, j)}")
    # IN l=0, q=0
    # W  k=0, i=10, j=0
    # dINlq/dWkji
    if l == 0:
        res = 0
    elif k >= l + 1:
        print(f"k >= l + 1 should not happen but {k} >= {l} + 1 happened")
        res = 0
    elif k == l and q != j:
        res = 0
    elif k == l and q == j:
        res = self.activation[l - 1][i]
    elif k == l - 1: # Special case for performance, same as k < l case
        res = self.activation[k][j] * self.weight[k + 1][j, q]
    elif k < l:
        res = sum(
            self.weight[l - 1][r, q]
            * self.activation[l - 1][r]
            * (1 - self.activation[l - 1][r]) * four(self, l - 1, r, k, i, j)
            for r in range(self.dlayers[l + 1]) # +1 for dlayers input
        )
        # res = sum(
        #     self.weight[l - 1][r, q] * star(self, l - 1, r, k, i, j)
        #     for r in range(self.dlayers[l + 1]) # +1 for dlayers input
        # )
    four.cache[args] = res
    return res


class NeuralNetwork:
    dlayers: List[int] # Layer description
    weight: List # List of (n + 1) * m matrices (+1 for bias)
    activation: List # List of n length vectors
    run: bool # Has the network been run with the current weights?

    def __init__(self, dlayers: List[int], params: List[int] = None):
        """
        dlayers: description of the network layers,
        params: Parameters to create_layers, if None, then randoms will be used
        """
        self.dlayers = dlayers
        self.run = False
        self.activation = np.array([
            np.concatenate((np.zeros(n), [1])) for n in dlayers[1:]
        ])
        # Set weights
        params = params if params is not None else self.get_random_params()
        self.create_layers(params)

    def __call__(self, params, ilayer):
        self.create_layers(params)
        return self.feedforward(ilayer)

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
        """
        Forward propagation of the network, fill activation vectors
        ilayer: Normalized input layer scaled in range [0, 1]
        returns: last layer activations (the guess)
        """
        # TODO: Try more activation functions as ReLU and others
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        inout = ilayer
        for layer in range(0, len(self.dlayers) - 1):
            inout = sigmoid(np.concatenate((inout, [1])) @ self.weight[layer])
            self.activation[layer] = np.concatenate((inout, [1]))
        return inout

    @property
    def params(self):
        """Return parameter list that can be used to recreate this network."""
        return np.array([w for m in self.weight for r in m for w in r])

    def get_error(self, expected: List[float]):
        """Return mean squared error, expected has an output-like structure."""
        return sum((g - e)**2 for g, e in zip(self.activation[-1], expected))

    def get_error_gradient(self, expected):
        L = len(self.dlayers) - 2 # Last layer index in weights and activations
        gradients = np.array([
            np.empty((n + 1, m)) for n, m in zip(self.dlayers, self.dlayers[1:])
        ])
        for k in reversed(range(L + 1)):
            n = self.dlayers[k]
            m = self.dlayers[k + 1]
            for j in range(m):
                #print(f"k,j,n,m={(k,j,n,m)}")
                for i in range(n + 1):  # +1 for bias neuron
                    # print(f"k,j,i,n,m={(k,j,i,n,m)}")
                    gradients[k][i, j] = sum(
                        (self.activation[L][q] - expected[q])
                        * (1 - self.activation[L][q])
                        * self.activation[L][q]
                        * four(self, L, q, k, i, j)
                        for q in range(self.dlayers[-1])
                    )
        return gradients

    def backpropagate(self, expected):
        gradients = self.get_error_gradient(expected)
        e = 0.5
        a = 0.5
        if hasattr(self, "oldgradients"):
            for w, g, o in zip(self.weight, gradients, self.oldgradients):
                w[...] -= e * g + a * o
        else:
            for w, g in zip(self.weight, gradients):
                w[...] -= e * g

        self.oldgradients = gradients
        del four.cache
        # self.weight -= gradients # TODO: for this to work self.weight should be np.array and not list
