from typing import Dict, List

import numpy as np
from numpy.random import randn

np.random.seed(1)


class NeuralNetwork:
    # n and m will refer to the size of current and next layer in comments
    dlayers: List[int]  # Layer description
    weight: List  # List of (n + 1) * m matrices (+1 for bias)
    activation: List  # List of n length vectors

    _dadw_cache: Dict

    def __init__(self, dlayers: List[int], params: List[int] = None):
        """
        dlayers: description of the network layers,
        params: Parameters to create_layers, if None, then randoms will be used
        """
        self.dlayers = dlayers
        self.activation = np.array(
            [np.concatenate((np.zeros(n), [1])) for n in dlayers]
        )
        # Set weights
        params = params if params is not None else self.get_random_params()
        self.create_layers(params)

        self._dadw_cache = {}

    # TODO: Understand and try other initializations as xavier and kaiming
    # see https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
    def get_random_params(self):
        # CUSTOM initialization heuristic
        # i / n to scale the sum result based on number of input weights
        # seems to make outputs stable
        # TODO: Should biases be initialized differently?
        return np.array(
            [
                i / np.sqrt(n + 1)
                for n, m in zip(self.dlayers, self.dlayers[1:])
                for i in randn((n + 1) * m)
            ]
        )

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
        l = len(self.dlayers)
        self.weight = [None] * (l - 1)
        wsizes = [(n + 1) * m for n, m in zip(self.dlayers, self.dlayers[1:])]
        offset_layer = np.concatenate([[0], np.cumsum(wsizes)])
        for k in range(l - 1):
            n = self.dlayers[k] + 1  # Neurons in current layer
            m = self.dlayers[k + 1]  # Neurons in next layer
            self.weight[k] = params[offset_layer[k] : offset_layer[k + 1]].reshape(
                (n, m)
            )

    def feedforward(self, ilayer):
        """
        Forward propagation of the network, fill activation vectors
        ilayer: Normalized input layer scaled in range [0, 1]
        returns: last layer activations (the guess)
        """
        # TODO: Try more activation functions as ReLU and others
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.activation[0] = np.concatenate((ilayer, [1]))
        for layer in range(1, len(self.dlayers)):
            self.activation[layer] = np.concatenate(
                (
                    sigmoid(self.activation[layer - 1] @ self.weight[layer - 1]),
                    [1],
                )
            )
        return self.activation[-1][:-1]  # Remove bias neuron from result

    @property
    def params(self):
        """Return parameter list that can be used to recreate this network."""
        return np.array([w for m in self.weight for r in m for w in r])

    def get_error(self, expected: List[float]):
        """Return mean squared error, expected has an output-like structure."""
        return sum((g - e) ** 2 for g, e in zip(self.activation[-1], expected))

    def dadw(self, l, q, k, i, j) -> float:
        """Return derivative of a^l_q with respect to w^k_ij."""
        # Memoization stuff
        args = (l, q, k, i, j)
        if args in self._dadw_cache:
            return self._dadw_cache[args]

        # Conditional factor of the multiplication
        if l < 0 or q < 0 or k < 0 or i < 0 or j < 0:  # Out of range indexes
            raise Exception(f"Negative parameter found: l, q, k, i, j={l, q, k, i, j}")
        elif l == 0:  # No weight affects an input neuron
            res = 0
        elif k >= l:  # Weight to the right of neuron
            res = 0
            raise Exception(f"{k=} >= {l=}, it should not happen")
        elif q == self.dlayers[l]:  # is bias neuron, nothing changes its value
            res = 0
        elif k == l - 1 and j != q:  # Weight just before neuron but disconnected
            res = 0
        elif k == l - 1 and j == q:  # Weight just before neuron and connected
            res = self.activation[k][i]
        # elif k == l - 2:  # Special case for performance, not needed for correctness
        #     res = ( # TODO: Uses sigmoid derivative, generalize.
        #         self.weight[l - 1][j, q]
        #         * self.activation[l - 1][j] * (1 - self.activation[l - 1][j])
        #         * self.activation[l - 2][i]
        #     )
        elif k < l - 1:
            res = sum(
                self.weight[l - 1][r, q] * self.dadw(l - 1, r, k, i, j)
                for r in range(self.activation[l - 1].size)
            )
        else:
            raise Exception("Should never reach this execution branch")

        # Multiply by derivative of activation function
        # TODO: Uses sigmoid derivative, generalize.
        res *= self.activation[l][q] * (1 - self.activation[l][q])  # g'(in^l_q)

        # Cache it
        self._dadw_cache[args] = res
        return res

    def get_error_gradient_slow(self, expected):
        L = len(self.dlayers) - 1  # Last layer index
        gradients = np.array(
            [np.empty((n + 1, m)) for n, m in zip(self.dlayers, self.dlayers[1:])]
        )

        assert all(
            g.shape == w.shape for g, w in zip(gradients, self.weight)
        ), "gradients is not the same shape as weights"

        for k in reversed(range(len(self.dlayers) - 1)):
            n = self.dlayers[k]
            m = self.dlayers[k + 1]
            for j in range(m):
                for i in range(n + 1):  # +1 for bias neuron
                    gradients[k][i, j] = sum(
                        (self.activation[L][q] - expected[q]) * self.dadw(L, q, k, i, j)
                        for q in range(self.dlayers[-1])
                    )
        return gradients

    def DADW_slow(self, l, q, k):
        """Same as DADW but using the theoretical and slow implementation."""
        if k == l - 1:
            res = np.array(
                [
                    [self.dadw(l, q, k, i, j) for j in range(self.dlayers[k + 1])]
                    for i in range(self.dlayers[k] + 1)
                ]
            )
        elif k < l - 1:
            res = np.zeros((self.dlayers[k] + 1, self.dlayers[k + 1]))
            for r in range(self.dlayers[l - 1]):
                res += self.weight[l - 1][r, q] * self.DADW_slow(l - 1, r, k)
            alq = self.activation[l][q]
            res *= alq * (1 - alq)
        else:
            print("This execution branch should not be reached.")
        return res

    def DADW(self, l, q, k):
        """Matrix of dadw with positions ij representing dadw(l, q, k, i, j)."""

        # Cache setup
        if not hasattr(self, "_DADW_cache"):
            self._DADW_cache = {}
        args = (l, q, k)
        if args in self._DADW_cache:
            return self._DADW_cache[args]

        alq = self.activation[l][q]
        if k == l - 1:
            res = np.zeros((self.dlayers[k] + 1, self.dlayers[k + 1]))
            res[:, q] = alq * (1 - alq) * self.activation[k]
        elif k < l - 1:
            res = (
                alq
                * (1 - alq)
                * sum(
                    self.weight[l - 1][r, q] * self.DADW(l - 1, r, k)
                    for r in range(self.dlayers[l - 1])
                )
            )
        else:
            print("This execution branch should not be reached.")

        self._DADW_cache[args] = res
        return res

    def get_error_gradient(self, expected):
        L = len(self.dlayers) - 1  # Last layer index
        gradients = np.array(
            [np.empty((n + 1, m)) for n, m in zip(self.dlayers, self.dlayers[1:])]
        )

        assert all(
            g.shape == w.shape for g, w in zip(gradients, self.weight)
        ), "gradients is not the same shape as weights"

        for k in reversed(range(len(self.dlayers) - 1)):
            gradients[k] = sum(
                (self.activation[L][q] - expected[q]) * self.DADW(L, q, k)
                for q in range(self.dlayers[-1])
            )
        return gradients

    def backpropagate(self, gradients=None, expected=None):
        assert gradients is not None or expected is not None
        if gradients is None:
            # gradients = self.get_error_gradient_slow(expected)
            gradients = self.get_error_gradient(expected)
        e = 1e-1  # learning rate
        a = 0 * e  # momentum
        if hasattr(self, "oldgradients"):
            for w, g, o in zip(self.weight, gradients, self.oldgradients):
                w[...] -= e * g * (1 - a) + a * o
        else:
            for w, g in zip(self.weight, gradients):
                w[...] -= e * g

        self.oldgradients = gradients
        # del dadw.cache
