from typing import Dict, List

import numpy as np
from numpy.random import randn

np.random.seed(1)

# TODO: Try more activation functions as ReLU and others
# TODO: Move activation function inside class
g = lambda x: 1 / (1 + np.exp(-x))  # Activation function
gprime = lambda h: g(h) * (1 - g(h))  # g'(in^l_q)


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
        self.fanin = [np.zeros(n) for n in [0] + dlayers[1:]]  # weighted sum for neuron
        self.activation = [np.concatenate((np.zeros(n), [1])) for n in dlayers]

        # Set weights
        params = params if params is not None else self.get_random_params()
        self.weight = self.weights_from_params(params)

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

    @property
    def params(self):
        """Return parameter list that can be used to recreate this network."""
        return np.array([w for m in self.weight for r in m for w in r])

    def weights_from_params(self, params) -> List[np.array]:
        """Map the flat params iterable to a proper weight matrix.

        params: Flat list of weights w^k_{i, j} from neuron i in layer k to j in k + 1
        to understand the param format see NeuralNetwork.params property
        """
        l = len(self.dlayers) - 1  # amount of weight matrices
        weights = [None] * l
        wsizes = [(n + 1) * m for n, m in zip(self.dlayers, self.dlayers[1:])]
        offset_layer = np.concatenate([[0], np.cumsum(wsizes)])
        for k in range(l):
            n = self.dlayers[k] + 1  # Neurons in current layer
            m = self.dlayers[k + 1]  # Neurons in next layer
            weights[k] = params[offset_layer[k] : offset_layer[k + 1]].reshape((n, m))
        return weights

    def feedforward(self, ilayer):
        """Forward propagation of the network, fill activation vectors.

        ilayer: Normalized input layer scaled in range [0, 1]
        returns: last layer activations (the guess)
        """
        self._dadw_cache = {}  # Previous cache invalid for this feedforward

        self.activation[0][:-1] = ilayer
        for k in range(1, len(self.dlayers)):
            self.fanin[k] = self.activation[k - 1] @ self.weight[k - 1]
            self.activation[k][:-1] = g(self.fanin[k])
        return self.activation[-1][:-1].copy()  # Remove bias neuron from result

    def get_error(self, tgt: List[float]):
        """Return mean squared error, target tgt has an output-like structure."""
        return sum((o - e) ** 2 for o, e in zip(self.activation[-1], tgt)) / len(tgt)

    def dadw(self, l, q, k, i, j) -> float:
        """Return derivative of a^l_q with respect to w^k_ij."""
        # Memoization stuff
        args = (l, q, k, i, j)
        if args in self._dadw_cache:
            return self._dadw_cache[args]

        # Range assertions
        assert l >= 0 and l < len(self.dlayers), f"out of range {l=}"
        assert k >= 0 and k < len(self.dlayers), f"out of range {k=}"
        assert i >= 0 and i < self.activation[k].size, f"out of range {i=}"
        assert j >= 0 and j < self.dlayers[k], f"out of range {j=}"

        # Usage assertions
        # while dadw is theoretically defined as 0 for these, we don't want them to run
        assert k < l, f"requesting dadw with weight right to the neuron {k=} >= {l=}"
        assert q != self.dlayers[l], f"requesting dadw of bias neuron a^{l=}_{q=}"

        # Conditional factor of the multiplication
        if l == 0:  # No weight affects an input neuron
            res = 0
        elif k == l - 1 and j != q:  # Weight just before neuron but disconnected
            res = 0
        elif k == l - 1 and j == q:  # Weight just before neuron and connected
            res = self.activation[k][i]
        elif k == l - 2:  # Special case for performance, not needed for correctness
            res = (
                self.weight[l - 1][j, q]
                * gprime(self.fanin[l - 1][j])
                * self.activation[k][i]
            )
        elif k < l - 1:
            res = sum(
                self.weight[l - 1][r, q] * self.dadw(l - 1, r, k, i, j)
                for r in range(self.dlayers[l - 1])
            )
        else:
            raise Exception("Should never reach this execution branch")

        # Multiply by derivative of activation function over the neuron's weighted sum
        res *= gprime(self.fanin[l][q])

        # Cache it
        self._dadw_cache[args] = res
        return res

    def get_gradients_slow(self, target) -> List[np.array]:
        """Matrix of each error gradient ∇E^k_{i, j} using dadw()."""
        L = len(self.dlayers) - 1  # Last layer index
        mseconst = 2 / self.dlayers[-1]
        gradients = [np.zeros_like(wm) for wm in self.weight]

        for k in reversed(range(L)):
            n, m = self.dlayers[k], self.dlayers[k + 1]
            for j in range(m):
                for i in range(n + 1):  # +1 for bias neuron
                    gradients[k][i, j] = mseconst * sum(
                        (self.activation[L][q] - target[q]) * self.dadw(L, q, k, i, j)
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
            # TODO: Careful here, I'm returning a reference and not a copy of this matrix
            return self._DADW_cache[args]

        alq = self.activation[l][q]
        if k == l - 1:
            res = np.zeros((self.dlayers[k] + 1, self.dlayers[k + 1]))
            # TODO: sigmoid specific, generalize.
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

    def get_error_gradient(self, target):
        L = len(self.dlayers) - 1  # Last layer index
        gradients = [
            np.empty((n + 1, m)) for n, m in zip(self.dlayers, self.dlayers[1:])
        ]

        assert all(
            g.shape == w.shape for g, w in zip(gradients, self.weight)
        ), "gradients is not the same shape as weights"

        for k in reversed(range(len(self.dlayers) - 1)):
            gradients[k] = sum(
                (self.activation[L][q] - target[q]) * self.DADW(L, q, k)
                for q in range(self.dlayers[-1])
            )
        return gradients

    def backpropagate(self, gradients=None, target=None):
        assert gradients is not None or target is not None
        if gradients is None:
            # gradients = self.get_error_gradient_slow(target)
            gradients = self.get_error_gradient(target)
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
