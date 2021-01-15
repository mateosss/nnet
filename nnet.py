from typing import Dict, List, Optional, Tuple

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
    weights: List  # List of (n + 1, m) arrays (+1 for bias). The synaptic efficacy
    activations: List  # List of (BATCH_SIZE, n + 1) arrays. The value of a neuron

    gradients: List  # List of (BATCH_SIZE, n + 1, m). Error gradients for each weight
    fanin: List  # List of (BATCH_SIZE, n) arrays. The linear input for a neuron

    _dadw_cache: Dict
    _DADW_cache: Dict
    _previous_gradients: List

    def __init__(self, dlayers: List[int], params: List[int] = None):
        """
        dlayers: description of the network layers,
        params: Parameters to create_layers, if None, then randoms will be used
        """
        self.dlayers = dlayers
        self.fanin = [np.zeros((1000, n)) for n in [0] + dlayers[1:]]
        self.activations = [np.zeros((1000, n + 1)) for n in dlayers]
        for k in range(len(dlayers)):
            self.activations[k][:, -1] = 1  # Bias neurons are 1

        # Set weights
        params = params if params is not None else self.get_random_params()
        self.weights = self.weights_from_params(params)

        self.gradients = [
            np.zeros_like((1000, n + 1, m)) for n, m in zip(dlayers, dlayers[1:])
        ]

        self._dadw_cache = {}
        self._DADW_cache = {}
        self._previous_gradients = [np.zeros_like(wm) for wm in self.weights]

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
        return np.array([w for m in self.weights for r in m for w in r])

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

    def feedforward(self, ilayer) -> np.array:
        """Forward propagation of the network, fill activation vectors.

        ilayer: Normalized input layer scaled in range [0, 1]
        returns: last layer activations (the guess)
        """
        self._dadw_cache = {}  # Previous cache invalid for this feedforward
        self._DADW_cache = {}

        self.activations[0][:, :-1] = ilayer
        for k in range(1, len(self.dlayers)):
            self.fanin[k] = self.activations[k - 1] @ self.weights[k - 1]
            self.activations[k][:, :-1] = g(self.fanin[k])
        return self.activations[-1][:, :-1].copy()  # Remove bias neuron from result

    def get_error(self, tgt: np.array):  # (BATCH_SIZE, #L) -> (#L,)
        """Return mean squared error, target tgt has an output-like structure."""
        return ((self.activations[-1][:, :-1] - tgt) ** 2).mean(axis=1)

    def dadw(self, l, q, k, i, j) -> float:
        """Return derivative of a^l_q with respect to w^k_ij."""
        # Memoization stuff
        args = (l, q, k, i, j)
        if args in self._dadw_cache:
            return self._dadw_cache[args]

        # Range assertions
        assert l >= 0 and l < len(self.dlayers), f"out of range {l=}"
        assert k >= 0 and k < len(self.dlayers), f"out of range {k=}"
        assert i >= 0 and i < self.activations[k].size, f"out of range {i=}"
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
            res = self.activations[k][i]
        elif k == l - 2:  # Special case for performance, not needed for correctness
            res = (
                self.weights[l - 1][j, q]
                * gprime(self.fanin[l - 1][j])
                * self.activations[k][i]
            )
        elif k < l - 1:
            res = sum(
                self.weights[l - 1][r, q] * self.dadw(l - 1, r, k, i, j)
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
        gradients = [np.zeros_like(wm) for wm in self.weights]

        for k in reversed(range(L)):
            n, m = self.dlayers[k], self.dlayers[k + 1]
            for j in range(m):
                for i in range(n + 1):  # +1 for bias neuron
                    gradients[k][i, j] = mseconst * sum(
                        (self.activations[L][q] - target[q]) * self.dadw(L, q, k, i, j)
                        for q in range(self.dlayers[-1])
                    )
        return gradients

    def DADW(self, l, q, k):
        """Read only matrix A^{l, q}_k of each derivative of dadw(i, j)."""
        args = (l, q, k)
        if args in self._DADW_cache:
            return self._DADW_cache[args]

        n, m = self.dlayers[k], self.dlayers[k + 1]
        res = np.zeros((1000, n + 1, m))
        if l == k + 1:
            derivatives = gprime(self.fanin[l][:, q, np.newaxis])
            columns = self.activations[k][:]
            res[:, :, q] = derivatives * columns
        elif l > k + 1:
            for r in range(self.dlayers[l - 1]):
                res += self.weights[l - 1][r, q] * self.DADW(l - 1, r, k)
            derivatives = gprime(self.fanin[l][:, q, np.newaxis, np.newaxis])
            res[...] *= derivatives
        else:
            raise Exception("This execution branch should not be reached.")

        res.setflags(write=False)  # As the result is cached, we make it readonly
        self._DADW_cache[args] = res
        return res

    def get_gradients(self, target: np.array) -> List[np.array]:
        """Matrix of each error gradient ∇E^k_{i, j} using DADW() matrices."""
        L = len(self.dlayers) - 1  # Last layer index
        mseconst = 2 / self.dlayers[L]
        gradients: List = [None for _ in self.weights]
        for k in reversed(range(L)):
            n, m = self.dlayers[k], self.dlayers[k + 1]
            summation = np.zeros((1000, n + 1, m))
            for q in range(self.dlayers[L]):
                tgtdiff = self.activations[L][:, q] - target[:, q]
                tgtdiff = tgtdiff[:, np.newaxis, np.newaxis]
                ALqk = self.DADW(L, q, k)
                summation += tgtdiff * ALqk
            gradients[k] = mseconst * summation
        return gradients

    def update_weights(self, gradients, lr=100, momentum=0.9):
        """Update weights using stochastic gradient decent."""
        prev_grads = self._previous_gradients
        first_run = not any(gm.any() for gm in prev_grads)
        if first_run:
            momentum = 0
        for wm, gm, pgm in zip(self.weights, gradients, prev_grads):
            wm[...] -= lr * gm * (1 - momentum) + pgm * momentum
        self._previous_gradients = gradients

    # def batch_eval(self, batch, batch_size, calc_grads=True):
    #     """Return mean losses and mean gradients (if calc_grads) over a batch.

    #     All batches must be not larger than batch_size.
    #     """
    #     batch_losses = np.empty(batch_size)
    #     batch_gradients: List = [None] * batch_size
    #     j = 0
    #     for j, (inputt, target) in enumerate(zip(*batch)):
    #         self.feedforward(inputt)
    #         batch_losses[j] = self.get_error(target)
    #         if calc_grads:
    #             batch_gradients[j] = self.get_gradients(target)

    #     # Needed for batches smaller than batch_size
    #     batch_losses = batch_losses[: j + 1]
    #     batch_gradients = batch_gradients[: j + 1]

    #     batch_loss = batch_losses.mean()
    #     if calc_grads:
    #         batch_gradient = [np.mean(grads, axis=0) for grads in zip(*batch_gradients)]
    #         return batch_loss, batch_gradient
    #     return batch_loss

    def batch_eval(self, batch, batch_size, calc_grads=True):
        """Return mean losses and mean gradients (if calc_grads) over a batch.

        All batches must be not larger than batch_size.
        """
        inputs, targets = batch
        outputs = self.feedforward(inputs)
        errors = self.get_error(targets)
        batch_loss = errors.mean()
        if calc_grads:
            gradients = self.get_gradients(targets)
            batch_gradients = [gms.mean(axis=0) for gms in gradients]
            return batch_loss, batch_gradients
        return batch_loss
