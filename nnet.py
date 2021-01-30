from functools import partial
from typing import Dict, List, Union

import numpy as _np

# not as np as we don't want the code to use numpy directly
# because it may be getting undesired float64 unintentionally

import dadw

_np.random.seed(1)

# TODO: The DTYPE should be a parameter of NeuralNetwork
# Use same DTYPE for everything
DTYPE = _np.dtype("float64")
Array = Union[_np.ndarray]
AXIS = _np.newaxis
array = partial(_np.array, dtype=DTYPE)
zeros = partial(_np.zeros, dtype=DTYPE)
zeros_like = partial(_np.zeros_like, dtype=DTYPE)
concatenate = _np.concatenate
cumsum = _np.cumsum
sqrt = partial(_np.sqrt, dtype=DTYPE)
exp = partial(_np.exp, dtype=DTYPE)
randn = lambda *a, **kw: _np.random.standard_normal(*a, **kw).astype(DTYPE)

# TODO: Try more activation functions as ReLU and others
# TODO: Move activation function inside class
g = lambda x: 1 / (1 + exp(-x))  # Activation function
gprime = lambda h: g(h) * (1 - g(h))  # g'(in^l_q)


class NeuralNetwork:
    # Note: under current layer k: read comments as if n, m = dlayers[k], dlayers[k + 1]
    dlayers: List[int]  # Layer description
    batch_size: int  # Amount of samples to process together
    weights: List  # List of (n + 1, m) arrays (+1 for bias). The synaptic efficacy
    activations: List  # List of (batch_size, n + 1) arrays. The value of a neuron

    gradients: List  # List of (batch_size, n + 1, m). Error gradients for each weight
    fanin: List  # List of (batch_size, n) arrays. The linear input for a neuron

    _dadw_cache: Dict
    _DADW_cache: Dict
    _previous_gradients: List

    def __init__(
        self, dlayers: List[int], batch_size: int = 0, params: List[int] = None
    ):
        """
        dlayers: description of the network layers,
        params: Parameters to create_layers, if None, then randoms will be used
        """
        self.dlayers = dlayers
        self.batch_size = batch_size

        # Set neurons
        bshape = lambda tup: (batch_size,) + tup if batch_size > 0 else tup
        self.fanin = [zeros(bshape((n,))) for n in [0] + dlayers[1:]]
        self.activations = [zeros(bshape((n + 1,))) for n in dlayers]
        for k in range(len(dlayers)):
            self.activations[k][..., -1] = 1  # Bias neurons are 1

        # Set weights
        params = params if params is not None else self.get_random_params()
        self.weights = self.weights_from_params(params)

        # TODO: Right now self.gradients could be removed as it is not really in use
        # however it may be useful to repurpose it to be the only malloc made in
        # the entire run of the network for improved performance
        # do this when sure that the mallocs are a real bottleneck
        self.gradients = [
            zeros(bshape((n + 1, m))) for n, m in zip(dlayers, dlayers[1:])
        ]

        self._dadw_cache = {}
        self._DADW_cache = {}
        self._previous_gradients = [zeros_like(wm) for wm in self.weights]

    # TODO: Understand and try other initializations as xavier and kaiming
    # see https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
    def get_random_params(self):
        # CUSTOM initialization heuristic
        # i / n to scale the sum result based on number of input weights
        # seems to make outputs stable
        # TODO: Should biases be initialized differently?
        return array(
            [
                i / sqrt(n + 1)
                for n, m in zip(self.dlayers, self.dlayers[1:])
                for i in randn((n + 1) * m)
            ]
        )

    @property
    def params(self):
        """Return parameter list that can be used to recreate this network."""
        return array([w for m in self.weights for r in m for w in r])

    @property
    def is_batched(self) -> bool:
        return self.batch_size != 0

    def weights_from_params(self, params) -> List[Array]:
        """Map the flat params iterable to a proper weight matrix.

        params: Flat list of weights w^k_{i, j} from neuron i in layer k to j in k + 1
        to understand the param format see NeuralNetwork.params property
        """
        l = len(self.dlayers) - 1  # amount of weight matrices
        weights = [None] * l
        wsizes = [(n + 1) * m for n, m in zip(self.dlayers, self.dlayers[1:])]
        offset_layer = concatenate([[0], cumsum(wsizes)])
        for k in range(l):
            n = self.dlayers[k] + 1  # Neurons in current layer
            m = self.dlayers[k + 1]  # Neurons in next layer
            weights[k] = params[offset_layer[k] : offset_layer[k + 1]].reshape((n, m))
        return weights

    def feedforward(self, ilayer) -> Array:
        """Forward propagation of the network, fill activation vectors.

        ilayer: Normalized input layer scaled in range [0, 1]
        returns: last layer activations (the guess)
        """
        self._dadw_cache = {}  # Previous cache invalid for this feedforward
        self._DADW_cache = {}

        self.activations[0][..., :-1] = ilayer
        for k in range(1, len(self.dlayers)):
            # TODO: Numpy matmul @ should be used here, however see cython matmul docstring
            # self.fanin[k] = self.activations[k - 1] @ self.weights[k - 1]
            dadw.matmul(self.activations[k - 1], self.weights[k - 1], self.fanin[k])
            self.activations[k][..., :-1] = g(self.fanin[k])
        return self.activations[-1][..., :-1].copy()  # Remove bias neuron from result

    def get_error(self, tgt: Array):  # (batch_size, #L) -> (#L,)
        """Return mean squared error, target tgt has an output-like structure."""
        axis = 1 if self.is_batched else 0
        return ((self.activations[-1][..., :-1] - tgt) ** 2).mean(axis)

    def py_dadw(self, l, q, k, i, j, b=0):  # -> DTYPE
        """Return derivative of a^l_q with respect to w^k_ij for batch sample b."""
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
            res = self.activations[k][b, i]
        elif k == l - 2:  # Special case for performance, not needed for correctness
            res = (
                self.weights[l - 1][j, q]
                * gprime(self.fanin[l - 1][b, j])
                * self.activations[k][b, i]
            )
        elif k < l - 1:
            res = sum(
                self.weights[l - 1][r, q] * self.py_dadw(l - 1, r, k, i, j)
                for r in range(self.dlayers[l - 1])
            )
        else:
            raise Exception("Should never reach this execution branch")

        # Multiply by derivative of activation function over the neuron's weighted sum
        res *= gprime(self.fanin[l][b, q])

        # Cache it
        self._dadw_cache[args] = res
        return res

    def py_get_gradients(self, target, b=0) -> List[Array]:
        """Matrix of each error gradient ∇E^k_{i, j} using dadw() for batch sample b."""
        L = len(self.dlayers) - 1  # Last layer index
        mseconst = 2 / self.dlayers[-1]
        gradients = [zeros_like(g) for g in self.gradients]
        for k in reversed(range(L)):
            for b in range(self.batch_size):
                n, m = self.dlayers[k], self.dlayers[k + 1]
                for j in range(m):
                    for i in range(n + 1):  # +1 for bias neuron
                        gradients[k][b, i, j] = mseconst * sum(
                            (self.activations[L][b, q] - target[b, q])
                            * self.py_dadw(L, q, k, i, j, b=b)
                            for q in range(self.dlayers[-1])
                        )
        return gradients

    def cy_DADW(self, l, q, k):
        return dadw.DADW(self, l, q, k)

    def cy_DADW_prepopulate(self):
        return dadw.DADW_prepopulate(self)

    def cy_get_gradients(self, target: Array) -> List[Array]:
        return dadw.get_gradients(self, target)

    def np_DADW(self, l, q, k):
        """Read only matrix A^{l, q}_k of each derivative of dadw(i, j)."""
        args = (l, q, k)
        if args in self._DADW_cache:
            return self._DADW_cache[args]

        res = zeros_like(self.gradients[k])  # (batch_size, n + 1, m)
        if l == k + 1:
            derivatives = gprime(self.fanin[l][..., q, AXIS])
            columns = self.activations[k][:]
            res[..., :, q] = derivatives * columns
        elif l > k + 1:
            for r in range(self.dlayers[l - 1]):
                res += self.weights[l - 1][r, q] * self.np_DADW(l - 1, r, k)
            derivatives = gprime(self.fanin[l][..., q, AXIS, AXIS])
            res[...] *= derivatives
            # for r in range(self.dlayers[l - 1]):
            #     mul = self.weights[l - 1][r, q] * self.DADW(l - 1, r, k)
            #     np.add(res, mul, out=res)
            # derivatives = gprime(self.fanin[l][..., q, AXIS, AXIS])
            # np.multiply(res, derivatives, out=res)
        else:
            raise Exception("This execution branch should not be reached.")

        res.setflags(write=False)  # As the result is cached, we make it readonly
        self._DADW_cache[args] = res
        return res

    def np_get_gradients(self, target: Array) -> List[Array]:
        """Matrix of each error gradient ∇E^k_{i, j} using DADW() matrices."""

        # Prefill cache, improves performance, unnecessary for correctness
        cache = self.cy_DADW_prepopulate()
        for q in range(16):
            self._DADW_cache[(2, q, 0)] = cache[q]

        L = len(self.dlayers) - 1  # Last layer index
        mseconst = 2 / self.dlayers[L]
        gradients: List = [None for _ in self.weights]
        for k in reversed(range(L)):
            summation = zeros_like(self.gradients[k])  # (batch_size, n + 1, m)
            for q in range(self.dlayers[L]):
                tgtdiff = self.activations[L][..., q] - target[..., q]
                tgtdiff = tgtdiff[..., AXIS, AXIS]
                ALqk = self.np_DADW(L, q, k)
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

    def batch_eval(self, batch, grads=True, hitrate=False):
        """Return mean losses, mean gradients (if grads) and hitrate (if hitrate) of a batch.

        All batches must be not larger than batch_size.
        """
        inputs, targets = batch
        batch_size = len(inputs)
        outputs = self.feedforward(inputs)
        errors = self.get_error(targets)
        batch_loss = errors.mean()
        results = [batch_loss]
        if grads:
            gradients = self.cy_get_gradients(targets)
            batch_gradients = [gms.mean(axis=0) for gms in gradients]
            results.append(batch_gradients)
        if hitrate:
            target_digits = targets.argmax(1)
            output_digits = outputs.argmax(1)
            batch_hitrate = sum(target_digits == output_digits) / batch_size
            results.append(batch_hitrate)
        if len(results) == 1:
            results = results[0]
        return results
