"""NeuralNetwork base class file from where all the other networks derive."""

from functools import partial
from typing import Dict, List, Union

# do not import as just np as we don't want the code to use numpy directly
# because it may be getting undesired float64 unintentionally
import numpy as _np

from . import cynet_native

# TODO: The DTYPE should be a parameter of NeuralNetwork
# Use same DTYPE for everything
DTYPE = _np.dtype("float32") # TESTMARK
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
rand = lambda *a, **kw: _np.random.rand(*a, **kw).astype(DTYPE)

# TODO: Try more activation functions as ReLU and others
# TODO: Move activation function inside class
g = lambda x: 1 / (1 + exp(-x))  # Activation function
gprime = lambda h: g(h) * (1 - g(h))  # g'(in^l_q)


class NeuralNetwork:
    # NOTE: read comments as if n, m = dlayers[k], dlayers[k + 1]
    dlayers: List[int]  # Layer description
    batch_size: int  # Amount of samples to process together
    weights: List  # List of (n + 1, m) arrays (+1 for bias). The synaptic efficacy
    activations: List  # List of (batch_size, n + 1) arrays. The value of a neuron
    fanins: List  # List of (batch_size, n) arrays. The linear input for a neuron

    _dadw_cache: Dict
    _DADW_cache: Dict
    _gradients: List  # List of (batch_size, n + 1, m). Storage for error gradients for each weight
    _velocities: List  # Previous velocities for SGD momentum

    def __init__(self, dlayers: List[int], batch_size: int, params: List[int] = None):
        """
        dlayers: description of the network layers,
        params: Parameters to create_layers, if None, then randoms will be used
        """
        self.dlayers = dlayers
        self.batch_size = batch_size

        # Set neurons
        self.fanins = [zeros((batch_size, n)) for n in [0] + dlayers[1:]]
        self.activations = [zeros((batch_size, n + 1)) for n in dlayers]
        for k in range(len(dlayers)):
            self.activations[k][:, -1] = 1  # Bias neurons are 1

        # Set weights
        params = params if params is not None else self.get_random_params()
        self.weights = self.weights_from_params(params)

        self._gradients = [
            zeros((batch_size, n + 1, m)) for n, m in zip(dlayers, dlayers[1:])
        ]

        self._dadw_cache = {}
        self._DADW_cache = {}
        self._velocities = [zeros_like(wm) for wm in self.weights]

    def get_random_params_custom(self):
        "Custom initialization heuristic, works better than kaiming for MNIST autoencoder."
        return array(
            [
                i / sqrt(n + 1)
                for n, m in zip(self.dlayers, self.dlayers[1:])
                for i in randn((n + 1) * m)
            ]
        )

    def get_random_params_kaiming_uniform(self):
        return array(
            [
                (i - 0.5) * sqrt(3 / (n + 1))
                for n, m in zip(self.dlayers, self.dlayers[1:])
                for i in rand((n + 1) * m)
            ]
        )

    def get_random_params(self):
        return self.get_random_params_kaiming_uniform()

    @property
    def params(self):
        """Return parameter list that can be used to recreate this network."""
        return array([w for m in self.weights for r in m for w in r])

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
        # TODO: invalidate cache in a better place, these one is a bit hidden
        self._dadw_cache = {}  # Previous cache invalid for this feedforward
        self._DADW_cache = {}

        self.activations[0][:, :-1] = ilayer
        for k in range(1, len(self.dlayers)):
            # TODO: Numpy matmul @ should be used here, however see cython matmul docstring
            # self.fanin[k] = self.activations[k - 1] @ self.weights[k - 1]
            # TODO: Use cython feedforward only in CyNet, make feedforward for NpNet and PyNet
            cynet_native.matmul(self.activations[k - 1], self.weights[k - 1], self.fanins[k])
            self.activations[k][:, :-1] = g(self.fanins[k])
        return self.activations[-1][:, :-1].copy()  # Remove bias neuron from result

    def get_error(self, tgt: Array):  # (batch_size, #L) -> (#L,)
        """Return mean squared error, target tgt has an output-like structure."""
        return ((self.activations[-1][:, :-1] - tgt) ** 2).mean(axis=1)

    def get_gradients(self, target: Array) -> List[Array]:
        """Matrix of each error gradient ∇E^k_{i, j} using DADW() matrices."""
        raise NotImplementedError

    def update_weights(self, gradients, lr=10, momentum=0.5):
        """Update weights using stochastic gradient decent with momentum.

        Reference: http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
        """
        for wm, gm, vm in zip(self.weights, gradients, self._velocities):
            vm[:, :] = -lr * gm + momentum * vm
            wm[:, :] += vm

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
            gradients = self.get_gradients(targets)
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
