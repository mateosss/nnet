"""NeuralNetwork unit tests"""

from dataclasses import dataclass

import numpy as np
from numpy.linalg import norm

from mnist import mnist
from nets.nnet import NeuralNetwork
from nets.pynet import PyNet
from nets.npnet import NpNet
from nets.cynet import CyNet

np.random.seed(1)  # TODO: Remove?
WTOL = 10  # weights must be within [-WTOL, +WTOL]
COMPLETENESS = 0.05  # ratio of loops that will be effectively tested
BATCH_SIZE = 1000 # TESTMARK
DTYPE = np.dtype("float32") # TESTMARK

# TODO: Needed for now, but it should be a cleaner way to set BATCH_SIZE and DTYPE globally
assert BATCH_SIZE == 1 and DTYPE == np.dtype(
    "float64"
), (
    "Modify all .py[x] files to use BATCH_SIZE = 1 and float64 when testing"
    "Search globally for TESTMARK to find where to apply the changes."
)

EPOCHS = mnist.load(batch_size=BATCH_SIZE)  # Epoch batches generator
BATCHES = next(EPOCHS)  # Epoch mini batches
BATCH = BATCHES[0]
# Single sample batch used for testing
INPUT = BATCH[0][0][np.newaxis, :]
TARGET = BATCH[1][0][np.newaxis, :]


@dataclass
class Loop:
    """Helper utility class to count test loops. Logs and skips loops."""

    def __init__(self, log_text, log_every, total_loops, cover_ratio) -> None:
        self.log_text = log_text
        self.log_every = log_every
        self.total_loops = total_loops
        self.cover_ratio = cover_ratio
        self._count = 0
        self._cover_progress = 0

    def __call__(self) -> bool:
        "Return if the loop should be executed based on specified cover_ratio"
        self._count += 1
        self._cover_progress += self.cover_ratio
        if self._count % self.log_every == 0:
            print(
                f"{self.log_text.format(self._count, self.total_loops)}{' ' * 16}\r",
                end="",
            )
        if self._count == self.total_loops:
            print(f"{' ' * 79}\r", end="")
        if self._cover_progress >= 1:
            self._cover_progress %= 1
            return True
        return False


def test_get_random_params():
    """It generates reasonable params."""
    dlayers = [1024, 32, 64, 47]
    nnet = NeuralNetwork(dlayers, batch_size=1)
    params = nnet.get_random_params()

    weights = [n * m for n, m in zip(dlayers, dlayers[1:])]
    biases = dlayers[1:]
    assert len(params) == sum(weights) + sum(biases)
    assert all(-WTOL <= p <= WTOL for p in params)


def test_weights_from_params():
    """It reshapes the flat list of params properly."""
    dlayers = [1024, 32, 64, 47]
    nnet = NeuralNetwork(dlayers, batch_size=1)
    params = nnet.get_random_params()
    weights = nnet.weights_from_params(params)

    wsizes = [(n + 1) * m for n, m in zip(dlayers, dlayers[1:])]

    assert len(weights) == len(dlayers) - 1
    assert all(w.size == wsize for w, wsize in zip(weights, wsizes)), "Weights shape"
    assert all(
        w.shape == (dlayers[i - 1] + 1, dlayers[i]) for i, w in enumerate(weights, 1)
    ), "Weights shape"
    assert all(-WTOL <= p <= WTOL for w in weights for p in np.nditer(w)), "Tolerance"
    assert all(a[0, -1] == 1 for a in nnet.activations), "Bias neuron is not 1"


def test_dadw(completeness=COMPLETENESS):
    """It is similar to a numerical gradient.

    It predicts a change in output neurons close enough to what a slight numerical
    nudge to each weight produces to the network outputs.

    Based on https://cs231n.github.io/neural-networks-3/#gradcheck
    """
    dlayers = [784, 16, 16, 10]
    L = len(dlayers) - 1  # Last layer index
    net = PyNet(dlayers, batch_size=BATCH_SIZE)

    # maximum relative difference between numerical and analytical dadw
    maxreldiff = 0
    epsilon = 1e-5

    iterations = 785 * 16 + 17 * 16 + 17 * 10  # hardcoded for specific dlayers
    loop = Loop("[dadw] {}/{}", 1000, iterations, completeness)
    outliers = 0
    for k, wmatrix in enumerate(net.weights):
        for (i, j), w in np.ndenumerate(wmatrix):
            if not loop():
                continue

            wmatrix[i, j] = w - epsilon
            a_out = net.feedforward(INPUT)
            wmatrix[i, j] = w + epsilon
            b_out = net.feedforward(INPUT)
            ndadw = (b_out - a_out) / (2 * epsilon)
            wmatrix[i, j] = w
            net.feedforward(INPUT)  # Refreshes net fanin and activation
            adadw = np.array([net.dadw(L, q, k, i, j) for q in range(dlayers[-1])])
            adadw = adadw[np.newaxis, :]

            normalizer = max(norm(adadw), norm(ndadw))
            if normalizer == 0:
                continue
            reldiff = norm(adadw - ndadw) / normalizer
            if reldiff > maxreldiff:
                maxreldiff = reldiff
                if maxreldiff > 1e-7:
                    outliers += 1
            assert outliers < 10, f"too many bad apples: {outliers}"
            assert maxreldiff <= 1e-6, f"{maxreldiff=} is too high"
    print(
        f"[dadw] maxreldiff={maxreldiff} between numeric and analytical dadw with {outliers} outliers >1e7"
    )


def test_get_gradients():
    """Check for equality and functioning of various gradient methods.

    Check if all the different gradient methods return the same matrix.
    Make one big update with the error gradient and assert it is almost perfect.
    """
    dlayers = [784, 16, 16, 10]
    pynet = PyNet(dlayers, batch_size=BATCH_SIZE)
    npnet = NpNet(dlayers, batch_size=BATCH_SIZE, params=pynet.params)
    cynet = CyNet(dlayers, batch_size=BATCH_SIZE, params=pynet.params)

    py_old_out = pynet.feedforward(INPUT)[0]
    np_old_out = npnet.feedforward(INPUT)[0]
    cy_old_out = cynet.feedforward(INPUT)[0]

    assert (py_old_out == np_old_out).all() and (np_old_out == cy_old_out).all()

    py_old_error = pynet.get_error(TARGET)[0]
    np_old_error = npnet.get_error(TARGET)[0]
    cy_old_error = cynet.get_error(TARGET)[0]

    assert py_old_error == np_old_error == cy_old_error

    py_grads = pynet.get_gradients(TARGET)
    np_grads = npnet.get_gradients(TARGET)
    cy_grads = cynet.get_gradients(TARGET)

    maxdiff = 0
    for k in range(len(dlayers) - 1):
        n, m = dlayers[k], dlayers[k + 1]
        for b in range(BATCH_SIZE):
            for i in range(n + 1):
                for j in range(m):
                    pyg = py_grads[k][b, i, j]
                    npg = np_grads[k][b, i, j]
                    cyg = cy_grads[k][b, i, j]
                    diff = max(abs(pyg - npg), abs(pyg - cyg), abs(npg - cyg))
                    if diff > maxdiff:
                        maxdiff = diff

    assert all(
        np.allclose(py_gm, np_gm, rtol=0, atol=1e-17)
        for py_gm, np_gm in zip(py_grads, np_grads)
    )
    assert all(
        np.allclose(py_gm, cy_gm, rtol=0, atol=1e-17)
        for py_gm, cy_gm in zip(py_grads, cy_grads)
    )

    for wm, gm in zip(pynet.weights, py_grads):
        wm[...] -= gm[0] * 1000

    new_out = pynet.feedforward(INPUT)[0]
    new_error = pynet.get_error(TARGET)[0]

    print(
        "[get_gradients] [step] "
        f"{py_old_error=:.6f}, "
        f"{new_error=:.6f}, "
        f"maxdiff between grads is {maxdiff}"
    )
    assert np.isclose(new_error, 0)
    assert np.allclose(new_out, TARGET[0])
