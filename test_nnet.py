"NeuralNetwork unit tests"

from dataclasses import dataclass
from random import random

import numpy as np
from numpy.linalg import norm

import mnist
from nnet import NeuralNetwork

# TODO: Comment intent of each individual assertion

np.random.seed(1)  # TODO: Remove?
GM = 10  # Gaussian bell curve maximum
COMPLETENESS = 1  # ratio of loops that will be effectively tested


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
    dlayers = [1024, 32, 64, 47]
    nnet = NeuralNetwork(dlayers)
    params = nnet.get_random_params()

    weights = [n * m for n, m in zip(dlayers, dlayers[1:])]
    biases = dlayers[1:]
    assert len(params) == sum(weights) + sum(biases)
    assert all(-GM <= p <= GM for p in params)


def test_create_layers():
    dlayers = [1024, 32, 64, 47]
    nnet = NeuralNetwork(dlayers)
    params = nnet.get_random_params()
    nnet.create_layers(params)

    weights = [(n + 1) * m for n, m in zip(dlayers, dlayers[1:])]

    # Weights assertions
    len(nnet.weight) == len(dlayers) - 1
    assert all(w.size == weights[i] for i, w in enumerate(nnet.weight))
    assert all(
        w.shape == (dlayers[i - 1] + 1, dlayers[i])
        for i, w in enumerate(nnet.weight, 1)
    )
    assert all(-GM <= p <= GM for w in nnet.weight for p in np.nditer(w))
    assert all(a[-1] == 1 for a in nnet.activation)


def test_numerical_vs_analytical_dadw(completeness=COMPLETENESS):
    dlayers = [784, 16, 16, 10]
    L = len(dlayers) - 1  # Last layer index
    net = NeuralNetwork(dlayers)

    label, image = next(mnist.read())
    ninput = [pixel / 255 for row in image for pixel in row]

    # maximum relative difference between numerical and analytical dadw
    maxreldiff = 0
    epsilon = 1e-5

    iterations = 785 * 16 + 17 * 16 + 17 * 10  # hardcoded for specific dlayers
    loop = Loop("dadw {} out of {}", 1000, iterations, completeness)
    for k, wmatrix in enumerate(net.weight):
        for (i, j), w in np.ndenumerate(wmatrix):
            if not loop():
                continue
            wmatrix[i, j] = w - epsilon
            a_out = net.feedforward(ninput)
            wmatrix[i, j] = w + epsilon
            b_out = net.feedforward(ninput)
            ndadw = (b_out - a_out) / (2 * epsilon)
            wmatrix[i, j] = w
            net.feedforward(ninput)  # Refreshes net fanin and activation
            adadw = np.array([net.dadw(L, q, k, i, j) for q in range(dlayers[-1])])
            # greatest relative difference
            diff = max(abs(ndadw - adadw)) / (
                min(min(abs(ndadw)), min(abs(adadw))) or 1
            )
            if diff > maxreldiff:
                maxreldiff = diff
                assert maxreldiff < 0.01, f"{maxreldiff=} should be less than 1%"
    print(f"maxreldiff={maxreldiff * 100:.5f}% between numeric and analytical dadw")


def test_numerical_gradient_checking(completeness=COMPLETENESS):
    """Compare error with dadw result with slightly moving each parameter.

    Based on https://cs231n.github.io/neural-networks-3/#gradcheck
    """
    label, image = next(mnist.read())
    ninput = [pixel / 255 for row in image for pixel in row]
    expected = [1 if i == label else 0 for i in range(10)]
    nnet = NeuralNetwork([784, 16, 16, 10])

    epsilon = 1e-5
    numgrad = [np.empty(wmatrix.shape) for wmatrix in nnet.weight]

    iterations = 785 * 16 + 17 * 16 + 17 * 10  # hardcoded for specific dlayers
    loop = Loop("numgrad {} out of {}", 1000, iterations, completeness)
    for k, wmatrix in enumerate(nnet.weight):
        for i, w in np.ndenumerate(wmatrix):
            if not loop():  # Use this to make the test quicker
                continue
            wmatrix[i] = w - epsilon
            nnet.feedforward(ninput)
            a = nnet.get_error(expected)
            wmatrix[i] = w + epsilon
            nnet.feedforward(ninput)
            b = nnet.get_error(expected)
            numgrad[k][i] = (b - a) / (2 * epsilon)  # centered formula
            wmatrix[i] = w
    error_gradient = nnet.get_error_gradient(expected)

    unit = lambda v: v / norm(v) if (v != 0).any() else np.zeros(v.shape)

    for k in range(len(nnet.weight)):
        ag = unit(error_gradient[k])
        ng = unit(numgrad[k])
        print(f"cs231 = {norm(ag - ng) / max(norm(ag), norm(ng))}")
        # http://cs231n.github.io/neural-networks-3/#gradcheck
        # CS231 way seems to not work, because it compares only magnitudes
        # and not the direction of the analitical and numerical gradients
        # what happens in this case is that the ag is waaays bigger
        # than the numerical, however, the direction seems to be pointing
        # in the same way by my custom and the derived from cs231 formulas
        # but, as this formulas are out of the hat, I don't know what
        # would be a good value for them
        # TODO: Investigate for a good formula, and what result to expect
        # TODO: Put a proper assertion in this test following that formula
