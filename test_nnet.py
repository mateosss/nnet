import unittest
import numpy as np
from numpy.linalg import norm
import mnist
from nnet import NeuralNetwork
# TODO: Comment intent of each individual assertion

np.random.seed(1)  # TODO: Remove?
GM = 10 # Gaussian bell curve maximum

class TestNeuralNetwork(unittest.TestCase):
    "NeuralNetwork unit tests"

    def test_get_random_params(self):
        dlayers = [1024, 32, 64, 47]
        nnet = NeuralNetwork(dlayers)
        params = nnet.get_random_params()

        weights = [n *  m for n, m in zip(dlayers, dlayers[1:])]
        biases = dlayers[1:]
        self.assertEqual(len(params), sum(weights) + sum(biases))
        self.assertTrue(all(-GM <= p <= GM for p in params))

    def test_create_layers(self):
        dlayers = [1024, 32, 64, 47]
        nnet = NeuralNetwork(dlayers)
        params = nnet.get_random_params()
        nnet.create_layers(params)

        weights = [(n + 1) *  m for n, m in zip(dlayers, dlayers[1:])]

        # Weights assertions
        self.assertEqual(len(nnet.weight), len(dlayers) - 1)
        self.assertTrue(
            all(w.size == weights[i] for i, w in enumerate(nnet.weight))
        )
        self.assertTrue(all(
            w.shape == (dlayers[i - 1] + 1, dlayers[i])
            for i, w in enumerate(nnet.weight, 1)
        ))
        self.assertTrue(all(
            -GM <= p <= GM for w in nnet.weight for p in np.nditer(w)
        ))

        # TODO: Check that all bias activations are 1

    def test_numerical_gradient_checking(self):
        label, image = next(mnist.read())
        ninput = [pixel / 255 for row in image for pixel in row]
        expected = [1 if i == label else 0 for i in range(10)]
        nnet = NeuralNetwork([784, 16, 16, 10])

        epsilon = 1e-5
        numgrad = [np.empty(wmatrix.shape) for wmatrix in nnet.weight]

        for k, wmatrix in enumerate(nnet.weight):
            for i, w in np.ndenumerate(wmatrix):
                wmatrix[i] = w - epsilon
                nnet.feedforward(ninput)
                a = nnet.get_error(expected)
                wmatrix[i] = w + epsilon
                nnet.feedforward(ninput)
                b = nnet.get_error(expected)
                numgrad[k][i] = (b - a) / 2 * epsilon
                wmatrix[i] = w
        error_gradient = nnet.get_error_gradient(expected)

        unit = lambda v: v / norm(v) if (v != 0).any() else np.zeros(v.shape)

        for k in range(len(nnet.weight)):
            ag = error_gradient[k]
            ng = numgrad[k]
            print(f"custom = {norm(unit(ag) - unit(ng))}")
            print(f"derived from cs231 = {norm(unit(ag) * norm(ng) - ng) / max(norm(ag), norm(ng))}")
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

    def test_nnet(self):
        pass


if __name__ == '__main__':
    unittest.main()
