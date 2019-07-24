import unittest
import numpy as np
import mnist
from nnet import NeuralNetwork

# TODO: Comment intent of each individual assertion

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

        epsilon = 1e-4
        numgrad = [np.empty(wmatrix.shape) for wmatrix in nnet.weight]

        for k, wmatrix in enumerate(nnet.weight):
            print(k)
            for i, w in np.ndenumerate(wmatrix):
                wmatrix[i] = w - epsilon
                nnet.feedforward(ninput)
                a = nnet.get_error(expected)
                wmatrix[i] = w + epsilon
                nnet.feedforward(ninput)
                b = nnet.get_error(expected)
                numgrad[k][i] = (b - a) / 2 * epsilon
                wmatrix[i] = w
        error_gradient = nnet.get_error_gradient2(expected)
        print(numgrad[-1])  # TODO: Remove print
        print(error_gradient[-1])  # TODO: Remove print
        print(numgrad / error_gradient)  # TODO: Remove print
        for k in range(len(nnet.weight)):
            print(np.linalg.norm(error_gradient[k] - numgrad[k]))
            print(np.linalg.norm(error_gradient[k] + numgrad[k]))
            print("k=", k,
                np.linalg.norm(error_gradient[k] - numgrad[k]) /
                np.linalg.norm(error_gradient[k] + numgrad[k])
            )
        # self.assertTrue(np.allclose(numgrad, error_gradient))

    def test_nnet(self):
        pass


if __name__ == '__main__':
    unittest.main()
