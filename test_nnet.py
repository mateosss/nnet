import unittest
import numpy as np
from nnet import NeuralNetwork

GM = 10 # Gaussian bell curve maximum

class TestNeuralNetwork(unittest.TestCase):
    "NeuralNetwork unit tests"

    def test_get_random_params(self):
        dlayers = [1024, 32, 64, 47]
        nnet = NeuralNetwork()
        nnet.set_layers_description(dlayers)
        params = nnet.get_random_params()

        weights = [n *  m for n, m in zip(dlayers, dlayers[1:])]
        biases = dlayers[1:]
        self.assertEqual(len(params), sum(weights) + sum(biases))
        self.assertTrue(all(-GM <= p and p <= GM for p in params))

    def test_create_layers(self):
        dlayers = [1024, 32, 64, 47]
        nnet = NeuralNetwork()
        nnet.set_layers_description(dlayers)
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
            -GM <= p and p <= GM for w in nnet.weight for p in np.nditer(w)
        ))

    def test_nnet(self):
        pass


if __name__ == '__main__':
    unittest.main()
