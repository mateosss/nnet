import unittest
import numpy as np
from nnet import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    "NeuralNetwork unit tests"

    def test_get_random_params(self):
        dlayers = [1024, 32, 64, 47]
        nnet = NeuralNetwork()
        nnet.set_layers_description(dlayers)
        params = list(nnet.get_random_params())

        weights = [n *  m for n, m in zip(dlayers, dlayers[1:])]
        biases = dlayers[1:]
        self.assertEqual(len(params), sum(weights) + sum(biases))
        self.assertTrue(all(0 <= p and p <= 1 for p in params))

    def test_create_layers(self):
        dlayers = [1024, 32, 64, 47]
        nnet = NeuralNetwork()
        nnet.set_layers_description(dlayers)
        params = nnet.get_random_params()
        nnet.create_layers(params)

        weights = [0] + [n *  m for n, m in zip(dlayers, dlayers[1:])]
        biases = [0] + dlayers[1:]

        # Weights assertions
        self.assertEqual(len(nnet.weight), len(dlayers))
        self.assertEqual(nnet.weight[0].size, 0)
        self.assertTrue(
            all(w.size == weights[i] for i, w in enumerate(nnet.weight))
        )
        self.assertTrue(all(
            w.shape == (dlayers[i - 1], dlayers[i])
            for i, w in enumerate(nnet.weight[1:], 1)
        ))
        self.assertTrue(all(
            0 <= p and p <= 1 for w in nnet.weight[1:] for p in np.nditer(w)
        ))

        # Bias assertions
        self.assertEqual(len(nnet.bias), len(dlayers))
        self.assertEqual(nnet.bias[0].size, 0)
        self.assertTrue(all(b.size == biases[i] for i, b in enumerate(nnet.bias)))
        self.assertTrue(all(
            b.shape == (dlayers[i],)
            for i, b in enumerate(nnet.bias[1:], 1)
        ))
        self.assertTrue(all(0 <= p and p <= 1 for b in nnet.bias for p in b))

    def test_nnet(self):
        pass


if __name__ == '__main__':
    unittest.main()
