from typing import List

from . import cynet_native
from .nnet import NeuralNetwork, Array


class CyNet(NeuralNetwork):
    """
    Cython powered Neural Network.

    This class directly calls into cython code, it is a rewrite of what NpNet
    already did plus other tricks for improved performance and parallelism.
    It is not optimal, but does the job
    """

    def DADW(self, l, q, k):
        """Matrix A^{l, q}_k of each derivative of dadw(i, j)."""
        return cynet_native.DADW(self, l, q, k)

    def get_gradients(self, target: Array) -> List[Array]:
        return cynet_native.get_gradients(self, target)
