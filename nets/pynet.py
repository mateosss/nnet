from typing import List

from .nnet import Array, NeuralNetwork, gprime


class PyNet(NeuralNetwork):
    """
    The slowest to run but the easiest to read of the networks,

    implements the derived equations almost identically as they are expressed.
    """

    def dadw(self, l, q, k, i, j, b=0):  # -> DTYPE
        """Return derivative of a^l_q with respect to w^k_ij for batch sample b."""
        # Memoization stuff
        args = (l, q, k, i, j)
        if args in self._dadw_cache:
            return self._dadw_cache[args]

        # Range assertions
        assert 0 <= l < len(self.dlayers), f"out of range {l=}"
        assert 0 <= k < len(self.dlayers), f"out of range {k=}"
        assert 0 <= i < self.activations[k].size, f"out of range {i=}"
        assert 0 <= j < self.dlayers[k], f"out of range {j=}"

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
                * gprime(self.fanins[l - 1][b, j])
                * self.activations[k][b, i]
            )
        elif k < l - 1:
            res = sum(
                self.weights[l - 1][r, q] * self.dadw(l - 1, r, k, i, j)
                for r in range(self.dlayers[l - 1])
            )
        else:
            raise Exception("Should never reach this execution branch")

        # Multiply by derivative of activation function over the neuron's weighted sum
        res *= gprime(self.fanins[l][b, q])

        # Cache it
        self._dadw_cache[args] = res
        return res

    def get_gradients(self, target) -> List[Array]:
        """Matrix of each error gradient ∇E^k_{i, j} using dadw() for batch sample b."""
        L = len(self.dlayers) - 1  # Last layer index
        mseconst = 2 / self.dlayers[-1]
        for k in reversed(range(L)):
            for b in range(self.batch_size):
                n, m = self.dlayers[k], self.dlayers[k + 1]
                for j in range(m):
                    for i in range(n + 1):  # +1 for bias neuron
                        self._gradients[k][b, i, j] = mseconst * sum(
                            (self.activations[L][b, q] - target[b, q])
                            * self.dadw(L, q, k, i, j, b=b)
                            for q in range(self.dlayers[-1])
                        )
        return self._gradients
