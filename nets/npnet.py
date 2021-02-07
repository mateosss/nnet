from typing import List

from .nnet import NeuralNetwork, zeros_like, gprime, AXIS, Array


class NpNet(NeuralNetwork):
    """Implements the matricization of the derived equations for performance gains with numpy."""

    def DADW(self, l, q, k):
        """Readonly matrix A^{l, q}_k of each derivative of dadw(i, j)."""
        args = (l, q, k)
        if args in self._DADW_cache:
            return self._DADW_cache[args]

        res = zeros_like(self._gradients[k])  # (batch_size, n + 1, m)
        if l == k + 1:
            derivatives = gprime(self.fanins[l][:, q, AXIS])
            columns = self.activations[k][:]
            res[:, :, q] = derivatives * columns
        elif l > k + 1:
            for r in range(self.dlayers[l - 1]):
                res += self.weights[l - 1][r, q] * self.DADW(l - 1, r, k)
            derivatives = gprime(self.fanins[l][:, q, AXIS, AXIS])
            res[:] *= derivatives
        else:
            raise Exception("This execution branch should not be reached.")

        res.setflags(write=False)  # As the result is cached, we make it readonly
        self._DADW_cache[args] = res
        return res

    def get_gradients(self, target: Array) -> List[Array]:
        """Matrix of each error gradient ∇E^k_{i, j} using DADW() matrices."""

        L = len(self.dlayers) - 1  # Last layer index
        mseconst = 2 / self.dlayers[L]
        for k in reversed(range(L)):
            summation = zeros_like(self._gradients[k])  # (batch_size, n + 1, m)
            for q in range(self.dlayers[L]):
                tgtdiff = self.activations[L][:, q] - target[:, q]
                tgtdiff = tgtdiff[:, AXIS, AXIS]
                ALqk = self.DADW(L, q, k)
                summation += tgtdiff * ALqk
            self._gradients[k] = mseconst * summation
        return self._gradients
