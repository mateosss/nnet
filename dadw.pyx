# distutils: extra_compile_args=-fopenmp -Ofast -march=native
# distutils: extra_link_args=-fopenmp
# cython: language_level=3, boundscheck=False, wraparound=False

import numpy as _np  # not as np as we don't want the code getting float64
from functools import partial
cimport cython
from libc.math cimport exp
from cython.parallel import prange
cdef size_t BATCH_SIZE = 1000

DTYPE = _np.float32
AXIS = _np.newaxis

cdef float _g(float x) nogil:
    return 1 / (1 + exp(-x))

cdef float _gprime(float h) nogil:
    return _g(h) * (1 - _g(h))

# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)  # Deactivate negative indexing.
cpdef DADW(self, size_t l, size_t q, size_t k):
    """Read only matrix A^{l, q}_k of each derivative of dadw(i, j)."""
    args = (l, q, k)
    if args in self._DADW_cache:
        return self._DADW_cache[args]

    res = _np.zeros_like(self.gradients[k], dtype=DTYPE)  # (batch_size, n + 1, m)
    cdef float [:, :, ::1] _res = res

    cdef size_t n = self.dlayers[k]
    cdef size_t m = self.dlayers[k + 1]
    cdef size_t prev_l_sz = self.dlayers[l - 1]

    cdef const float[:, ::1] weights = self.weights[l - 1]
    cdef const float[:, ::1] fanins = self.fanin[l]
    cdef const float[:, ::1] activations = self.activations[k]

    cdef size_t b, i, j
    cdef float _fanin, derivative, activation
    cdef float w
    cdef const float[:, :, :] prev_A




    # if l == k + 1:
    #     fanin = self.fanin[l][0:BATCH_SIZE, q, AXIS]
    #     derivatives = gprime(fanin)
    #     columns = self.activations[k][:]
    #     res[..., :, q] = derivatives * columns
    if l == k + 1:
        # for b in range(BATCH_SIZE):
        for b in prange(BATCH_SIZE, nogil=True):
            for i in range(n + 1):
                _fanin = fanins[b, q]
                derivative = _gprime(_fanin)
                activation = activations[b, i]
                _res[b, i, q] = derivative * activation
    elif l > k + 1:
        for r in range(prev_l_sz):
            prev_A = DADW(self, l - 1, r, k)
            w = weights[r, q]
            # for b in range(BATCH_SIZE):
            for b in prange(BATCH_SIZE, nogil=True):
                for i in range(n + 1):
                    for j in range(m):
                        _res[b, i, j] += w * prev_A[b, i, j]
        _fanin = fanins[b, q]
        derivative = _gprime(_fanin)
        for b in range(BATCH_SIZE):
            for i in range(n + 1):
                for j in range(m):
                    _res[b, i, j] *= derivative

    # elif l > k + 1:
    #     for r in range(self.dlayers[l - 1]):
    #         res += self.weights[l - 1][r, q] * self.DADW(l - 1, r, k)
    #     fanin = self.fanin[l][0:BATCH_SIZE, q, AXIS, AXIS]
    #     derivatives = gprime(fanin)
    #     res[...] *= derivatives
    else:
        raise Exception("This execution branch should not be reached.")




    res.setflags(write=False)  # As the result is cached, we make it readonly
    self._DADW_cache[args] = res
    return res
