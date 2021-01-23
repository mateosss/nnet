# distutils: extra_compile_args=-fopenmp -Ofast -march=native
# distutils: extra_link_args=-fopenmp
# cython: language_level=3, boundscheck=False, wraparound=False

import numpy as _np  # not as np as we don't want the code getting float64
from functools import partial
cimport cython
from libc.math cimport exp
from cython.parallel import prange
from cython.parallel import parallel, threadid
cdef size_t BATCH_SIZE = 1000

DTYPE = _np.float32
AXIS = _np.newaxis

cdef float _g(float x) nogil:
    return 1 / (1 + exp(-x))

cdef float _gprime(float h) nogil:
    return _g(h) * (1 - _g(h))

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
    cdef const float[:, ::1] fanins_prev = self.fanin[l - 1]
    cdef const float[:, ::1] activations = self.activations[k]

    cdef size_t b, i, j
    cdef float fanin, derivative, activation
    cdef float fanin_prev, derivative_prev
    cdef float w
    cdef const float[:, :, :] prev_A

    if l == k + 1:
        # for b in range(BATCH_SIZE):
        for b in prange(BATCH_SIZE, nogil=True):
            fanin = fanins[b, q]
            derivative = _gprime(fanin)
            for i in range(n + 1):
                activation = activations[b, i]
                _res[b, i, q] = derivative * activation
    elif l == k + 2:
        # for b in range(BATCH_SIZE):
        for b in prange(BATCH_SIZE, nogil=True):
            fanin = fanins[b, q]
            derivative = _gprime(fanin)
            for i in range(n + 1):
                activation = activations[b, i]
                for j in range(m):
                    fanin_prev = fanins_prev[b, j]
                    derivative_prev = _gprime(fanin_prev)
                    w = weights[j, q]
                    _res[b, i, j] = derivative * w * derivative_prev * activation
    elif l > k + 1:
        for r in range(prev_l_sz):
            prev_A = DADW(self, l - 1, r, k)
            w = weights[r, q]
            # for b in range(BATCH_SIZE):
            for b in prange(BATCH_SIZE, nogil=True):
                for i in range(n + 1):
                    for j in range(m):
                        _res[b, i, j] += w * prev_A[b, i, j]
        for b in range(BATCH_SIZE):
            fanin = fanins[b, q]
            derivative = _gprime(fanin)
            for i in range(n + 1):
                for j in range(m):
                    _res[b, i, j] *= derivative
    else:
        raise Exception("This execution branch should not be reached.")

    res.setflags(write=False)  # As the result is cached, we make it readonly
    self._DADW_cache[args] = res
    return res




cdef void DADW_pre(
    object self, float [:, :, :, ::1] cache, size_t l, size_t q, size_t k,
    size_t n, size_t m, size_t prev_l_sz,
    const float[:, ::1] weights,
    const float[:, ::1] fanins,
    const float[:, ::1] fanins_prev,
    const float[:, ::1] activations,
) nogil:

    cdef size_t b, i, j
    cdef float fanin, derivative, activation
    cdef float fanin_prev, derivative_prev
    cdef float w
    cdef const float[:, :, :] prev_A

    # if l == k + 2:
    for b in range(BATCH_SIZE):
        fanin = fanins[b, q]
        derivative = _gprime(fanin)
        for i in range(n + 1):
            activation = activations[b, i]
            for j in range(m):
                fanin_prev = fanins_prev[b, j]
                derivative_prev = _gprime(fanin_prev)
                w = weights[j, q]
                cache[q, b, i, j] = derivative * w * derivative_prev * activation

def DADW_prepopulate(self):
    "Precalculate some of the needed DADW cache in a multithread burst"
    cdef size_t l = 2
    cdef size_t q
    cdef size_t k = 0
    cdef size_t num_threads = 4 # TODO: Get this number from OMP_NUM_THREADS envvar

    cdef float[:, :, :, ::1] cache = _np.zeros((16, 1000, 785, 16), dtype=DTYPE)
    cdef size_t n = self.dlayers[k]
    cdef size_t m = self.dlayers[k + 1]
    cdef size_t prev_l_sz = self.dlayers[l - 1]

    if m % num_threads != 0:
        print(f"[W] m={m} % num_threads={num_threads} != 0, some threads will remain idle while others work")

    cdef const float[:, ::1] weights = self.weights[l - 1]
    cdef const float[:, ::1] fanins = self.fanin[l]
    cdef const float[:, ::1] fanins_prev = self.fanin[l - 1]
    cdef const float[:, ::1] activations = self.activations[k]

    for q in prange(m, nogil=True, num_threads=num_threads):
        DADW_pre(
            self, cache, l, q, k,
            n, m, prev_l_sz,
            weights, fanins, fanins_prev, activations
        )
    return cache
