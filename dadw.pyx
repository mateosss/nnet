# distutils: extra_compile_args=-fopenmp -Ofast -march=native
# distutils: extra_link_args=-fopenmp
# cython: language_level=3, boundscheck=False, wraparound=False

# TODO: Remove unused underscores

import os
import numpy as _np  # not as np as we don't want the code getting float64

from cython.parallel import prange

cimport cython
from libc.math cimport exp
from libc.string cimport memset

cdef size_t BATCH_SIZE = 1000 # TODO: Redundant, already in main.py

assert os.getenv("OMP_NUM_THREADS"), "Unset OMP_NUM_THREADS envvar"
cdef size_t OMP_NUM_THREADS = int(os.getenv("OMP_NUM_THREADS"))

DTYPE = _np.dtype("float32")
cdef size_t DTYPE_SIZE = DTYPE.itemsize
AXIS = _np.newaxis

cdef float _g(float x) nogil: # TODO: Redundant, already in nnet
    return 1 / (1 + exp(-x))

cdef float _gprime(float h) nogil: # TODO: Redundant, already in nnet
    return _g(h) * (1 - _g(h))

cdef void zerofill3(float[:, :, ::1] out, size_t X, size_t Y, size_t Z) nogil:
    cdef size_t x, y, z
    for x in prange(X, nogil=True):
        memset(&out[x,0,0], 0, Y * Z * DTYPE_SIZE)

cpdef void matmul(float[:, :] A, float[:, :] B, float[:, :] out):
    """matrix multiply A (n x m) and B (m x l) into out (n x l)
    Needed as numpy matmul @ was working unpredictably. See commit bb58858
    Running with `kernprof -l main.py` was faster than running `python main.py`.
    """
    cdef size_t i, j, k
    cdef float s
    cdef size_t n = A.shape[0], m = A.shape[1]
    cdef size_t mm = B.shape[0], l = B.shape[1]

    for i in range(n):
        for j in range(l):
            s = 0
            for k in range(m):
                s += A[i, k] * B[k, j]
            out[i, j] = s

cpdef DADW(self, size_t l, size_t q, size_t k, float[:, :, ::1] out):
    """Read only matrix A^{l, q}_k of each derivative of dadw(i, j).

    out must be a matrix of size (BATCH_SIZE, n + 1, m) to overwrite.
    """

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
    cdef const float[:, :, ::1] prev_A

    if l == k + 1:
        zerofill3(out, BATCH_SIZE, n + 1, m)
        for b in prange(BATCH_SIZE, nogil=True):
            fanin = fanins[b, q]
            derivative = _gprime(fanin)
            for i in range(n + 1):
                activation = activations[b, i]
                out[b, i, q] = derivative * activation
    elif l == k + 2:
        for b in prange(BATCH_SIZE, nogil=True):
            fanin = fanins[b, q]
            derivative = _gprime(fanin)
            for i in range(n + 1):
                activation = activations[b, i]
                for j in range(m):
                    fanin_prev = fanins_prev[b, j]
                    derivative_prev = _gprime(fanin_prev)
                    w = weights[j, q]
                    out[b, i, j] = derivative * w * derivative_prev * activation
    elif l > k + 1:
        zerofill3(out, BATCH_SIZE, n + 1, m)
        for r in range(prev_l_sz):
            prev_A = self._DADW_cache[(l - 1, r, k)]
            w = weights[r, q]
            for b in prange(BATCH_SIZE, nogil=True):
                for i in range(n + 1):
                    for j in range(m):
                        out[b, i, j] += w * prev_A[b, i, j]
        for b in range(BATCH_SIZE):
            fanin = fanins[b, q]
            derivative = _gprime(fanin)
            for i in range(n + 1):
                for j in range(m):
                    out[b, i, j] *= derivative
    else:
        raise Exception("This execution branch should not be reached.")

def get_gradients(object self, float[:, ::1] target):
    """Matrix of each error gradient ∇E^k_{i, j} using DADW() matrices."""

    cdef size_t L = len(self.dlayers) - 1  # Last layer index
    cdef size_t sL = self.dlayers[L] # Last layer size
    cdef float mseconst = 2 / self.dlayers[L]
    cdef size_t n, m, k, q, b, i, j

    cdef float[:, ::1] outputs = self.activations[L]
    cdef float[::1] tgtdiff = _np.zeros(BATCH_SIZE, dtype=DTYPE)
    cdef float[:, :, ::1] summation
    cdef float[:, :, ::1] ALqk

    cdef float summ = 0
    cdef float[:, ::1] fanin1 = self.fanin[1]
    cdef float[:, ::1] fanin2 = self.fanin[2]
    cdef float[:, ::1] activation0 = self.activations[0]
    cdef float[:, ::1] weight1 = self.weights[1]

    cdef float[:, :, :, ::1] cache = DADW_prepopulate(self)
    for q in range(16):
        self._DADW_cache[(2, q, 0)] = cache[q]

    gradients = [None for _ in self.weights]

    k = 1
    print(f"k={k}")
    summation = _np.zeros_like(self.gradients[k], dtype=DTYPE)  # (batch_size, n + 1, m)
    ALqk = _np.zeros_like(self.gradients[k], dtype=DTYPE)
    n = self.dlayers[k]
    m = self.dlayers[k + 1]
    for q in range(sL):
        for b in range(BATCH_SIZE):
            tgtdiff[b] = outputs[b, q] - target[b, q]
        DADW(self, L, q, k, ALqk)
        for b in prange(BATCH_SIZE, nogil=True):
            for i in range(n + 1):
                for j in range(m):
                    summation[b, i, j] += tgtdiff[b] * ALqk[b, i, j]
    for b in prange(BATCH_SIZE, nogil=True):
        for i in range(n + 1):
            for j in range(m):
                summation[b, i, j] *= mseconst
    gradients[k] = _np.asarray(summation)

    k = 0
    print(f"k={k}")
    summation = _np.zeros_like(self.gradients[k], dtype=DTYPE)  # (batch_size, n + 1, m)
    ALqk = _np.zeros_like(self.gradients[k], dtype=DTYPE)
    n = self.dlayers[k]
    m = self.dlayers[k + 1]
    for b in prange(BATCH_SIZE, nogil=True):
        for i in range(n + 1):
            for j in range(m):
                ALqk[b, i, j] = mseconst * _gprime(fanin1[b, j]) * activation0[b, i]
    for b in prange(BATCH_SIZE, nogil=True):
        for j in range(m):
            summ = 0
            for q in range(sL):
                summ += (outputs[b, q] - target[b, q]) * _gprime(fanin2[b, q]) * weight1[j, q]
            for i in range(n + 1):
                ALqk[b, i, j] *= summ
    gradients[k] = _np.asarray(ALqk)

    return gradients

cdef void DADW_pre(
    object self, float [:, :, :, ::1] cache, size_t l, size_t q, size_t k,
    size_t n, size_t m, size_t prev_l_sz,
    const float[:, ::1] weights,
    const float[:, ::1] fanins,
    const float[:, ::1] fanins_prev,
    const float[:, ::1] activations,
) nogil:
    "This function is a copy of DADW with for the l == k + 2 case"

    cdef size_t b, i, j
    cdef float fanin, derivative, activation
    cdef float fanin_prev, derivative_prev
    cdef float w
    cdef const float[:, :, ::1] prev_A

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

    if len(self.dlayers) <= 3:
        return
    assert len(self.dlayers) == 4 # Read TODO comment below

    # TODO: This prefill is needed for performance in networks of more than
    # three layers (counting input and output layers)
    # This function right now is designed solely for the case of four layers
    # But it is possible to generalize it with the following pseudo algorithm:
    # for k = L - 1 - 2 downto k = 0 do
    #     for l = k + 2 to l = L - 1 do
    #         use previous l generated cache to
    #         cache DADW(l, 0, k), ..., DADW(l, #l, k)
    # Note that in the inner loop, iterations before the last one are unused
    # thus their memory can be reused to prevent more mallocs.
    # Right now, this function is doing just one loop of the previous algorithm.

    cdef size_t l = 2
    cdef size_t q
    cdef size_t k = 0

    cdef float[:, :, :, ::1] cache = _np.zeros((16, 1000, 785, 16), dtype=DTYPE)
    cdef size_t n = self.dlayers[k]
    cdef size_t m = self.dlayers[k + 1]
    cdef size_t prev_l_sz = self.dlayers[l - 1]

    cdef size_t num_threads = min(OMP_NUM_THREADS, m)

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
