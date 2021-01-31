# distutils: extra_compile_args=-fopenmp -Ofast -march=native
# distutils: extra_link_args=-fopenmp
# cython: language_level=3, boundscheck=False, wraparound=False

# TODO: Remove unused underscores

import os
import numpy as _np
# TODO: Importing numpy as _np instead of np like in nnet.py to
# originally prevent from not putting dtype, but I'm explicitly writing
# dtype=DTYPE on all calls, decide to rename _np to np, or to do the same
# as in nnet.py (wrap np partial functions into ones that pass dtype=DTYPE)

from cython.parallel import prange

cimport cython
from libc.math cimport exp
from libc.string cimport memset

cdef size_t BATCH_SIZE = 1000 # TODO: Redundant, already in main.py

assert os.getenv("OMP_NUM_THREADS"), "Unset OMP_NUM_THREADS envvar"
cdef size_t OMP_NUM_THREADS = int(os.getenv("OMP_NUM_THREADS"))

# NOTE: Change to double and float64 these two lines for float64 usage
ctypedef float real
DTYPE = _np.dtype("float32")

cdef size_t DTYPE_SIZE = DTYPE.itemsize
AXIS = _np.newaxis

cdef real _g(real x) nogil: # TODO: Redundant, already in nnet
    return 1 / (1 + exp(-x)) # TODO: exp is for double, use expf for floats, available in newer versions of cython

cdef real _gprime(real h) nogil: # TODO: Redundant, already in nnet
    return _g(h) * (1 - _g(h))

cdef void zerofill3(real[:, :, ::1] out, size_t X, size_t Y, size_t Z) nogil:
    cdef size_t x, y, z
    for x in prange(X, nogil=True):
        memset(&out[x,0,0], 0, Y * Z * DTYPE_SIZE)

cpdef void matmul(real[:, :] A, real[:, :] B, real[:, :] out):
    """matrix multiply A (n x m) and B (m x l) into out (n x l)
    Needed as numpy matmul @ was working unpredictably. See commit bb58858
    Running with `kernprof -l main.py` was faster than running `python main.py`.
    """
    cdef size_t i, j, k
    cdef real s
    cdef size_t n = A.shape[0], m = A.shape[1]
    cdef size_t mm = B.shape[0], l = B.shape[1]

    for i in range(n):
        for j in range(l):
            s = 0
            for k in range(m):
                s += A[i, k] * B[k, j]
            out[i, j] = s

cpdef DADW(self, size_t l, size_t q, size_t k, real[:, :, ::1] out):
    """Read only matrix A^{l, q}_k of each derivative of dadw(i, j).

    out must be a matrix of size (BATCH_SIZE, n + 1, m) to overwrite.
    """

    cdef size_t n = self.dlayers[k]
    cdef size_t m = self.dlayers[k + 1]
    cdef size_t prev_l_sz = self.dlayers[l - 1]

    cdef const real[:, ::1] weights = self.weights[l - 1]
    cdef const real[:, ::1] fanins = self.fanins[l]
    cdef const real[:, ::1] fanins_prev = self.fanins[l - 1]
    cdef const real[:, ::1] activations = self.activations[k]

    cdef size_t b, i, j
    cdef real fanin, derivative, activation
    cdef real fanin_prev, derivative_prev
    cdef real w
    cdef const real[:, :, ::1] prev_A

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

def get_gradients(object self, real[:, ::1] target):
    """Matrix of each error gradient ∇E^k_{i, j} using DADW() matrices."""

    cdef size_t L = len(self.dlayers) - 1  # Last layer index
    cdef size_t sL = self.dlayers[L] # Last layer size
    cdef real mseconst = 2 / self.dlayers[L]
    cdef size_t n, m, k, q, b, i, j

    cdef real[:, ::1] outputs = self.activations[L]
    cdef real[:, :, ::1] ALqk

    cdef real a1i, gh1j, summ = 0 # summ = 0 needed or cython complains
    cdef real[:, ::1] fanin1 = self.fanins[1]
    cdef real[:, ::1] fanin2 = self.fanins[2]
    cdef real[:, ::1] activation0 = self.activations[0]
    cdef real[:, ::1] activation1 = self.activations[1]
    cdef real[:, ::1] weight1 = self.weights[1]

    gradients = [None for _ in self.weights]

    # Explicit iteration k=1 for better performance
    k = 1
    n = self.dlayers[k]
    m = self.dlayers[k + 1]
    ALqk = _np.empty((BATCH_SIZE, n + 1, m), dtype=DTYPE)
    for b in prange(BATCH_SIZE, nogil=True):
        for i in range(n + 1):
            a1i = activation1[b, i]
            for j in range(m):
                ALqk[b, i, j] = (
                    mseconst * (outputs[b, j] - target[b, j])
                    * _gprime(fanin2[b, j]) * a1i
                )
    gradients[k] = _np.asarray(ALqk)

    # Explicit iteration k=0 for better performance
    k = 0
    n = self.dlayers[k]
    m = self.dlayers[k + 1]
    ALqk = _np.empty((BATCH_SIZE, n + 1, m), dtype=DTYPE)
    for b in prange(BATCH_SIZE, nogil=True):
        for j in range(m):
            summ = 0
            for q in range(sL):
                summ += (outputs[b, q] - target[b, q]) * _gprime(fanin2[b, q]) * weight1[j, q]
            gh1j = _gprime(fanin1[b, j])
            for i in range(n + 1):
                ALqk[b, i, j] = mseconst * gh1j * activation0[b, i]
                ALqk[b, i, j] *= summ # NOTE: *= needed or cython complains
    gradients[k] = _np.asarray(ALqk)

    return gradients

cdef void DADW_pre(
    object self, real [:, :, :, ::1] cache, size_t l, size_t q, size_t k,
    size_t n, size_t m, size_t prev_l_sz,
    const real[:, ::1] weights,
    const real[:, ::1] fanins,
    const real[:, ::1] fanins_prev,
    const real[:, ::1] activations,
) nogil:
    "This function is a copy of DADW with for the l == k + 2 case"

    cdef size_t b, i, j
    cdef real fanin, derivative, activation
    cdef real fanin_prev, derivative_prev
    cdef real w
    cdef const real[:, :, ::1] prev_A

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

    cdef real[:, :, :, ::1] cache = _np.zeros((16, BATCH_SIZE, 785, 16), dtype=DTYPE)
    cdef size_t n = self.dlayers[k]
    cdef size_t m = self.dlayers[k + 1]
    cdef size_t prev_l_sz = self.dlayers[l - 1]

    cdef size_t num_threads = min(OMP_NUM_THREADS, m)

    if m % num_threads != 0:
        print(f"[W] m={m} % num_threads={num_threads} != 0, some threads will remain idle while others work")

    cdef const real[:, ::1] weights = self.weights[l - 1]
    cdef const real[:, ::1] fanins = self.fanins[l]
    cdef const real[:, ::1] fanins_prev = self.fanins[l - 1]
    cdef const real[:, ::1] activations = self.activations[k]

    for q in prange(m, nogil=True, num_threads=num_threads):
        DADW_pre(
            self, cache, l, q, k,
            n, m, prev_l_sz,
            weights, fanins, fanins_prev, activations
        )
    return cache
