import numpy as np
from mnist import mnist

def test_mnist_shuffle():
    a = np.arange(1000)
    b = np.arange(1000)
    mnist.shuffle(a, b)
    assert all(a == b)

    A = np.arange(100 * 10).reshape(100, 10)
    AA = A.copy()
    idx = np.arange(100)
    mnist.shuffle(AA, idx)
    assert (AA == A[idx]).all()
