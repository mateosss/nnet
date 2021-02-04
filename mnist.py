import os
import struct

import matplotlib.pyplot as plt
import numpy as np

DTYPE = np.dtype("float32")
MNIST_PATH = "./mnist_training_data"


def load(dataset="training", batch_size=10, path=MNIST_PATH):
    """Import and process mnist dataset.
    Return generator of lists of batches
    with batch shape: (images: batch_size x 784, labels: batch_size x 10)
    Based on https://gist.github.com/akesling/5358964
    """

    assert dataset in ["training", "testing"]
    prefix = {"training": "train", "testing": "t10k"}[dataset]
    fname_img = os.path.join(path, f"{prefix}-images-idx3-ubyte")
    fname_lbl = os.path.join(path, f"{prefix}-labels-idx1-ubyte")

    with open(fname_lbl, "rb") as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        assert num % batch_size == 0
        lblt = np.fromfile(flbl, dtype=np.int8)
        lbl = np.zeros((num, 10), dtype=DTYPE)
        for i, row in enumerate(lbl):
            row[lblt[i]] = 1

    with open(fname_img, "rb") as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        assert num % batch_size == 0
        img = np.fromfile(fimg, dtype=np.uint8)
        img = img.astype(DTYPE)
        img /= 255
        img.shape = (num, rows * cols)

    while True:
        batched_imgs = img.reshape(num // batch_size, batch_size, rows * cols)
        batched_lbls = lbl.reshape((num // batch_size, batch_size, 10))
        yield list(zip(batched_imgs, batched_lbls))
        shuffle(img, lbl)


def shuffle(a, b) -> None:
    """Shuffle arrays a and b inplace moving elements over axis 0 to the same indices"""
    prev_rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(prev_rng_state)
    np.random.shuffle(b)


def show(image: np.array) -> None:
    if len(image.shape) == 1:
        image = image.reshape(28, 28)
    plt.axis("off")
    plt.imshow(image, cmap="gray")
    plt.show()
    plt.close()
