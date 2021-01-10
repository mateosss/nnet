import os
import struct
from typing import Iterator, Tuple

import matplotlib.pyplot as plt
import numpy as np

"""
Based on https://gist.github.com/akesling/5358964
"""


def read(
    dataset: str = "training", path: str = "./mnist_training_data"
) -> Iterator[Tuple[str, np.ndarray]]:
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset == "training":
        fname_img = os.path.join(path, "train-images-idx3-ubyte")
        fname_lbl = os.path.join(path, "train-labels-idx1-ubyte")
    elif dataset == "testing":
        fname_img = os.path.join(path, "t10k-images-idx3-ubyte")
        fname_lbl = os.path.join(path, "t10k-labels-idx1-ubyte")
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, "rb") as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, "rb") as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield (img[i], lbl[i])


def show(image: np.ndarray) -> None:
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    plt.axis("off")
    plt.imshow(image, cmap="gray")
    plt.show()
    plt.close()
