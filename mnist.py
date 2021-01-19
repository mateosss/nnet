from main import BATCH_SIZE
import os
import struct
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load(
    dataset: str = "training", batch_size=10, path: str = "./mnist_training_data"
) -> List[Tuple[np.array, np.array]]:
    """Import mnist dataset.
    Return list of batches like (batch_size x 784 images, batch_size x 10 labels)
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
        lbl = np.zeros((num, 10), dtype=np.float32)
        for i, row in enumerate(lbl):
            row[lblt[i]] = 1
        lbl = lbl.reshape((num // BATCH_SIZE, BATCH_SIZE, 10))

    with open(fname_img, "rb") as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        assert num % batch_size == 0
        img = np.fromfile(fimg, dtype=np.uint8)
        img = img.astype(np.float32)
        img = img.reshape(num // batch_size, batch_size, rows * cols)
        img /= 255
    return list(zip(img, lbl))


def show(image: np.array) -> None:
    if len(image.shape) == 1:
        image = image.reshape(28, 28)
    plt.axis("off")
    plt.imshow(image, cmap="gray")
    plt.show()
    plt.close()
