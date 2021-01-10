"Main user of nnet module"

import itertools as it
from time import time

import numpy as np

import mnist
from genetic_nnet import NeuralGA
from nnet import NeuralNetwork

# TODO: Improve program print output
# TODO: Add argparse options to configure the run

DLAYERS = [784, 16, 16, 10]
EPOCHS = 16
BATCH_SIZE = 1000

# TODO: Hardcoding dataset lengths
TEST_BATCHES = 60000 // BATCH_SIZE
TRAIN_BATCHES = 10000 // BATCH_SIZE

LOG_SAMPLE_FREQ = 10000  # How many samples between logs
assert LOG_SAMPLE_FREQ % BATCH_SIZE == 0, "should be multiples"
LOG_FREQ = LOG_SAMPLE_FREQ / BATCH_SIZE  # How many batches between logs

# TODO: Save runs to a separate folder and don't overwrite them
def save_params(params):
    with open("best.py", "w") as f:  # TODO: What a way to save data :)
        f.write(f"params = {str(params.tolist())}")


def load_params():
    # TODO: Stop using best.py for saving # pylint: disable=E0401
    from best import params

    return np.array(params)


def genetic_main():
    ga = NeuralGA(DLAYERS, 10)
    save_params(ga.best)
    return ga.best


def dataset(dataset_type="training"):
    """Use type training or testing."""
    data = mnist.read(dataset_type)
    while True:
        pair_imglbl = tuple(zip(*it.islice(data, BATCH_SIZE)))
        if not pair_imglbl:
            return
        images, labels = pair_imglbl
        images = np.array(images).reshape(BATCH_SIZE, 28 * 28)
        labels = np.array([[1 if q == l else 0 for q in range(10)] for l in labels])
        yield images, labels


def test(net: NeuralNetwork, epoch):
    itime = time()
    train_cumloss = 0
    test_cumloss = 0
    for (inputs, targets) in dataset("training"):
        outputs = net.feedforward(inputs)
        errors = net.get_error(targets)
        loss = errors.mean()
        train_cumloss += loss
    for (inputs, targets) in dataset("testing"):
        outputs = net.feedforward(inputs)
        errors = net.get_error(targets)
        loss = errors.mean()
        train_cumloss += loss
    train_avgloss = train_cumloss / TRAIN_BATCHES
    test_avgloss = test_cumloss / TEST_BATCHES
    test_time = time() - itime
    print(
        f"[E] [{epoch + 1}] {train_avgloss=:.6f} {test_avgloss=:.6f} {test_time=:.2f}s"
    )


def train_epoch(net: NeuralNetwork, epoch):
    log_loss = 0
    log_time = time()
    for i, (inputs, targets) in enumerate(dataset("training")):
        print(f">>> evaluating batch {i=}")
        outputs = net.feedforward(inputs)
        errors = net.get_error(targets)
        loss = errors.mean()
        gradients = net.get_gradients(targets)
        batch_gradient = [np.mean(grads, axis=0) for grads in zip(*gradients)]
        print(f">>> updating weights")
        net.update_weights(batch_gradient)

        # Assert improved loss
        print(f">>> asserting improvement")
        net.feedforward(inputs)
        new_loss = net.get_error(targets).mean()
        if new_loss > loss:
            print(f"[W] loss increased by {100 * (new_loss - loss) / loss:.2f}%")

        log_loss += loss
        if i % LOG_FREQ == LOG_FREQ - 1:
            print(
                f"[TR] [{epoch + 1}, {(i + 1) * BATCH_SIZE}] [{time() - log_time:.2f}s] "
                f"loss: {log_loss / LOG_FREQ:.3f}"
            )
            log_time = time()
            log_loss = 0


def train(net: NeuralNetwork):
    itime = time()
    for epoch in range(EPOCHS):
        print(f">>> start {epoch=} train")
        train_epoch(net, epoch)
        print(f">>> start {epoch=} test")
        test(net, epoch)
    print(f"[FINISH] Training finished in {time() - itime:.2f}s.")


def main():
    net = NeuralNetwork(DLAYERS)
    print(">>> datasets initialized")
    train(net)


if __name__ == "__main__":
    # best_params = genetic_main()
    main()
