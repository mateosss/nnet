"Main user of nnet module"

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

# Assertion needed for using dataset()
assert 60000 % BATCH_SIZE == 0 and 10000 % BATCH_SIZE == 0

LOG_SAMPLE_FREQ = 1000  # How many samples between logs
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


def test(trainbatches, testbatches, net: NeuralNetwork, epoch):
    itime = time()
    train_cumloss = 0
    test_cumloss = 0
    for batch in trainbatches:
        train_cumloss += net.batch_eval(batch, BATCH_SIZE, calc_grads=False)
    for batch in testbatches:
        test_cumloss += net.batch_eval(batch, BATCH_SIZE, calc_grads=False)
    train_avgloss = train_cumloss / len(trainbatches)
    test_avgloss = test_cumloss / len(testbatches)
    test_time = time() - itime
    print(
        f"[E] [{epoch + 1}] {train_avgloss=:.6f} {test_avgloss=:.6f} {test_time=:.2f}s"
    )


def train_epoch(trainbatches, net: NeuralNetwork, epoch):
    log_loss = 0
    log_time = time()
    for i, batch in enumerate(trainbatches):
        loss, gradients = net.batch_eval(batch, BATCH_SIZE)
        net.update_weights(gradients)

        # Assert improved loss
        # new_loss = net.batch_eval(batch, BATCH_SIZE, calc_grads=False)
        # if new_loss > loss:
        #     print(f"[W] loss increased by {100 * (new_loss - loss) / loss:.2f}%")

        log_loss += loss
        if i % LOG_FREQ == LOG_FREQ - 1:
            print(
                f"[TR] [{epoch + 1}, {(i + 1) * BATCH_SIZE}] [{time() - log_time:.2f}s] "
                f"avgloss: {log_loss / LOG_FREQ}"
            )
            log_time = time()
            log_loss = 0


def train(net: NeuralNetwork, trainbatches, testbatches):
    itime = time()
    for epoch in range(EPOCHS):
        train_epoch(trainbatches, net, epoch)
        test(trainbatches, testbatches, net, epoch)
    print(f"[FINISH] Training finished in {time() - itime:.2f}s.")


def main():
    net = NeuralNetwork(DLAYERS, BATCH_SIZE)
    trainbatches = mnist.load("training", BATCH_SIZE)
    testbatches = mnist.load("testing", BATCH_SIZE)
    train(net, trainbatches, testbatches)


if __name__ == "__main__":
    # best_params = genetic_main()
    main()
