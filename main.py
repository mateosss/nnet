"Main user of nnet module"

from typing import List
import numpy as np
from numpy import mean
import mnist
from nnet import nnet
from genetic_nnet import NeuralGA

# TODO: Try other hyperparameters
DLAYERS = [784, 16, 16, 10]

def save_params(params):
    with open("best.py", "w") as f: # TODO: What a way to save data :)
        f.write(f"params = {str(params)}")

def load_params():
    from best import params # TODO: Stop using best.py for saving # pylint: disable=E0401
    return params

def genetic_main():
    ga = NeuralGA([784, 16, 16, 10], 10)
    save_params(ga.best)
    return ga.best

# TODO: Report network confidence and cost, not only hits/misses
def test_and_report_against(nn, samples, print_every=50):
    hits = 0
    for i, (label, image) in enumerate(samples):
        ninput = [pixel / 255 for row in image for pixel in row] # Normalized
        guess = nn.feedforward(ninput) # network guess
        expected = [1 if i == label else 0 for i in range(10)]
        guessed = [1 if i == max(guess) else 0 for i in guess] # pragmatic guess
        hits += guessed == expected
        if (i + 1) % print_every == 0:
            print(
                f"Run {i + 1}/{len(samples)}\n"
                f"[guess] squashed [0, 1] = {np.round(np.interp(guess, [guess.min(), guess.max()], [0, 1]), 6)}\n"
                f"[expected] = {expected}\n"
            )
    misses = len(samples) - hits
    print(f"\nHits: {hits} | Misses: {misses} -> {100 * hits / len(samples)}%")

def test_trained(params=None, head=100, tail=100):
    "Tests a network with params against first `head` and last `tail` examples"
    params = params if params else load_params()
    nnet.set_layers_description(DLAYERS)
    nnet.create_layers(params)
    mnist_db = list(mnist.read())
    print("[KNOWN]")
    test_and_report_against(nnet, mnist_db[:head]) # Training dataset
    print("[UNKNOWN]")
    test_and_report_against(nnet, mnist_db[-tail:]) # Unknown dataset


if __name__ == '__main__':
    best_params = genetic_main()
    test_trained(best_params)
