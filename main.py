"Main user of nnet module"

import numpy as np
import mnist
from nnet import NeuralNetwork
from genetic_nnet import NeuralGA

# TODO: Improve program print output
# TODO: Add argparse options to configure the run

# TODO: Try other hyperparameters
DLAYERS = [784, 16, 16, 10]

# TODO: Save runs to a separate folder and don't overwrite them
def save_params(params):
    with open("best.py", "w") as f: # TODO: What a way to save data :)
        f.write(f"params = {str(params.tolist())}")

def load_params():
    from best import params # TODO: Stop using best.py for saving # pylint: disable=E0401
    return np.array(params)

def genetic_main():
    ga = NeuralGA(DLAYERS, 10)
    save_params(ga.best)
    return ga.best

def backpropagation_main():
    labels, images = zip(*list(mnist.read()))
    batch_size = 16
    ninputs = [
        [pixel / 255 for row in image for pixel in row]
        for image in images[:batch_size]
    ]
    expected_outputs = [
        [1 if i == label else 0 for i in range(10)]
        for label in labels[:batch_size]
    ]
    print("here")

    # ninput = list(range(2))
    # nnet = NeuralNetwork([2, 1, 3, 2], params=None)
    nnet = NeuralNetwork(DLAYERS, params=None)
    # nnet = NeuralNetwork(DLAYERS, params=load_params())
    cost_gradient = [None] * batch_size
    cost = np.empty(batch_size)
    for j in range(1024):
        try:
            for i, (ninput, expected) in enumerate(zip(ninputs, expected_outputs)):
                guess = nnet.feedforward(ninput)
                cost[i] = nnet.get_error(expected)
                cost_gradient[i] = nnet.get_error_gradient(expected)
        except KeyboardInterrupt:
            break
        print(f"[{j + 1}] cost = {cost.mean()}")
        batch_gradient = np.mean(cost_gradient, axis=0)
        nnet.backpropagate(gradients=batch_gradient)
    # guess = nnet.feedforward(ninput)
    # cost = nnet.get_error(expected)
    # print(f"[{i + 1}] cost = {cost}")
    save_params(nnet.params)

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
                f"""[guess] squashed [0, 1] = {np.round(np.interp(
                    guess, [guess.min(), guess.max()], [0, 1]
                ), 6)}\n"""
                f"[expected] = {expected}\n"
            )
    misses = len(samples) - hits
    print(f"\nHits: {hits} | Misses: {misses} -> {100 * hits / len(samples)}%")

def test_trained(params=None, head=100, tail=100):
    "Tests a network with params against first `head` and last `tail` examples"
    params = params if params is not None else load_params()
    nnet = NeuralNetwork(DLAYERS, params)
    mnist_db = list(mnist.read())
    print("[KNOWN]")
    test_and_report_against(nnet, mnist_db[:head]) # Training dataset
    print("[UNKNOWN]")
    test_and_report_against(nnet, mnist_db[-tail:]) # Unknown dataset


if __name__ == '__main__':
    # best_params = genetic_main()
    # test_trained(best_params)
    backpropagation_main()
