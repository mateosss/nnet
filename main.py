"Main user of nnet module"

from typing import List
import numpy as np
from numpy import mean
import mnist
from nnet import nnet
from genetic_nnet import GANeuralNetwork, NeuralGA

def main():
    "Forward propagation specification"
    # TODO: Try other hyperparameters
    dlayers = [784, 16, 16, 10]
    nnet.set_layers_description(dlayers)
    params = list(nnet.get_random_params())

    BATCH_SIZE = 1
    costs = [None] * BATCH_SIZE
    for i, (label, image) in zip(range(BATCH_SIZE), mnist.read()):
        # mnist.show(image)
        # Run network
        ninput = [pixel / 255 for row in image for pixel in row] # Normalized
        guess: List[int] = nnet(params, ninput)
        print(f"guess={guess}")
        # Cost calculation
        expected = [1 if i == label else 0 for i in range(10)]
        costs[i] = sum((g - e)**2 for g, e in zip(guess, expected))
        print(f"Running {i + 1}/{BATCH_SIZE} | Cost {costs[i]} \r", end='')
    cost = mean(costs)
    print(f"Average cost of {cost} after {BATCH_SIZE} runs")

def genetic_main():
    ga = NeuralGA([784, 16, 16, 10], 350)
    with open("best.py", "w") as f:
        # print(str(ga.best))
        f.write("params = ")
        f.write(str(ga.best))
    return ga.best

def testing_one_of_the_trained(best=None):
    if not best:
        import best
        best = best.params
    dlayers = [784, 16, 16, 10]
    nnet.set_layers_description(dlayers)
    params = best

    mnist_db = list(mnist.read())
    label, image = mnist_db[25001]
    mnist.show(image)

    # Run network
    ninput = [pixel / 255 for row in image for pixel in row] # Normalized
    guess = nnet(params, ninput)
    expected = [1 if i == label else 0 for i in range(10)]
    print(f"guess as int = {(guess * 1e8).astype(int)}")
    print(f"guess squashed as int = {np.interp(guess, [guess.min(), guess.max()], [0, 1]).astype(int)}")
    print(f"guess squashed = {np.round(np.interp(guess, [guess.min(), guess.max()], [0, 1]), 6)}")

    print(f"expected = {expected}")

if __name__ == '__main__':
    best_params = genetic_main()
    testing_one_of_the_trained(best_params)
