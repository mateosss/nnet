"Main user of nnet module"

from typing import List
from numpy import mean
import mnist
from nnet import nnet

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

if __name__ == '__main__':
    main()
