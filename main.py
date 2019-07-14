"Main user of nnet module"

from typing import List
import mnist
from nnet import nnet

def main():
    "Forward propagation specification"
    dlayers = [784, 16, 16, 10]
    nnet.set_layers_description(dlayers)
    label, image = next(mnist.read())
    mnist.show(image)
    ninput = [pixel for row in image for pixel in row]
    params = nnet.get_random_params()
    guess: List[int] = nnet(params, ninput)
    print(guess)

if __name__ == '__main__':
    main()
