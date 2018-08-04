from typing import Union, Optional, List, Tuple, Iterator
from random import random
from math import exp
import mnist

class Neuron:

    previous_layer: Optional['Layer']
    input_weights: List[float]
    bias: float
    activation: float

    def __init__(self, previous_layer: Optional['Layer']) -> None:
        self.previous_layer = previous_layer
        self.input_weights = []
        self.bias = 0
        self.activation = 0

    def sigmoid(self, x: float) -> float:
        return 1 / (1 + exp(-x))

    def get_activation(self) -> float:
        if not self.activation and self.previous_layer: # If it is an input neuron, should have a predefined activation and there is no need to calculate a new one
            self.activation = self.sigmoid(sum(
                    prev_neuron.get_activation() * weight
                    for prev_neuron, weight in zip(self.previous_layer.neurons, self.input_weights)
                ) - self.bias)
        return self.activation

    def __str__(self) -> str:
        return f'activation={self.activation} | len(input_weights)={len(self.input_weights)} | bias={self.bias}{" | input" if self.previous_layer is None else ""}'

class Layer:

    previous_layer: Optional['Layer']
    neurons: List[Neuron]

    def __init__(self, neuron_count: int, previous_layer: Optional['Layer']) -> None:
        self.previous_layer = previous_layer
        self.neurons = [Neuron(previous_layer) for n in range(neuron_count)]


class NeuralNetwork:

    layers: List[Layer]

    def __init__(self, layers: List[int], dna: Union[bool, List[int]] = False) -> None:
        self.layers = [Layer(layers[0], None)]
        for i, neuron_count in enumerate(layers[1:]):
            self.layers.append(Layer(neuron_count, self.layers[i]))
        self.genrand_weights_and_biases()

    def genrand_weights_and_biases(self) -> None:
        previous_layer_neurons = len(self.layers[0].neurons)
        for layer in self.layers[1:]:
            for neuron in layer.neurons:
                neuron.input_weights = [random() for i in range(previous_layer_neurons)]
                neuron.bias = random()
            previous_layer_neurons = len(layer.neurons)

    def get_neurons(self) -> Iterator[Neuron]:
        return (neuron for layer in self.layers for neuron in layer.neurons)

    def set_dna(self, dna: List[int]) -> None:
        last_neuron_at = 0
        for i, neuron in enumerate(self.get_neurons()):
            for j, weight in enumerate(neuron.input_weights):
                neuron.input_weights[j] = dna[last_neuron_at + i]
            last_neuron_at = last_neuron_at + j + 1
            neuron.bias = dna[last_neuron_at]

    def get_dna(self) -> List[float]:
        dna: List[float] = []
        for layer in self.layers[1:]:
            for neuron in layer.neurons:
                for weight in neuron.input_weights:
                    dna.append(weight)
                dna.append(neuron.bias)
        return dna

    def run(self, input_values: List[float]) -> List[float]:
        assert len(input_values) == len(self.layers[0].neurons), f'This networks accepts {len(self.layers[0].neurons)} but you sent {len(input_values)}'

        for neuron, input in zip(self.layers[0].neurons, input_values):
            neuron.activation = input

        res: List[float] = []
        for neuron in self.layers[-1].neurons:
            res.append(neuron.get_activation())
        return res


if __name__ == '__main__':
    n = NeuralNetwork([784, 16, 16, 10])

    label, image = next(mnist.read())
    input = [pixel for row in image for pixel in row] # TODO is one image with the number five
    run = n.run(input)

    # TODO The comparisson should be made in a proper place
    perfect_run = [1 if i is int(label) else 0 for i in range(len(run))]
    cost = sum((returned - expected)**2 for returned, expected in zip(run, perfect_run))
