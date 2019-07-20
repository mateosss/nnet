from typing import List
from random import sample
from numpy import mean
from numpy.random import randn
import mnist
from nnet import NeuralNetwork
from genetics import Subject, GeneticAlgorithm

class GANeuralNetwork(Subject, NeuralNetwork):

    __mnist_db = list(mnist.read())

    _genome: List[float]
    _fitness: float

    def __init__(self, params):
        """
        Precondition: use set_layers_description() before any instanciation
        so dlayers is initialized
        """
        super().__init__(GANeuralNetwork.dlayers, params)
        self._genome = params
        self._fitness = None

    @classmethod
    def create_random(cls):
        return GANeuralNetwork(cls.get_random_params())

    @property
    def genome(self) -> List[float]:
        return self._genome

    @property
    def fitness(self) -> float:
        return self.batch_cost() if not self._fitness else self._fitness

    def batch_cost(self, batch_size=10, random_samples=False):
        "Runs a random minibatch and returns average network cost"
        costs = [None] * batch_size
        db = (
            sample(GANeuralNetwork.__mnist_db, batch_size) if random_samples
            else GANeuralNetwork.__mnist_db[:batch_size]
        )
        for i, (label, image) in enumerate(db): # TODO: parallelize runs
            # Run network
            ninput = [pixel / 255 for row in image for pixel in row] # Normalized
            guess = self(self.genome, ninput)
            # Cost calculation
            expected = [1 if i == label else 0 for i in range(10)]
            costs[i] = sum((g - e)**2 for g, e in zip(guess, expected))
        cost = mean(costs)
        self._fitness = -cost
        # print(f"Average cost of {cost} after {batch_size} runs")
        return self._fitness

    # TODO: Think more about this and make it
    # Maybe a urand in [c +- d] range with c = (min + max) / 2, d = max - min
    @staticmethod
    def mutate(gen):
        return gen + randn()

    @classmethod
    def set_layers_description(cls, dlayers):
        """
        Override of NeuralNetwork method that makes it static
        dlayers will be used as a static attribute of GANeuralNetwork class
        """
        cls.dlayers = dlayers

    @classmethod
    def get_random_params(cls):
        return super().get_random_params(cls)


class NeuralGA(GeneticAlgorithm):
    def __init__(self, dlayers, iterations, **kwargs):
        GANeuralNetwork.set_layers_description(dlayers)
        super().__init__(GANeuralNetwork, iterations, **kwargs)
