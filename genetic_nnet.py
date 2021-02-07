from time import time
from typing import List
from random import sample
from numpy import mean
from numpy.random import randn
import mnist
from nnet import NeuralNetwork
from genetics import Subject, GeneticAlgorithm

BATCH_SIZE = 10

class GANeuralNetwork(Subject, NeuralNetwork):

    __mnist_db = mnist.load(batch_size=BATCH_SIZE)
    __batch_gen = (batch for epoch in __mnist_db for batch in epoch)

    _genome: List[float]
    _fitness: float

    def __init__(self, params):
        """
        Precondition: use set_layers_description() before any instanciation
        so dlayers is initialized
        """
        super().__init__(GANeuralNetwork.dlayers, BATCH_SIZE, params=params)
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
        if not self._fitness:
            self._fitness = -self.batch_cost()
        return self._fitness

    def batch_cost(self):
        random_batch = next(GANeuralNetwork.__batch_gen)
        error = self.batch_eval(random_batch, grads=False)
        return error

    def test(self):
        itime = time()
        train_cumloss = train_cumhits = 0
        epoch = next(mnist.load(batch_size=self.batch_size))
        nof_batches = int(60000 / BATCH_SIZE)
        for batch in epoch:
            cumloss, cumhits = self.batch_eval(batch, grads=False, hitrate=True)
            train_cumloss += cumloss
            train_cumhits += cumhits
        train_avgloss = train_cumloss / nof_batches
        train_hitrate = train_cumhits / nof_batches
        test_time = time() - itime
        print(
            f"[E] {train_avgloss=:.6f} {train_hitrate=:.2f}% {test_time=:.2f}s"
        )

    # TODO: Think more about this and make it
    # Maybe a urand in [c +- d] range with c = (min + max) / 2, d = max - min
    @staticmethod
    def mutate(gen):
        return gen + randn()

    @classmethod
    def set_layers_description(cls, dlayers):
        """
        Override of NeuralNetwork method to make it static.
        dlayers will be used as a static attribute of GANeuralNetwork class
        """
        cls.dlayers = dlayers

    @classmethod
    def get_random_params(cls):
        return super().get_random_params_custom(cls)


class NeuralGA(GeneticAlgorithm):
    def __init__(self, dlayers, iterations, **kwargs):
        GANeuralNetwork.set_layers_description(dlayers)
        super().__init__(GANeuralNetwork, iterations, **kwargs)
