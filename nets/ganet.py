"""
Network that learns through genetic algorithms.

This module contains generic classes for genetic algorithms and a
neural network specialization which updates its weights with them.
"""

from time import time
from typing import List, Type

import numpy as np
from numpy.random import rand, randn

import mnist
from .nnet import NeuralNetwork

BATCH_SIZE = 10

# Generic Genetic Algorithm Classes

class Subject:
    @classmethod
    def create_random(cls):
        raise NotImplementedError

    @property
    def genome(self) -> List:
        raise NotImplementedError

    @property
    def fitness(self) -> float:
        raise NotImplementedError

    @staticmethod
    def mutate(gen):
        raise NotImplementedError


class GeneticAlgorithm:
    # TODO: Parallelize

    population: List[Subject]
    mutation_rate: float
    crossover_rate: float
    SubjectClass: Type[Subject]

    def __init__(
        self,
        SubjectClass,
        iterations,
        size=8,
        mutation_rate=(5 / 1000),
        crossover_rate=0.7,
    ):
        assert size % 2 == 0
        self.SubjectClass = SubjectClass
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.initialize_population(size)
        self.run(iterations)

    def run(self, iterations):
        popsize = len(self.population)
        for i in range(iterations):
            if i % 100 == 0:
                self.dump_round_stats(i, iterations)
            if i % 5000 == 4999:
                self.population[0].test()
            self.roulette_sort()
            self.population += self.best_half_offspring()
            self.population += self.best_half_offspring()
            self.fitness_sort()
            self.population[popsize:] = []  # Keep only the fitest

    def dump_round_stats(self, round, max):
        "Prints a round summary"
        print(f"Round {round + 1}/{max} - Average: {self.get_average_costs()}")
        for i, s in enumerate(self.population):
            print(f" | [{i + 1}] - {s.fitness}")

    def get_average_costs(self):
        return np.mean([s.fitness for s in self.population])

    @property
    def best(self):
        return self.population[0].genome

    def initialize_population(self, size) -> None:
        S = self.SubjectClass
        self.population = [S.create_random() for _ in range(size)]

    def roulette_sort(self) -> None:
        "Reorders population list based on fitness and luck"
        self.population.sort(key=lambda s: s.fitness * rand(), reverse=True)

    def best_half_offspring(self) -> List[Subject]:
        """
        Assumes self.population is ordered from the most to the less fit
        Returns offspring of length (len(p) / 4) of the best p=population half
        """
        p = self.population  # alias
        return [self.crossover(p[i], p[i + 1]) for i in range(0, len(p) // 2, 2)]

    def fitness_sort(self) -> None:
        "Reorders population list based on fitness"
        self.population.sort(key=lambda s: s.fitness, reverse=True)

    def crossover(self, p, q):  # -> Subject
        "Makes a new subject out of p and q, with p being the fitest parent"
        S = self.SubjectClass
        genome_length = len(p.genome)
        gen_choice = rand(genome_length) < self.crossover_rate
        will_mutate = rand(genome_length) < self.mutation_rate
        child_genome = np.where(gen_choice, p.genome, q.genome)
        child_genome[will_mutate] = S.mutate(child_genome[will_mutate])
        return S(child_genome)


# Neural Network


class GaNet(Subject, NeuralNetwork):

    dlayers: List

    __mnist_db = mnist.load(batch_size=BATCH_SIZE)
    __batch_gen = (batch for epoch in __mnist_db for batch in epoch)

    _genome: List[float]
    _fitness: float

    def __init__(self, params):
        """
        Precondition: use set_layers_description() before any instanciation
        so dlayers is initialized
        """
        super().__init__(GaNet.dlayers, BATCH_SIZE, params=params)
        self._genome = params
        self._fitness = None

    @classmethod
    def create_random(cls):
        return GaNet(cls.get_random_params())

    @property
    def genome(self) -> List[float]:
        return self._genome

    @property
    def fitness(self) -> float:
        if not self._fitness:
            self._fitness = -self.batch_cost()
        return self._fitness

    def batch_cost(self):
        random_batch = next(GaNet.__batch_gen)
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
        print(f"[E] {train_avgloss=:.6f} {train_hitrate=:.2f}% {test_time=:.2f}s")

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
        GaNet.set_layers_description(dlayers)
        super().__init__(GaNet, iterations, **kwargs)
