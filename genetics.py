from typing import List, Type
from numpy.random import rand
import numpy as np

class Subject:

    def __init__(self, genome, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def create_random():
        raise NotImplementedError

    # TODO: Memoize genome
    @property
    def genome(self) -> List:
        raise NotImplementedError

    # TODO: Memoize fitness result, it should never change
    @property
    def fitness(self) -> float:
        raise NotImplementedError

    # TODO: Think more about this
    # Maybe a urand in [c +- d] range with c = (min + max) / 2, d = max - min
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
        self, SubjectClass, iterations,
        size=8, mutation_rate=(1 / 1000), crossover_rate=0.7
    ):
        assert size % 2 == 0
        self.SubjectClass = SubjectClass
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.initialize_population(size)
        self.run(iterations)

    def run(self, iterations):
        popsize = len(self.population)
        for _ in range(iterations):
            self.roulette_sort()
            self.population += self.best_half_offspring()
            self.population += self.best_half_offspring()
            self.fitness_sort()
            self.population[popsize:] = [] # Keep only the fitest

    def initialize_population(self, size) -> None:
        S = self.SubjectClass
        self.population = [S.create_random() for _ in range(size)]

    def roulette_sort(self) -> None:
        "Reorders population list based on fitness and luck"
        self.population.sort(key=lambda s: s.fitness * rand())

    def best_half_offspring(self) -> List[Subject]:
        """
        Assumes self.population is ordered from the most to the less fit
        Returns offspring of length (len(p) / 4) of the best p=population half
        """
        p = self.population # alias
        return [self.crossover(p[i], p[i + 1]) for i in range(0, len(p) // 2, 2)]

    def fitness_sort(self) -> None:
        "Reorders population list based on fitness"
        self.population.sort(key=lambda s: s.fitness)

    def crossover(self, p, q) -> Subject:
        "Makes a new subject out of p and q, with p being the fitest parent"
        S = self.SubjectClass
        genome_length = len(p.genome)
        gen_choice = rand(genome_length) < self.crossover_rate
        will_mutate = rand(genome_length) < self.mutation_rate
        child_genome = np.where(gen_choice, p.genome, q.genome)
        child_genome[will_mutate] = S.mutate(child_genome[will_mutate])
        return S(child_genome)
