import torch
import numpy as np
from typing import Callable, Tuple


class GeneticThicknessOptimizer:
    """
    Generic genetic algorithm for thin-film thickness optimization.

    Each individual is a vector of layer thicknesses [nm].
    Fitness is computed via a user-supplied callable.
    """

    def __init__(
        self,
        fitness_fn: Callable[[torch.Tensor], float],
        n_layers: int,
        population_size: int = 40,
        mutation_rate: float = 0.2,
        mutation_scale: float = 5.0,
        crossover_rate: float = 0.7,
        elite_fraction: float = 0.2,
        thickness_bounds: Tuple[float, float] = (1e-3, 300.0),
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        fitness_fn
            Callable that takes a (n_layers,) tensor and returns scalar fitness
            (lower is better, e.g. RMSE).
        n_layers
            Number of layers to optimize.
        population_size
            Number of individuals per generation.
        mutation_rate
            Probability of mutating each gene.
        mutation_scale
            Std-dev of Gaussian mutation [nm].
        crossover_rate
            Probability of performing crossover.
        elite_fraction
            Fraction of best individuals kept unchanged.
        thickness_bounds
            (min_nm, max_nm)
        """
        self.fitness_fn = fitness_fn
        self.n_layers = n_layers
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        self.tmin, self.tmax = thickness_bounds
        self.device = device
        self.dtype = dtype

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.population = None
        self.fitness = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize_population(self, init_guess: torch.Tensor | None = None):
        """
        Initialize population around init_guess or uniformly random.
        """
        pop = []

        for _ in range(self.population_size):
            if init_guess is None:
                individual = torch.empty(
                    self.n_layers, device=self.device, dtype=self.dtype
                ).uniform_(self.tmin, self.tmax)
            else:
                noise = torch.randn_like(init_guess) * self.mutation_scale
                individual = (init_guess + noise).clamp(self.tmin, self.tmax)

            pop.append(individual)

        self.population = pop

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------
    def evaluate_population(self):
        fitness = []
        for ind in self.population:
            f = self.fitness_fn(ind)
            fitness.append(f)
        self.fitness = torch.tensor(fitness, device=self.device)

    # ------------------------------------------------------------------
    # Selection (elitism + tournament)
    # ------------------------------------------------------------------
    def select_parents(self):
        n_elite = max(1, int(self.elite_fraction * self.population_size))
        elite_idx = torch.argsort(self.fitness)[:n_elite]
        elites = [self.population[i] for i in elite_idx]

        return elites

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------
    def crossover(self, parent1: torch.Tensor, parent2: torch.Tensor):
        if torch.rand(1).item() > self.crossover_rate:
            return parent1.clone(), parent2.clone()

        point = torch.randint(1, self.n_layers, (1,)).item()

        child1 = torch.cat([parent1[:point], parent2[point:]])
        child2 = torch.cat([parent2[:point], parent1[point:]])

        return child1, child2

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def mutate(self, individual: torch.Tensor):
        mask = torch.rand(self.n_layers, device=self.device) < self.mutation_rate
        noise = torch.randn(self.n_layers, device=self.device) * self.mutation_scale
        individual = individual + mask * noise
        return individual.clamp(self.tmin, self.tmax)

    # ------------------------------------------------------------------
    # One generation step
    # ------------------------------------------------------------------
    def step(self):
        self.evaluate_population()
        elites = self.select_parents()

        new_population = elites.copy()

        while len(new_population) < self.population_size:
            idx = torch.randint(0, len(elites), (2,))
            parent1 = elites[idx[0]]
            parent2 = elites[idx[1]]

            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            new_population.extend([child1, child2])

        self.population = new_population[: self.population_size]

    # ------------------------------------------------------------------
    # Run GA
    # ------------------------------------------------------------------
    def run(self, n_generations: int, verbose: bool = True):
        for gen in range(n_generations):
            self.step()

            if verbose:
                best_idx = torch.argmin(self.fitness)
                print(
                    f"GA Gen {gen:3d} | "
                    f"Best RMSE = {self.fitness[best_idx]:.4e} | "
                    f"d = {self.population[best_idx].detach().cpu().numpy()}"
                )

        best_idx = torch.argmin(self.fitness)
        return self.population[best_idx].detach()
