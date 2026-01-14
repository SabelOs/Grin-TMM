import torch
import numpy as np

class GeneticThicknessOptimizer:
    """
    GA optimizing:
    [ d1, d2, ..., f1, f2, ... ]
    """

    def __init__(
        self,
        fitness_fn,
        n_params,
        bounds_thickness,
        bounds_fraction,
        population_size=40,
        mutation_rate=0.25,
        elite_fraction=0.2,
        device="cpu",
        mutation_scale_volume_fraction = 0.02,
        mutation_scale_thickness = 1,
    ):
        self.fitness_fn = fitness_fn
        self.n_params = n_params
        self.tmin, self.tmax = bounds_thickness
        self.f_bounds = bounds_fraction
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_scale_volume_fraction = mutation_scale_volume_fraction
        self.elite_fraction = elite_fraction
        self.device = device
        self.mutation_scale_thickness = mutation_scale_thickness
    
    def _project_bounds(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enforce physical bounds on genome:
        [ thicknesses | volume fractions ]
        """
        x = x.clone()

        n_layers = len(self.f_bounds)

        # --- Thickness bounds ---
        x[:n_layers] = x[:n_layers].clamp(self.tmin, self.tmax)

        # --- Volume fraction bounds ---
        for i, (fmin, fmax) in enumerate(self.f_bounds):
            idx = n_layers + i
            x[idx] = x[idx].clamp(fmin, fmax)

        return x

    def initialize(self, d_init, f_init):
        """
        Initialize population using individual-level mutation probability.
        """
        self.population = []

        base = torch.cat([d_init, f_init]).to(self.device)
        n_layers = len(self.f_bounds)

        for _ in range(self.population_size):
            ind = base.clone()

            # --- Individual-level mutation roll ---
            if torch.rand(1, device=self.device) < self.mutation_rate:
                # Thickness mutation
                ind[:n_layers] += (
                    torch.randn(n_layers, device=self.device)
                    * self.mutation_scale_thickness
                )
            if torch.rand(1, device=self.device) < self.mutation_rate:
                # Volume fraction mutation
                ind[n_layers:] += (
                    torch.randn(n_layers, device=self.device)
                    * self.mutation_scale_volume_fraction
                )

            ind = self._project_bounds(ind)
            self.population.append(ind)

    def evaluate(self):
        self.fitness = torch.tensor(
            [self.fitness_fn(self._project_bounds(ind)) for ind in self.population],
            device=self.device
        )

    def step(self):
        self.evaluate()

        idx = torch.argsort(self.fitness)
        n_elite = max(1, int(self.elite_fraction * len(idx)))
        elites = [self.population[i] for i in idx[:n_elite]]

        new_population = elites.copy()
        n_layers = len(self.f_bounds)

        while len(new_population) < self.population_size:
            parent = elites[np.random.randint(len(elites))]
            child = parent.clone()

            # --- Individual-level mutation rolls ---
            if torch.rand(1, device=self.device) < self.mutation_rate:
                # Thickness mutation
                child[:n_layers] += (
                    torch.randn(n_layers, device=self.device)
                    * self.mutation_scale_thickness
                )

            if torch.rand(1, device=self.device) < self.mutation_rate:
                # Volume fraction mutation
                child[n_layers:] += (
                    torch.randn(n_layers, device=self.device)
                    * self.mutation_scale_volume_fraction
                )

            child = self._project_bounds(child)
            new_population.append(child)

        self.population = new_population[:self.population_size]


    def run(self, generations):
        for g in range(generations):
            self.step()
            best = torch.argmin(self.fitness)
            print(f"GA Gen {g:03d} | RMSE={self.fitness[best]:.4}")

        best = torch.argmin(self.fitness)
        return self.population[best].detach()
