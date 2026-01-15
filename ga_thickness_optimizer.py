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
        #define thickness and volume fraction bounds

    
    def _project_bounds(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enforce physical bounds on genome:
        [ thicknesses | volume fractions ]
        """
        x = x.clone()

        n_layers = int(self.n_params/2) #divide by 2 as there are allways volume fractiona and layerthickness per layer
        n_fractions = len(self.f_bounds)  # number of volume-fraction genes
        
        # --- Thickness bounds ---
        x[:n_layers] = x[:n_layers].clamp(self.tmin, self.tmax)

        # --- Volume fraction bounds ---
        for i in range(n_fractions):
            fmin, fmax = self.f_bounds[i]
            x[n_layers + i] = x[n_layers + i].clamp(fmin, fmax)
        return x

    def initialize(self, d_init, f_init):
        """
        Initialize population:
        - First individual is exactly the initial guess
        - Remaining individuals are noisy variants
        """
        self.population = []

        base = torch.cat([d_init, f_init]).to(self.device)
        n_layers = len(self.f_bounds)

        # --- First individual: exact initial guess ---
        self.population.append(self._project_bounds(base.clone()))

        # --- Remaining individuals: noisy variants ---
        for _ in range(self.population_size - 1):
            ind = base.clone()

            # Thickness noise
            ind[:n_layers] += (
                torch.randn(n_layers, device=self.device)
                * self.mutation_scale_thickness
            )

            # Volume fraction noise
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
            # --- Select two parents ---
            parent1 = elites[np.random.randint(len(elites))]
            parent2 = elites[np.random.randint(len(elites))]

            # --- Gene-wise crossover ---
            mask = torch.rand(self.n_params, device=self.device) < 0.5
            child = torch.where(mask, parent1, parent2).clone()

            # --- Individual-level mutation ---
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
            
            #Print the info of the best species
            best = torch.argmin(self.fitness)
            best_ind = self.population[best]

            n_layers = len(self.f_bounds)
            d_vals = best_ind[:n_layers]
            f_vals = best_ind[n_layers:]

            d_str = ", ".join([f"{d:.2f}" for d in d_vals])
            f_str = ", ".join([f"{f:.4f}" for f in f_vals])

            print(
                f"GA Gen {g:03d} | RMSE={self.fitness[best]:.4} | "
                f"d = [{d_str}] | f = [{f_str}]"
            )

        best = torch.argmin(self.fitness)
        return self.population[best].detach()
