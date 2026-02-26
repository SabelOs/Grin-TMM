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
        crossover_fraction = 0.8,
        stall_generations=20,
        stall_increase_mutation_factor_thickness=2,
        stall_increase_mutation_factor_volume_fraction = 2,
        stall_increase_crossover_fraction=2,
        RMSE_convergence_threshold=0.01,
    ):
        self.fitness_fn = fitness_fn
        self.n_params = n_params
        
        if isinstance(bounds_thickness[0], (list, tuple)):
            self.thickness_bounds = bounds_thickness
        else:
            self.thickness_bounds = None
            self.tmin, self.tmax = bounds_thickness
        
        self.f_bounds = bounds_fraction
        self.population_size = population_size
        self.device = device

        self.mutation_rate = mutation_rate
        self.base_mutation_rate = mutation_rate

        self.crossover_fraction = crossover_fraction
        self.base_crossover_fraction = crossover_fraction

        self.elite_fraction = elite_fraction

        self.stall_generations = stall_generations
        self.stall_increase_mutation_factor_thickness = stall_increase_mutation_factor_thickness
        self.stall_increase_mutation_factor_volume_fraction = stall_increase_mutation_factor_volume_fraction
        self.stall_increase_crossover_fraction = stall_increase_crossover_fraction

        self.RMSE_convergence_threshold = RMSE_convergence_threshold

        self.mutation_scale_volume_fraction = mutation_scale_volume_fraction
        self.mutation_scale_thickness = mutation_scale_thickness
        self.boosted = False
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
        for i in range(n_layers):
            if self.thickness_bounds is not None:
                tmin, tmax = self.thickness_bounds[i]
            else:
                tmin, tmax = self.tmin, self.tmax

            x[i] = x[i].clamp(tmin, tmax)

        # --- Volume fraction bounds ---
        for i, (fmin, fmax) in enumerate(self.f_bounds):
            x[n_layers + i] = x[n_layers + i].clamp(fmin, fmax)

            # --- Physical coupling ---
            if x[i] < 0.1:
                x[n_layers + i] = 0.0
        return x

    def initialize(self, d_init, f_init):
        """
        Initialize population:
        - First individual is exactly the initial guess
        - Remaining individuals are uniformly random within bounds
        """
        self.population = []

        base = torch.cat([d_init, f_init]).to(self.device)
        n_layers = len(self.f_bounds)

        # --- First individual: exact initial guess ---
        self.population.append(self._project_bounds(base.clone()))

        # --- Remaining individuals: random within bounds ---
        for _ in range(self.population_size - 1):
            ind = torch.empty(self.n_params, device=self.device)

            # Thickness genes
            for i in range(n_layers):
                if self.thickness_bounds is not None:
                    tmin, tmax = self.thickness_bounds[i]
                else:
                    tmin, tmax = self.tmin, self.tmax

                ind[i] = torch.rand((), device=self.device) * (tmax - tmin) + tmin

            # Volume fraction genes (each with its own bounds)
            for i, (fmin, fmax) in enumerate(self.f_bounds):
                if ind[i] > 0.1:
                    ind[n_layers + i] = torch.rand(1, device=self.device) * (fmax - fmin) + fmin
                else:
                    ind[n_layers + i] = fmin

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

        elite_stack = torch.stack(elites)  # shape: (n_elite, n_params)

        while len(new_population) < self.population_size:
            # --- Select primary elite parent ---
            primary_idx = torch.randint(0, n_elite, (), device=self.device)
            child = elite_stack[primary_idx].clone()

            # --- Gene-wise crossover ---
            # --- effective crossover fraction ---
            if self.boosted:
                crossover_fraction_eff = min(
                    self.base_crossover_fraction * self.stall_increase_crossover_fraction,
                    1.0,
                )
            else:
                crossover_fraction_eff = self.base_crossover_fraction

            crossover_rolls = torch.rand(self.n_params, device=self.device)
            crossover_mask = crossover_rolls > crossover_fraction_eff

            if crossover_mask.any():
                secondary_indices = torch.randint(
                    0, n_elite, (crossover_mask.sum(),), device=self.device
                )
                gene_indices = torch.nonzero(crossover_mask, as_tuple=False).squeeze(1)

                child[gene_indices] = elite_stack[
                    secondary_indices, gene_indices
                ]

            # --- Gene-wise mutation ---
            # --- effective mutation rate ---
            if self.boosted:
                max_mutation_size_thickness = self.mutation_scale_thickness * self.stall_increase_mutation_factor_thickness
                max_mutation_size_volume_fraction = self.mutation_scale_volume_fraction * self.stall_increase_mutation_factor_volume_fraction
            else:
                max_mutation_size_thickness = self.mutation_scale_thickness
                max_mutation_size_volume_fraction = self.mutation_scale_volume_fraction

            mutation_mask = torch.rand(self.n_params, device=self.device) < self.base_mutation_rate


            # Thickness mutations
            if mutation_mask[:n_layers].any():
                idx = mutation_mask[:n_layers]
                child[:n_layers][idx] += (
                    torch.randn(idx.sum(), device=self.device)
                    * max_mutation_size_thickness
                )

            # Volume fraction mutations — ONLY if thickness >= 0.1
            for i in range(n_layers):
                if child[i] >= 0.1 and mutation_mask[i]: #mutation individually mutation_mask[n_layers + i]:
                    child[n_layers + i] += (
                        torch.randn((), device=self.device)
                        * max_mutation_size_volume_fraction 
                    )

            child = self._project_bounds(child)
            new_population.append(child)

        self.population = new_population[:self.population_size]



    def run(self, generations):
        best_rmse = float("inf")
        stall_counter = 0

        for g in range(generations):
            self.step()

            best = torch.argmin(self.fitness)
            current_rmse = self.fitness[best].item()
            best_ind = self.population[best]

            # --- RMSE convergence stopping ---
            if current_rmse <= self.RMSE_convergence_threshold:
                print(
                    f"✓ RMSE threshold reached at gen {g:03d} "
                )
                self.boosted = False
                n_layers = len(self.f_bounds)
                self.print_fitting_results(g, best_ind[:n_layers], best_ind[n_layers:],current_rmse)
                return best_ind.detach()

            # --- stall detection ---
            if abs(current_rmse - best_rmse) < 1e-12:
                stall_counter += 1
            else:
                stall_counter = 0
                best_rmse = current_rmse
                self.boosted = False  # ← clean reset on improvement

            # --- mid-stall boost ---
            if (
                stall_counter >= self.stall_generations // 2
                and not self.boosted
            ):
                self.boosted = True
                print(
                    f"⚠ Stall detected at gen {g:03d} — "
                    f"enabling boosted mutation & crossover"
                )

            # --- full stall stop ---
            if stall_counter >= self.stall_generations:
                print(
                    f"⏹ GA stalled for {self.stall_generations} generations "
                )
                self.boosted = False
                n_layers = len(self.f_bounds)
                self.print_fitting_results(g, best_ind[:n_layers], best_ind[n_layers:],current_rmse)
                return best_ind.detach()

            # --- final-generation print ---
            if g == generations - 1:
                """
                n_layers = len(self.f_bounds)
                d_vals = best_ind[:n_layers]
                f_vals = best_ind[n_layers:]

                d_str = ", ".join([f"{d:.2f}" for d in d_vals])
                f_str = ", ".join([f"{f:.4f}" for f in f_vals])

                print(
                    f"GA Gen {g:03d} | RMSE={current_rmse:.4f} | "
                    f"d = [{d_str}] | f = [{f_str}]"
                )"""
                n_layers = len(self.f_bounds)
                self.print_fitting_results(g, best_ind[:n_layers], best_ind[n_layers:],current_rmse)

        best = torch.argmin(self.fitness)
        self.boosted = False
        return self.population[best].detach()

    
    def inject_elites(self, guesses):
        """
        guesses: list of tensors shaped (n_params,)
        Inject guesses into the population (replacing worst individuals).
        """
        guesses = [
            self._project_bounds(g.to(self.device))
            for g in guesses
        ]

        # Evaluate current population
        self.evaluate()
        idx = torch.argsort(self.fitness, descending=True)

        for i, g in enumerate(guesses):
            if i >= len(idx):
                break
            self.population[idx[i]] = g

    def print_fitting_results(self, g, d_vals, f_vals, current_rmse):
        d_str = ", ".join([f"{d:.2f}" for d in d_vals])
        f_str = ", ".join([f"{f:.4f}" for f in f_vals])

        print(
            f"GA Gen {g:03d} | RMSE={current_rmse:.4f} | "
            f"d = [{d_str}] | f = [{f_str}]"
            )