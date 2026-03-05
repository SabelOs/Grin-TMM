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
        n_layers,
        bounds_thickness,
        bounds_fraction,
        inclusions_per_layer,
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
        smart_mutation_scaling = False,
    ):
        self.fitness_fn = fitness_fn
        self.n_layers = n_layers
        
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

        self.smart_mutation_scaling = smart_mutation_scaling
        self.inclusions_per_layer = torch.tensor(inclusions_per_layer)
        self.n_fractions = int(self.inclusions_per_layer.sum().item())
        self.n_genes = self.n_layers + self.n_fractions

        #define thickness and volume fraction bounds

    
    def _project_bounds(self, x: torch.Tensor) -> torch.Tensor:

        x = x.clone()

        # --- Thickness bounds ---
        for i in range(self.n_layers):
            if self.thickness_bounds is not None:
                tmin, tmax = self.thickness_bounds[i]
            else:
                tmin, tmax = self.tmin, self.tmax

            x[i] = x[i].clamp(tmin, tmax)

        # --- Volume fraction bounds ---
        frac_offset = 0

        for layer_idx, n_inc in enumerate(self.inclusions_per_layer):
            for j in range(int(n_inc)):
                fmin, fmax = self.f_bounds[frac_offset]

                gene_idx = self.n_layers + frac_offset
                x[gene_idx] = x[gene_idx].clamp(fmin, fmax)

                # physical coupling
                if x[layer_idx] < 0.1:
                    x[gene_idx] = 0.0

                frac_offset += 1

        return x

    def initialize(self, d_init, f_init):
        """
        Initialize population:
        - First individual is exactly the initial guess
        - Remaining individuals are uniformly random within bounds
        """
        self.population = []

        base = torch.cat([d_init, f_init]).to(self.device)
        n_layers = self.n_layers

        # --- First individual: exact initial guess ---
        self.population.append(self._project_bounds(base.clone()))

        # --- Remaining individuals: random within bounds ---
        for _ in range(self.population_size - 1):
            ind = torch.empty(self.n_genes, device=self.device)

            # Thickness genes
            for i in range(n_layers):
                if self.thickness_bounds is not None:
                    tmin, tmax = self.thickness_bounds[i]
                else:
                    tmin, tmax = self.tmin, self.tmax

                ind[i] = torch.rand((), device=self.device) * (tmax - tmin) + tmin

            # Volume fraction genes
            frac_offset = 0

            for layer_idx, n_inc in enumerate(self.inclusions_per_layer):
                for j in range(int(n_inc)):
                    fmin, fmax = self.f_bounds[frac_offset]
                    gene_idx = self.n_layers + frac_offset

                    if ind[layer_idx] > 0.1:
                        ind[gene_idx] = (
                            torch.rand((), device=self.device) * (fmax - fmin) + fmin
                        )
                    else:
                        ind[gene_idx] = 0.0

                    frac_offset += 1

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
        n_layers = self.n_layers

        elite_stack = torch.stack(elites)  # shape: (n_elite, n_params)

        # -------------------------------------------------
        # SMART MUTATION SCALING (computed once per generation)
        # -------------------------------------------------
        if self.smart_mutation_scaling:

            # Use best elite (index 0 because fitness is sorted)
            best_individual = elites[0]


            # Minimum mutation amplitudes
            min_thickness_scale = 1.0        # nm
            min_fraction_scale = 0.02

            # --- Thickness scaling ---
            thickness_scales = torch.zeros(self.n_layers, device=self.device)
            for i in range(self.n_layers):
                if self.thickness_bounds is not None:
                    tmin, _ = self.thickness_bounds[i]
                else:
                    tmin = self.tmin

                raw_scale = (best_individual[i] - tmin) / 6.0

                # Clamp to minimum
                thickness_scales[i] = torch.maximum(
                    raw_scale,
                    torch.tensor(min_thickness_scale, device=self.device)
                )

            # --- Volume fraction scaling ---
            fraction_scales = torch.zeros(self.n_fractions, device=self.device)

            for i, (fmin, _) in enumerate(self.f_bounds):
                raw_scale = (
                    best_individual[self.n_layers + i] - fmin
                ) / 6.0

                fraction_scales[i] = torch.maximum(
                    raw_scale,
                    torch.tensor(0.02, device=self.device)
                )

        else:
            thickness_scales = None
            fraction_scales = None

        while len(new_population) < self.population_size:
            # --- Select primary elite parent ---
            primary_idx = torch.randint(0, n_elite, (), device=self.device)
            child = elite_stack[primary_idx].clone()

                        # --- Gene-wise mutation ---
            # --- effective mutation rate ---
            if self.boosted:
                max_mutation_size_thickness = (
                    self.mutation_scale_thickness
                    * self.stall_increase_mutation_factor_thickness
                )
                max_mutation_size_volume_fraction = (
                    self.mutation_scale_volume_fraction
                    * self.stall_increase_mutation_factor_volume_fraction
                )
            else:
                max_mutation_size_thickness = self.mutation_scale_thickness
                max_mutation_size_volume_fraction = self.mutation_scale_volume_fraction

            mutation_mask = (
                torch.rand(self.n_genes, device=self.device)
                < self.base_mutation_rate
            )

            # Thickness mutations
            if mutation_mask[:self.n_layers].any():
                idx = mutation_mask[:self.n_layers]

                if self.smart_mutation_scaling:
                    local_scales = thickness_scales[idx]
                    child[:self.n_layers][idx] += (
                        torch.randn(idx.sum(), device=self.device)
                        * local_scales
                        * (
                            self.stall_increase_mutation_factor_thickness
                            if self.boosted
                            else 1.0
                        )
                    )
                else:
                    child[:self.n_layers][idx] += (
                        torch.randn(idx.sum(), device=self.device)
                        * max_mutation_size_thickness
                    )

            # Volume fraction mutations — ONLY if thickness >= 0.1
            frac_offset = 0

            for layer_idx, n_inc in enumerate(self.inclusions_per_layer):

                for j in range(n_inc):
                    gene_idx = self.n_layers + frac_offset

                    if child[layer_idx] >= 0.1 and mutation_mask[gene_idx]:

                        if self.smart_mutation_scaling:
                            local_scale = fraction_scales[frac_offset]
                            boost_factor = (
                                self.stall_increase_mutation_factor_volume_fraction
                                if self.boosted
                                else 1.0
                            )

                            child[gene_idx] += (
                                torch.randn((), device=self.device)
                                * local_scale
                                * boost_factor
                            )
                        else:
                            child[gene_idx] += (
                                torch.randn((), device=self.device)
                                * max_mutation_size_volume_fraction
                            )

                    frac_offset += 1

            child = self._project_bounds(child)
            new_population.append(child)

        self.population = new_population[:self.population_size]



    def run(self, generations):
        best_rmse = float("inf")
        stall_counter = 0

        n_thickness = self.n_layers
        n_fraction = int(self.inclusions_per_layer.sum())

        for g in range(generations):
            self.step()

            best = torch.argmin(self.fitness)
            current_rmse = self.fitness[best].item()
            best_ind = self.population[best]

            # --- RMSE convergence stopping ---
            if current_rmse <= self.RMSE_convergence_threshold:
                print(f"✓ RMSE threshold reached at gen {g:03d}")
                self.boosted = False

                d_vals = best_ind[:n_thickness]
                f_vals = best_ind[n_thickness:n_thickness + n_fraction]

                self.print_fitting_results(g, d_vals, f_vals, current_rmse)
                return best_ind.detach()

            # --- stall detection ---
            if abs(current_rmse - best_rmse) < 1e-12:
                stall_counter += 1
            else:
                stall_counter = 0
                best_rmse = current_rmse
                self.boosted = False  # reset on improvement

            # --- mid-stall boost ---
            if stall_counter >= self.stall_generations // 2 and not self.boosted:
                self.boosted = True
                print(
                    f"⚠ Stall detected at gen {g:03d} — "
                    f"enabling boosted mutation & crossover"
                )

            # --- full stall stop ---
            if stall_counter >= self.stall_generations:
                print(
                    f"⏹ GA stalled for {self.stall_generations} generations"
                )
                self.boosted = False

                d_vals = best_ind[:n_thickness]
                f_vals = best_ind[n_thickness:n_thickness + n_fraction]

                self.print_fitting_results(g, d_vals, f_vals, current_rmse)
                return best_ind.detach()

            # --- final-generation print ---
            if g == generations - 1:
                d_vals = best_ind[:n_thickness]
                f_vals = best_ind[n_thickness:n_thickness + n_fraction]

                self.print_fitting_results(g, d_vals, f_vals, current_rmse)

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

        print(f"\nGA Gen {g:03d} | RMSE={current_rmse:.4}")
        print("-------------------------------------------------")

        frac_offset = 0

        for layer_idx in range(self.n_layers):

            d = d_vals[layer_idx].item()
            print(f"Layer {layer_idx+1}: thickness = {d:.2f} nm")

            n_inc = int(self.inclusions_per_layer[layer_idx].item())

            if n_inc == 0:
                print("   No inclusions")
            else:
                for j in range(n_inc):
                    f_val = f_vals[frac_offset].item()
                    print(f"   Inclusion {j+1}: f = {f_val:.2f}")
                    frac_offset += 1

        print("-------------------------------------------------\n")