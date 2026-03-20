#%% ================== Imports =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#%% ================== User settings =====================

# Define your datasets here: "Label": "filename"
datasets = {
    "\sigma = 6": "benchmark_pop50_gen1000_sigma5_mutrate0_3_mutratestall_5_elite_0_1_stall50_crossover0_8_bugfix-added-mutationstallincrease-smartScaling.csv",
    "\sigma = 5": "benchmark_pop100_gen1000_sigma5_mutrate0_1_mutratestall_5_elite_0_1_stall50_crossover0_8_bugfix-added-mutationstallincrease-smartScaling.csv",
    "\sigma = 4": "benchmark_pop50_gen1000_sigma4_mutrate0_3_mutratestall_5_elite_0_1_stall50_crossover0_8_bugfix-added-mutationstallincrease-smartScaling_2.csv",
    "\sigma = 3": "benchmark_pop50_gen1000_sigma3_mutrate0_3_mutratestall_5_elite_0_1_stall50_crossover0_8_bugfix-added-mutationstallincrease-smartScaling_2.csv",
    "\sigma = 2.5": "benchmark_pop50_gen1000_sigma2_5_mutrate0_3_mutratestall_5_elite_0_1_stall50_crossover0_8_bugfix-added-mutationstallincrease-smartScaling_2.csv",
    "\sigma = 2": "benchmark_pop50_gen1000_sigma2_mutrate0_3_mutratestall_5_elite_0_1_stall50_crossover0_8_bugfix-added-mutationstallincrease-smartScaling_2.csv",
    "\sigma = 1 and 6": "benchmark_pop50_gen1000_sigma1_and_6_mutrate0_3_mutratestall_5_elite_0_1_stall50_crossover0_8_bugfix-added-mutationstallincrease-smartScaling_2.csv",
}

additional_folder = ""  # optional subfolder

base_path = Path(__file__).parent / additional_folder

# Output directory
out_dir = base_path / "plots_multi"
out_dir.mkdir(exist_ok=True)

# Choose colormap (good options: viridis, plasma, inferno, cividis)
cmap = plt.cm.viridis
#%% ================== Plot - log=====================
plt.figure(figsize=(6, 4))
n = len(datasets)
for i, (label, filename) in enumerate(datasets.items()):

    file_path = base_path / filename

    # Load data
    df = pd.read_csv(file_path)

    # Sort to be safe
    df = df.sort_values(["spectrum", "wavelength_nm"])

    # Compute RMSE per spectrum
    rmse_per_spec = (
        df.groupby("spectrum")["RMSE"]
        .first()
    )
    # Pick color from gradient
    color = cmap(i / (n - 1)) if n > 1 else cmap(0.5)
                                                 
    # Plot
    plt.plot(
        rmse_per_spec.index,
        rmse_per_spec.values,
        marker="o",
        label=label,
        color = color
    )

#================== Formatting =====================
plt.xlabel("Spectrum index")
plt.ylabel("RMSE")
plt.yscale("log")  # <-- log scale as requested
plt.title("Fit error per spectrum (comparison)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save
plt.savefig(out_dir / "RMSE_vs_spectrum_comparison_log.png", dpi=300)

plt.show()
# %%
#%% ================== Plot =====================
plt.figure(figsize=(6, 4))

for i, (label, filename) in enumerate(datasets.items()):

    file_path = base_path / filename

    # Load data
    df = pd.read_csv(file_path)

    # Sort to be safe
    df = df.sort_values(["spectrum", "wavelength_nm"])

    # Compute RMSE per spectrum
    rmse_per_spec = (
        df.groupby("spectrum")["RMSE"]
        .first()
    )
    # Pick color from gradient
    color = cmap(i / (n - 1)) if n > 1 else cmap(0.5)
                                                 
    # Plot
    plt.plot(
        rmse_per_spec.index,
        rmse_per_spec.values,
        marker="o",
        label=label,
        color = color
    )


#================== Formatting =====================
plt.xlabel("Spectrum index")
plt.ylabel("RMSE")
plt.title("Fit error per spectrum (comparison)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save
plt.savefig(out_dir / "RMSE_vs_spectrum_comparison.png", dpi=300)

plt.show()
# %%
