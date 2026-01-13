#%% ================== Imports =====================
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from matplotlib.colors import TwoSlopeNorm

from winspec import SpeFile
import tmm_fast.gym_multilayerthinfilm as mltf
from tmm_fast import coh_tmm
from ga_thickness_optimizer import GeneticThicknessOptimizer

#%% ================== Functions =====================
def moving_average_same(a, n=5):
    return uniform_filter1d(a, size=n, mode="reflect")

#%% ================= USER SETTINGS =================
path = str(Path(__file__).parent)

# -------- Material nk files --------
pathCu   = path + "/OpticalConstants/nk_Cu.txt"
pathCu2O = path + "/OpticalConstants/nk_Cu2O.txt"
pathCuO  = path + "/OpticalConstants/nk_CuO.txt"
material_path_list = [pathCu, pathCu2O, pathCuO]
material_names = ["Cu", "Cu2O", "CuO"]

# -------- SPE files --------
SPE_file  = path + "/GRIN.SPE"
Lamp_file = path + "/Substrate.SPE"

# -------- Initial thickness for LAST spectrum (nm) --------
init_thickness_nm = [22.0, 1.0, 1.0]

# -------- Smoothing --------
enable_smoothing = True
smoothing_window = 5

# -------- Wavelength cut --------
enable_wl_cut = True
wl_opt_min = 500.0
wl_opt_max = 950.0

# -------- Optimization --------
n_steps = 600
learning_rate = 0.5
printing_interval = 300

# -------- Angular averaging --------
angle_min_deg = 0.0
angle_max_deg = 1.0 #np.arcsin(0.55) * 180/np.pi for objective in reality
n_angles = 1
#===================================================

#%% ================= Load SPE data =================
wl_nm = SpeFile(SPE_file).xaxis.astype(np.float64)
df_intensities = SpeFile(SPE_file).data[:, :, 0]
lamp_intensities = SpeFile(Lamp_file).data[0, :, 0]

n_spectra = df_intensities.shape[0]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#%% ================= Preprocessing =================
if enable_smoothing:
    for i in range(n_spectra):
        df_intensities[i] = moving_average_same(df_intensities[i], smoothing_window)
    lamp_intensities = moving_average_same(lamp_intensities, smoothing_window)

df_transmission = df_intensities / lamp_intensities

if enable_wl_cut:
    wl_mask = (wl_nm >= wl_opt_min) & (wl_nm <= wl_opt_max)
    wl_nm = wl_nm[wl_mask]
    df_transmission = df_transmission[:, wl_mask]

n_wl = wl_nm.size
lambda_nm = torch.tensor(wl_nm, dtype=torch.float64, device=device)

#%% ================= Refractive indices =================
N_np = mltf.get_N(
    material_path_list,
    wl_nm.min(),
    wl_nm.max(),
    points=n_wl,
    complex_n=True
)

n_stack_np = np.vstack([
    np.ones_like(wl_nm),
    N_np[0],
    N_np[1],
    N_np[2],
    np.ones_like(wl_nm)
])

N_torch = torch.tensor(
    n_stack_np,
    dtype=torch.complex128,
    device=device
).unsqueeze(0)

#%% ================= Angles =================
thetas = torch.deg2rad(
    torch.linspace(angle_min_deg, angle_max_deg, n_angles,
                   dtype=torch.float64, device=device)
)

#%% ================= Storage containers =================
records = []   # list of dicts → pandas DataFrame

current_init = init_thickness_nm.copy()

#%% ================= Sequential optimization (GENETIC ALGORITHM) =================
for spec_no in range(n_spectra - 1, -1, -1):

    print(f"\n=== GA fitting spectrum {spec_no} ===")

    target_T = torch.tensor(
        df_transmission[spec_no],
        dtype=torch.float64,
        device=device
    )

    # ------------------------------------------------------------
    # Fitness functions
    # ------------------------------------------------------------
    def fitness_fn_ga(d_layers_nm: torch.Tensor) -> float:
        """GA version: returns float"""
        with torch.no_grad():
            loss = fitness_fn_torch(d_layers_nm)
        return loss.item()


    def fitness_fn_torch(d_layers_nm: torch.Tensor) -> torch.Tensor:
        """
        Differentiable RMSE for gradient descent
        """
        d_full = torch.cat([
            torch.tensor([np.inf], device=device),
            d_layers_nm,
            torch.tensor([np.inf], device=device),
        ])

        result = coh_tmm(
            pol="s",
            N=N_torch,
            T=d_full.unsqueeze(0),
            Theta=thetas,
            lambda_vacuum=lambda_nm,
            device=device
        )

        T_sim = result["T"][0].mean(dim=0)
        rmse = torch.sqrt(torch.mean((T_sim - target_T) ** 2))
        return rmse

    # ------------------------------------------------------------
    # Initialize GA
    # ------------------------------------------------------------
    ga = GeneticThicknessOptimizer(
        fitness_fn=fitness_fn_ga,
        n_layers=3,
        population_size=60,
        mutation_rate=0.25,
        mutation_scale=5.0,     # nm
        crossover_rate=0.7,
        elite_fraction=0.2,
        thickness_bounds=(1e-3, 300.0),
        device=device,
        dtype=torch.float64,
    )

    ga.initialize_population(
        init_guess=torch.tensor(current_init, device=device)
    )

    # ------------------------------------------------------------
    # Run GA
    # ------------------------------------------------------------
    best_thickness = ga.run(
        n_generations=50,
        verbose=True
    )

    # ------------------------------------------------------------
    # Gradient-based refinement (Adam)
    # ------------------------------------------------------------
    d_opt = torch.nn.Parameter(
        best_thickness.clone().detach()
    )

    optimizer = torch.optim.Adam(
        [d_opt],
        lr=0.05   # good starting point for nm-scale problems
    )

    n_refine_steps = 200

    for step in range(n_refine_steps):
        optimizer.zero_grad()

        loss = fitness_fn_torch(d_opt)
        loss.backward()
        optimizer.step()

        # Hard physical bounds
        with torch.no_grad():
            d_opt.clamp_(1e-3, 300.0)

        if step % 50 == 0:
            print(
                f"  Adam step {step:3d} | RMSE = {loss.item():.4e} | "
                f"d = {d_opt.detach().cpu().numpy()}"
            )

    best_thickness = d_opt.detach()

    # ------------------------------------------------------------
    # Final forward pass for storage
    # ------------------------------------------------------------
    d_nm = torch.cat([
        torch.tensor([np.inf], device=device),
        best_thickness,
        torch.tensor([np.inf], device=device),
    ])

    result = coh_tmm(
        pol="s",
        N=N_torch,
        T=d_nm.unsqueeze(0),
        Theta=thetas,
        lambda_vacuum=lambda_nm,
        device=device
    )

    T_sim = result["T"][0].mean(dim=0)

    # ------------------------------------------------------------
    # Save results (UNCHANGED FORMAT)
    # ------------------------------------------------------------
    T_sim_np = T_sim.detach().cpu().numpy()
    thicknesses = best_thickness.detach().cpu().numpy()
    rmse_val = fitness_fn_ga(best_thickness)

    for i_wl, wl in enumerate(wl_nm):
        records.append({
            "spectrum": spec_no,
            "wavelength_nm": wl,
            "T_exp": df_transmission[spec_no, i_wl],
            "T_fit": T_sim_np[i_wl],
            "Cu_nm": thicknesses[0],
            "Cu2O_nm": thicknesses[1],
            "CuO_nm": thicknesses[2],
            "RMSE": rmse_val
        })

    # ------------------------------------------------------------
    # Warm start next spectrum
    # ------------------------------------------------------------
    current_init = thicknesses.tolist()


#%% ================= Save results =================
df_results = pd.DataFrame.from_records(records)

out_base = Path(path) / "tmm_fit_results"
df_results.to_pickle(out_base.with_suffix(".pkl"))
df_results.to_csv(out_base.with_suffix(".csv"), index=False)

print("\nSaved results to:")
print(out_base.with_suffix(".pkl"))
print(out_base.with_suffix(".csv"))

""" #%% ================= Example plot =================
last_spec = n_spectra - 1
df_last = df_results[df_results["spectrum"] == last_spec]

plt.figure()
plt.plot(df_last["wavelength_nm"], df_last["T_fit"], "r--", label="TMM fit")
plt.plot(df_last["wavelength_nm"], df_last["T_exp"], "k", label="Measured")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmittance")
plt.legend()
plt.tight_layout()
plt.savefig("comparison_last_spectrum.png", dpi=300)
plt.show()

# %%
# RMSE vs spectrum
df_results.groupby("spectrum")["RMSE"].first().plot()

# Thickness evolution
df_results.groupby("spectrum")[["Cu_nm","Cu2O_nm","CuO_nm"]].first().plot()

# Residuals
df_results["residual"] = df_results["T_fit"] - df_results["T_exp"]


# ================= Prepare 2D grids =================
# Sort to be safe
df = df_results.sort_values(["spectrum", "wavelength_nm"])

spectra = df["spectrum"].unique()
wavelengths = df["wavelength_nm"].unique()

n_spec = spectra.size
n_wl = wavelengths.size

# Reshape into 2D arrays: (spectrum, wavelength)
T_exp_2d = df["T_exp"].values.reshape(n_spec, n_wl)
T_fit_2d = df["T_fit"].values.reshape(n_spec, n_wl)
T_diff_2d = T_fit_2d - T_exp_2d

# ================= Plot settings =================
extent = [
    wavelengths.min(), wavelengths.max(),
    spectra.min(), spectra.max()
]

aspect = "auto"
origin = "lower"

# ================= 1) Experimental transmission =================
plt.figure(figsize=(8, 5))
plt.imshow(
    T_exp_2d,
    extent=extent,
    aspect=aspect,
    origin=origin,
    cmap="viridis"
)
plt.colorbar(label="Transmission (exp)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Spectrum index")
plt.title("Experimental transmission")
plt.tight_layout()
plt.savefig("T_exp_2D.png", dpi=300)
plt.show()

# ================= 2) Simulated transmission =================
plt.figure(figsize=(8, 5))
plt.imshow(
    T_fit_2d,
    extent=extent,
    aspect=aspect,
    origin=origin,
    cmap="viridis"
)
plt.colorbar(label="Transmission (fit)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Spectrum index")
plt.title("Simulated transmission (TMM fit)")
plt.tight_layout()
plt.savefig("T_fit_2D.png", dpi=300)
plt.show()

# ================= 3) Difference (fit - exp) =================
# Diverging colormap with white at zero
vmax = np.max(np.abs(T_diff_2d))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

plt.figure(figsize=(8, 5))
plt.imshow(
    T_diff_2d,
    extent=extent,
    aspect=aspect,
    origin=origin,
    cmap="seismic",   # temperature-like, white at 0
    norm=norm
)
plt.colorbar(label="Δ Transmission (fit − exp)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Spectrum index")
plt.title("Transmission difference (fit − experiment)")
plt.tight_layout()
plt.savefig("T_diff_2D.png", dpi=300)
plt.show()

# %%
 """