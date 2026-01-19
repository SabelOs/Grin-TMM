#%% ================== Imports =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import re
from collections import defaultdict
#%% ================== User settings =====================
# Path to results (pickle preferred)
fileName = "sample5_Cu_Cu2O-CuSphere_CuO_50xObj.csv"
results_base = Path(__file__).parent / fileName


use_pickle = True   # set False to use CSV instead

# Output directory for plots
outDirName = "plots_" + fileName[:-4]
out_dir = Path(__file__).parent / outDirName
out_dir.mkdir(exist_ok=True)

#%% ================== Load results =====================
df_results = pd.read_csv(results_base.with_suffix(".csv"))

print("Loaded results:")
print(df_results.head())
print(df_results.columns[10:])
#%% ================== Basic info =====================
spectra = df_results["spectrum"].unique()
wavelengths = df_results["wavelength_nm"].unique()

n_spec = spectra.size
n_wl = wavelengths.size

# Sort to be safe
df = df_results.sort_values(["spectrum", "wavelength_nm"])

#%% ================== 1) Last spectrum comparison =====================
last_spec = spectra.max()
last_spec = 1
df_last = df[df["spectrum"] == last_spec]

plt.figure(figsize=(6, 4))
plt.plot(df_last["wavelength_nm"], df_last["T_exp"], "k", label="Measured")
plt.plot(df_last["wavelength_nm"], df_last["T_fit"], "r--", label="TMM fit")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmittance")
plt.title(f"Spectrum {last_spec}: Fit vs Experiment")
plt.legend()
plt.tight_layout()
#plt.savefig(out_dir / "comparison_last_spectrum.png", dpi=300)
plt.show()

#%% ================== 2) RMSE vs spectrum =====================
plt.figure(figsize=(6, 4))
(
    df.groupby("spectrum")["RMSE"]
    .first()
    .plot(marker="o")
)
plt.xlabel("Spectrum index")
plt.ylabel("RMSE")
plt.title("Fit error per spectrum")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "RMSE_vs_spectrum.png", dpi=300)
plt.show()

#%% ================== 3) Thickness evolution =====================
"""
plt.figure(figsize=(6, 4))
(
    df.groupby("spectrum")[["Cu_nm", "Cu2O_nm", "CuO_nm"]]
    .first()
    .plot()
)
plt.xlabel("Spectrum index")
plt.ylabel("Thickness (nm)")
plt.title("Layer thickness evolution")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "thickness_evolution.png", dpi=300)
plt.show()
"""
addMaterialFractions = False  # <<< USER OPTION

# --- find material indices automatically ---
material_indices = sorted(
    int(m.group(1))
    for c in df.columns
    if (m := re.match(r"material_(\d+)_thickness_nm", c))
)

# --- container for effective thickness contributions ---
# key = material name, value = list of per-spectrum thicknesses
material_thickness = defaultdict(list)

# --- group by spectrum ---
grouped = df.groupby("spectrum").first()

for spectrum, row in grouped.iterrows():

    # temporary storage per spectrum
    spectrum_contrib = defaultdict(float)

    for i in material_indices:
        d = row.get(f"material_{i}_thickness_nm", 0.0)
        if pd.isna(d) or d == 0:
            continue

        # --- matrix ---
        mat_name = row.get(f"material_{i}_name", f"material_{i}")
        f_mat = row.get(f"material_{i}_volume_fraction", 1.0)

        spectrum_contrib[(mat_name, "matrix", i)] += f_mat * d

        # --- inclusion ---
        inc_name = row.get(f"inclusion_{i}_name", None)
        if inc_name and not pd.isna(inc_name):
            f_inc = row.get(f"inclusion_{i}_volume_fraction", 0.0)
            spectrum_contrib[(inc_name, "inclusion", i)] += f_inc * d

    # --- merge or keep separate ---
    if addMaterialFractions:
        merged = defaultdict(float)
        for (name, _, _), val in spectrum_contrib.items():
            merged[name] += val
        for name in merged:
            material_thickness[name].append(merged[name])
    else:
        for key, val in spectrum_contrib.items():
            material_thickness[key].append(val)

# --- build dataframe for plotting ---
plot_df = pd.DataFrame(material_thickness, index=grouped.index)

# --- build legend labels ---
legend_labels = []
for col in plot_df.columns:
    if isinstance(col, tuple):
        name, role, idx = col
        label = f"{name} ({role}, layer {idx})"
    else:
        label = col
    legend_labels.append(label)

# --- plot ---
plt.figure(figsize=(6, 4))
plot_df.plot(ax=plt.gca())

plt.xlabel("Spectrum index")
plt.ylabel("Effective thickness (nm)")
plt.title("Volume-fraction–corrected thickness evolution")
plt.grid(True, alpha=0.3)

plt.legend(
    legend_labels,
    title="Materials",
    fontsize=9,
    title_fontsize=10
)

plt.tight_layout()
plt.savefig(out_dir / "thickness_evolution.png", dpi=300)
plt.show()
#%% ================== 4) Residuals =====================
df["residual"] = df["T_fit"] - df["T_exp"]

#%% ================== Prepare 2D grids =====================
T_exp_2d = df["T_exp"].values.reshape(n_spec, n_wl)
T_fit_2d = df["T_fit"].values.reshape(n_spec, n_wl)
T_diff_2d = T_fit_2d - T_exp_2d

extent = [
    wavelengths.min(), wavelengths.max(),
    spectra.min(), spectra.max()
]

aspect = "auto"
origin = "lower"

#%% ================== 5) Experimental transmission (2D) =====================
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
plt.savefig(out_dir / "T_exp_2D.png", dpi=300)
plt.show()

#%% ================== 6) Simulated transmission (2D) =====================
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
plt.savefig(out_dir / "T_fit_2D.png", dpi=300)
plt.show()

#%% ================== 7) Difference map (fit - exp) =====================
vmax = np.max(np.abs(T_diff_2d))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

plt.figure(figsize=(8, 5))
plt.imshow(
    T_diff_2d,
    extent=extent,
    aspect=aspect,
    origin=origin,
    cmap="seismic",
    norm=norm
)
plt.colorbar(label="Δ Transmission (fit − exp)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Spectrum index")
plt.title("Transmission difference (fit − experiment)")
plt.tight_layout()
plt.savefig(out_dir / "T_diff_2D.png", dpi=300)
plt.show()

print("\nAll plots saved to:", out_dir)


# %%
