#%% ================== Imports =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import re
#%% ================== User settings =====================
# Path to results (pickle preferred)
results_base = Path(__file__).parent / "tmm_fit_results_pop30_step100"

use_pickle = True   # set False to use CSV instead

# Output directory for plots
out_dir = Path(__file__).parent / "plots"
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
last_spec = 19
df_last = df[df["spectrum"] == last_spec]

plt.figure(figsize=(6, 4))
plt.plot(df_last["wavelength_nm"], df_last["T_exp"], "k", label="Measured")
plt.plot(df_last["wavelength_nm"], df_last["T_fit"], "r--", label="TMM fit")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmittance")
plt.title(f"Spectrum {last_spec}: Fit vs Experiment")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "comparison_last_spectrum.png", dpi=300)
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

# --- find all material indices automatically ---
material_indices = sorted(
    {
        int(m.group(1))
        for c in df.columns
        if (m := re.match(r"material_(\d+)_thickness_nm", c))
    }
)

# --- build thickness dataframe and legend labels ---
thickness_cols = []
legend_labels = []

for i in material_indices:
    t_col = f"material_{i}_thickness_nm"
    thickness_cols.append(t_col)

    # extract metadata from first spectrum (assumed constant per layer)
    row = df.iloc[0]

    mat_name  = row.get(f"material_{i}_name", f"material_{i}")
    mat_shape = row.get(f"material_{i}_shape", None)

    label = mat_name
    if mat_shape and str(mat_shape) != "nan":
        label += f" ({mat_shape})"

    # inclusion info (optional)
    inc_name = row.get(f"inclusion_{i}_name", None)
    if inc_name and str(inc_name) != "nan":
        inc_shape = row.get(f"inclusion_{i}_shape", "")
        inc_vf    = row.get(f"inclusion_{i}_volume_fraction", "")

        inc_label = inc_name
        if inc_shape and str(inc_shape) != "nan":
            inc_label += f", {inc_shape}"
        if inc_vf and str(inc_vf) != "nan":
            inc_label += f", vf={inc_vf:.2f}"

        label += f"\n+ {inc_label}"

    legend_labels.append(label)

# --- group by spectrum and plot ---
plt.figure(figsize=(6, 4))

(
    df.groupby("spectrum")[thickness_cols]
      .first()
      .plot(ax=plt.gca())
)

plt.xlabel("Spectrum index")
plt.ylabel("Thickness (nm)")
plt.title("Layer thickness evolution")
plt.grid(True, alpha=0.3)

plt.legend(
    legend_labels,
    title="Layers",
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
