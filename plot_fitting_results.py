#%% ================== Imports =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import re
from collections import defaultdict
#%% ================== User settings =====================
# Path to results (pickle preferred)
fileName = "sample9_Cu_Cu-Cu2O-chain-CuO-sphere-60s_2W.csv"
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

#%% ================== 1) Spectrum comparison every 5th =====================

spectra_ids = sorted(df["spectrum"].unique())

for spec in spectra_ids[::5]:
    df_spec = df[df["spectrum"] == spec]

    plt.figure(figsize=(6, 4))
    plt.plot(df_spec["wavelength_nm"], df_spec["T_exp"], "k", label="Measured")
    plt.plot(df_spec["wavelength_nm"], df_spec["T_fit"], "r--", label="TMM fit")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmittance")
    plt.title(f"Spectrum {spec}: Fit vs Experiment")
    plt.legend()
    plt.tight_layout()

    savestr = f"comparison_{spec}_spectrum.png"
    plt.savefig(out_dir / savestr, dpi=300)
    plt.show()
    plt.close()  # IMPORTANT: prevents memory buildup


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
#plt.yscale('log')
plt.show()

#%% ================== 3) Thickness evolution (NEW STRUCTURE) =====================

grouped = df.groupby("spectrum").first()
spectra_index = grouped.index

# -------------------------------------------------
# Detect layers automatically
# -------------------------------------------------
layer_indices = sorted(
    int(m.group(1))
    for c in df.columns
    if (m := re.match(r"layer_(\d+)_thickness_nm", c))
)

# -------------------------------------------------
# Detect inclusions per layer automatically
# -------------------------------------------------
layer_inclusions = {}

for i in layer_indices:
    inc_indices = sorted(
        int(m.group(1))
        for c in df.columns
        if (m := re.match(rf"layer_{i}_inc_(\d+)_material", c))
    )
    layer_inclusions[i] = inc_indices

# =================================================
# 1) INDIVIDUAL CONTRIBUTION PLOT
# =================================================

rows_individual = []

for _, row in grouped.iterrows():

    spectrum_dict = {}

    for i in layer_indices:

        thickness = row[f"layer_{i}_thickness_nm"]
        matrix_mat = row[f"layer_{i}_matrix"]
        matrix_fraction = row.get(f"layer_{i}_matrix_fraction", 1.0)

        # --- matrix contribution ---
        spectrum_dict[(matrix_mat, "matrix", i)] = thickness * matrix_fraction

        # --- inclusions ---
        for j in layer_inclusions[i]:

            inc_mat = row.get(f"layer_{i}_inc_{j}_material", None)
            if pd.isna(inc_mat):
                continue

            inc_frac = row.get(f"layer_{i}_inc_{j}_fraction", 0.0)
            spectrum_dict[(inc_mat, "inclusion", i)] = thickness * inc_frac

    rows_individual.append(spectrum_dict)

plot_df_ind = pd.DataFrame(rows_individual, index=spectra_index).fillna(0)

plt.figure(figsize=(6,4))
plot_df_ind.plot(ax=plt.gca())
plt.xlabel("Spectrum index")
plt.ylabel("Effective thickness (nm)")
plt.title("Individual material contributions")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "thickness_individual.png", dpi=300)
plt.show()


# =================================================
# 2) ACCUMULATED BY MATERIAL (matrix+inclusions)
# =================================================

rows_accumulated = []

for _, row in grouped.iterrows():

    spectrum_dict = {}

    for i in layer_indices:

        thickness = row[f"layer_{i}_thickness_nm"]
        matrix_mat = row[f"layer_{i}_matrix"]
        matrix_fraction = row.get(f"layer_{i}_matrix_fraction", 1.0)

        spectrum_dict[matrix_mat] = (
            spectrum_dict.get(matrix_mat, 0.0)
            + thickness * matrix_fraction
        )

        for j in layer_inclusions[i]:

            inc_mat = row.get(f"layer_{i}_inc_{j}_material", None)
            if pd.isna(inc_mat):
                continue

            inc_frac = row.get(f"layer_{i}_inc_{j}_fraction", 0.0)

            spectrum_dict[inc_mat] = (
                spectrum_dict.get(inc_mat, 0.0)
                + thickness * inc_frac
            )

    rows_accumulated.append(spectrum_dict)

plot_df_acc = pd.DataFrame(rows_accumulated, index=spectra_index).fillna(0)

plt.figure(figsize=(6,4))
plot_df_acc.plot(ax=plt.gca())
plt.xlabel("Spectrum index")
plt.ylabel("Accumulated effective thickness (nm)")
plt.title("Accumulated material thickness")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "thickness_accumulated.png", dpi=300)
plt.show()

#%%
# =================================================
# 3) NEW STACKED LAYER / INCLUSION PLOT
# =================================================

color_map = {
    "Cu": "orange",
    "Cu2O": "red",
    "CuO": "green",
    "Vacuum": "blue",
}

fig, ax = plt.subplots(figsize=(7,5))

base_offset = np.zeros(len(grouped))
legend_handles = {}
spectra_index = grouped.index

for i in layer_indices:

    thickness_series = grouped[f"layer_{i}_thickness_nm"].values
    matrix_mat = grouped[f"layer_{i}_matrix"].iloc[0]

    layer_offset = base_offset.copy()

    # ---- create neighbour-aware mask ----
    threshold = 1.0
    below = thickness_series < threshold
    mask = np.zeros_like(below, dtype=bool)

    if len(below) > 1:
        mask[1:-1] = below[1:-1] & below[:-2] & below[2:]
        mask[0] = below[0] & below[1]
        mask[-1] = below[-1] & below[-2]
    else:
        mask[:] = below

    # ---------- inclusions ----------
    for j in layer_inclusions[i]:

        inc_mat = grouped[f"layer_{i}_inc_{j}_material"].iloc[0]
        if pd.isna(inc_mat):
            continue

        inc_frac_series = grouped[f"layer_{i}_inc_{j}_fraction"].values
        inc_height = thickness_series * inc_frac_series

        color = color_map.get(inc_mat, "gray")

        ax.fill_between(
            spectra_index,
            layer_offset,
            layer_offset + inc_height,
            color=color,
            alpha=0.3,
            linestyle="--"
        )

        inc_line_height = layer_offset + inc_height
        inc_line_height = np.where(mask, np.nan, inc_line_height)

        line, = plt.plot(
            spectra_index,
            inc_line_height,
            color=color,
            linestyle="--"
        )

        # --- Add legend entry only once per material ---
        if inc_mat not in legend_handles:
            legend_handles[inc_mat] = line

        layer_offset += inc_height

    # ---------- matrix ----------
    matrix_color = color_map.get(matrix_mat, "gray")

    ax.fill_between(
        spectra_index,
        base_offset,
        base_offset + thickness_series,
        color=matrix_color,
        alpha=0.15
    )

    # --- Hide line where layer thinner than 1 nm ---
    line_height = base_offset + thickness_series

    # Do not hide thin regions for the first layer
    if i != layer_indices[0]:
        line_height = np.where(mask, np.nan, line_height)

    line, = ax.plot(
        spectra_index,
        line_height,
        color=matrix_color,
        linewidth=3
    )

    if matrix_mat not in legend_handles:
        legend_handles[matrix_mat] = line

    base_offset += thickness_series


# ================= MAIN AXIS STYLE =================
ax.set_xlabel("Spectrum index", fontsize=14)
ax.set_ylabel("Layer height / nm", fontsize=14)
#ax.set_title("Layer stack evolution (inclusions stacked)", fontsize=16)

ax.tick_params(axis='both', labelsize=14, width = 2, length = 4)

# thicker frame instead of grid
for spine in ax.spines.values():
    spine.set_linewidth(2)

# legend
ax.legend(
    legend_handles.values(),
    legend_handles.keys(),
    title="Materials",
    fontsize=12,
    title_fontsize=13,
    loc = 2,
)


# =================================================
# RMSE INSET
# =================================================
ax_inset = inset_axes(
    ax, 
    width="30%",   # relative to main axes
    height="30%",  # relative to main axes
    loc='upper right',  # anchor corner
    borderpad=0     # optional padding
)

rmse_series = grouped["RMSE"]

ax_inset.plot(
    spectra_index,
    rmse_series,
    color="black",
    linewidth=2
)

# highlight current spectrum position if desired
# ax_inset.axvline(current_spec, color="red", linestyle="--", linewidth=2)

ax_inset.set_xlabel("Spec", fontsize=12)
ax_inset.set_ylabel("RMSE", fontsize=12)

ax_inset.tick_params(axis='both', labelsize=14, width = 2, length = 4)

# thick frame
for spine in ax_inset.spines.values():
    spine.set_linewidth(2)

# remove grid
ax_inset.grid(False)


plt.tight_layout()
plt.savefig(out_dir / "thickness_stacked_layers.png", dpi=300)
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
