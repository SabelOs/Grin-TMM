#%% ================== Imports =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import re

#%% ================== User settings =====================
fileName = "Sample9-remeasured-2W-60s-Cu-Cu2O-CuO-scale0_6-for-Thesis.csv"
additional_folder = ""

results_base = Path(__file__).parent / additional_folder / fileName

dx_per_spec = 10.0  # µm per spectrum

outDirName = "plots_" + fileName[:-4]
out_dir = Path(__file__).parent / additional_folder / outDirName
out_dir.mkdir(exist_ok=True)

#%% ================== Load results =====================
df_results = pd.read_csv(results_base.with_suffix(".csv"))

#%% ================== Basic info =====================
df = df_results.sort_values(["spectrum", "wavelength_nm"])

spectra = np.sort(df["spectrum"].unique())
wavelengths = np.sort(df["wavelength_nm"].unique())

# ---- REAL AXIS ----
x_axis = spectra * dx_per_spec  # Δx in µm

n_spec = spectra.size
n_wl = wavelengths.size

#%% ================== 1) Spectrum comparison =====================
for spec in spectra:
    df_spec = df[df["spectrum"] == spec]
    dx = spec * dx_per_spec

    plt.figure(figsize=(6, 4))
    plt.plot(df_spec["wavelength_nm"], df_spec["T_exp"], "k", label="Measured")
    plt.plot(df_spec["wavelength_nm"], df_spec["T_fit"], "r--", label="TMM fit")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmittance")
    plt.title(f"Spectrum {spec} (Δx = {dx:.1f} µm)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_dir / f"comparison_{spec}_spectrum.png", dpi=300)
    plt.close()

#%% ================== 2) RMSE vs Δx =====================
rmse_vals = df.groupby("spectrum")["RMSE"].first()

plt.figure(figsize=(6, 4))
plt.plot(x_axis, rmse_vals, marker="o")
plt.xlabel("Δx / µm")
plt.ylabel("RMSE")
#plt.title("Fit error vs position")
plt.grid(True, alpha=0.3)
plt.tight_layout()
#plt.savefig(out_dir / "RMSE_vs_dx.png", dpi=300)

#%% ================== 3) Thickness evolution =====================
grouped = df.groupby("spectrum").first()
spectra_index = grouped.index
x_axis_grouped = spectra_index * dx_per_spec

# ---- detect layers ----
layer_indices = sorted(
    int(m.group(1))
    for c in df.columns
    if (m := re.match(r"layer_(\d+)_thickness_nm", c))
)

# ---- detect inclusions ----
layer_inclusions = {}
for i in layer_indices:
    inc_indices = sorted(
        int(m.group(1))
        for c in df.columns
        if (m := re.match(rf"layer_{i}_inc_(\d+)_material", c))
    )
    layer_inclusions[i] = inc_indices

# =================================================
# INDIVIDUAL CONTRIBUTIONS
# =================================================
rows_individual = []

for _, row in grouped.iterrows():
    spectrum_dict = {}

    for i in layer_indices:
        thickness = row[f"layer_{i}_thickness_nm"]
        matrix_mat = row[f"layer_{i}_matrix"]
        matrix_fraction = row.get(f"layer_{i}_matrix_fraction", 1.0)

        spectrum_dict[(matrix_mat, "matrix", i)] = thickness * matrix_fraction

        for j in layer_inclusions[i]:
            inc_mat = row.get(f"layer_{i}_inc_{j}_material", None)
            if pd.isna(inc_mat):
                continue

            inc_frac = row.get(f"layer_{i}_inc_{j}_fraction", 0.0)
            spectrum_dict[(inc_mat, "inclusion", i)] = thickness * inc_frac

    rows_individual.append(spectrum_dict)

plot_df_ind = pd.DataFrame(rows_individual, index=x_axis_grouped).fillna(0)

plt.figure(figsize=(6,4))
plot_df_ind.plot(ax=plt.gca())
plt.xlabel("Δx / µm")
plt.ylabel("Effective thickness / nm")
#plt.title("Individual material contributions")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# =================================================
# ACCUMULATED
# =================================================
rows_accumulated = []

for _, row in grouped.iterrows():
    spectrum_dict = {}

    for i in layer_indices:
        thickness = row[f"layer_{i}_thickness_nm"]
        matrix_mat = row[f"layer_{i}_matrix"]
        matrix_fraction = row.get(f"layer_{i}_matrix_fraction", 1.0)

        spectrum_dict[matrix_mat] = spectrum_dict.get(matrix_mat, 0.0) + thickness * matrix_fraction

        for j in layer_inclusions[i]:
            inc_mat = row.get(f"layer_{i}_inc_{j}_material", None)
            if pd.isna(inc_mat):
                continue

            inc_frac = row.get(f"layer_{i}_inc_{j}_fraction", 0.0)
            spectrum_dict[inc_mat] = spectrum_dict.get(inc_mat, 0.0) + thickness * inc_frac

    rows_accumulated.append(spectrum_dict)

plot_df_acc = pd.DataFrame(rows_accumulated, index=x_axis_grouped).fillna(0)

plt.figure(figsize=(6,4))
plot_df_acc.plot(ax=plt.gca())
plt.xlabel("Δx / µm")
plt.ylabel("Accumulated thickness / nm")
#plt.title("Accumulated material thickness")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# =================================================
# STACKED PLOT
# =================================================
color_map = {
    "Cu": [0.98,0.42,0.14],
    "Cu2O": [0.98,0.67,0.15],
    "CuO": [0.57,0.57,0.57],
    "Vacuum": "blue",
}

fig, ax = plt.subplots(figsize=(7,5))
base_offset = np.zeros(len(grouped))
legend_handles = {}

n_layers = len(layer_indices)

for idx, i in enumerate(layer_indices):

    # --- enforce drawing hierarchy ---
    # lower layers → higher zorder (drawn on top)
    z_base = 100 - idx * 10

    thickness_series = grouped[f"layer_{i}_thickness_nm"].values
    matrix_mat = grouped[f"layer_{i}_matrix"].iloc[0]

    layer_offset = base_offset.copy()

    # ---------- inclusions ----------
    for j in layer_inclusions[i]:
        inc_mat = grouped[f"layer_{i}_inc_{j}_material"].iloc[0]
        if pd.isna(inc_mat):
            continue

        inc_frac_series = grouped[f"layer_{i}_inc_{j}_fraction"].values
        inc_height = thickness_series * inc_frac_series

        color = color_map.get(inc_mat, "gray")

        ax.fill_between(
            x_axis_grouped,
            layer_offset,
            layer_offset + inc_height,
            color=color,
            alpha=1,
            zorder=z_base
        )

        line, = ax.plot(
            x_axis_grouped,
            layer_offset + inc_height,
            color=color,
            linestyle="--",
            linewidth=3,
            zorder=z_base + 1
        )

        if inc_mat not in legend_handles:
            legend_handles[inc_mat] = line

        layer_offset += inc_height

    # ---------- matrix ----------
    matrix_color = color_map.get(matrix_mat, "gray")

    ax.fill_between(
        x_axis_grouped,
        base_offset,
        base_offset + thickness_series,
        color=matrix_color,
        alpha=0.15,
        zorder=z_base
    )

    line, = ax.plot(
        x_axis_grouped,
        base_offset + thickness_series,
        color=matrix_color,
        linewidth=3,
        zorder=z_base + 2
    )

    if matrix_mat not in legend_handles:
        legend_handles[matrix_mat] = line

    base_offset += thickness_series


ax.set_xlabel("Δx / µm", fontsize=14)
ax.set_ylabel("Layer height / nm", fontsize=14)

#ax.legend(legend_handles.values(), legend_handles.keys(), title="Materials",loc="lower right")

# ---- RMSE inset ----
ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper right')
rmse_series = grouped["RMSE"]

ax_inset.plot(x_axis_grouped, rmse_series, color="black", linewidth=2)
ax_inset.set_xlabel("Δx / µm")
ax_inset.set_ylabel("RMSE")
ax_inset.tick_params(axis='both', labelsize=10)
#ax_inset.yaxis.tick_right()
#ax_inset.yaxis.set_label_position("right")
# main axis ticks
ax.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.savefig(out_dir / "thickness_stacked_layers.png", dpi=300)
plt.show()

# =================================================
# STACKED PLOT Spectrum Index
# =================================================
color_map = {
    "Cu": [0.98,0.42,0.14],
    "Cu2O": [0.98,0.67,0.15],
    "CuO": [0.57,0.57,0.57],
    "Vacuum": "blue",
}

fig, ax = plt.subplots(figsize=(7,5))
base_offset = np.zeros(len(grouped))
legend_handles = {}

n_layers = len(layer_indices)
x_index = np.arange(n_spec)

for idx, i in enumerate(layer_indices):

    # --- enforce drawing hierarchy ---
    # lower layers → higher zorder (drawn on top)
    z_base = 100 - idx * 10

    thickness_series = grouped[f"layer_{i}_thickness_nm"].values
    matrix_mat = grouped[f"layer_{i}_matrix"].iloc[0]

    layer_offset = base_offset.copy()

    # ---------- inclusions ----------
    for j in layer_inclusions[i]:
        inc_mat = grouped[f"layer_{i}_inc_{j}_material"].iloc[0]
        if pd.isna(inc_mat):
            continue

        inc_frac_series = grouped[f"layer_{i}_inc_{j}_fraction"].values
        inc_height = thickness_series * inc_frac_series

        color = color_map.get(inc_mat, "gray")

        ax.fill_between(
            x_index,
            layer_offset,
            layer_offset + inc_height,
            color=color,
            alpha=1,
            zorder=z_base
        )

        line, = ax.plot(
            x_index,
            layer_offset + inc_height,
            color=color,
            linestyle="--",
            linewidth=3,
            zorder=z_base + 1
        )

        if inc_mat not in legend_handles:
            legend_handles[inc_mat] = line

        layer_offset += inc_height

    # ---------- matrix ----------
    matrix_color = color_map.get(matrix_mat, "gray")

    ax.fill_between(
        x_index,
        base_offset,
        base_offset + thickness_series,
        color=matrix_color,
        alpha=0.15,
        zorder=z_base
    )

    line, = ax.plot(
        x_index,
        base_offset + thickness_series,
        color=matrix_color,
        linewidth=3,
        zorder=z_base + 2
    )

    if matrix_mat not in legend_handles:
        legend_handles[matrix_mat] = line

    base_offset += thickness_series


ax.set_xlabel("Spectrum Index", fontsize=14)
ax.set_ylabel("Layer height / nm", fontsize=14)

ax.legend(legend_handles.values(), legend_handles.keys(), title="Materials")

# ---- RMSE inset ----
ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper left', borderpad=1)
rmse_series = grouped["RMSE"]

ax_inset.plot(x_index, rmse_series, color="black", linewidth=2)
ax_inset.set_xlabel("Spectrum Index")
ax_inset.set_ylabel("RMSE")
ax_inset.tick_params(axis='both', labelsize=10)
ax_inset.yaxis.tick_right()
ax_inset.yaxis.set_label_position("right")
# main axis ticks
ax.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.savefig(out_dir / "thickness_stacked_layers-index.png", dpi=300)
plt.show()

#%% ================== 4) Residuals =====================
df["residual"] = df["T_fit"] - df["T_exp"]

#%% ================== Prepare 2D grids =====================
T_exp_2d = df["T_exp"].values.reshape(n_spec, n_wl)
T_fit_2d = df["T_fit"].values.reshape(n_spec, n_wl)
T_diff_2d = T_fit_2d - T_exp_2d

extent = [
    wavelengths.min(), wavelengths.max(),
    x_axis.min(), x_axis.max()
]

#%% ================== 5) Experimental =====================
plt.figure(figsize=(8, 5))
plt.imshow(T_exp_2d, extent=extent, aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(label="Transmission (exp)")
plt.xlabel("Wavelength / nm")
plt.ylabel("Δx / µm")
#plt.title("Experimental transmission")
plt.tight_layout()
plt.savefig(out_dir / "T_exp_2D.png", dpi=300)

#%% ================== 6) Simulated =====================
plt.figure(figsize=(8, 5))
plt.imshow(T_fit_2d, extent=extent, aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(label="Transmission (fit)")
plt.xlabel("Wavelength / nm")
plt.ylabel("Δx / µm")
#plt.title("Simulated transmission")
plt.tight_layout()
plt.savefig(out_dir / "T_fit_2D.png", dpi=300)

#%% ================== 7) Difference =====================
vmax = np.max(np.abs(T_diff_2d))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

plt.figure(figsize=(8, 5))
plt.imshow(T_diff_2d, extent=extent, aspect="auto", origin="lower", cmap="seismic", norm=norm)
plt.colorbar(label="Δ Transmission")
plt.xlabel("Wavelength / nm")
plt.ylabel("Δx / µm")
#plt.title("Fit - Experiment")
plt.tight_layout()
plt.savefig(out_dir / "T_diff_2D.png", dpi=300)

print("\nAll plots saved to:", out_dir)

# %%
