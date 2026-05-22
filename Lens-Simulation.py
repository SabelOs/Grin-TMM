#%% =========================================================
#   GRIN Lens Wavefront / OPL Analysis
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import torch

# ============================================================
# USER SETTINGS
# ============================================================

# ------------------------------------------------------------
# Mode selection
# ------------------------------------------------------------

USE_EXPERIMENTAL_DATA = False

# ------------------------------------------------------------
# File settings
# ------------------------------------------------------------

fileName= "sample9_Cu_Cu2O_CuO-120s_3W_scale-0_6"
additional_folder = ""
#fileName = "sample9_Cu_Cu-Cu2O-sphere_CuO-40s_2W_scale-0_6.csv"
#additional_folder = "Sample-9-Grin-2W-40s"

# ------------------------------------------------------------
# Spatial calibration
# ------------------------------------------------------------

dx_per_spec = 10.0  # µm per spectrum

# ------------------------------------------------------------
# Selected wavelength for single profile plot
# ------------------------------------------------------------

selected_wavelength = 600  # nm

# ------------------------------------------------------------
# Waterfall wavelength sweep
# ------------------------------------------------------------

waterfall_lambda_min = 500   # nm
waterfall_lambda_max = 1000  # nm
waterfall_lambda_step = 50   # nm

# ------------------------------------------------------------
# Plot limits
# ------------------------------------------------------------

xlim_profile = (-250, 250)  # µm
ylim_profile = None

xlim_map = (500, 1000)      # nm
ylim_map = (-250, 250)      # µm

# ------------------------------------------------------------
# Manual simulation mode
# ------------------------------------------------------------

# ============================================================
# PARABOLIC GRIN PROFILE
# ============================================================

R_lens = 250  # µm

manual_x_positions = np.linspace(
    0,
    R_lens,
    101
)

Cu2O_profile = 100 * (
    manual_x_positions / R_lens
)**2

CuO_profile = 100 - Cu2O_profile

manual_thickness = {

    "Cu": np.zeros_like(
        manual_x_positions
    ),

    "Cu2O": Cu2O_profile,

    "CuO": CuO_profile
}
"""
# ============================================================
# LINEAR GRIN PROFILE
# ============================================================

R_lens = 250  # µm

manual_x_positions = np.linspace(
    0,
    R_lens,
    101
)

Cu2O_profile = 100 * (
    manual_x_positions / R_lens
)

CuO_profile = 100 - Cu2O_profile

manual_thickness = {

    "Cu": np.zeros_like(
        manual_x_positions
    ),

    "Cu2O": Cu2O_profile,

    "CuO": CuO_profile
}
"""
# ============================================================
# Torch settings
# ============================================================

device = "cpu"
dtype = torch.complex128

# ============================================================
# Optical constants
# ============================================================

base_path = Path(__file__).parent

materials = {
    "Cu":   base_path / "OpticalConstants/nk_Cu.txt",
    "Cu2O": base_path / "OpticalConstants/nk_Cu2O.txt",
    "CuO":  base_path / "OpticalConstants/nk_CuO.txt",
}

# ============================================================
# Load experimental data
# ============================================================

results_base = (
    Path(__file__).parent
    / additional_folder
    / fileName
)

if USE_EXPERIMENTAL_DATA:

    df_results = pd.read_csv(
        results_base.with_suffix(".csv")
    )

    df = df_results.sort_values(
        ["spectrum", "wavelength_nm"]
    )

    spectra = np.sort(df["spectrum"].unique())

    wavelengths_fit = np.sort(
        df["wavelength_nm"].unique()
    )

    x_axis = spectra * dx_per_spec

    n_spec = spectra.size

# ============================================================
# Load optical constants
# ============================================================

def load_nk_file(path):
    """
    Expected format:

    wavelength[nm]   n   k
    """

    data = np.loadtxt(path)

    wl = data[:, 0]
    n = data[:, 1]
    k = data[:, 2]

    N = n + 1j * k

    return wl, N

# ------------------------------------------------------------
# Load all materials
# ------------------------------------------------------------

material_data = {}

for name, path in materials.items():

    wl, N = load_nk_file(path)

    material_data[name] = {
        "wl": wl,
        "N": N
    }

# ============================================================
# Build COMMON interpolation wavelength grid
# ============================================================

wl_min_common = max(
    np.min(material_data[m]["wl"])
    for m in materials
)

wl_max_common = min(
    np.max(material_data[m]["wl"])
    for m in materials
)

wl_grid = np.arange(
    int(np.ceil(wl_min_common)),
    int(np.floor(wl_max_common)) + 1,
    1
)

print(
    f"Common wavelength range: "
    f"{wl_grid[0]} - {wl_grid[-1]} nm"
)

# ============================================================
# Interpolate refractive indices
# ============================================================

N_interp = {}

for name in materials:

    wl = material_data[name]["wl"]
    N = material_data[name]["N"]

    n_interp = np.interp(
        wl_grid,
        wl,
        np.real(N)
    )

    k_interp = np.interp(
        wl_grid,
        wl,
        np.imag(N)
    )

    N_interp[name] = n_interp + 1j * k_interp

# ============================================================
# Thickness source selection
# ============================================================

if USE_EXPERIMENTAL_DATA:

    # ========================================================
    # EXPERIMENTAL DATA
    # ========================================================

    grouped = df.groupby("spectrum").first()

    # --------------------------------------------------------
    # detect layers
    # --------------------------------------------------------

    layer_indices = sorted(
        int(m.group(1))
        for c in df.columns
        if (m := re.match(
            r"layer_(\d+)_thickness_nm",
            c
        ))
    )

    # --------------------------------------------------------
    # detect inclusions
    # --------------------------------------------------------

    layer_inclusions = {}

    for i in layer_indices:

        inc_indices = sorted(
            int(m.group(1))
            for c in df.columns
            if (m := re.match(
                rf"layer_{i}_inc_(\d+)_material",
                c
            ))
        )

        layer_inclusions[i] = inc_indices

    # --------------------------------------------------------
    # Build accumulated thickness table
    # --------------------------------------------------------

    rows_accumulated = []

    for _, row in grouped.iterrows():

        spectrum_dict = {}

        for i in layer_indices:

            thickness = row[
                f"layer_{i}_thickness_nm"
            ]

            # ------------------------------------------------
            # matrix contribution
            # ------------------------------------------------

            matrix_mat = row[
                f"layer_{i}_matrix"
            ]

            matrix_fraction = row.get(
                f"layer_{i}_matrix_fraction",
                1.0
            )

            matrix_contribution = (
                thickness * matrix_fraction
            )

            spectrum_dict[matrix_mat] = (
                spectrum_dict.get(matrix_mat, 0.0)
                + matrix_contribution
            )

            # ------------------------------------------------
            # inclusion contributions
            # ------------------------------------------------

            for j in layer_inclusions[i]:

                inc_mat = row.get(
                    f"layer_{i}_inc_{j}_material",
                    None
                )

                if pd.isna(inc_mat):
                    continue

                inc_frac = row.get(
                    f"layer_{i}_inc_{j}_fraction",
                    0.0
                )

                inc_contribution = (
                    thickness * inc_frac
                )

                spectrum_dict[inc_mat] = (
                    spectrum_dict.get(inc_mat, 0.0)
                    + inc_contribution
                )

        rows_accumulated.append(spectrum_dict)

    # --------------------------------------------------------
    # Final dataframe
    # --------------------------------------------------------

    plot_df_acc = pd.DataFrame(
        rows_accumulated,
        index=grouped.index
    ).fillna(0)

    plot_df_acc.index = (
        plot_df_acc.index * dx_per_spec
    )

else:

    # ========================================================
    # MANUAL SIMULATION MODE
    # ========================================================

    plot_df_acc = pd.DataFrame(
        manual_thickness,
        index=manual_x_positions
    ).fillna(0)

# ============================================================
# Final spatial axis
# ============================================================

x_positions = plot_df_acc.index.values

print(plot_df_acc.head())

# ============================================================
# Compute OPL(lambda, x)
# ============================================================

n_x = len(x_positions)
n_lambda = len(wl_grid)

OPL = np.zeros(
    (n_x, n_lambda),
    dtype=np.complex128
)

# ------------------------------------------------------------
# loop over positions
# ------------------------------------------------------------

for ix, x in enumerate(x_positions):

    opl_x = np.zeros(
        n_lambda,
        dtype=np.complex128
    )

    thicknesses = plot_df_acc.iloc[ix]

    for material in materials:

        if material not in thicknesses:
            continue

        d_nm = thicknesses[material]

        opl_x += (
            N_interp[material] * d_nm
        )

    OPL[ix, :] = opl_x

# ============================================================
# Selected wavelength profile
# ============================================================

idx_wl = np.argmin(
    np.abs(wl_grid - selected_wavelength)
)

opl_profile = np.real(
    OPL[:, idx_wl]
)

# ============================================================
# MIRROR PROFILE
# ============================================================

x_mirror = np.concatenate([
    -x_positions[::-1][:-1],
    x_positions
])

opl_mirror = np.concatenate([
    opl_profile[::-1][:-1],
    opl_profile
])

# ============================================================
# Plot single wavelength profile
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    x_mirror,
    opl_mirror,
    linewidth=2
)

plt.xlabel(r"Position on lens $x$ / µm")

plt.ylabel(
    r"Optical path length OPL / nm"
)

plt.title(
    rf"GRIN lens OPL profile at "
    rf"$\lambda = {wl_grid[idx_wl]}$ nm"
)

plt.grid(True)

if xlim_profile is not None:
    plt.xlim(xlim_profile)

if ylim_profile is not None:
    plt.ylim(ylim_profile)

plt.tight_layout()
plt.show()

# ============================================================
# Waterfall wavelength sweep
# ============================================================

lambda_values = np.arange(
    waterfall_lambda_min,
    waterfall_lambda_max + 1,
    waterfall_lambda_step
)

plt.figure(figsize=(9, 6))

# ------------------------------------------------------------
# determine x-range mask for visible region
# ------------------------------------------------------------

if xlim_profile is not None:

    xmask = (
        (x_mirror >= xlim_profile[0]) &
        (x_mirror <= xlim_profile[1])
    )

else:

    xmask = np.ones_like(
        x_mirror,
        dtype=bool
    )

# ------------------------------------------------------------
# determine GLOBAL y-limits ONLY inside visible region
# ------------------------------------------------------------

global_min = np.inf
global_max = -np.inf

profiles_all = []

for wl in lambda_values:

    idx = np.argmin(
        np.abs(wl_grid - wl)
    )

    profile = np.real(
        OPL[:, idx]
    )

    profile_mirror = np.concatenate([
        profile[::-1][:-1],
        profile
    ])

    profiles_all.append(
        (wl, profile_mirror)
    )

    # --------------------------------------------
    # only visible cutout contributes
    # --------------------------------------------

    visible_profile = profile_mirror[xmask]

    global_min = min(
        global_min,
        np.min(visible_profile)
    )

    global_max = max(
        global_max,
        np.max(visible_profile)
    )

# ------------------------------------------------------------
# actual plotting
# ------------------------------------------------------------

for wl, profile_mirror in profiles_all:

    plt.plot(
        x_mirror,
        profile_mirror,
        linewidth=2,
        label=f"{wl}"
    )

plt.xlabel(r"Position on lens $x$ / µm")

plt.ylabel(
    r"Optical path length OPL / nm"
)

plt.title(
    "GRIN lens OPL wavelength sweep"
)

plt.legend(
    title=r"Wavelength $\lambda$ / nm",
    ncol=2
)

plt.grid(True)

# ------------------------------------------------------------
# apply x limits
# ------------------------------------------------------------

if xlim_profile is not None:
    plt.xlim(xlim_profile)

# ------------------------------------------------------------
# apply y limits from visible cutout
# ------------------------------------------------------------

plt.ylim(global_min, global_max)

plt.tight_layout()
plt.show()