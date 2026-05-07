#%% ================== Imports =====================
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import tmm_fast.gym_multilayerthinfilm as mltf
from tmm_fast import coh_tmm

#%% ================= SETTINGS =====================
device = "cpu"
path = Path(__file__).parent
outFilename = "random-structure"
materials = {
    "Cu":   path / "OpticalConstants/nk_Cu.txt",
    "Cu2O": path / "OpticalConstants/nk_Cu2O.txt",
    "CuO":  path / "OpticalConstants/nk_CuO.txt",
    "Vacuum": path / "OpticalConstants/nk_Vacuum.txt",
}

n_spectra = 40

# wavelength grid (match your real data!)
wl_nm = np.linspace(450, 950, 600)
lambda_nm = torch.tensor(wl_nm, dtype=torch.float64, device=device)

#%% ================= REFRACTIVE INDICES =====================
N_np = mltf.get_N(
    list(map(str, materials.values())),
    wl_nm.min(),
    wl_nm.max(),
    points=len(wl_nm),
    complex_n=True
)

mat_index = {k: i for i, k in enumerate(materials.keys())}

#%% ================= LAYER EVOLUTION =====================
# build thickness evolution (nm)

thicknesses = []

# --- 1) first 10 spectra: pure Cu ---
for _ in range(10):
    thicknesses.append([40.0, 0.0, 0.0])

# --- 2) transition (2 spectra) ---
thicknesses.append([25.0, 35.0, 5.0])
thicknesses.append([10.0, 75.0, 5.0])

# --- 3) Cu gone ---
thicknesses.append([0.0, 85.0, 5.0])

# --- 4) linear oxide transition ---
remaining = n_spectra - len(thicknesses)

cu2o_start = 85.0
cuo_start = 5.0
total = cu2o_start + cuo_start  # ~90 nm

for i in range(remaining):
    t = i / (remaining - 1)

    cu2o = cu2o_start * (1 - t)
    cuo  = total - cu2o

    thicknesses.append([0.0, cu2o, cuo])

thicknesses = np.array(thicknesses)
thicknesses = thicknesses[::-1] #invert to account for actual data structure

#%% ================= SPECTRA PLOTS =====================
color_map = {
    "Cu": [0.98,0.42,0.14],
    "Cu2O": [0.98,0.67,0.15],
    "CuO": [0.57,0.57,0.57],
    "Vacuum": "blue",
}
fig, ax = plt.subplots(figsize=(7,5))

spectra_index = np.arange(n_spectra)
base_offset = np.zeros(n_spectra)

labels = ["Cu", "Cu2O", "CuO"]

for i, mat in enumerate(labels):

    layer = thicknesses[:, i]

    ax.fill_between(
        spectra_index,
        base_offset,
        base_offset + layer,
        color=color_map[mat],
        alpha=0.3,
        label=mat
    )

    ax.plot(
        spectra_index,
        base_offset + layer,
        color=color_map[mat],
        linewidth=2
    )

    base_offset += layer

ax.set_xlabel("Spectrum index", fontsize=14)
ax.set_ylabel("Layer height / nm", fontsize=14)

ax.legend(loc = "upper right")
ax.tick_params(axis='both', labelsize=12)

for spine in ax.spines.values():
    spine.set_linewidth(2)

outFilename = "easy-structure"
plt.tight_layout()
plt.savefig(path / str(outFilename + "_spectra"), dpi=300)
plt.show()

#%% ================= RANDOMIZED LAYER EVOLUTION =================

n_spectra = 10

thicknesses = []

# --- first spectrum: pure Cu ---
thicknesses.append([40.0, 0.0, 0.0])

# --- parameters controlling randomness ---
total_thickness_target = 90.0

cu_decay_scale = 6.0      # how fast Cu disappears
noise_scale = 5.0        # randomness strength
min_thickness = 0.0

for i in range(1, n_spectra):

    # --- Cu decays but with randomness ---
    base_cu = 40.0 * np.exp(-i / cu_decay_scale)
    cu = base_cu + np.random.normal(0, noise_scale)
    cu = np.clip(cu, 0.0, 40.0)

    # --- remaining thickness goes to oxides ---
    remaining = total_thickness_target - cu
    remaining = max(remaining, 0.0)

    # --- random split between Cu2O and CuO ---
    split = np.random.rand()

    cu2o = remaining * split
    cuo  = remaining * (1 - split)

    # --- add noise to oxides too ---
    cu2o += np.random.normal(0, noise_scale)
    cuo  += np.random.normal(0, noise_scale)

    # keep physical
    cu2o = max(cu2o, min_thickness)
    cuo  = max(cuo, min_thickness)

    # --- renormalize to keep total thickness stable ---
    total = cu + cu2o + cuo
    if total > 0:
        scale = total_thickness_target / total
        cu *= scale
        cu2o *= scale
        cuo *= scale

    thicknesses.append([cu, cu2o, cuo])

thicknesses = np.array(thicknesses)
thicknesses = thicknesses[::-1] #invert to account for actual data structure


#%% ================= SPECTRA PLOTS =====================
color_map = {
    "Cu": [0.98,0.42,0.14],
    "Cu2O": [0.98,0.67,0.15],
    "CuO": [0.57,0.57,0.57],
    "Vacuum": "blue",
}
fig, ax = plt.subplots(figsize=(7,5))

spectra_index = np.arange(n_spectra)
base_offset = np.zeros(n_spectra)

labels = ["Cu", "Cu2O", "CuO"]

for i, mat in enumerate(labels):

    layer = thicknesses[:, i]

    ax.fill_between(
        spectra_index,
        base_offset,
        base_offset + layer,
        color=color_map[mat],
        alpha=0.3,
        label=mat
    )

    ax.plot(
        spectra_index,
        base_offset + layer,
        color=color_map[mat],
        linewidth=2
    )

    base_offset += layer

ax.set_xlabel("Spectrum index", fontsize=14)
ax.set_ylabel("Layer height / nm", fontsize=14)

ax.legend(loc = "upper right")
ax.tick_params(axis='both', labelsize=12)

for spine in ax.spines.values():
    spine.set_linewidth(2)

outFilename = "random-fixed-height"
plt.tight_layout()
plt.savefig(path / str(outFilename + "_spectra"), dpi=300)
plt.show()

#%% ================= FIXED STRUCTURE (USER DEFINED) =================

Cu_nm   = [0, 0, 12, 0, 13, 31, 20, 35, 40, 33]
Cu2O_nm = [30, 38, 9, 76, 1, 2, 9, 17, 7, 26]
CuO_nm  = [0, 0, 10, 0, 9, 33, 24, 29, 48, 33]

thicknesses = np.array(list(zip(Cu_nm, Cu2O_nm, CuO_nm)))

n_spectra = len(thicknesses)



#%% ================= SPECTRA PLOTS =====================
color_map = {
    "Cu": [0.98,0.42,0.14],
    "Cu2O": [0.98,0.67,0.15],
    "CuO": [0.57,0.57,0.57],
    "Vacuum": "blue",
}
fig, ax = plt.subplots(figsize=(7,5))

spectra_index = np.arange(n_spectra)
base_offset = np.zeros(n_spectra)

labels = ["Cu", "Cu2O", "CuO"]

for i, mat in enumerate(labels):

    layer = thicknesses[:, i]

    ax.fill_between(
        spectra_index,
        base_offset,
        base_offset + layer,
        color=color_map[mat],
        alpha=0.3,
        label=mat
    )

    ax.plot(
        spectra_index,
        base_offset + layer,
        color=color_map[mat],
        linewidth=2
    )

    base_offset += layer

ax.set_xlabel("Spectrum index", fontsize=14)
ax.set_ylabel("Layer height / nm", fontsize=14)

ax.legend(loc = "upper right")
ax.tick_params(axis='both', labelsize=12)

for spine in ax.spines.values():
    spine.set_linewidth(2)

outFilename = "complete-random"
plt.tight_layout()
plt.savefig(path / str(outFilename + "_spectra"), dpi=300)
plt.show()

#%% ================= FORWARD MODEL =====================
def simulate_T(d):

    N_list = [
        torch.ones_like(lambda_nm, dtype=torch.complex128)  # air
    ]

    for mat_name, thickness in zip(["Cu", "Cu2O", "CuO"], d):

        n = torch.tensor(
            N_np[mat_index[mat_name]],
            dtype=torch.complex128,
            device=device
        )
        N_list.append(n)

    # substrate = air
    N_list.append(torch.ones_like(lambda_nm, dtype=torch.complex128))

    N = torch.stack(N_list).unsqueeze(0)

    d_full = torch.tensor(
        [np.inf, *d, np.inf],
        dtype=torch.float64,
        device=device
    )

    T = coh_tmm(
        pol="s",
        N=N,
        T=d_full.unsqueeze(0),
        Theta=torch.zeros(1),
        lambda_vacuum=lambda_nm,
        device=device,
    )["T"][0]

    return T.cpu().numpy()

#%% ================= GENERATE DATA =====================
T_exp_all = []

for i, d in enumerate(thicknesses):
    print(f"Simulating spectrum {i+1}/{n_spectra}")
    T = simulate_T(d)
    T_exp_all.append(np.squeeze(T))

T_exp_all = np.array(T_exp_all)

# optional: add noise
noise_level = 0.003
T_exp_all += noise_level * np.random.randn(*T_exp_all.shape)
T_exp_all = np.clip(T_exp_all, 0, 1)

#%% ================= SAVE =====================
np.save(path / str(outFilename + "_benchmark_T.npy"), T_exp_all)
np.save(path / str(outFilename + "_benchmark_wl.npy"), wl_nm)
# CSV version (optional)
#df = pd.DataFrame(T_exp_all)
#df.insert(0, "wavelength_nm", wl_nm)
#df.to_csv(path / "benchmark_T.csv", index=False)

print("Saved benchmark dataset")

#%% 