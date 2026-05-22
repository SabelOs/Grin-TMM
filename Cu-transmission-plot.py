#%% ================== Imports =====================
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

import tmm_fast.gym_multilayerthinfilm as mltf
from tmm_fast import coh_tmm

#%% ================= User settings =================

device = "cpu"
dtype = torch.complex128

# -------- Paths --------
base_path = Path(__file__).parent

materials = {
    "Cu":   base_path / "OpticalConstants/nk_Cu.txt",
    "Cu2O": base_path / "OpticalConstants/nk_Cu2O.txt",
    "CuO":  base_path / "OpticalConstants/nk_CuO.txt",
}

# -------- Wavelength range --------
wl_min = 400.0
wl_max = 1000.0
n_wl   = 800

# -------- Angle --------
theta = torch.tensor([0.0], device=device)

layers = [
    {
        "name": "Cu",
        "material": "Cu",
        "thickness": 10.0,
    },
]

#%% ================= Load optical constants =================

wl_nm = np.linspace(wl_min, wl_max, n_wl)

paths = [str(p) for p in materials.values()]

N_np = mltf.get_N(
    paths,
    wl_min,
    wl_max,
    points=n_wl,
    complex_n=True
)

material_index = {name: i for i, name in enumerate(materials)}

N_torch = {
    name: torch.tensor(N_np[i], device=device, dtype=dtype)
    for name, i in material_index.items()
}

lambda_nm = torch.tensor(wl_nm, device=device, dtype=torch.float64)

#%% ================= Thickness sweep =================

thickness_values = np.arange(10, 61, 5)

fig, ax = plt.subplots(figsize=(8,5))

# Custom colormap:
# rgb(255,188,27) -> rgb(250,107,36)

custom_colors = [
    (255/255, 188/255, 27/255),
    (212/255, 47/255, 11/255),
]

cmap = LinearSegmentedColormap.from_list(
    "custom_orange",
    custom_colors
)
norm = plt.Normalize(
    vmin=min(thickness_values),
    vmax=max(thickness_values)
)

#% ================= Sweep loop =================

for thickness in thickness_values:

    N_list = []

    for layer in layers:

        # pure Cu layer
        n_eff = N_torch[layer["material"]]

        N_list.append(n_eff)

    # -------- Build stack --------

    N_stack = torch.stack(
        [torch.ones_like(N_list[0])] + N_list + [torch.ones_like(N_list[0])],
        dim=0
    ).unsqueeze(0)

    d_stack = torch.tensor(
        [
            np.inf,
            thickness,   # swept Cu thickness
            np.inf
        ],
        device=device,
        dtype=torch.float64
    ).unsqueeze(0)

    # -------- TMM --------

    result = coh_tmm(
        pol="s",
        N=N_stack,
        T=d_stack,
        Theta=theta,
        lambda_vacuum=lambda_nm,
        device=device
    )

    T_sim = result["T"][0,0]

    ax.plot(
        wl_nm,
        T_sim.detach().cpu().numpy(),
        color=cmap(norm(thickness)),
        linewidth=2,
        label=f"{thickness} nm"
    )

#% ================= Plot cosmetics =================

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

#cbar = fig.colorbar(sm, ax=ax)
#cbar.set_label("Cu thickness (nm)", rotation=270, labelpad=15)

ax.set_xlabel("Wavelength / nm", fontsize=14)
ax.set_ylabel("Transmission", fontsize=14)
#ax.set_title("Transmission of pure Cu layers")
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)
ax.grid(True)

ax.legend(
    title="Layer thickness",
    loc="upper right",
    fontsize=11,
)

fig.tight_layout()
plt.savefig("Cu-Transmission-Thickness-Plot")
plt.show()
# %%
