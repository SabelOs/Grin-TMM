#%% ================== Imports =====================
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

import tmm_fast.gym_multilayerthinfilm as mltf
from tmm_fast import coh_tmm

from bruggemann_mixing_new import bruggeman_n_multi


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


#%% ================= Layer definition =================
"""
            # fixed inclusion
            {
                "material": "Cu2O",
                "shape": "sphere",
                "fraction": 0.0
            },"""
layers = [

    {
        "name": "Cu_layer",
        "matrix": "Cu",
        "shape": "sphere",
        "thickness": 40.0,

        "inclusions": [
            # fixed inclusion
            {
                "material": "Cu2O",
                "shape": "sphere",
                "fraction": 0.0
            },
            # fixed inclusion
            {
                "material": "CuO",
                "shape": "sphere",
                "sweep": (0.0, 0.2, 10),
            },

        ],
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


#%% ================= Find sweep variable =================

sweep_layer = None
sweep_inc = None
sweep_range = None

for i, layer in enumerate(layers):

    inclusions = layer.get("inclusions") or []

    for j, inc in enumerate(inclusions):

        if "sweep" in inc:
            sweep_layer = i
            sweep_inc = j
            sweep_range = inc["sweep"]
            break

    if sweep_layer is not None:
        break


# -------- determine sweep values --------

if sweep_layer is None:

    # no sweep → run once
    f_values = [None]
    sweep_mode = False

else:

    f_values = np.linspace(*sweep_range)
    sweep_mode = True


#%% ================= Plot =================

fig, ax = plt.subplots(figsize=(8,5))

cmap = plt.cm.coolwarm
norm = plt.Normalize(vmin=min(f_values), vmax=max(f_values))


#%% ================= Sweep loop =================

for f in f_values:

    N_list = []

    for i, layer in enumerate(layers):

        n_matrix = N_torch[layer["matrix"]]
        shape_matrix = layer["shape"]

        inclusions = layer.get("inclusions") or []

        if len(inclusions) == 0:

            n_eff = n_matrix

        else:

            n_list = []
            f_list = []
            shape_list = []

            f_sum = 0.0

            for j, inc in enumerate(inclusions):

                n_inc = N_torch[inc["material"]]

                if "fraction" in inc:
                    f_val = inc["fraction"]

                elif "sweep" in inc and i == sweep_layer and j == sweep_inc:
                    f_val = f

                else:
                    f_val = 0.0

                f_sum += f_val

                n_list.append(n_inc)
                f_list.append(torch.tensor(f_val, device=device))
                shape_list.append(inc["shape"])

            f_matrix = max(0.0, 1.0 - f_sum)

            n_list.append(n_matrix)
            f_list.append(torch.tensor(f_matrix, device=device))
            shape_list.append(shape_matrix)

            n_eff = bruggeman_n_multi(
                n_list=n_list,
                f_list=f_list,
                shape_list=shape_list,
            )

        N_list.append(n_eff)


    # -------- Build stack --------

    N_stack = torch.stack(
        [torch.ones_like(N_list[0])] + N_list + [torch.ones_like(N_list[0])],
        dim=0
    ).unsqueeze(0)

    d_stack = torch.tensor(
        [np.inf] +
        [layer["thickness"] for layer in layers] +
        [np.inf],
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
        color=cmap(norm(f)),
        linewidth=2
    )


#%% ================= Plot cosmetics =================

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Swept inclusion fraction", rotation=270, labelpad=15)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Transmission")
ax.set_title("Bruggeman sanity check")

ax.grid(True)
fig.tight_layout()
plt.show()