#%% ================== Imports =====================
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

import tmm_fast.gym_multilayerthinfilm as mltf
from tmm_fast import coh_tmm

from bruggemann_mixing import bruggeman_n

#%% ================= User settings =================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.complex128

# -------- Paths --------
base_path = Path(__file__).parent
pathCu   = base_path / "OpticalConstants/nk_Cu.txt"
pathCu2O = base_path / "OpticalConstants/nk_Cu2O.txt"

# -------- Wavelength range --------
wl_min = 400.0
wl_max = 1000.0
n_wl   = 800

# -------- Layer thickness (nm) --------
d_eff_nm = 25.0

# -------- Volume fractions to test --------
f_Cu2O_list = np.linspace(0.0, 1, 20)

# -------- Geometry --------
shape_Cu   = "sphere"   # metallic connectivity
shape_Cu2O = "sphere"

# -------- Angle --------
theta = torch.tensor([0.0], device=device)

#%% ================= Load optical constants =================
wl_nm = np.linspace(wl_min, wl_max, n_wl)

N_np = mltf.get_N(
    [str(pathCu), str(pathCu2O)],
    wl_min,
    wl_max,
    points=n_wl,
    complex_n=True
)

n_Cu   = torch.tensor(N_np[0], device=device, dtype=dtype)
n_Cu2O = torch.tensor(N_np[1], device=device, dtype=dtype)

lambda_nm = torch.tensor(wl_nm, device=device, dtype=torch.float64)

#%% ================= Plot =================
fig, ax = plt.subplots(figsize=(8, 5))

cmap = plt.cm.coolwarm   # red → blue
norm = plt.Normalize(vmin=min(f_Cu2O_list), vmax=max(f_Cu2O_list))

for f_Cu2O in f_Cu2O_list:

    f = torch.tensor(f_Cu2O, device=device)

    # -------- Bruggeman effective index --------
    n_eff = bruggeman_n(
        n1=n_Cu2O,
        n2=n_Cu,
        f1=f,
        shape1=shape_Cu2O,
        shape2=shape_Cu,
    )

    # -------- Build multilayer stack --------
    N_stack = torch.stack([
        torch.ones_like(n_eff),
        n_eff,
        torch.ones_like(n_eff)
    ], dim=0).unsqueeze(0)

    d_stack = torch.tensor(
        [np.inf, d_eff_nm, np.inf],
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

    T_sim = result["T"][0, 0]

    ax.plot(
        wl_nm,
        T_sim.detach().cpu().numpy(),
        color=cmap(norm(f_Cu2O)),
        linewidth=2
    )

# -------- Colorbar --------
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Cu$_2$O volume fraction", rotation=270, labelpad=15)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Transmission")
ax.set_title("Bruggeman sanity check: Cu film with Cu$_2$O inclusions")
ax.grid(True)

fig.tight_layout()
plt.show()


 # %%
