#%% Imports
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from winspec import SpeFile
import tmm_fast.gym_multilayerthinfilm as mltf
from tmm_fast import coh_tmm

#%% ================= USER SETTINGS =================
path = str(Path(__file__).parent)

# Material nk files
pathCu   = path + "/OpticalConstants/nk_Cu.txt"
pathCu2O = path + "/OpticalConstants/nk_Cu2O.txt"
pathCuO  = path + "/OpticalConstants/nk_CuO.txt"
material_path_list = [pathCu, pathCu2O, pathCuO]

# SPE files
SPE_file  = path + "/GRIN.SPE"
Lamp_file = path + "/Substrate.SPE"

# Initial thickness guesses [nm] (Cu, Cu2O, CuO)
init_thickness_nm = [22.0, 1.0, 1.0]
spec_no = 69

wl_min = 900
wl_max = 400
# Optimization
n_steps = 300
learning_rate = 0.5

# Angular averaging (brightfield, small NA)
angle_min_deg = 0.0
angle_max_deg = 1.0
n_angles = 5
#===================================================


#%% Load SPE data
wl_nm = SpeFile(SPE_file).xaxis.astype(np.float64)
df_intensities = SpeFile(SPE_file).data[:, :, 0]
lamp_intensities = SpeFile(Lamp_file).data[0, :, 0]

target_T = torch.tensor(
    df_intensities[spec_no, :] / lamp_intensities,
    dtype=torch.float64
)

lambda_nm = torch.tensor(wl_nm, dtype=torch.float64)

#%% Load dispersive refractive indices
N_np = mltf.get_N(
    material_path_list,
    wl_nm.min(),
    wl_nm.max(),
    points=len(wl_nm),
    complex_n=True
)

# Stack: vacuum | Cu | Cu2O | CuO | vacuum
n_stack = np.vstack([
    np.ones_like(wl_nm),
    N_np[0],
    N_np[1],
    N_np[2],
    np.ones_like(wl_nm)
])

N_torch = torch.tensor(n_stack, dtype=torch.complex128).unsqueeze(0)

#%% Trainable thickness tensor (nm)
d_init_nm = torch.tensor(
    [np.inf] + init_thickness_nm + [np.inf],
    dtype=torch.float64,
    requires_grad=True
)
T_torch = d_init_nm.unsqueeze(0)

#%% Angles (radians)
thetas = torch.deg2rad(
    torch.linspace(angle_min_deg, angle_max_deg, n_angles, dtype=torch.float64)
)

#%% Optimizer and loss
optimizer = torch.optim.Adam([d_init_nm], lr=learning_rate)
loss_fn = torch.nn.MSELoss()

#%% Optimization loop
for step in range(n_steps):

    optimizer.zero_grad()

    result = coh_tmm(
        pol="s",
        N=N_torch,
        T=T_torch,
        Theta=thetas,
        lambda_vacuum=lambda_nm,
        device="cpu"
    )

    T_sim = result["T"][0].mean(dim=0)

    loss = loss_fn(T_sim, target_T)
    loss.backward()
    optimizer.step()

    # Physical constraints
    with torch.no_grad():
        d_init_nm[1:-1].clamp_(1.0, 300.0)

    if step % 20 == 0:
        print(
            f"Step {step:3d} | "
            f"Loss = {loss.item():.4e} | "
            f"Thicknesses (nm) = {d_init_nm[1:-1].detach().numpy()}"
        )

#%% Final plot
plt.figure()
plt.plot(wl_nm, T_sim.detach().numpy(), "r--", label="TMM fit")
plt.plot(wl_nm, target_T.numpy(), "k", label="Measured")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmittance")
plt.legend()
plt.tight_layout()
plt.savefig("comparison.png", dpi=300)
plt.show()

# %%
