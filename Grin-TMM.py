#%% Imports
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from winspec import SpeFile
import tmm_fast.gym_multilayerthinfilm as mltf
from tmm_fast import coh_tmm

#%% ================== Functions =====================
def moving_average_same(a, n=5):
    if n <= 1:
        return a
    kernel = np.ones(n) / n
    return np.convolve(a, kernel, mode="same")



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

# Smoothing
enable_smoothing = True
smoothing_window = 5   # must be odd, e.g. 3,5,7

# Wavelength cutout for optimization (nm)
enable_wl_cut = True
wl_opt_min = 500.0
wl_opt_max = 950.0

# Optimization
n_steps = 300
learning_rate = 0.5

# Angular averaging (brightfield, small NA)
angle_min_deg = 0.0
angle_max_deg = 1.0
n_angles = 1
#===================================================


#%% Load SPE data
wl_nm = SpeFile(SPE_file).xaxis.astype(np.float64)
df_intensities = SpeFile(SPE_file).data[:, :, 0]
lamp_intensities = SpeFile(Lamp_file).data[0, :, 0]

# select device to run on:
if torch.cuda.is_available():
    Computation_device = "cuda"
else:
    Computation_device = "cpu"


# Optional Smoothing
if enable_smoothing:
    df_intensities[spec_no,:] = moving_average_same(df_intensities[spec_no,:],smoothing_window)
    lamp_intensities = moving_average_same(lamp_intensities,smoothing_window)

transmission_data = df_intensities[spec_no, :] / lamp_intensities

#Optional wavelength cut
if enable_wl_cut:
    wl_mask = (wl_nm >= wl_opt_min) & (wl_nm <= wl_opt_max)
    transmission_data = transmission_data[wl_mask]
    wl_nm = wl_nm[wl_mask]


#Create torch vectors
target_T = torch.tensor(
    transmission_data,
    dtype=torch.float64,
    device=Computation_device
)

lambda_nm = torch.tensor(
    wl_nm, 
    dtype=torch.float64,
    device=Computation_device
)

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

N_torch = torch.tensor(
    n_stack, 
    dtype=torch.complex128, 
    device = Computation_device
).unsqueeze(0)

#%% Trainable thickness tensor (nm)
d_init_nm = torch.tensor(
    [np.inf] + init_thickness_nm + [np.inf],
    dtype=torch.float64,
    requires_grad=True,
    device = Computation_device
)
T_torch = d_init_nm.unsqueeze(0)

#%% Angles (radians)
thetas = torch.deg2rad(
    torch.linspace(angle_min_deg, angle_max_deg, n_angles, dtype=torch.float64, device = Computation_device)
)

#%% Optimizer and loss
optimizer = torch.optim.Adam([d_init_nm], lr=learning_rate)
loss_fn = torch.nn.MSELoss()

#%% Debugging:
print("Thickness tensor device:", d_init_nm.device)
print("N tensor device:", N_torch.device)
print("Lambda device:", lambda_nm.device)

#%% Optimization loop
for step in range(n_steps):

    optimizer.zero_grad()

    result = coh_tmm(
        pol="s",
        N=N_torch,
        T=T_torch,
        Theta=thetas,
        lambda_vacuum=lambda_nm,
        device=Computation_device
    )

    T_sim = result["T"][0].mean(dim=0)

    loss = loss_fn(T_sim, target_T)
    loss.backward()
    optimizer.step()

    # Physical constraints
    with torch.no_grad():
        d_init_nm[1:-1].clamp_(1.0, 300.0)
    
    if step == 0:
        print("CUDA available:", torch.cuda.is_available())
        print("Allocated GPU memory:", torch.cuda.memory_allocated())

    if step % 100 == 0:
        print(
            f"Step {step:3d} | "
            f"Loss = {loss.item():.4e} | "
            f"Thicknesses (nm) = {d_init_nm[1:-1].detach().cpu().numpy()}"
        )

#%% Final plot

plt.figure()
plt.plot(wl_nm, T_sim.cpu().detach().numpy(), "r--", label="TMM fit")
plt.plot(wl_nm, target_T.cpu().numpy(), "k", label="Measured")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmittance")
plt.legend()
plt.tight_layout()
plt.savefig("comparison.png", dpi=300)
plt.show()

# %%
