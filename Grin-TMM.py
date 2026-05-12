#%% ================== Imports =====================
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d

from winspec import SpeFile
import tmm_fast.gym_multilayerthinfilm as mltf
from tmm_fast import coh_tmm
from ga_thickness_optimizer_new import GeneticThicknessOptimizer
#from bruggemann_mixing import bruggeman_n
from bruggemann_mixing_new import bruggeman_n_multi
import time
from datetime import timedelta, datetime
from zoneinfo import ZoneInfo

#%% ================== Helper =====================
def moving_average_same(a, n=5):
    return uniform_filter1d(a, size=n, mode="reflect")

#%% ================= USER SETTINGS =================
path = str(Path(__file__).parent)

material_mixing = True   # <<< MASTER SWITCH

materials = {
    "Cu":   path + "/OpticalConstants/nk_Cu.txt",
    "Cu2O": path + "/OpticalConstants/nk_Cu2O.txt",
    "CuO":  path + "/OpticalConstants/nk_CuO.txt",
    "Vacuum": path + "/OpticalConstants/nk_Vacuum.txt",
}

# --- Spectrum-dependent layer bound overrides ---
# key = (n_spec - spec), same convention as secondary_guesses
layer_bounds_overrides = {}
# layer_bounds_overrides = {
#     12: {
#         "CuO": (0.0, 100.0),
#     }
# }

"""{
            "material": "Cu2O",
            "shape": "sphere",
            "fraction_init": 0.0,
            "bounds": (0.0, 0.3),
        }"""

layers = [
    {
        "name" : "Cu",
        "matrix" : "Cu",
        "shape" : "sphere",
        "thickness_init" : 40.0,
        "thickness_bounds" : (0.0, 60.0),
        "inclusions": None,
    },
    {
        "name": "Cu2O",
        "matrix": "Cu2O",
        "shape": "sphere",
        "thickness_init": 0.0,
        "thickness_bounds": (0.0, 250.0),
        "inclusions": None,
    },
    {
        "name" : "CuO",
        "matrix" : "CuO",
        "shape" : "sphere",
        "thickness_init" : 0.0,
        "thickness_bounds" : (0.0, 100.0),
        "inclusions": None,
    },
]


#Place guesses for specific spectrum here: NOTE the number is the n_spec - spec, i.e. the one that is printed in the console like this: "=== Fitting spectrum x / N ==="
secondary_guesses = {}
"""secondary_guesses = {
    19: [
        torch.tensor([27.80, 39.36, 4.11, 3.02, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([27.80, 39.36, 5, 3.02, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([25, 39.36, 0, 3.02, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([29.80, 39.36, 4, 3.02, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
    ]
}"""


#--------- File Settings -----------
SPE_file  = path + "/16_03_2026-Sample9-remeasured/GRIN-2W-60s.SPE"
Lamp_file = path + "/16_03_2026-Sample9-remeasured/Substrate-2W.SPE"

#test_CSV = path + "/sampCu9.csv"

exclude_even_spectra = True #This option is only used for the case where no automatic shutter is located at the spectrometer and there allways need to be one "flush" spectrum
substrateSpectrum_no = 2 #Select which of the lamp spectrums is used (in case of single spectrum use 0)

spectra_fitting_range = -1 #set to -1 to fit all spectra imported
#saveName = "sample9-remeasured_Cu_Cu2O_CuO-40s_3W_scale-0_65"
saveName = "Sample9-remeasured-2W-60s-Cu-Cu2O-CuO-scale0_6-for-Thesis"

#-------- GA Settings -------------
device = "cpu"
pop_size = 50
generations = 1000
smart_mutation_scaling = True
mutation_scale_thickness = 5 #5 best value usually
mutation_scale_volume_fraction= 0.035 #guessed value because sigma= (xmax-xmin) / 6
elite_percentage = 0.1
mutation_rate = 0.5
crossover_fraction = 0.8
redo_on_rmse_jump = False

stall_generations = 50
stall_increase_mutation_factor_thickness = 2.0
stall_increase_mutation_factor_volume_fraction = 2.0
stall_increase_crossover_fraction = 0.8
increase_mutation_rate_stall = 2 
sigma = [2,6]

RMSE_convergence_threshold = 0.0

scaling_parameter = 0.6 #0.56 scales the transmission amplitude by this factor (used for calibration afterwards)

# -------- Wavelength cut -------- 
enable_wl_cut = True 
wl_opt_min = 450.0 
wl_opt_max = 950.0

#%% ================= Load data =================
wl_nm = SpeFile(SPE_file).xaxis.astype(np.float64)
if exclude_even_spectra:
    I = SpeFile(SPE_file).data[1::2, :, 0]  # spectra 0, 2, 4, ...
else:
    I = SpeFile(SPE_file).data[:, :, 0]
I_lamp = SpeFile(Lamp_file).data[substrateSpectrum_no, :, 0]

I = np.array([moving_average_same(x, 5) for x in I])
I_lamp = moving_average_same(I_lamp, 5)

T_exp_all = np.multiply(I / I_lamp, scaling_parameter)


if np.any((T_exp_all < 0) | (T_exp_all > 1)):
    print("WARNING: T_exp_all contains values outside the [0, 1] interval.\n")
    print(f"T_exp_all min = {T_exp_all.min():.3g}, max = {T_exp_all.max():.3g}\n")

if np.any(T_exp_all > 1):
    T_exp_all = T_exp_all / np.max(T_exp_all)
    print("Corrected for values >1!\n")

if np.any((T_exp_all < 0) | (T_exp_all > 1)):
    print("WARNING: T_exp_all still has values outside [0, 1] interval.\n")
    print(f"T_exp_all min = {T_exp_all.min():.3g}, max = {T_exp_all.max():.3g}\n")

if enable_wl_cut: 
    wl_mask = (wl_nm >= wl_opt_min) & (wl_nm <= wl_opt_max) 
    wl_nm = wl_nm[wl_mask] 
    T_exp_all = T_exp_all[:, wl_mask]

    I_lamp = I_lamp[wl_mask]
    I = I[:,wl_mask]

#======== BENCHMARK IMPORT ========
# --- load wavelength axis ---
#wl_nm = np.load("easy-structure_benchmark_wl.npy")
# --- load transmission ---
#T_exp_all = np.load("easy-structure_benchmark_T.npy")

lambda_nm = torch.tensor(wl_nm, dtype=torch.float64, device=device)
"""
CSV_DF = pd.read_csv(test_CSV, delimiter='\t', header = 1)

wl_nm = CSV_DF['Wavelength (nm)'].to_numpy()
T_csv = CSV_DF['%T'].to_numpy()

# convert percent → fraction if necessary
if T_csv.max() > 1:
    T_csv = T_csv / 100.0

# make it 2D like SPE pipeline
T_exp_all = T_csv[np.newaxis, :]   # shape (1, N)

# convert to torch
lambda_nm = torch.tensor(wl_nm, dtype=torch.float64, device=device)
T_exp_all = torch.tensor(T_exp_all, dtype=torch.float64, device=device)
"""
n_spec = T_exp_all.shape[0]

if spectra_fitting_range == -1:
    spectra_fitting_range = n_spec

#%% Test plotting code

# plt.figure()
# plt.plot(wl_nm,I[-1,:],color="red",label="Cu")
# plt.plot(wl_nm, I_lamp,label="Lamp")
# plt.xlabel("Wavelength / nm", fontsize=14)
# plt.ylabel("Counts", fontsize=14)
# plt.legend(fontsize=12)
# print("WL Shape:" + str(wl_nm.shape))
# print("Lamp Shape:" + str(I_lamp.shape))
# print("Transmission Shape:" + str(T_exp_all[0].shape))
# plt.savefig("test-lamp-plot.png")
#%%
plt.figure()
plt.plot(wl_nm,T_exp_all[-1,:],color="red")
plt.savefig("test-transmission-plot.png")
#%% ================= Refractive indices =================
N_np = mltf.get_N(
    list(materials.values()),
    wl_nm.min(),
    wl_nm.max(),
    points=len(wl_nm),
    complex_n=True
)

mat_index = {k: i for i, k in enumerate(materials.keys())}

#%% ================= Optimization =================
records = []

init_d = torch.tensor(
    [l["thickness_init"] for l in layers],
    dtype=torch.float64,
    device=device,
)

inclusions_per_layer = []

for layer in layers:
    if layer.get("inclusions"):
        inclusions_per_layer.append(len(layer["inclusions"]))
    else:
        inclusions_per_layer.append(0)

init_f = []

for layer in layers:
    if layer.get("inclusions"):
        for inc in layer["inclusions"]:
            init_f.append(inc["fraction_init"])

init_f = torch.tensor(init_f, dtype=torch.float64, device=device)

fraction_bounds = []

for layer in layers:
    if layer.get("inclusions"):
        for inc in layer["inclusions"]:
            fraction_bounds.append(inc["bounds"])


def fitness_torch(d, f, target_T):

    N_list = [torch.ones_like(lambda_nm, dtype=torch.complex128, device=device)]

    frac_offset = 0

    for i, layer in enumerate(layers):
        n_mat = torch.tensor(
            N_np[mat_index[layer["matrix"]]],
            dtype=torch.complex128,
            device=device,
        )

        if material_mixing and layer.get("inclusions"):

            n_list = []
            f_list = []
            shape_list = []

            f_sum = 0.0

            # --- inclusions ---
            for j, inc in enumerate(layer["inclusions"]):

                fi = f[frac_offset].clamp(*inc["bounds"])
                f_sum += fi

                n_inc = torch.tensor(
                    N_np[mat_index[inc["material"]]],
                    dtype=torch.complex128,
                    device=device,
                )

                n_list.append(n_inc)
                f_list.append(fi)
                shape_list.append(inc["shape"])

                frac_offset += 1

            # --- matrix fraction ---
            f_matrix = torch.clamp(1.0 - f_sum, min=0.0)

            n_list.append(n_mat)
            f_list.append(f_matrix)
            shape_list.append(layer["shape"])

            n_eff = bruggeman_n_multi(
                n_list=n_list,
                f_list=f_list,
                shape_list=shape_list,
            )

            N_list.append(n_eff)

        else:
            N_list.append(n_mat)

    # substrate (air)
    N_list.append(torch.ones_like(lambda_nm, dtype=torch.complex128, device=device))

    N = torch.stack(N_list).unsqueeze(0)

    d_full = torch.cat([
        torch.tensor([np.inf], device=device),
        d.to(device),
        torch.tensor([np.inf], device=device),
    ])

    T_sim = coh_tmm(
        pol="s",
        N=N,
        T=d_full.unsqueeze(0),
        Theta=torch.zeros(1),
        lambda_vacuum=lambda_nm,
        device=device,
    )["T"][0]

    return torch.sqrt(torch.mean((T_sim - target_T) ** 2))

#%% ================= Main loop =================
start_time = time.time()
prev_rmse = None

for spec in range(n_spec - 1, n_spec - spectra_fitting_range - 1, -1):

    print(f"\n=== Fitting spectrum {n_spec - spec} / {spectra_fitting_range} ===")
    
    target_T = torch.tensor(T_exp_all[spec], dtype=torch.float64, device=device)
    
    # --- Resolve active thickness bounds for this spectrum ---
    spec_idx = n_spec - spec

    thickness_bounds = []
    for layer in layers:
        # start with default
        tb = layer.get("thickness_bounds", (1e-3, 300.0))

        # apply overrides if active
        for k, overrides in layer_bounds_overrides.items():
            if spec_idx >= k and layer["name"] in overrides:
                tb = overrides[layer["name"]]

        thickness_bounds.append(tb)
    
    def fitness_ga(x):
        with torch.no_grad():
            d = x[:len(layers)]
            f = x[len(layers):]
            return fitness_torch(d, f, target_T).item()

    ga = GeneticThicknessOptimizer(
        fitness_fn=fitness_ga,
        n_layers=len(layers),
        bounds_thickness=thickness_bounds,
        bounds_fraction=fraction_bounds,
        population_size=pop_size,
        mutation_rate=mutation_rate,
        elite_fraction=elite_percentage,
        device=device,
        mutation_scale_volume_fraction=mutation_scale_volume_fraction,
        mutation_scale_thickness=mutation_scale_thickness,
        crossover_fraction=crossover_fraction,
        stall_generations=stall_generations,
        stall_increase_mutation_factor_thickness=stall_increase_mutation_factor_thickness,
        stall_increase_mutation_factor_volume_fraction= stall_increase_mutation_factor_volume_fraction,
        stall_increase_crossover_fraction=stall_increase_crossover_fraction,
        RMSE_convergence_threshold=RMSE_convergence_threshold,
        smart_mutation_scaling = smart_mutation_scaling,
        inclusions_per_layer = inclusions_per_layer,
        increase_mutation_rate_stall = increase_mutation_rate_stall,
        sigma = sigma,
    )

    ga.initialize(init_d, init_f)

    # --- inject secondary guesses if provided ---
    if (n_spec - spec) in secondary_guesses:
        ga.inject_elites(secondary_guesses[n_spec - spec])
        print("Injected Spectrum\n")

    best = ga.run(generations)


    d_best = best[:len(layers)]
    f_best = best[len(layers):]

    #set the best fitting options from the last spectrum as init guess for next one
    init_d = d_best
    init_f = f_best

    rmse = fitness_torch(d_best, f_best, target_T).item()

    #Compute the final optimization once (to safe)
    with torch.no_grad():
        frac_offset = 0
        N_list = [torch.ones_like(lambda_nm, dtype=torch.complex128)]

        for i, layer in enumerate(layers):
            n_mat = torch.tensor(
                N_np[mat_index[layer["matrix"]]],
                dtype=torch.complex128,
                device=device,
            )

            if material_mixing and layer.get("inclusions"):

                n_list = []
                f_list = []
                shape_list = []

                f_sum = 0.0

                for j, inc in enumerate(layer["inclusions"]):

                    f_val = f_best[frac_offset].clamp(*inc["bounds"])
                    f_sum += f_val

                    n_inc = torch.tensor(
                        N_np[mat_index[inc["material"]]],
                        dtype=torch.complex128,
                        device=device,
                    )

                    n_list.append(n_inc)
                    f_list.append(f_val)
                    shape_list.append(inc["shape"])

                    frac_offset += 1

                # matrix fraction
                f_matrix = torch.clamp(1.0 - f_sum, min=0.0)

                n_list.append(n_mat)
                f_list.append(f_matrix)
                shape_list.append(layer["shape"])

                n_eff = bruggeman_n_multi(
                    n_list=n_list,
                    f_list=f_list,
                    shape_list=shape_list,
                )

                N_list.append(n_eff)
            else:
                N_list.append(n_mat)

        N_list.append(torch.ones_like(lambda_nm))
        N = torch.stack(N_list).unsqueeze(0)

        d_full = torch.cat([
            torch.tensor([np.inf],device=device),
            d_best.to(device),
            torch.tensor([np.inf],device=device),
        ])

        T_sim = coh_tmm(
            pol="s",
            N=N,
            T=d_full.unsqueeze(0),
            Theta=torch.zeros(1),
            lambda_vacuum=lambda_nm,
            device=device,
        )["T"][0].mean(dim=0).cpu().numpy()

    rmse = fitness_torch(d_best, f_best, target_T).item()

    # -------- Save structured info --------
    for i_wl, (wl, Texp) in enumerate(zip(wl_nm, target_T.cpu().numpy())):
        
        frac_offset = 0
        
        row = {
            "spectrum": spec,
            "wavelength_nm": wl,
            "T_exp": Texp,
            "T_fit": T_sim[i_wl],
            "RMSE": rmse,
        }

        for i, layer in enumerate(layers):

            row[f"layer_{i+1}_matrix"] = layer["matrix"]
            row[f"layer_{i+1}_thickness_nm"] = d_best[i].item()
            row[f"layer_{i+1}_matrix_shape"] = layer["shape"]

            inclusions = layer.get("inclusions") or []

            if len(inclusions) == 0:
                row[f"layer_{i+1}_matrix_fraction"] = 1.0
                continue

            f_sum = 0.0

            for j, inc in enumerate(inclusions):

                f_val = f_best[frac_offset].item()
                f_sum += f_val

                row[f"layer_{i+1}_inc_{j+1}_material"] = inc["material"]
                row[f"layer_{i+1}_inc_{j+1}_shape"] = inc["shape"]
                row[f"layer_{i+1}_inc_{j+1}_fraction"] = f_val

                frac_offset += 1

            row[f"layer_{i+1}_matrix_fraction"] = 1.0 - f_sum

        records.append(row)

    # ---- Timing information ----
    now = time.time()
    total_time_running = now - start_time

    completed_specs = completed_specs = n_spec - spec

    estimated_total_time = (
        total_time_running / completed_specs
    ) * spectra_fitting_range

    time_left = max(estimated_total_time - total_time_running, 0)

    elapsed_str = str(timedelta(seconds=int(total_time_running)))
    remaining_str = str(timedelta(seconds=int(time_left)))

    eta_time = datetime.now(ZoneInfo("Europe/Berlin")) + timedelta(
        seconds=int(time_left)
    )

    print(
        f"[TIME] Elapsed: {elapsed_str} | "
        f"Remaining: {remaining_str} | "
        f"ETA: {eta_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

#%% ================= Save =================
df = pd.DataFrame(records)
outFileName = saveName + ".csv"
out = Path(path) / outFileName
df.to_csv(out, index=False)
print("Saved:", out)
