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
from bruggemann_mixing import bruggeman_n
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
layer_bounds_overrides = {
    1: {
        "Cu": (29.97, 44.95),
    },
    2: {
        "Cu": (29.7, 44.55),
    },
    3: {
        "Cu": (29.45, 44.17),
    },
    4: {
        "Cu": (29.45, 44.17),
    },
    5: {
        "Cu": (28.81, 43.22),
    },
    6: {
        "Cu": (28.07, 42.1),
    },
    7: {
        "Cu": (27.62, 41.44),
    },
    8: {
        "Cu": (26.79, 40.18),
    },
    9: {
        "Cu": (26.32, 39.47),
    },
    10: {
        "Cu": (25.55, 38.32),
    },
    11: {
        "Cu": (24.66, 37.0),
    },
    12: {
        "Cu": (26.08, 39.12),
        "CuO": (10.76, 16.14),
    },
    13: {
        "Cu": (25.17, 37.75),
        "CuO": (12.43, 18.64),
    },
    14: {
        "Cu": (24.21, 36.31),
        "CuO": (14.0, 21.01),
    },
    15: {
        "Cu": (22.44, 33.66),
        "CuO": (15.56, 23.34),
    },
    16: {
        "Cu": (20.67, 31.01),
        "CuO": (17.09, 25.64),
    },
    17: {
        "Cu": (19.14, 28.72),
        "CuO": (18.27, 27.41),
    },
    18: {
        "Cu": (18.19, 27.29),
        "CuO": (19.51, 29.26),
    },
    19: {
        "Cu": (6.58, 9.86),
        "Cu2O": (59.64, 89.46),
        "CuO": (26.32, 39.48),
    },
    20: {
        "Cu": (5.18, 7.77),
        "Cu2O": (52.6, 78.91),
        "CuO": (27.35, 41.03),
    },
    21: {
        "Cu": (3.57, 5.35),
        "Cu2O": (48.43, 72.65),
        "CuO": (23.6, 35.4),
    },
    22: {
        "Cu2O": (47.29, 70.93),
        "CuO": (13.06, 19.58),
    },
    23: {
        "Cu2O": (51.34, 77.01),
        "CuO": (8.04, 12.07),
    },
    24: {
        "Cu2O": (52.09, 78.14),
        "CuO": (6.86, 10.29),
    },
    25: {
        "Cu2O": (52.32, 78.48),
        "CuO": (6.5, 9.74),
    },
    26: {
        "Cu2O": (52.48, 78.73),
        "CuO": (6.15, 9.22),
    },
    27: {
        "Cu2O": (52.48, 78.73),
        "CuO": (6.06, 9.08),
    },
    28: {
        "Cu2O": (52.39, 78.58),
        "CuO": (6.03, 9.04),
    },
    29: {
        "Cu2O": (52.38, 78.57),
        "CuO": (6.04, 9.05),
    },
    30: {
        "Cu2O": (52.35, 78.52),
        "CuO": (6.04, 9.05),
    },
    31: {
        "Cu2O": (52.25, 78.37),
        "CuO": (6.21, 9.32),
    },
    32: {
        "Cu2O": (51.83, 77.74),
        "CuO": (6.65, 9.97),
    },
    33: {
        "Cu2O": (51.17, 76.76),
        "CuO": (7.56, 11.34),
    },
    34: {
        "Cu2O": (50.58, 75.87),
        "CuO": (8.46, 12.69),
    },
    35: {
        "Cu2O": (48.69, 73.03),
        "CuO": (10.81, 16.22),
    },
    36: {
        "Cu2O": (46.93, 70.39),
        "CuO": (12.86, 19.3),
    },
    37: {
        "Cu2O": (45.98, 68.97),
        "CuO": (14.08, 21.12),
    },
    38: {
        "Cu": (0.38, 0.56),
        "Cu2O": (43.89, 65.84),
        "CuO": (20.49, 30.73),
    },
    39: {
        "Cu2O": (41.34, 62.01),
        "CuO": (19.06, 28.59),
    },
    40: {
        "Cu2O": (37.36, 56.04),
        "CuO": (23.21, 34.82),
    },
    41: {
        "Cu2O": (33.6, 50.39),
        "CuO": (27.04, 40.55),
    },
    42: {
        "Cu2O": (29.1, 43.65),
        "CuO": (32.03, 48.05),
    },
    43: {
        "Cu2O": (15.89, 23.84),
        "CuO": (48.3, 72.46),
    },
    44: {
        "Cu2O": (11.77, 17.65),
        "CuO": (53.61, 80.42),
    },
    45: {
        "Cu2O": (10.0, 15.0),
        "CuO": (55.3, 82.94),
    },
    46: {
        "Cu2O": (8.37, 12.55),
        "CuO": (57.14, 85.71),
    },
    47: {
        "Cu2O": (5.15, 7.73),
        "CuO": (60.21, 90.31),
    },
    48: {
        "Cu2O": (3.06, 4.58),
        "CuO": (62.18, 93.27),
    },
    49: {
        "Cu2O": (1.87, 2.81),
        "CuO": (62.99, 94.48),
    },
    50: {
        "Cu2O": (0.34, 0.51),
        "CuO": (64.41, 96.61),
    },
    51: {
        "Cu": (0.79, 1.19),
        "Cu2O": (3.23, 4.85),
        "CuO": (64.63, 96.94),
    },
    52: {
        "Cu": (0.79, 1.19),
        "Cu2O": (3.43, 5.15),
        "CuO": (64.46, 96.69),
    },
}
layer_bounds_overrides = {}

"""{
            "material": "Cu2O",
            "shape": "sphere",
            "fraction_init": 0.0,
            "bounds": (0.0, 0.3),
        }"""

layers = [
    {
        "name": "Cu",
        "matrix": "Cu",
        "shape": "sphere",
        "thickness_init": 34.59,
        "thickness_bounds": (0.0, 45.0),
        "inclusion": None,
    },
    {
        "name": "Cu2O",
        "matrix": "Cu2O",
        "shape": "sphere",
        "thickness_init": 0.0,
        "thickness_bounds": (0.0, 150.0),  # ← default BEFORE override
        "inclusion": {
            "material": "Cu",
            "shape": "sphere",
            "fraction_init": 0.0,
            "bounds": (0.0, 1.0),
        }
    },
    {
        "name": "CuO",
        "matrix": "CuO",
        "shape": "sphere",
        "thickness_init": 0.0,
        "thickness_bounds": (0.0, 150.0),  # ← default BEFORE override
        "inclusion": None,
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
#SPE_file  = path + "/Sample1_BiggestGrin.SPE"
#Lamp_file = path + "/Substrate_Xe.SPE"#"/Substrate-20xObj.SPE"
SPE_file  = path + "/Grin-2W-120s.SPE"
Lamp_file = path + "/Substrate-Grin-2W-120s.SPE"

#test_CSV = path + "/sampCu9.csv"

exclude_even_spectra = True #This option is only used for the case where no automatic shutter is located at the spectrometer and there allways need to be one "flush" spectrum
substrateSpectrum_no = 2 #Select which of the lamp spectrums is used (in case of single spectrum use 0)

spectra_fitting_range = -1 #set to -1 to fit all spectra imported
saveName = "sample9_Cu_Cu2O-Cu_sphere_CuO-120s_2_0W"

#-------- GA Settings -------------
device = "cpu"
pop_size = 50
generations = 100
mutation_scale_thickness = 5
mutation_scale_volume_fraction= 0.035 #guessed value because sigma= (xmax-xmin) / 6
elite_percentage = 0.1
mutation_rate = 0.1
crossover_fraction = 0.8
redo_on_rmse_jump = False

stall_generations = 30
stall_increase_mutation_factor_thickness = 2.0
stall_increase_mutation_factor_volume_fraction = 2.0
stall_increase_crossover_fraction = 0.8

RMSE_convergence_threshold = 0.001

scaling_parameter = 0.56 #scales the transmission amplitude by this factor (used for calibration afterwards)

"""
Default Values for GA with 25nm copper film:
device = "cpu"
pop_size = 30
generations = 80
mutation_scale_thickness = 3
mutation_scale_volume_fraction= 0.05
elite_percentage = 0.1
mutation_rate = 0.05
"""
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
"""
plt.figure()
plt.plot(wl_nm,I[-1,:],color="red")
plt.plot(wl_nm, I_lamp)

print("WL Shape:" + str(wl_nm.shape))
print("Lamp Shape:" + str(I_lamp.shape))
print("Transmission Shape:" + str(T_exp_all[0].shape))
plt.savefig("test-lamp-plot.png")"""
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

init_f = torch.tensor(
    [l["inclusion"]["fraction_init"] if l["inclusion"] else 0.0 for l in layers],
    dtype=torch.float64,
    device=device,
)

fraction_bounds = [
    l["inclusion"]["bounds"] if l["inclusion"] else (0.0, 0.0)
    for l in layers
]

def fitness_torch(d, f, target_T):

    N_list = [torch.ones_like(lambda_nm, dtype=torch.complex128,device=device)]

    for i, layer in enumerate(layers):
        n_mat = torch.tensor(
            N_np[mat_index[layer["matrix"]]],
            dtype=torch.complex128,
            device=device,
        )

        if material_mixing and layer["inclusion"]:
            inc = layer["inclusion"]
            fi = f[i].clamp(*inc["bounds"])
            n_inc = torch.tensor(
                N_np[mat_index[inc["material"]]],
                dtype=torch.complex128,
                device=device,
            )
            n_eff = bruggeman_n(
                n1=n_mat,
                n2=n_inc,
                f1=1-fi,
                shape1=layer["shape"],
                shape2=inc["shape"],
            )
            N_list.append(n_eff)
        else:
            N_list.append(n_mat)

    N_list.append(torch.ones_like(lambda_nm))
    N = torch.stack(N_list).unsqueeze(0)

    d_full = torch.cat([
        torch.tensor([np.inf],device=device),
        d.to(device),
        torch.tensor([np.inf],device=device),
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
        n_params=2 * len(layers),
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

    #Mutation Annealing in case of RMSE Jump!
    rmse = fitness_torch(d_best, f_best, target_T).item()

    rmse_jump = False
    if prev_rmse is not None:
        if rmse > 1.5 * prev_rmse:
            rmse_jump = True

    prev_rmse = rmse

    if rmse_jump and redo_on_rmse_jump:
        print("⚠ RMSE jump detected — re-running GA with boosted mutation")

        ga = GeneticThicknessOptimizer(
            fitness_fn=fitness_ga,
            n_params=2 * len(layers),
            bounds_thickness=(1e-3, 300.0),
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
            stall_increase_mutation_factor_volume_fraction = stall_increase_mutation_factor_volume_fraction,
            stall_increase_crossover_fraction=stall_increase_crossover_fraction,
            RMSE_convergence_threshold=RMSE_convergence_threshold,
        )

        ga.initialize(init_d, init_f)

        # inject secondary guesses if present
        if spec in secondary_guesses:
            ga.inject_elites(secondary_guesses[spec])

        best = ga.run(int(generations * 0.5))
        
        d_best = best[:len(layers)]
        f_best = best[len(layers):]

        #set the best fitting options from the last spectrum as init guess for next one
        init_d = d_best
        init_f = f_best


    #Compute the final optimization once (to safe)
    with torch.no_grad():
        N_list = [torch.ones_like(lambda_nm, dtype=torch.complex128)]

        for i, layer in enumerate(layers):
            n_mat = torch.tensor(
                N_np[mat_index[layer["matrix"]]],
                dtype=torch.complex128,
                device=device,
            )

            if material_mixing and layer["inclusion"]:
                inc = layer["inclusion"]
                fi = f_best[i].clamp(*inc["bounds"])
                n_inc = torch.tensor(
                    N_np[mat_index[inc["material"]]],
                    dtype=torch.complex128,
                    device = device,
                )
                n_eff = bruggeman_n(
                    n1=n_mat,
                    n2=n_inc,
                    f1=1-fi,
                    shape1=layer["shape"],
                    shape2=inc["shape"],
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
    for i_wl, (wl, Texp) in enumerate(zip(wl_nm, target_T.numpy())):
        row = {
            "spectrum": spec,
            "wavelength_nm": wl,
            "T_exp": Texp,
            "T_fit": T_sim[i_wl],   # <<< ADDED
            "RMSE": rmse,
        }

        for i, layer in enumerate(layers):
            row[f"material_{i+1}_name"] = layer["matrix"]
            row[f"material_{i+1}_thickness_nm"] = d_best[i].item()
            row[f"material_{i+1}_shape"] = layer["shape"]

            if layer["inclusion"]:
                row[f"material_{i+1}_volume_fraction"] = 1.0 - f_best[i].item()
                row[f"inclusion_{i+1}_name"] = layer["inclusion"]["material"]
                row[f"inclusion_{i+1}_shape"] = layer["inclusion"]["shape"]
                row[f"inclusion_{i+1}_volume_fraction"] = f_best[i].item()
            else:
                row[f"material_{i+1}_volume_fraction"] = 1.0
                row[f"inclusion_{i+1}_name"] = None
                row[f"inclusion_{i+1}_shape"] = None
                row[f"inclusion_{i+1}_volume_fraction"] = None

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
