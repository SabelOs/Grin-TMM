#%% ================== Imports =====================
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from scipy.ndimage import uniform_filter1d

from winspec import SpeFile
import tmm_fast.gym_multilayerthinfilm as mltf
from tmm_fast import coh_tmm
from ga_thickness_optimizer import GeneticThicknessOptimizer
from bruggemann_mixing import bruggeman_n

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
}

layers = [
    {
        "name": "Cu",
        "matrix": "Cu",
        "shape": "sphere",
        "thickness_init": 22.0,
        "inclusion": {
            "material": "Cu2O",
            "shape": "sphere",
            "fraction_init": 0.01,
            "bounds": (0.0, 0.3),
        },
    },
    {
        "name": "Cu2O",
        "matrix": "Cu2O",
        "shape": "sphere",
        "thickness_init": 1.0,
        "inclusion": None,
    },
    {
        "name": "CuO",
        "matrix": "CuO",
        "shape": "sphere",
        "thickness_init": 1.0,
        "inclusion": None,
    },
]

SPE_file  = path + "/GRIN.SPE"
Lamp_file = path + "/Substrate.SPE"

device = "cpu"
pop_size = 20
generations = 20
#%% ================= Load data =================
wl_nm = SpeFile(SPE_file).xaxis.astype(np.float64)
I = SpeFile(SPE_file).data[:, :, 0]
I_lamp = SpeFile(Lamp_file).data[0, :, 0]

I = np.array([moving_average_same(x, 5) for x in I])
I_lamp = moving_average_same(I_lamp, 5)

T_exp_all = I / I_lamp

lambda_nm = torch.tensor(wl_nm, dtype=torch.float64, device=device)

n_spec = T_exp_all.shape[0]
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
    dtype=torch.float64
)

init_f = torch.tensor(
    [l["inclusion"]["fraction_init"] if l["inclusion"] else 0.0 for l in layers],
    dtype=torch.float64
)

fraction_bounds = [
    l["inclusion"]["bounds"] if l["inclusion"] else (0.0, 0.0)
    for l in layers
]

def fitness_torch(d, f, target_T):

    N_list = [torch.ones_like(lambda_nm, dtype=torch.complex128)]

    for i, layer in enumerate(layers):
        n_mat = torch.tensor(
            N_np[mat_index[layer["matrix"]]],
            dtype=torch.complex128,
        )

        if material_mixing and layer["inclusion"]:
            inc = layer["inclusion"]
            fi = f[i].clamp(*inc["bounds"])
            n_inc = torch.tensor(
                N_np[mat_index[inc["material"]]],
                dtype=torch.complex128,
            )
            n_eff = bruggeman_n(
                n1=n_mat,
                n2=n_inc,
                f1=fi,
                shape1=layer["shape"],
                shape2=inc["shape"],
            )
            N_list.append(n_eff)
        else:
            N_list.append(n_mat)

    N_list.append(torch.ones_like(lambda_nm))
    N = torch.stack(N_list).unsqueeze(0)

    d_full = torch.cat([
        torch.tensor([np.inf]),
        d,
        torch.tensor([np.inf]),
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
for spec in range(T_exp_all.shape[0]):
    
    print(f"\n=== Fitting spectrum {spec + 1} / {n_spec} ===")
    
    target_T = torch.tensor(T_exp_all[spec], dtype=torch.float64)

    def fitness_ga(x):
        with torch.no_grad():
            d = x[:len(layers)]
            f = x[len(layers):]
            return fitness_torch(d, f, target_T).item()

    ga = GeneticThicknessOptimizer(
        fitness_fn=fitness_ga,
        n_params=2 * len(layers),
        bounds_thickness=(1e-3, 300.0),
        bounds_fraction=fraction_bounds,
        population_size = pop_size,
    )

    ga.initialize(init_d, init_f)
    best = ga.run(generations)

    d_best = best[:len(layers)]
    f_best = best[len(layers):]

    #Compute the final optimization once (to safe)
    with torch.no_grad():
        N_list = [torch.ones_like(lambda_nm, dtype=torch.complex128)]

        for i, layer in enumerate(layers):
            n_mat = torch.tensor(
                N_np[mat_index[layer["matrix"]]],
                dtype=torch.complex128,
            )

            if material_mixing and layer["inclusion"]:
                inc = layer["inclusion"]
                fi = f_best[i].clamp(*inc["bounds"])
                n_inc = torch.tensor(
                    N_np[mat_index[inc["material"]]],
                    dtype=torch.complex128,
                )
                n_eff = bruggeman_n(
                    n1=n_mat,
                    n2=n_inc,
                    f1=fi,
                    shape1=layer["shape"],
                    shape2=inc["shape"],
                )
                N_list.append(n_eff)
            else:
                N_list.append(n_mat)

        N_list.append(torch.ones_like(lambda_nm))
        N = torch.stack(N_list).unsqueeze(0)

        d_full = torch.cat([
            torch.tensor([np.inf]),
            d_best,
            torch.tensor([np.inf]),
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


#%% ================= Save =================
df = pd.DataFrame(records)
out = Path(path) / "tmm_fit_results.csv"
df.to_csv(out, index=False)
print("Saved:", out)
