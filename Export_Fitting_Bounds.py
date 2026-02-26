import pandas as pd
import re
from pathlib import Path

# ================= USER SETTINGS =================

fileName = "sample9_Cu_Cu2O_CuO-120s_2_0W_test1.csv"

relative_interval = 0.20   # 0.10 = ±10%, 0.0 = exact value
round_digits = 2           # rounding of printed values

# =================================================

results_path = Path(__file__).parent / fileName
df = pd.read_csv(results_path)

# --- find material indices automatically ---
material_indices = sorted(
    int(m.group(1))
    for c in df.columns
    if (m := re.match(r"material_(\d+)_thickness_nm", c))
)

# --- group by spectrum ---
grouped = df.groupby("spectrum").first()

max_spectrum = grouped.index.max()

layer_bounds_overrides = {}

# =================================================
# =============== MAIN PROCESSING =================
# =================================================

for spectrum, row in grouped.iterrows():

    formatted_spectrum = int((max_spectrum + 1) - spectrum)

    material_bounds = {}

    for i in material_indices:

        thickness = row.get(f"material_{i}_thickness_nm", None)

        if thickness is None or pd.isna(thickness) or thickness == 0:
            continue

        mat_name = row.get(f"material_{i}_name", f"material_{i}")

        # Apply relative interval
        delta = thickness * relative_interval
        lower = thickness - delta
        upper = thickness + delta

        lower = round(lower, round_digits)
        upper = round(upper, round_digits)

        material_bounds[mat_name] = (lower, upper)

    if material_bounds:
        layer_bounds_overrides[formatted_spectrum] = material_bounds

# =================================================
# ================== PRINT OUTPUT =================
# =================================================

print("layer_bounds_overrides = {")

for spec in sorted(layer_bounds_overrides.keys()):
    print(f"    {spec}: {{")
    for mat, bounds in layer_bounds_overrides[spec].items():
        print(f'        "{mat}": ({bounds[0]}, {bounds[1]}),')
    print("    },")
    
print("}")