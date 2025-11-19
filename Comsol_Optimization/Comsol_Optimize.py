import mph
from pathlib import Path
import sys
import csv
from scipy.optimize import minimize
import os

# --- Baseline values from your COMSOL GUI ---
baseline_values = {
    "V_rf": 300,
    "V_dc": 50,
    "V_endcap": 10,
    "rod_spacing": 0.005,
    "rod_radius": 0.002,
    "rod_length": 0.04,
    "endcap_offset": 0.001
}

# --- Target values for normalization ---
targets = {
    "depth_eV": 5.0,     # want >= 5 eV
    "offset_mm": 0.0001,    # want ~0 mm
    "P_est_mW": 1000.0      # want ~1000 mW
}

# --- Weights for each objective ---
weights = {
    "depth_eV": 1.0,
    "offset_mm": 10.0,
    "P_est_mW": 0.1
}

def find_model_file(preferred_name: str = "3d_pole_trap - Copy.mph") -> Path:
    cwd = Path(__file__).resolve().parent
    preferred_path = cwd / preferred_name
    if preferred_path.exists():
        print(f"Using model file: {preferred_path}")
        return preferred_path

    candidates = list(cwd.glob("*.mph"))
    if candidates:
        print("Preferred model not found. Available .mph files in the folder:")
        for i, p in enumerate(candidates, 1):
            print(f"  {i}. {p.name}")
        fallback = candidates[0]
        print(f"Falling back to: {fallback}")
        return fallback

    print(f"No .mph model file found in {cwd}. Please place your COMSOL model there or provide the correct path.")
    sys.exit(2)

def try_eval(model, name):
    try:
        return float(model.evaluate(name))
    except Exception:
        return None
    
def objective(depth_eV, offset_mm, P_est_mW):
    # Normalized scores relative to targets
    depth_score  = depth_eV / (targets["depth_eV"] + 1e-9)
    offset_score = (targets["offset_mm"] + 1e-9) / (offset_mm + 1e-9)
    power_score  = (targets["P_est_mW"] + 1e-9) / (P_est_mW + 1e-9)
    if offset_mm > 10:
        return -1e6

    # Weighted sum
    score = (weights["depth_eV"] * depth_score) \
          + (weights["offset_mm"] * offset_score) \
          + (weights["P_est_mW"] * power_score)
    return score

def run_trial(params, model, writer, f):
    # unpack params
    V_rf, V_dc, V_endcap, rod_spacing, rod_radius, rod_length, endcap_offset = params

    # set COMSOL parameters
    model.parameter("V_rf", V_rf)
    model.parameter("V_dc", V_dc)
    model.parameter("V_endcap", V_endcap)
    model.parameter("rod_spacing", rod_spacing)
    model.parameter("rod_radius", rod_radius)
    model.parameter("rod_length", rod_length)
    model.parameter("endcap_offset", endcap_offset)

    print("Running trial with params:", params)

    try:
        model.solve()
    except Exception as e:
        print("COMSOL study run failed:", e)
        return 1e6
    
    print('solved Trial')

    depth_eV   = try_eval(model, "depth_eV")
    offset_mm  = try_eval(model, "offset_mm")
    P_est_mW   = try_eval(model, "P_est_mW")
    print("depth_eV:", depth_eV, "offset_mm:", offset_mm, "P_est_mW:", P_est_mW)


    score = objective(depth_eV, offset_mm, P_est_mW)
    print("Optimizer result:", -score)

    try:
        # write a row using the provided DictWriter and flush the underlying file
        writer.writerow({
                "V_rf": V_rf, "V_dc": V_dc, "V_endcap": V_endcap,
                "rod_spacing": rod_spacing, "rod_radius": rod_radius,
                "rod_length": rod_length, "endcap_offset": endcap_offset,
                "depth_eV": depth_eV, "offset_mm": offset_mm,
                "P_est_mW": P_est_mW, "score": score
            })
        f.flush()
        os.fsync(f.fileno())

        print("Row written")
    except Exception as e:
        print("Failed to write row:", e)

    return -score  # minimize negative score
def main():
    model_path = find_model_file()
    print("Starting COMSOL client...")
    client = mph.start(cores=8, version="6.3") #THE CORE COUNT IS SO IMPORTANT GODDAMNIT KEEP IT 8

    try:
        print(f"Loading model: {model_path}")
        model = client.load(str(model_path))

        # --- Print all COMSOL parameters (expression + value) ---
        print("\n📋 All COMSOL parameters:")
        exprs = model.parameters()
        for name, expr in exprs.items():
            val = model.parameter(name)
            print(f"  {name:<20} | Expression: {expr:<10} | Value: {val}")
        x0 = [baseline_values["V_rf"], baseline_values["V_dc"], baseline_values["V_endcap"],
                  baseline_values["rod_spacing"], baseline_values["rod_radius"],
                  baseline_values["rod_length"], baseline_values["endcap_offset"]]

            

        with open("optimization_log.csv", "w", newline="") as f:
            fieldnames = ["V_rf","V_dc","V_endcap","rod_spacing","rod_radius",
                                "rod_length","endcap_offset","depth_eV","offset_mm","P_est_mW","score"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # run optimizer, passing both the DictWriter and the open file handle for flushing
            result = minimize(lambda p: run_trial(p, model, writer, f),
                            x0, method="Nelder-Mead", options={"maxiter": 50})
        
        
        model.save()
        client.remove(model)

    except Exception as e:
        print("❌ Exception occurred:")
        print(e)
        try:
            client.remove_all()
        except Exception:
            pass
        raise

if __name__ == "__main__":
    main()