import mph
from scipy.optimize import minimize
from pathlib import Path
import sys
from tqdm import tqdm
import logging
import traceback

logging.basicConfig(filename="optimization_log.txt", level=logging.INFO, format="%(message)s")

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

# --- Physical bounds (0× to 5× baseline) ---
param_bounds = {
    "V_rf":           (0, 1500),
    "V_dc":           (0, 250),
    "endcap_dc":      (0, 50),
    "rod_spacing":    (0, 0.020),
    "rod_radius":     (0, 0.010),
    "rod_length":     (0, 0.200),
    "endcap_offset":  (0, 0.005),
    "V_endcap":       (0, 50)
}

def unscale(normalized):
    keys = list(param_bounds.keys())
    return [
        param_bounds[key][0] + normalized[i] * (param_bounds[key][1] - param_bounds[key][0])
        for i, key in enumerate(keys)
    ]

def main():
    model_path = find_model_file()
    print("Starting COMSOL client (mph.start)...")
    client = mph.start(cores=2, version="6.3")

    try:
        print(f"Loading model from: {model_path}")
        model = client.load(str(model_path))

        # --- Baseline evaluation ---
        print("\n📊 Evaluating baseline configuration...")
        baseline_values = {
            "V_rf": 300,
            "V_dc": 50,
            "endcap_dc": 10,
            "rod_spacing": 0.004,
            "rod_radius": 0.002,
            "rod_length": 0.04,
            "endcap_offset": 0.001,
            "V_endcap": 10
        }

        for k, v in baseline_values.items():
            model.parameter(k, v)

        baseline_depth = baseline_power = baseline_offset = None
        try:
            model.solve()
            baseline_depth = float(model.evaluate("depth_eV"))
            baseline_power = float(model.evaluate("P_est_mW"))
            baseline_offset = float(model.evaluate("offset_m"))

            eps = 1e-9
            alpha, beta = 1.0, 1e8
            baseline_cost = alpha * (baseline_power / (baseline_depth + eps)) + beta * (baseline_offset ** 2)

            print(f"Baseline depth_eV={baseline_depth:.6f}, P_est_mW={baseline_power:.2f}, offset_m={baseline_offset:.6f}")
            print(f"Baseline Power/Depth={baseline_power/(baseline_depth+eps):.2f}, OffsetPenalty={beta*baseline_offset**2:.2f}")
            print(f"Baseline Cost={baseline_cost:.2f}\n")

        except Exception:
            print("⚠️ Baseline solve failed. Continuing to optimization...")

        iteration_counter = tqdm(total=50, desc="Optimizing", unit="iter")

        def objective(normalized_params):
            try:
                values = unscale(normalized_params)
                keys = list(param_bounds.keys())
                for k, v in zip(keys, values):
                    model.parameter(k, v)

                try:
                    model.solve()
                except Exception:
                    logging.info(f"⚠️ Solve failed for params: {normalized_params}")
                    iteration_counter.update(1)
                    return 1e6

                depth_eV = float(model.evaluate("depth_eV"))
                P_est_mW = float(model.evaluate("P_est_mW"))
                offset_m = float(model.evaluate("offset_m"))

                eps = 1e-9
                alpha, beta = 1.0, 1e8
                cost = alpha * (P_est_mW / (depth_eV + eps)) + beta * (offset_m ** 2)

                param_log = ", ".join([f"{k}={v:.4f}" for k, v in zip(keys, values)])
                logging.info(f"{param_log} | depth_eV={depth_eV:.6f}, P_est_mW={P_est_mW:.2f}, offset_m={offset_m:.6f} | "
                             f"Power/Depth={P_est_mW/(depth_eV+eps):.2f}, OffsetPenalty={beta*offset_m**2:.2f} | Cost={cost:.2f}")

                iteration_counter.update(1)
                print(f"Iter: {param_log} | depth_eV={depth_eV:.6f}, P_est_mW={P_est_mW:.2f}, offset_m={offset_m:.6f} | Cost={cost:.2f}")

                return cost

            except Exception:
                logging.info(f"⚠️ Evaluation error for params: {normalized_params}")
                iteration_counter.update(1)
                return 1e6

        initial_guess = [0.4] * len(param_bounds)
        bounds = [(0, 1)] * len(param_bounds)

        print("Running optimization...")
        result = minimize(objective, initial_guess, bounds=bounds, method="Powell", options={"maxiter": 50})

        iteration_counter.close()

        final_values = unscale(result.x)
        keys = list(param_bounds.keys())
        print("\n✅ Optimization complete:")
        for k, v in zip(keys, final_values):
            print(f"{k:<15} = {v:.6f}")
        print(f"Final cost     = {result.fun:.6f}")

        # --- Improvement report ---
        try:
            final_depth = float(model.evaluate("depth_eV"))
            final_power = float(model.evaluate("P_est_mW"))
            final_offset = float(model.evaluate("offset_m"))

            if baseline_depth and baseline_power and baseline_offset:
                depth_change = 100 * (final_depth - baseline_depth) / baseline_depth
                power_change = 100 * (final_power - baseline_power) / baseline_power
                offset_change = 100 * (final_offset - baseline_offset) / baseline_offset

                print("\n📈 Improvement Report:")
                print(f"Trap depth change: {depth_change:+.2f}%")
                print(f"Power change:      {power_change:+.2f}%")
                print(f"Offset change:     {offset_change:+.2f}%")
        except Exception:
            print("⚠️ Could not compute improvement report.")

        model.save()
        client.remove(model)

    except Exception:
        print("An exception occurred while loading/solving the model:")
        traceback.print_exc()
        try:
            client.remove_all()
        except Exception:
            pass
        raise

if __name__ == "__main__":
    main()
