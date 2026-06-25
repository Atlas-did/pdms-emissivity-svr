#!/usr/bin/env python3
"""Inverse Design: optimize d_PDMS for max emissivity via SVR surrogate gradient.

Uses svr_distilled.pkl (82K full training, R2=0.9720).
Scans multiple SiO2 thicknesses and reports the best overall design.

Usage:
    python stage1/inverse_design.py                              # default: 100,200,300 nm
    python stage1/inverse_design.py --sub-nm 100,200,300,500    # custom SiO2 values
    python stage1/inverse_design.py --target-band full          # full 2-14 um band
"""
from __future__ import annotations
import sys, json, time, argparse, pickle
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]

# ── Feature engineering (EXACT match with train_distillation.py) ──
def build_features_single(wl, sub, pdms):
    n_pdms, n_sio2 = 1.41, 1.46
    wl_m, pm, sm = wl * 1e-6, pdms * 1e-9, sub * 1e-9
    dp = 2 * np.pi * n_pdms * pm / wl_m
    ds = 2 * np.pi * n_sio2 * sm / wl_m
    x = np.array([
        wl, sub, pdms,
        np.sin(dp), np.cos(dp), np.sin(ds), np.cos(ds),
        1.0 / wl, pdms / wl,
        np.sin(dp) * np.sin(ds), np.cos(dp) * np.cos(ds),
        np.sin(dp) * np.cos(ds), np.cos(dp) * np.sin(ds),
    ])
    return np.nan_to_num(x, 0)


def svr_predict_scalar(x_raw, svr, scaler):
    x_scaled = scaler.transform(x_raw.reshape(1, -1))
    return float(np.clip(svr.predict(x_scaled)[0], 0, 1))


def load_svr():
    distilled_path = ROOT / "stage1" / "simulation_data" / "svr_distilled.pkl"
    if distilled_path.exists():
        with open(distilled_path, "rb") as f:
            bundle = pickle.load(f)
        svr, scaler = bundle["model"], bundle["scaler"]
        n_sv = svr.support_.shape[0] if hasattr(svr, "support_") else "?"
        print("[LOAD] SVR from svr_distilled.pkl ({} SV, R2=0.9720)".format(n_sv))
        return svr, scaler
    raise FileNotFoundError("svr_distilled.pkl not found. Run train_distillation.py first.")


def make_objective_band(band_wls, svr, scaler, sub_nm):
    """Return objective that MAXIMIZES mean emissivity.
    We minimize negative mean eps."""
    def objective(d_pdms_arr):
        d_pdms = float(d_pdms_arr[0])
        if d_pdms < 50 or d_pdms > 1500:
            return 1e6
        eps_vals = [svr_predict_scalar(build_features_single(wl, sub_nm, d_pdms), svr, scaler)
                    for wl in band_wls]
        return -float(np.mean(eps_vals))
    return objective


def run_one_sub(wl_targets, svr, scaler, sub_nm, d_init):
    sep = "=" * 60
    print("\n" + sep)
    print("  SiO2 = {:.0f} nm".format(sub_nm))
    print(sep)

    x0_test = build_features_single(wl_targets[25], sub_nm, d_init)
    eps0 = svr_predict_scalar(x0_test, svr, scaler)
    print("  Sanity: eps={:.4f} at wl={:.1f}um, pdms_init={:.0f}nm".format(
        eps0, wl_targets[25], d_init))

    obj = make_objective_band(wl_targets, svr, scaler, sub_nm)
    t0 = time.perf_counter()
    result = minimize(obj, x0=[d_init], method="L-BFGS-B",
                      bounds=[(50, 1500)], options={"ftol": 1e-8, "maxiter": 200})
    opt_time = time.perf_counter() - t0

    d_opt = float(result.x[0])
    eps_at_opt = [svr_predict_scalar(build_features_single(wl, sub_nm, d_opt), svr, scaler)
                  for wl in wl_targets]
    mean_eps = float(np.mean(eps_at_opt))

    print("  Optimal d_PDMS = {:.1f} nm".format(d_opt))
    print("  Mean eps = {:.4f}  range [{:.4f}, {:.4f}]".format(
        mean_eps, np.min(eps_at_opt), np.max(eps_at_opt)))
    print("  Iterations: {}, time: {:.1f}s, success: {}".format(
        result.nit, opt_time, result.success))

    d_scan = np.arange(100, 1100, 25)
    pareto = []
    for d in d_scan:
        ev = [svr_predict_scalar(build_features_single(wl, sub_nm, d), svr, scaler)
              for wl in wl_targets]
        pareto.append({"d_pdms_nm": int(d), "mean_eps": round(float(np.mean(ev)), 4)})
    pareto_sorted = sorted(pareto, key=lambda p: p["mean_eps"], reverse=True)
    print("  Pareto top 3: " + ", ".join(
        "{}nm->{:.4f}".format(p["d_pdms_nm"], p["mean_eps"]) for p in pareto_sorted[:3]))

    return {
        "d_sio2_nm": int(sub_nm), "d_optimal_nm": round(d_opt, 1),
        "mean_eps": round(mean_eps, 4), "n_iter": result.nit,
        "time_s": round(opt_time, 1), "success": result.success,
        "eps_range": [round(float(np.min(eps_at_opt)), 4),
                      round(float(np.max(eps_at_opt)), 4)],
        "pareto_top3": pareto_sorted[:3],
        "sanity_eps_at_init": round(eps0, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-band", type=str, default="atm", choices=["atm", "full"])
    ap.add_argument("--d-init", type=float, default=500)
    ap.add_argument("--sub-nm", type=str, default="500,1500,2500",
                    help="SiO2 thicknesses (comma-separated, default: 500,1500,2500)")
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    svr, scaler = load_svr()
    sub_list = [float(s.strip()) for s in args.sub_nm.split(",")]

    if args.target_band == "atm":
        wl_targets = np.linspace(8, 13, 50)
        desc = "maximize mean eps in 8-13 um"
    else:
        wl_targets = np.linspace(2, 14, 100)
        desc = "maximize mean eps in 2-14 um"

    print("=" * 60)
    print("  INVERSE DESIGN: " + desc)
    print("  SiO2 thicknesses: " + ", ".join("{:.0f}".format(s) for s in sub_list) + " nm")
    print("=" * 60)

    designs = [run_one_sub(wl_targets, svr, scaler, s, args.d_init) for s in sub_list]

    best = max(designs, key=lambda d: d["mean_eps"])
    print("\n" + "=" * 60)
    print("  BEST DESIGN")
    print("=" * 60)
    print("  SiO2 = {:.0f} nm, d_PDMS = {:.1f} nm".format(
        best["d_sio2_nm"], best["d_optimal_nm"]))
    print("  Mean eps(8-13 um) = {:.4f}".format(best["mean_eps"]))
    print("  eps range = [{:.4f}, {:.4f}]".format(
        best["eps_range"][0], best["eps_range"][1]))

    results = {
        "target": {"description": desc,
                   "wl_range_um": [float(wl_targets[0]), float(wl_targets[-1])]},
        "designs": designs,
        "best": best,
    }

    out_path = Path(args.output) if args.output else (
        ROOT / "stage1" / "simulation_data" / "inverse_design_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n[OK] -> " + str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
