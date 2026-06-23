#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablation study -- PARALLEL Halton search.
Each Halton trial runs as a separate process. ~5x wall-clock speedup.

Usage:
    python stage1/run_ablation_parallel.py --max-samples 90000 --n-iter 10
    python stage1/run_ablation_parallel.py --max-samples 8000 --n-iter 5   # quick test
"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Prevent Windows from sleeping while this script runs
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    print("[INFO] System sleep prevented while this script is running")


def build_features(df):
    wl = df["wl"].values.astype(float)
    sio = df["sub"].values.astype(float)
    pdms = df["pdms"].values.astype(float)
    n_pdms, n_sio2 = 1.41, 1.46
    wl_m, pm, sm = wl * 1e-6, pdms * 1e-9, sio * 1e-9
    dp = 2 * np.pi * n_pdms * pm / wl_m
    ds = 2 * np.pi * n_sio2 * sm / wl_m
    X = np.column_stack([wl, sio, pdms,
        np.sin(dp), np.cos(dp), np.sin(ds), np.cos(ds),
        1.0 / wl, pdms / wl,
        np.sin(dp) * np.sin(ds), np.cos(dp) * np.cos(ds),
        np.sin(dp) * np.cos(ds), np.cos(dp) * np.sin(ds)])
    return np.nan_to_num(X, 0)


def _trial(args):
    """Single Halton trial -- runs in its own process."""
    i, C, gamma, eps, X_tr_hs, X_te_hs, y_tr_h, y_te_h, cache_mb = args
    t0 = time.perf_counter()
    svr = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=eps, cache_size=cache_mb)
    svr.fit(X_tr_hs, y_tr_h)
    score = float(r2_score(y_te_h, np.clip(svr.predict(X_te_hs), 0, 1)))
    elapsed = time.perf_counter() - t0
    return {"trial": i, "C": C, "gamma": gamma, "eps": eps,
            "r2": score, "time_s": round(elapsed, 1)}


def train_eval(X, y, groups, C=500, gamma=1.0, eps=0.01, cache_mb=2000, label=""):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    scaler = StandardScaler()
    svr = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=eps, cache_size=cache_mb)
    t0 = time.perf_counter()
    svr.fit(scaler.fit_transform(X_tr), y_tr)
    ts = time.perf_counter() - t0
    yp = np.clip(svr.predict(scaler.transform(X_te)), 0, 1)
    r2v = float(r2_score(y_te, yp))
    if label:
        print(f"  {label}: n_train={len(X_tr)}, n_test={len(X_te)}, "
              f"R2={r2v:.4f}, time={ts:.1f}s, SV={svr.support_.shape[0]}")
    return {"r2": r2v,
            "rmse": float(np.sqrt(mean_squared_error(y_te, yp))),
            "mae": float(mean_absolute_error(y_te, yp)),
            "train_time_s": round(ts, 2),
            "n_train": len(X_tr), "n_test": len(X_te),
            "n_support_vectors": int(svr.support_.shape[0])}


def halton_search_parallel(X, y, groups, n_iter=10, workers=4, cache_mb=2000,
                           quick=False):
    """Parallel Halton search: each trial = one process."""
    if quick:
        C_vals = [100, 200, 500]
        gamma_vals = [0.05, 0.1, 0.5]
        eps_vals = [0.001, 0.01]
    else:
        C_vals = [100, 200, 500, 1000, 2000, 5000]
        gamma_vals = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        eps_vals = [0.001, 0.005, 0.01, 0.05]

    print(f"  Parallel Halton search: {n_iter} trials on {workers} workers "
          f"(C={C_vals}, gamma={gamma_vals}, eps={eps_vals})...")
    print(f"  Training on {int(len(y)*0.7)} samples, "
          f"testing on {int(len(y)*0.3)}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_tr_h, X_te_h = X[train_idx], X[test_idx]
    y_tr_h, y_te_h = y[train_idx], y[test_idx]
    scaler_h = StandardScaler()
    X_tr_hs = scaler_h.fit_transform(X_tr_h)
    X_te_hs = scaler_h.transform(X_te_h)

    rng = np.random.RandomState(42)
    tasks = []
    for i in range(n_iter):
        C = int(rng.choice(C_vals))
        gamma = float(rng.choice(gamma_vals))
        eps = float(rng.choice(eps_vals))
        tasks.append((i + 1, C, gamma, eps, X_tr_hs, X_te_hs,
                      y_tr_h, y_te_h, cache_mb))

    t0 = time.perf_counter()
    with Pool(processes=workers) as pool:
        trial_results = pool.map(_trial, tasks)
    search_time = round(time.perf_counter() - t0, 2)

    # Find best
    trial_results.sort(key=lambda t: t["r2"], reverse=True)
    best = trial_results[0]
    best_params = {"C": best["C"], "gamma": best["gamma"], "eps": best["eps"]}
    best_score = best["r2"]

    # Print all trials
    for t in sorted(trial_results, key=lambda x: x["trial"]):
        mark = " *" if t["trial"] == best["trial"] else ""
        print(f"  [{t['trial']:2d}/{n_iter}] C={t['C']:4d} gamma={t['gamma']:.3f} "
              f"eps={t['eps']:.4f} -> R2={t['r2']:.4f}{mark} "
              f"({t['time_s']:.0f}s)")

    return best_params, search_time, round(best_score, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=8000,
                    help="Samples for final evaluation (A1-A3)")
    ap.add_argument("--search-samples", type=int, default=5000,
                    help="Samples for Halton search (smaller = faster)")
    ap.add_argument("--n-iter", type=int, default=10)
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel workers (0=auto: min(n_iter, cpu_count))")
    ap.add_argument("--cache-mb", type=int, default=2000,
                    help="SVR cache per worker in MB")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    workers = args.workers if args.workers > 0 else min(args.n_iter, cpu_count())
    print(f"CPU cores: {cpu_count()}, workers: {workers}")

    out_path = Path(args.output) if args.output else (
        ROOT / "stage1" / "simulation_data" / "ablation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        """Save intermediate results after each step so crashes don't lose everything."""
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  [checkpoint] saved to {out_path}")

    csv_path = ROOT / "stage1" / "simulation_data" / "tmm_emissivity_data.csv"
    if not csv_path.exists():
        print(f"[FAIL] {csv_path}")
        return 1
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    df.rename(columns={cols[0]: "wl", cols[1]: "sub", cols[2]: "pdms",
                       cols[3]: "R", cols[4]: "T", cols[5]: "eps"}, inplace=True)
    df["sum"] = df["R"] + df["T"] + df["eps"]
    df_clean = df[np.abs(df["sum"] - 1.0) <= 1e-3].copy()
    n_filtered = len(df) - len(df_clean)
    print(f"[LOAD] {len(df)} rows, {n_filtered} failed energy check, "
          f"using {len(df_clean)}")
    if len(df_clean) > args.max_samples:
        df_clean = df_clean.sample(n=args.max_samples, random_state=42)
        print(f"  downsampled to {len(df_clean)}")

    X = build_features(df_clean)
    y = df_clean["eps"].values.astype(float)
    groups = df_clean["pdms"].values

    # Halton search subset (small, for speed)
    if len(df_clean) > args.search_samples:
        rng_s = np.random.RandomState(42)
        idx_s = rng_s.choice(len(df_clean), args.search_samples, replace=False)
        X_search, y_search, groups_search = X[idx_s], y[idx_s], groups[idx_s]
    else:
        X_search, y_search, groups_search = X, y, groups

    results = {}
    print(f"\n{'='*60}\n  Ablation Study ({len(df_clean)} samples, "
          f"{X.shape[1]} features, {workers} workers)\n{'='*60}")

    # A1: Full model
    print("\n[A1] Full model (Halton search, energy filter)...")
    bp, ss, bs = halton_search_parallel(X_search, y_search, groups_search, args.n_iter,
                                        workers=workers, cache_mb=args.cache_mb,
                                        quick=args.quick)
    C_bp, g_bp, e_bp = bp["C"], bp["gamma"], bp["eps"]
    m1 = train_eval(X, y, groups, C=C_bp, gamma=g_bp, eps=e_bp,
                    cache_mb=args.cache_mb, label="  final eval")
    results["A1_full_model"] = {
        "description": "Halton search + energy filter (baseline)",
        "hyperparams": bp, "search_time_s": ss, **m1}
    print(f"  => R2={m1['r2']:.4f}, RMSE={m1['rmse']:.4f}, search={ss}s")
    checkpoint()

    # A2: No energy filter
    print("\n[A2] No energy filter (all data, same params)...")
    df_all = df.copy()
    if len(df_all) > args.max_samples:
        df_all = df_all.sample(n=args.max_samples, random_state=42)
    X_all = build_features(df_all)
    y_all = df_all["eps"].values.astype(float)
    groups_all = df_all["pdms"].values
    m2 = train_eval(X_all, y_all, groups_all, C=C_bp, gamma=g_bp, eps=e_bp,
                    cache_mb=args.cache_mb, label="  final eval")
    results["A2_no_energy_filter"] = {
        "description": f"No energy filter ({n_filtered} extra pts)",
        "extra_samples": n_filtered, **m2}
    print(f"  => R2={m2['r2']:.4f} (delta vs A1: {m2['r2']-m1['r2']:+.4f})")
    checkpoint()

    # A3: Random split (data leakage demo)
    print("\n[A3] Random split (NOT thickness-grouped) -- data leakage demo...")
    from sklearn.model_selection import train_test_split
    X_tr3, X_te3, y_tr3, y_te3 = train_test_split(
        X, y, test_size=0.3, random_state=42)
    sc3 = StandardScaler()
    svr3 = SVR(kernel="rbf", C=C_bp, gamma=g_bp, epsilon=e_bp,
               cache_size=args.cache_mb)
    t0 = time.perf_counter()
    svr3.fit(sc3.fit_transform(X_tr3), y_tr3)
    ts3 = time.perf_counter() - t0
    yp3 = np.clip(svr3.predict(sc3.transform(X_te3)), 0, 1)
    m3 = {"r2": float(r2_score(y_te3, yp3)),
          "rmse": float(np.sqrt(mean_squared_error(y_te3, yp3))),
          "mae": float(mean_absolute_error(y_te3, yp3)),
          "train_time_s": round(ts3, 2)}
    results["A3_random_split"] = {
        "description": "Random split (inflated R2, NOT comparable)",
        "note": "Data leakage from adjacent wavelengths", **m3}
    print(f"  => R2={m3['r2']:.4f} (inflated by data leakage vs "
          f"A1={m1['r2']:.4f})")
    checkpoint()

    # Create a 5k subset for fast grid search
    rng5k = np.random.RandomState(42)
    idx5k_all = rng5k.choice(len(df_clean), min(5000, len(df_clean)), replace=False)
    X_5k = X[idx5k_all]
    y_5k = y[idx5k_all]
    groups_5k = groups[idx5k_all]

    # A4: Grid search (on small 5000-sample subset for speed)
    print("\n[A4] Grid search (3x3x2=18 combos, on 5k subset)...")
    t0 = time.perf_counter()
    pg = {"C": [100, 500, 1000], "gamma": [0.1, 0.5, 1.0],
          "epsilon": [0.001, 0.01]}
    X_gs, y_gs, groups_gs = X_5k, y_5k, groups_5k
    gss4 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx4, test_idx4 = next(gss4.split(X_gs, y_gs, groups_gs))
    X_tr4, X_te4 = X_gs[train_idx4], X_gs[test_idx4]
    y_tr4, y_te4 = y_gs[train_idx4], y_gs[test_idx4]
    scaler4 = StandardScaler()
    X_tr4s = scaler4.fit_transform(X_tr4)
    gs = GridSearchCV(SVR(kernel='rbf', cache_size=args.cache_mb), pg,
                      cv=2, scoring='r2', n_jobs=1, verbose=1)
    # n_jobs=1 avoids nested parallelism: outer Pool already uses `workers`
    # processes for Halton trials; letting GridSearchCV fork more would oversubscribe
    # CPU cores and degrade throughput.
    print(f"  Fitting 18 combos x 2-fold CV = 36 fits on {len(X_gs)} samples "
          f"(n_jobs=1, outer pool handles parallelism)...")
    gs.fit(X_tr4s, y_tr4)
    gs_time = round(time.perf_counter() - t0, 2)
    yp4 = np.clip(gs.best_estimator_.predict(scaler4.transform(X_te4)), 0, 1)
    m4 = {"best_params": gs.best_params_, "search_time_s": gs_time,
          "r2": float(r2_score(y_te4, yp4)),
          "rmse": float(np.sqrt(mean_squared_error(y_te4, yp4))),
          "mae": float(mean_absolute_error(y_te4, yp4))}
    results["A4_grid_search"] = {
        "description": "Grid search (18 combos) instead of Halton", **m4}
    print(f"  => R2={m4['r2']:.4f}, search={gs_time}s, "
          f"params={gs.best_params_}")
    checkpoint()

    # A5: Reference from test_results
    print("\n[A5] Reference: production SVR from test_results.json...")
    tres_path = ROOT / "stage1" / "simulation_data" / "test_results.json"
    if tres_path.exists():
        tres = json.load(open(tres_path))
        results["A5_global_svr_reference"] = {
            "description": "Global SVR (paper Table 2, production model)",
            "r2": tres["r2"], "rmse": tres["rmse"], "mae": tres["mae"],
            "split_protocol": tres["split_protocol"]}
        print(f"  => R2={tres['r2']:.4f}, RMSE={tres['rmse']:.4f}, "
              f"MAE={tres['mae']:.4f}")
    else:
        print(f"  [SKIP] {tres_path} not found")
    checkpoint()

    # Final save (also done incrementally after each step)

    print(f"\n{'='*60}\n  Summary\n{'='*60}")
    print(f"  {'Configuration':<35} {'R2':>7}  {'RMSE':>7}  {'MAE':>7}")
    print(f"  {'-'*35} {'-'*7}  {'-'*7}  {'-'*7}")
    for k, r in results.items():
        if "r2" in r:
            print(f"  {r['description'][:35]:<35} {r['r2']:7.4f}  "
                  f"{r.get('rmse',0):7.4f}  {r.get('mae',0):7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
