#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robustness: 5-seed SVR in PARALLEL using multiprocessing.
Each seed runs on its own CPU core. ~5x wall-clock speedup vs sequential.

Usage:
    python stage1/run_robustness_parallel.py --max-samples 90000
    python stage1/run_robustness_parallel.py --max-samples 90000 --workers 3
    python stage1/run_robustness_parallel.py --max-samples 8000   # quick test
"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GroupShuffleSplit
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


def train_one_seed(args_tuple):
    """Train SVR with a given seed. args_tuple = (X, y, groups_list, seed, cache_mb)"""
    X, y, groups_list, seed, cache_mb = args_tuple
    groups = np.array(groups_list)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    svr = SVR(kernel='rbf', C=500, gamma=1.0, epsilon=0.01, cache_size=cache_mb)

    t0 = time.perf_counter()
    svr.fit(scaler.fit_transform(X_tr), y_tr)
    train_s = time.perf_counter() - t0

    yp = np.clip(svr.predict(scaler.transform(X_te)), 0, 1)
    return {
        "seed": seed,
        "r2": float(r2_score(y_te, yp)),
        "rmse": float(np.sqrt(mean_squared_error(y_te, yp))),
        "mae": float(mean_absolute_error(y_te, yp)),
        "train_time_s": round(train_s, 2),
        "n_train": len(X_tr), "n_test": len(X_te),
        "n_support_vectors": int(svr.support_.shape[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=90000)
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel workers (0=auto: min(5, cpu_count))")
    ap.add_argument("--cache-mb", type=int, default=2000,
                    help="SVR cache per worker in MB (higher = faster for large datasets)")
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    workers = args.workers if args.workers > 0 else min(len(seeds), cpu_count())
    print(f"CPU cores: {cpu_count()}, workers: {workers}, seeds: {seeds}")

    csv_path = ROOT / "stage1" / "simulation_data" / "tmm_emissivity_data.csv"
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    df.rename(columns={cols[0]: "wl", cols[1]: "sub", cols[2]: "pdms",
                       cols[3]: "R", cols[4]: "T", cols[5]: "eps"}, inplace=True)
    df = df[np.abs(df["R"] + df["T"] + df["eps"] - 1.0) <= 1e-3]
    if len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42)
    X = build_features(df)
    y = df["eps"].values.astype(float)
    groups_list = df["pdms"].tolist()
    print(f"Data: {len(df)} samples, {X.shape[1]} features, "
          f"cache={args.cache_mb}MB/worker")

    # Each worker gets the full data arrays (read-only, shared via fork on Linux/Mac,
    # pickled on Windows -- still fine since X is ~8MB for 83k samples)
    tasks = [(X, y, groups_list, s, args.cache_mb) for s in seeds]

    print(f"\nRunning {len(seeds)} seeds on {workers} cores...")
    print(f"  Est. wall-clock: ~{5*3600/workers/3600:.0f}h "
          f"(vs ~{5*3600/3600:.0f}h sequential)\n")

    t_start = time.perf_counter()
    with Pool(processes=workers) as pool:
        all_results = pool.map(train_one_seed, tasks)
    total_t = time.perf_counter() - t_start

    # Sort by seed
    all_results.sort(key=lambda r: r["seed"])
    for r in all_results:
        print(f"  seed={r['seed']:3d}: R2={r['r2']:.4f}, RMSE={r['rmse']:.4f}, "
              f"MAE={r['mae']:.4f}, time={r['train_time_s']:.1f}s")
    sum_fit = sum(r["train_time_s"] for r in all_results)
    speedup = sum_fit / total_t
    ideal = len(seeds)  # perfect linear scaling
    print(f"  wall-clock: {total_t/3600:.1f}h")
    print(f"  speedup: {speedup:.1f}x (ideal={ideal:.0f}x; "
          f"note: pickle/startup overhead on Windows makes this lower than ideal)")

    # Aggregate
    r2s = [r["r2"] for r in all_results]
    summary = {
        "meta": {"n_samples": len(df), "n_seeds": len(seeds), "seeds": seeds,
                 "workers": workers, "wall_clock_s": round(total_t, 1),
                 "speedup": round(speedup, 1),
                 "ideal_speedup": ideal},
        "individual_runs": all_results,
        "aggregate": {
            "r2": {"mean": round(float(np.mean(r2s)), 4),
                   "std": round(float(np.std(r2s)), 4)},
            "rmse": {"mean": round(float(np.mean([r["rmse"] for r in all_results])), 4),
                     "std": round(float(np.std([r["rmse"] for r in all_results])), 4)},
            "mae": {"mean": round(float(np.mean([r["mae"] for r in all_results])), 4),
                    "std": round(float(np.std([r["mae"] for r in all_results])), 4)},
        }
    }

    out_path = Path(args.output) if args.output else (
        ROOT / "stage1" / "simulation_data" / "robustness_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved to {out_path}")
    print(f"  R2  = {summary['aggregate']['r2']['mean']:.4f} "
          f"+- {summary['aggregate']['r2']['std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
