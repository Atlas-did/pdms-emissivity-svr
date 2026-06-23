#!/usr/bin/env python3
"""Robustness: 5-seed SVR training with mean+-std reporting.
Saves to stage1/simulation_data/robustness_results.json"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

def build_features(df):
    wl=df["wl"].values.astype(float); sio=df["sub"].values.astype(float)
    pdms=df["pdms"].values.astype(float)
    n_pdms,n_sio2=1.41,1.46; wl_m,pm,sm=wl*1e-6,pdms*1e-9,sio*1e-9
    dp=2*np.pi*n_pdms*pm/wl_m; ds=2*np.pi*n_sio2*sm/wl_m
    X=np.column_stack([wl,sio,pdms,
        np.sin(dp),np.cos(dp),np.sin(ds),np.cos(ds),
        1.0/wl,pdms/wl,
        np.sin(dp)*np.sin(ds),np.cos(dp)*np.cos(ds),
        np.sin(dp)*np.cos(ds),np.cos(dp)*np.sin(ds)])
    return np.nan_to_num(X,0)

def train_one_seed(df, X, y, seed):
    """Train SVR with a given random seed, return metrics"""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, df["pdms"].values))
    X_tr, X_te = X[train_idx], X[test_idx]; y_tr, y_te = y[train_idx], y[test_idx]
    scaler = StandardScaler()
    svr = SVR(kernel='rbf', C=500, gamma=1.0, epsilon=0.01)
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
    ap.add_argument("--max-samples", type=int, default=15000)
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    csv_path = ROOT/"stage1"/"simulation_data"/"tmm_emissivity_data.csv"
    print(f"[LOAD] {csv_path.name}...")
    t_load = time.perf_counter()
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    df.rename(columns={cols[0]:"wl",cols[1]:"sub",cols[2]:"pdms",cols[3]:"R",cols[4]:"T",cols[5]:"eps"}, inplace=True)
    df = df[np.abs(df["R"]+df["T"]+df["eps"]-1.0) <= 1e-3]
    if len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42)
    X = build_features(df); y = df["eps"].values.astype(float)
    print(f"  {len(df)} samples, load time={time.perf_counter()-t_load:.1f}s")

    print(f"\nRunning {len(seeds)} seeds on {len(df)} samples ({X.shape[1]} features)...")
    print(f"  Est. {(len(df)*0.7):.0f} train / {(len(df)*0.3):.0f} test per seed")
    all_results = []
    t_start = time.perf_counter()
    for i, seed in enumerate(seeds):
        r = train_one_seed(df, X, y, seed)
        all_results.append(r)
        elapsed = time.perf_counter() - t_start
        eta = (elapsed/(i+1))*(len(seeds)-i-1) if i+1 < len(seeds) else 0
        print(f"  [{i+1}/{len(seeds)}] seed={seed:3d}: R2={r['r2']:.4f}, RMSE={r['rmse']:.4f}, "
              f"MAE={r['mae']:.4f}, time={r['train_time_s']:.1f}s, ETA {eta:.0f}s")

    total_t = time.perf_counter() - t_start
    print(f"  total time: {total_t:.1f}s")

    # Aggregate
    r2s = [r["r2"] for r in all_results]
    rmses = [r["rmse"] for r in all_results]
    maes = [r["mae"] for r in all_results]
    times = [r["train_time_s"] for r in all_results]
    svs = [r["n_support_vectors"] for r in all_results]

    summary = {
        "meta": {"n_samples": len(df), "n_seeds": len(seeds), "seeds": seeds,
                 "total_time_s": round(total_t, 1)},
        "individual_runs": all_results,
        "aggregate": {
            "r2": {"mean": round(float(np.mean(r2s)), 4), "std": round(float(np.std(r2s)), 4)},
            "rmse": {"mean": round(float(np.mean(rmses)), 4), "std": round(float(np.std(rmses)), 4)},
            "mae": {"mean": round(float(np.mean(maes)), 4), "std": round(float(np.std(maes)), 4)},
            "train_time_s": {"mean": round(float(np.mean(times)), 2), "std": round(float(np.std(times)), 2)},
            "support_vectors": {"mean": round(float(np.mean(svs)), 0), "std": round(float(np.std(svs)), 0)},
        }
    }

    out_path = Path(args.output) if args.output else ROOT/"stage1"/"simulation_data"/"robustness_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved to {out_path}")
    agg = summary["aggregate"]
    print(f"  R2  = {agg['r2']['mean']:.4f} +- {agg['r2']['std']:.4f}")
    print(f"  RMSE = {agg['rmse']['mean']:.4f} +- {agg['rmse']['std']:.4f}")
    print(f"  MAE = {agg['mae']['mean']:.4f} +- {agg['mae']['std']:.4f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
