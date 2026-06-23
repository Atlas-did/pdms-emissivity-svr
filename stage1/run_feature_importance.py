#!/usr/bin/env python3
"""Feature importance: permutation importance of 13 SVR features.
Saves to stage1/simulation_data/feature_importance.json"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GroupShuffleSplit

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

FEATURE_NAMES = [
    "wavelength (um)", "SiO2 thickness (nm)", "PDMS thickness (nm)",
    "sin(delta_PDMS)", "cos(delta_PDMS)", "sin(delta_SiO2)", "cos(delta_SiO2)",
    "1/wavelength", "d_PDMS / wavelength",
    "sin_PDMS * sin_SiO2", "cos_PDMS * cos_SiO2",
    "sin_PDMS * cos_SiO2", "cos_PDMS * sin_SiO2"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=10000)
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    csv_path = ROOT/"stage1"/"simulation_data"/"tmm_emissivity_data.csv"
    print(f"[LOAD] {csv_path.name}...")
    t_load = time.perf_counter()
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    df.rename(columns={cols[0]:"wl",cols[1]:"sub",cols[2]:"pdms",cols[3]:"R",cols[4]:"T",cols[5]:"eps"}, inplace=True)
    df = df[np.abs(df["R"]+df["T"]+df["eps"]-1.0) <= 1e-3]
    if len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42)
    print(f"  {len(df)} samples, load time={time.perf_counter()-t_load:.1f}s")

    X = build_features(df)
    y = df["eps"].values.astype(float)
    groups = df["pdms"].values

    # Train SVR
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_tr, X_te = X[train_idx], X[test_idx]; y_tr, y_te = y[train_idx], y[test_idx]
    scaler = StandardScaler()
    svr = SVR(kernel='rbf', C=500, gamma=1.0, epsilon=0.01)
    print(f"Training SVR on {len(X_tr)} samples...")
    t_train = time.perf_counter()
    svr.fit(scaler.fit_transform(X_tr), y_tr)
    train_t = time.perf_counter() - t_train
    yp = np.clip(svr.predict(scaler.transform(X_te)), 0, 1)
    from sklearn.metrics import r2_score
    print(f"  done in {train_t:.1f}s, R2={r2_score(y_te,yp):.4f}, SV={svr.support_.shape[0]}")

    # Permutation importance
    print(f"Computing permutation importance on {len(X_te)} test samples (5 repeats)...")
    X_te_s = scaler.transform(X_te)
    t0 = time.perf_counter()
    r = permutation_importance(svr, X_te_s, y_te, n_repeats=5, random_state=42, n_jobs=1)
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s")

    # Sort by importance
    idx = np.argsort(r.importances_mean)[::-1]
    results = []
    for i in idx:
        results.append({
            "rank": int(np.where(idx==i)[0][0])+1,
            "feature": FEATURE_NAMES[i],
            "importance_mean": float(r.importances_mean[i]),
            "importance_std": float(r.importances_std[i]),
        })
        print(f"  #{results[-1]['rank']:2d} {FEATURE_NAMES[i]:<30s}  "
              f"{r.importances_mean[i]:+.4f} +/- {r.importances_std[i]:.4f}")

    out = {
        "meta": {"n_samples": len(df), "n_repeats": 5, "elapsed_s": round(elapsed,1)},
        "features": results,
        "top3": [f["feature"] for f in results[:3]],
    }

    out_path = Path(args.output) if args.output else ROOT/"stage1"/"simulation_data"/"feature_importance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved to {out_path}")
    print(f"Top 3 features: {', '.join(out['top3'])}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
