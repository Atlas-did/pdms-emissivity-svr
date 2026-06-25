#!/usr/bin/env python3
"""RF Distillation: RF(teacher) -> SVR(student). Incremental checkpoint every step."""
from __future__ import annotations
import sys, json, time, argparse, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "stage1" / "simulation_data"

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

def train_eval_svr(X_tr, X_te, y_tr, y_te, C=500, gamma=1.0, eps=0.01, label=""):
    scaler = StandardScaler()
    svr = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=eps, cache_size=2000)
    t0 = time.perf_counter()
    svr.fit(scaler.fit_transform(X_tr), y_tr)
    train_s = time.perf_counter() - t0
    yp = np.clip(svr.predict(scaler.transform(X_te)), 0, 1)
    r2 = float(r2_score(y_te, yp))
    rmse = float(np.sqrt(mean_squared_error(y_te, yp)))
    mae = float(mean_absolute_error(y_te, yp))
    if label:
        print(f"  {label}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, "
              f"time={train_s:.1f}s, SV={svr.support_.shape[0]}")
    return {"r2": r2, "rmse": rmse, "mae": mae, "train_time_s": round(train_s, 2),
            "n_support_vectors": int(svr.support_.shape[0])}, svr, scaler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=8000)
    ap.add_argument("--rf-n-estimators", type=int, default=200)
    ap.add_argument("--rf-max-depth", type=int, default=20)
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    out_path = Path(args.output) if args.output else (OUT_DIR / "distillation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(data):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  [checkpoint] {out_path.name}")

    csv_path = ROOT / "stage1" / "simulation_data" / "tmm_emissivity_data.csv"
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    df.rename(columns={cols[0]: "wl", cols[1]: "sub", cols[2]: "pdms",
                       cols[3]: "R", cols[4]: "T", cols[5]: "eps"}, inplace=True)
    df = df[np.abs(df["R"] + df["T"] + df["eps"] - 1.0) <= 1e-3]
    if len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42)
    n_total = len(df)
    print(f"[LOAD] {n_total} clean samples")

    X13 = build_features(df)
    y_tmm = df["eps"].values.astype(float)
    groups = df["pdms"].values
    sep = "=" * 60

    # Step 1: RF Teacher
    print(f"\n{sep}\n  Step 1: Train RF Teacher\n{sep}")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X13, y_tmm, groups))
    X_tr, X_te = X13[train_idx], X13[test_idx]
    y_tr_tmm, y_te_tmm = y_tmm[train_idx], y_tmm[test_idx]

    rf = RandomForestRegressor(n_estimators=args.rf_n_estimators, max_depth=args.rf_max_depth,
                               random_state=42, n_jobs=-1)
    t0 = time.perf_counter()
    rf.fit(X_tr, y_tr_tmm)
    rf_train_time = time.perf_counter() - t0
    y_rf_pred = np.clip(rf.predict(X_te), 0, 1)
    rf_r2 = float(r2_score(y_te_tmm, y_rf_pred))
    rf_rmse = float(np.sqrt(mean_squared_error(y_te_tmm, y_rf_pred)))
    print(f"  RF Teacher: R2={rf_r2:.4f}, RMSE={rf_rmse:.4f}, train={rf_train_time:.1f}s")

    # Step 2: Soft Labels
    print(f"\n{sep}\n  Step 2: Generate RF Soft Labels\n{sep}")
    t0 = time.perf_counter()
    y_rf_all = np.clip(rf.predict(X13), 0, 1)
    print(f"  Generated {len(y_rf_all):,} soft labels in {time.perf_counter()-t0:.1f}s")

    # Step 3: SVR Baseline (TMM truth)
    print(f"\n{sep}\n  Step 3: SVR on TMM truth (baseline)\n{sep}")
    svr_baseline, _, _ = train_eval_svr(X_tr, X_te, y_tr_tmm, y_te_tmm, C=500, gamma=1.0, eps=0.01,
                                         label="SVR_baseline(TMM)")
    checkpoint({"meta": {"n_samples": n_total}, "teacher": {"r2": rf_r2, "rmse": rf_rmse},
                "student_baseline": svr_baseline, "status": "step3_complete"})

    # Step 4: SVR Distilled (RF labels)
    print(f"\n{sep}\n  Step 4: SVR Student - DISTILLED (on RF predictions)\n{sep}")
    y_tr_rf = y_rf_all[train_idx]
    y_te_rf = y_rf_all[test_idx]
    svr_distilled, svr_model, svr_scaler = train_eval_svr(
        X_tr, X_te, y_tr_rf, y_te_rf, C=500, gamma=1.0, eps=0.01, label="SVR_distilled(RF)")

    # Step 5: Distilled SVR on TMM Truth
    print(f"\n{sep}\n  Step 5: Distilled SVR -> TMM Truth\n{sep}")
    y_distilled_pred = np.clip(svr_model.predict(svr_scaler.transform(X_te)), 0, 1)
    distilled_r2_on_tmm = float(r2_score(y_te_tmm, y_distilled_pred))
    distilled_rmse_on_tmm = float(np.sqrt(mean_squared_error(y_te_tmm, y_distilled_pred)))
    distilled_mae_on_tmm = float(mean_absolute_error(y_te_tmm, y_distilled_pred))
    print(f"  Distilled SVR -> TMM truth: R2={distilled_r2_on_tmm:.4f}, "
          f"RMSE={distilled_rmse_on_tmm:.4f}, MAE={distilled_mae_on_tmm:.4f}")

    # Summary
    print()
    print(sep)
    print("  DISTILLATION SUMMARY")
    print(sep)
    hdr = "  {:30} {:18} {:18} {:>8}".format("Model", "Trained on", "Tested on", "R2")
    print(hdr)
    div = "  {:30} {:18} {:18} {:>8}".format("-"*30, "-"*18, "-"*18, "-"*8)
    print(div)
    row1 = "  {:30} {:18} {:18} {:>8.4f}".format("RF Teacher", "TMM truth", "TMM truth", rf_r2)
    print(row1)
    row2 = "  {:30} {:18} {:18} {:>8.4f}".format("SVR (no distill)", "TMM truth", "TMM truth", svr_baseline["r2"])
    print(row2)
    row3 = "  {:30} {:18} {:18} {:>8.4f}".format("SVR (distilled)", "RF preds", "RF preds", svr_distilled["r2"])
    print(row3)
    row4 = "  {:30} {:18} {:18} {:>8.4f}".format("SVR (distilled)->TMM", "RF preds", "TMM truth *", distilled_r2_on_tmm)
    print(row4)
    delta = distilled_r2_on_tmm - svr_baseline["r2"]
    print()
    print("  SVR baseline -> distilled gain on TMM truth: {:.4f} R2".format(delta))

    # Save final
    results = {
        "meta": {"n_samples": n_total, "method": "RF->SVR distillation"},
        "teacher": {"r2": rf_r2, "rmse": rf_rmse, "train_time_s": round(rf_train_time, 1)},
        "student_baseline": svr_baseline,
        "student_distilled": {
            "trained_on": "RF_predictions",
            "r2_vs_RF": svr_distilled["r2"], "rmse_vs_RF": svr_distilled["rmse"],
            "r2_vs_TMM": distilled_r2_on_tmm, "rmse_vs_TMM": distilled_rmse_on_tmm,
            "mae_vs_TMM": distilled_mae_on_tmm,
            "n_support_vectors": svr_distilled["n_support_vectors"],
            "train_time_s": svr_distilled["train_time_s"],
        },
        "gain_on_TMM": {
            "delta_R2": round(delta, 4),
            "delta_RMSE": round(distilled_rmse_on_tmm - svr_baseline["rmse"], 4),
        },
    }
    checkpoint(results)

    rf_path = OUT_DIR / "rf_teacher.pkl"
    svr_path = OUT_DIR / "svr_distilled.pkl"
    with open(rf_path, "wb") as f:
        pickle.dump(rf, f)
    with open(svr_path, "wb") as f:
        pickle.dump({"model": svr_model, "scaler": svr_scaler}, f)
    print(f"[OK] RF Teacher -> {rf_path}")
    print(f"[OK] SVR Distilled -> {svr_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
