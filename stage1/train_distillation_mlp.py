#!/usr/bin/env python3
"""RF -> MLP distillation.
Key: RF trained on RAW features (no scaler). MLP trains on SCALED features.
RF labels are scaler-independent numbers; we generate them from raw X,
then train MLP on scaled X with those labels.

Usage:
    python stage1/train_distillation_mlp.py --max-samples 90000
"""

from __future__ import annotations
import sys, json, time, pickle, argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "stage1" / "simulation_data"


def build_features(df):
    """EXACT copy from train_distillation.py.
    Physical optical phase terms: n_PDMS=1.41, n_SiO2=1.46.
    RF was trained on RAW output of this function (NO StandardScaler).
    """
    wl = df["wl"].values.astype(float)
    sio = df["sub"].values.astype(float)
    pdms = df["pdms"].values.astype(float)
    n_pdms, n_sio2 = 1.41, 1.46
    wl_m = wl * 1e-6
    pm = pdms * 1e-9
    sm = sio * 1e-9
    dp = 2 * np.pi * n_pdms * pm / wl_m
    ds = 2 * np.pi * n_sio2 * sm / wl_m
    X = np.column_stack([
        wl, sio, pdms,
        np.sin(dp), np.cos(dp), np.sin(ds), np.cos(ds),
        1.0 / wl, pdms / wl,
        np.sin(dp) * np.sin(ds), np.cos(dp) * np.cos(ds),
        np.sin(dp) * np.cos(ds), np.cos(dp) * np.sin(ds),
    ])
    return np.nan_to_num(X, 0)


def load_data(max_samples=90000):
    csv_path = ROOT / "stage1" / "simulation_data" / "tmm_emissivity_data.csv"
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    df.rename(columns={
        cols[0]: "wl", cols[1]: "sub", cols[2]: "pdms",
        cols[3]: "R", cols[4]: "T", cols[5]: "eps",
    }, inplace=True)
    df = df[np.abs(df["R"] + df["T"] + df["eps"] - 1.0) <= 1e-3]
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    return df


def load_rf_teacher():
    rf_path = OUT_DIR / "rf_teacher.pkl"
    if not rf_path.exists():
        print("[ERROR] rf_teacher.pkl not found.")
        sys.exit(1)
    with open(rf_path, "rb") as f:
        rf = pickle.load(f)
    return rf


def train_eval_mlp(X_tr, X_te, y_tr, y_te, hidden_layers, label=""):
    t0 = time.perf_counter()
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu", solver="adam", alpha=1e-5,
        batch_size=256, learning_rate_init=0.001,
        max_iter=500, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20,
        random_state=42,
    )
    mlp.fit(X_tr, y_tr)
    train_s = time.perf_counter() - t0
    yp = np.clip(mlp.predict(X_te), 0, 1)
    r2 = float(r2_score(y_te, yp))
    rmse = float(np.sqrt(mean_squared_error(y_te, yp)))
    mae = float(mean_absolute_error(y_te, yp))
    n_iter = mlp.n_iter_ if hasattr(mlp, "n_iter_") else 0
    if label:
        lay_str = "x".join(map(str, hidden_layers))
        print("  {}: R2={:.4f}  RMSE={:.4f}  MAE={:.4f}  time={:.1f}s  iter={}".format(
            label, r2, rmse, mae, train_s, n_iter))
    return {
        "r2": r2, "rmse": rmse, "mae": mae,
        "train_time_s": round(train_s, 2),
        "hidden_layers": list(hidden_layers),
        "n_iter": int(n_iter) if n_iter is not None else 0,
    }, mlp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=90000)
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    out_path = Path(args.output) if args.output else (OUT_DIR / "distillation_mlp_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(data):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("  [checkpoint] " + out_path.name)

    sep = "=" * 60

    # ===== LOAD =====
    print(sep)
    print("  RF -> MLP Distillation (v3: RF=raw features, MLP=scaled features)")
    print(sep)
    df = load_data(args.max_samples)
    n_total = len(df)
    X_raw = build_features(df)          # 13-dim raw features for RF
    y_tmm = df["eps"].values
    groups = df["pdms"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X_raw, y_tmm, groups))

    # RF uses RAW features (as in train_distillation.py)
    X_tr_raw, X_te_raw = X_raw[train_idx], X_raw[test_idx]
    y_tr_tmm, y_te_tmm = y_tmm[train_idx], y_tmm[test_idx]

    # MLP uses SCALED features
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_raw)
    X_te_s = scaler.transform(X_te_raw)

    print("  Data: {} total, {} train, {} test".format(n_total, len(X_tr_raw), len(X_te_raw)))

    # ===== STEP 1: RF Teacher =====
    print("\n  Step 1: Load RF Teacher + Generate Soft Labels")
    rf = load_rf_teacher()

    # RF prediction on RAW features (RF was trained without scaler!)
    t0 = time.perf_counter()
    y_rf_train = np.clip(rf.predict(X_tr_raw), 0, 1)
    y_rf_test = np.clip(rf.predict(X_te_raw), 0, 1)
    y_rf_all = np.clip(rf.predict(X_raw), 0, 1)

    # Sanity check: RF on raw features should give R2 ~0.9998
    rf_r2 = float(r2_score(y_te_tmm, y_rf_test))
    rf_rmse = float(np.sqrt(mean_squared_error(y_te_tmm, y_rf_test)))
    print("  RF Teacher: R2={:.4f}  RMSE={:.4f}  ({} soft labels in {:.1f}s)".format(
        rf_r2, rf_rmse, len(y_rf_all), time.perf_counter() - t0))

    results = {
        "meta": {
            "n_samples": n_total,
            "method": "RF teacher -> MLP student distillation",
            "rf_teacher_r2": round(rf_r2, 4),
            "rf_teacher_rmse": round(rf_rmse, 4),
        },
        "models": {},
    }

    # ===== STEP 2: MLP Baseline (TMM truth, scaled features) =====
    print("\n  Step 2: MLP Baseline (train on TMM truth, scaled features)")
    configs = [(128, 64, 32), (256, 128, 64), (512, 256, 128)]
    for layers in configs:
        name = "mlp_baseline_" + "x".join(map(str, layers))
        info, _ = train_eval_mlp(X_tr_s, X_te_s, y_tr_tmm, y_te_tmm, layers, label=name)
        results["models"][name] = info
        checkpoint(results)

    # ===== STEP 3: MLP Distilled (RF labels, scaled features) =====
    print("\n  Step 3: MLP Distilled (train on RF predictions, scaled features)")
    for layers in configs:
        name = "mlp_distilled_" + "x".join(map(str, layers))
        info, mlp_model = train_eval_mlp(X_tr_s, X_te_s, y_rf_train, y_rf_test, layers, label=name)

        # Evaluate on TMM TRUTH
        y_pred_tmm = np.clip(mlp_model.predict(X_te_s), 0, 1)
        r2_on_tmm = float(r2_score(y_te_tmm, y_pred_tmm))
        rmse_on_tmm = float(np.sqrt(mean_squared_error(y_te_tmm, y_pred_tmm)))
        mae_on_tmm = float(mean_absolute_error(y_te_tmm, y_pred_tmm))

        results["models"][name] = info
        results["models"][name + "_on_TMM"] = {
            "r2": r2_on_tmm, "rmse": rmse_on_tmm, "mae": mae_on_tmm,
            "note": "Trained on RF labels, tested on TMM truth",
        }

        baseline_name = "mlp_baseline_" + "x".join(map(str, layers))
        baseline_r2 = results["models"][baseline_name]["r2"]
        delta = r2_on_tmm - baseline_r2
        print("    -> On TMM truth: R2={:.4f} (delta vs MLP baseline: {:+.4f})".format(r2_on_tmm, delta))
        checkpoint(results)

    # ===== STEP 4: Compare with SVR =====
    print("\n  Step 4: Compare with SVR distillation")
    svr_path = OUT_DIR / "distillation_results.json"
    if svr_path.exists():
        svr_data = json.loads(svr_path.read_text(encoding="utf-8"))
        svr_bl = svr_data["student_baseline"]["r2"]
        svr_ds = svr_data["student_distilled"]["r2_vs_TMM"]
        print("  SVR baseline  (TMM->TMM):   R2={:.4f}".format(svr_bl))
        print("  SVR distilled (RF->TMM):    R2={:.4f}".format(svr_ds))

        best_mlp = max(v["r2"] for k, v in results["models"].items() if k.endswith("_on_TMM"))
        results["comparison"] = {
            "svr_baseline_r2": svr_bl,
            "svr_distilled_r2": svr_ds,
            "mlp_best_distilled_r2": best_mlp,
            "mlp_vs_svr_delta": round(best_mlp - svr_ds, 4),
        }
        print("  MLP distilled (RF->TMM):    R2={:.4f}  (delta vs SVR: {:+.4f})".format(
            best_mlp, best_mlp - svr_ds))

    checkpoint(results)
    print("\n[OK] -> " + str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())