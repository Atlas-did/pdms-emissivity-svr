#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train GPR on TMM data for uncertainty quantification (P2.1).

Saves model + prediction samples + confidence intervals.
Reuses the same feature engineering protocol as stage1 SVRTrainer.

Usage:
    python stage1/train_gpr_uncertainty.py
    python stage1/train_gpr_uncertainty.py --max-train 3000 --max-test 800
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "stage1" / "simulation_data" / "tmm_emissivity_data.csv"
DEFAULT_OUT = ROOT / "stage1" / "simulation_data" / "gpr_uncertainty_results.json"
DEFAULT_MODEL = ROOT / "stage1" / "simulation_data" / "gpr_uncertainty_model.pkl"


def _build_features(X_raw: np.ndarray) -> np.ndarray:
    """Feature engineering matching stage1 SVRTrainer._build_features."""
    n_pdms = 1.41
    n_sio2 = 1.46
    wl = X_raw[:, 0]
    d_sio2 = X_raw[:, 1]
    d_pdms = X_raw[:, 2]
    wl_m = wl * 1e-6
    pdms_m = d_pdms * 1e-9
    sio2_m = d_sio2 * 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_pdms = 2 * np.pi * n_pdms * pdms_m / wl_m
        delta_sio2 = 2 * np.pi * n_sio2 * sio2_m / wl_m
        inv_wl = 1.0 / wl
        d_over_wl = d_pdms / wl
    X = np.column_stack([
        X_raw,
        np.sin(delta_pdms), np.cos(delta_pdms),
        np.sin(delta_sio2), np.cos(delta_sio2),
        inv_wl, d_over_wl,
        np.sin(delta_pdms) * np.sin(delta_sio2),
        np.cos(delta_pdms) * np.cos(delta_sio2),
        np.sin(delta_pdms) * np.cos(delta_sio2),
        np.cos(delta_pdms) * np.sin(delta_sio2),
    ])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def train_gpr_paper_quality(
    csv_path: Path,
    out_path: Path,
    model_path: Path,
    max_train: int = 2000,
    max_test: int = 500,
    random_state: int = 42,
) -> None:
    """Train GPR with tuned kernel and save model + prediction samples."""

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    X_raw = df[["波长_μm", "基底厚度_nm", "PDMS厚度_nm"]].values.astype(float)
    y = df["发射率ε"].values.astype(float)
    print(f"  Total samples: {len(X_raw)}")

    # Sub-sample for GPR (O(n³) scaling)
    n_total = min(max_train + max_test, len(X_raw))
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X_raw), size=n_total, replace=False)
    X_raw_s, y_s = X_raw[idx], y[idx]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw_s, y_s, test_size=float(max_test) / n_total, random_state=random_state
    )
    print(f"  Train: {len(X_train_raw)}, Test: {len(X_test_raw)}")

    # Feature engineering
    X_train = _build_features(X_train_raw)
    X_test = _build_features(X_test_raw)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Composite kernel: ConstantKernel * RBF + WhiteKernel
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    )

    print("Training GPR (this may take 1-5 minutes for ~2000 samples)...")
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=random_state,
    )

    t0 = time.time()
    gpr.fit(X_train_s, y_train)
    train_time_s = time.time() - t0
    print(f"  Training done in {train_time_s:.1f}s")
    print(f"  Learned kernel: {gpr.kernel_}")

    # Predict with std
    y_pred, y_std = gpr.predict(X_test_s, return_std=True)
    y_pred = np.clip(y_pred, 0.0, 1.0)
    y_std = np.clip(y_std, 0.0, None)

    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    # Within-2sigma coverage (calibration check)
    within_2sigma = np.abs(y_pred - y_test) <= 2.0 * y_std
    coverage_2sigma = float(np.mean(within_2sigma))

    # Per-wavelength-band analysis
    wl_test = X_test_raw[:, 0]
    atm_mask = (wl_test >= 8.0) & (wl_test <= 13.0)
    atm_mae = float(mean_absolute_error(y_test[atm_mask], y_pred[atm_mask])) if np.any(atm_mask) else None
    atm_std = float(np.mean(y_std[atm_mask])) if np.any(atm_mask) else None

    # Per-thickness-band analysis
    th_test = X_test_raw[:, 2]
    thin_mask = th_test < 400
    thick_mask = th_test > 700
    thin_mae = float(mean_absolute_error(y_test[thin_mask], y_pred[thin_mask])) if np.any(thin_mask) else None
    thick_mae = float(mean_absolute_error(y_test[thick_mask], y_pred[thick_mask])) if np.any(thick_mask) else None

    results: Dict[str, Any] = {
        "meta": {
            "action": "gpr_uncertainty_training",
            "date": time.strftime("%Y-%m-%d"),
            "n_train": int(len(X_train_raw)),
            "n_test": int(len(X_test_raw)),
            "train_time_s": round(train_time_s, 2),
            "random_state": random_state,
        },
        "metrics": {
            "r2": round(r2, 6),
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "coverage_2sigma": round(coverage_2sigma, 4),
            "mean_std": round(float(np.mean(y_std)), 6),
            "median_std": round(float(np.median(y_std)), 6),
            "atm_window_mae": round(atm_mae, 6) if atm_mae is not None else None,
            "atm_window_mean_std": round(atm_std, 6) if atm_std is not None else None,
            "thin_film_mae": round(thin_mae, 6) if thin_mae is not None else None,
            "thick_film_mae": round(thick_mae, 6) if thick_mae is not None else None,
        },
        "samples": {
            "y_test_sample": y_test[:500].tolist(),
            "y_pred_sample": y_pred[:500].tolist(),
            "y_std_sample": y_std[:500].tolist(),
            "X_test_sample_wl": X_test_raw[:500, 0].tolist(),
            "X_test_sample_th": X_test_raw[:500, 2].tolist(),
        },
        "kernel": str(gpr.kernel_),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    model_bundle = {"gpr": gpr, "scaler": scaler, "kernel_str": str(gpr.kernel_)}
    joblib.dump(model_bundle, model_path)

    print(f"\n✅ GPR model saved to: {model_path}")
    print(f"✅ Results saved to: {out_path}")
    print(f"  R²={r2:.6f}  RMSE={rmse:.6f}  MAE={mae:.6f}")
    print(f"  2σ coverage = {coverage_2sigma:.2%}")
    print(f"  Mean uncertainty (std) = {float(np.mean(y_std)):.6f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Train GPR for uncertainty quantification")
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to TMM emissivity CSV")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSON path")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="Output model path")
    ap.add_argument("--max-train", type=int, default=2000, help="Max training samples")
    ap.add_argument("--max-test", type=int, default=500, help="Max test samples")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ TMM CSV not found: {csv_path}")
        return 1

    try:
        train_gpr_paper_quality(
            csv_path=Path(args.csv),
            out_path=Path(args.out),
            model_path=Path(args.model),
            max_train=args.max_train,
            max_test=args.max_test,
            random_state=args.random_state,
        )
        return 0
    except Exception as e:
        print(f"❌ GPR training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
