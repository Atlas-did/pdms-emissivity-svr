#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick baselines for paper table (Polynomial / GPR).

Design goals:
- Match the existing baseline protocol in `stage1/main_pdms_svr_bandscan_full.py`:
  - X = [波长_μm, 基底厚度_nm, PDMS厚度_nm]
  - y = 发射率ε
  - train_test_split(test_size=0.2, random_state=42)
  - StandardScaler fitted on X_train
  - Metrics: R2/RMSE/MAE on the (optionally subsampled) test set
  - Timing: train_time_s and pred_time_ms (per-point average)
- Be CPU-friendly by allowing small subsamples.

Outputs:
- stage1/simulation_data/baseline_comparison_gpr_poly_quick.json

Note:
This script is meant for preliminary estimates only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "stage1" / "simulation_data" / "tmm_emissivity_data.csv"
DEFAULT_OUT = ROOT / "stage1" / "simulation_data" / "baseline_comparison_gpr_poly_quick.json"
DEFAULT_MANIFEST = ROOT / "stage1" / "simulation_data" / "run_manifest_latest.json"


def _load_manifest_run_id_best_effort(path: Path) -> str | None:
    """Best-effort read of manifest_run_id for traceability.

    Preference:
    1) manifest['lineage']['manifest_run_id']
    2) manifest['run_id']
    """
    try:
        if not path.exists() or not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None
        lineage = obj.get("lineage")
        if isinstance(lineage, dict):
            rid = lineage.get("manifest_run_id")
            if isinstance(rid, str) and rid.strip():
                return rid.strip()
        rid2 = obj.get("run_id")
        if isinstance(rid2, str) and rid2.strip():
            return rid2.strip()
        return None
    except Exception:
        return None


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def _sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    try:
        if not path.exists() or not path.is_file():
            return None
        import hashlib

        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _build_data_fingerprint(csv_path: Path, *, rows_loaded: int | None = None) -> Dict[str, Any]:
    fp: Dict[str, Any] = {
        "csv": str(csv_path),
        "rows_loaded": int(rows_loaded) if rows_loaded is not None else None,
        "size_bytes": None,
        "mtime": None,
        "sha256": None,
    }
    try:
        if csv_path.exists() and csv_path.is_file():
            st = csv_path.stat()
            fp["size_bytes"] = int(st.st_size)
            fp["mtime"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime))
            # Compute sha256 only for modest-size CSVs to avoid accidental slowdowns.
            if int(st.st_size) <= 50 * 1024 * 1024:
                fp["sha256"] = _sha256_of_file(csv_path)
    except Exception:
        pass
    return fp


def _ensure_meta_fields(doc: Dict[str, Any], *, manifest_run_id: str, action: str) -> None:
    meta = doc.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        doc["meta"] = meta

    meta["run_id"] = manifest_run_id
    meta["manifest_run_id"] = manifest_run_id
    meta["action"] = action


def _backfill_only(out_path: Path, *, manifest_run_id: str, action: str) -> None:
    if not out_path.exists() or not out_path.is_file():
        raise FileNotFoundError(f"Output JSON not found for backfill: {out_path}")
    with out_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise TypeError(f"Output JSON must be an object/dict: {out_path}")

    _ensure_meta_fields(doc, manifest_run_id=manifest_run_id, action=action)
    _save_json(out_path, doc)


def _subsample_rows(X: np.ndarray, y: np.ndarray, max_n: int | None, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    if max_n is None or max_n <= 0 or len(X) <= max_n:
        return X, y
    idx = rng.choice(len(X), size=int(max_n), replace=False)
    return X[idx], y[idx]


def _fit_predict_timed(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Tuple[float, float, np.ndarray]:
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time_s = time.time() - t0

    t1 = time.time()
    y_pred = model.predict(X_test)
    pred_time_s = time.time() - t1

    pred_time_ms = (pred_time_s / max(1, len(X_test))) * 1000.0
    return float(train_time_s), float(pred_time_ms), np.asarray(y_pred, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="Path to tmm_emissivity_data.csv")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Output JSON path")

    ap.add_argument(
        "--manifest-run-id",
        type=str,
        default="",
        help="Traceability anchor. If empty, read from run_manifest_latest.json when available.",
    )
    ap.add_argument(
        "--manifest-path",
        type=str,
        default=str(DEFAULT_MANIFEST),
        help="Path to run_manifest_latest.json (used to auto-detect manifest_run_id).",
    )

    ap.add_argument(
        "--backfill-only",
        action="store_true",
        help="Only backfill meta.run_id/meta.manifest_run_id/meta.action into an existing output JSON; do not read CSV or train models.",
    )

    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)

    # Optional global downsample before split (keeps memory/IO smaller)
    ap.add_argument("--max-total", type=int, default=0, help="If >0, sample this many rows from full CSV before split")

    # Per-model subsamples after split
    ap.add_argument("--max-train-poly", type=int, default=20000)
    ap.add_argument("--max-test-poly", type=int, default=5000)
    ap.add_argument("--max-train-gpr", type=int, default=1500)
    ap.add_argument("--max-test-gpr", type=int, default=500)

    ap.add_argument("--poly-degree", type=int, default=2)
    ap.add_argument("--poly-alpha", type=float, default=1e-3, help="Ridge alpha for Polynomial baseline")

    ap.add_argument("--gpr-alpha", type=float, default=1e-6, help="Added noise term for GPR")
    ap.add_argument("--gpr-length-scale", type=float, default=1.0)
    ap.add_argument("--gpr-const", type=float, default=1.0)
    ap.add_argument("--gpr-noise", type=float, default=1e-3)

    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    manifest_path = Path(args.manifest_path)

    manifest_run_id = (args.manifest_run_id or "").strip()
    if not manifest_run_id:
        manifest_run_id = _load_manifest_run_id_best_effort(manifest_path) or ""
    if not manifest_run_id:
        # Fallback: keep the file self-describing even when no manifest exists.
        manifest_run_id = time.strftime("%Y%m%d_%H%M%S")

    action_name = "baselines_gpr_poly_quick"

    if bool(args.backfill_only):
        _backfill_only(out_path, manifest_run_id=manifest_run_id, action=action_name)
        print(f"Backfilled meta.run_id/meta.manifest_run_id into: {out_path}")
        return 0

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"波长_μm", "基底厚度_nm", "PDMS厚度_nm", "发射率ε"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"CSV missing required columns: {sorted(missing)}")

    if args.max_total and args.max_total > 0 and len(df) > args.max_total:
        df = df.sample(n=int(args.max_total), random_state=int(args.random_state)).reset_index(drop=True)

    X = df[["波长_μm", "基底厚度_nm", "PDMS厚度_nm"]].values.astype(float)
    y = df["发射率ε"].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(args.test_size), random_state=int(args.random_state)
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rng = np.random.RandomState(int(args.random_state))

    results: Dict[str, Dict[str, Any]] = {}

    # Polynomial baseline: PolynomialFeatures + Ridge
    poly = Pipeline([
        ("poly", PolynomialFeatures(degree=int(args.poly_degree), include_bias=False)),
        ("ridge", Ridge(alpha=float(args.poly_alpha), random_state=int(args.random_state))),
    ])
    X_train_poly, y_train_poly = _subsample_rows(X_train_s, y_train, int(args.max_train_poly), rng)
    X_test_poly, y_test_poly = _subsample_rows(X_test_s, y_test, int(args.max_test_poly), rng)

    train_time_s, pred_time_ms, y_pred = _fit_predict_timed(poly, X_train_poly, y_train_poly, X_test_poly)
    r2 = float(r2_score(y_test_poly, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test_poly, y_pred)))
    mae = float(mean_absolute_error(y_test_poly, y_pred))
    results[f"Polynomial (deg={int(args.poly_degree)}) + Ridge"] = {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "train_time_s": train_time_s,
        "pred_time_ms": pred_time_ms,
        "n_train_used": int(len(X_train_poly)),
        "n_test_used": int(len(X_test_poly)),
        "y_pred_sample": [float(v) for v in y_pred[:1000]],
        "y_test_sample": [float(v) for v in y_test_poly[:1000]],
        "params": {
            "poly_degree": int(args.poly_degree),
            "ridge_alpha": float(args.poly_alpha),
        },
    }

    # GPR baseline (small sample only)
    kernel = ConstantKernel(constant_value=float(args.gpr_const), constant_value_bounds="fixed") * RBF(
        length_scale=float(args.gpr_length_scale), length_scale_bounds="fixed"
    ) + WhiteKernel(noise_level=float(args.gpr_noise), noise_level_bounds="fixed")
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(args.gpr_alpha),
        normalize_y=True,
        optimizer=None,  # keep it deterministic + fast
        random_state=int(args.random_state),
    )

    X_train_gpr, y_train_gpr = _subsample_rows(X_train_s, y_train, int(args.max_train_gpr), rng)
    X_test_gpr, y_test_gpr = _subsample_rows(X_test_s, y_test, int(args.max_test_gpr), rng)

    train_time_s, pred_time_ms, y_pred = _fit_predict_timed(gpr, X_train_gpr, y_train_gpr, X_test_gpr)
    r2 = float(r2_score(y_test_gpr, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test_gpr, y_pred)))
    mae = float(mean_absolute_error(y_test_gpr, y_pred))
    results["GPR (fixed kernel, quick)"] = {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "train_time_s": train_time_s,
        "pred_time_ms": pred_time_ms,
        "n_train_used": int(len(X_train_gpr)),
        "n_test_used": int(len(X_test_gpr)),
        "y_pred_sample": [float(v) for v in y_pred[:1000]],
        "y_test_sample": [float(v) for v in y_test_gpr[:1000]],
        "params": {
            "alpha": float(args.gpr_alpha),
            "kernel": str(kernel),
            "optimizer": None,
            "normalize_y": True,
        },
    }

    out = {
        "meta": {
            "run_id": manifest_run_id,
            "manifest_run_id": manifest_run_id,
            "action": action_name,
            "date": time.strftime("%Y-%m-%d"),
            "csv": str(csv_path),
            "rows_loaded": int(len(df)),
            "data_fingerprint": _build_data_fingerprint(csv_path, rows_loaded=int(len(df))),
            "split_protocol": f"train_test_split:test_size={float(args.test_size)}:random_state={int(args.random_state)}",
            "split": {"test_size": float(args.test_size), "random_state": int(args.random_state)},
            "scaler": "StandardScaler(on X_train)",
            "quick_estimate": {
                "timing_scope": "train_time_s=fit-only; pred_time_ms=predict-only per-point",
                "global_presplit_downsample": {"enabled": bool(args.max_total and args.max_total > 0), "max_total": int(args.max_total)},
                "subsample_after_split": {
                    "max_train_poly": int(args.max_train_poly),
                    "max_test_poly": int(args.max_test_poly),
                    "max_train_gpr": int(args.max_train_gpr),
                    "max_test_gpr": int(args.max_test_gpr),
                },
                "polynomial": {"degree": int(args.poly_degree), "ridge_alpha": float(args.poly_alpha)},
                "gpr": {
                    "alpha": float(args.gpr_alpha),
                    "kernel_fixed": {
                        "const": float(args.gpr_const),
                        "length_scale": float(args.gpr_length_scale),
                        "noise": float(args.gpr_noise),
                    },
                    "optimizer": None,
                    "normalize_y": True,
                },
            },
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "results": results,
    }

    _save_json(out_path, out)
    print(f"Wrote: {out_path}")
    for name, r in results.items():
        print(
            f"- {name}: R2={r['r2']:.6f}, RMSE={r['rmse']:.6f}, MAE={r['mae']:.6f}, "
            f"train={r['train_time_s']:.3f}s, pred={r['pred_time_ms']:.6f}ms/pt, "
            f"n_train={r['n_train_used']}, n_test={r['n_test_used']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
