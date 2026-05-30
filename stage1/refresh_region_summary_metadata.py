#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

if __package__ is None or __package__ == "":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(_HERE)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from stage1.main_pdms_svr_bandscan_full import Config, SVRTrainer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "stage1" / "simulation_data"
SUMMARY_PATH = OUT / "region_training_summary.json"
REGION_MODELS_DIR = OUT / "region_models"


def main() -> int:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary: {SUMMARY_PATH}")

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    results = summary.get("results", [])
    if not isinstance(results, list):
        raise ValueError("region_training_summary.json: results must be a list")

    run_id = str(summary.get("run_id") or summary.get("meta", {}).get("run_id") or "manual_refresh")
    cfg = Config()
    cfg.load()
    tmp_trainer = SVRTrainer(cfg)

    for item in results:
        region = str(item.get("region", ""))
        if not region:
            continue
        region_dir = REGION_MODELS_DIR / region
        model_path = region_dir / "svr_model.pkl"
        scaler_path = region_dir / "scaler.pkl"
        csv_path = region_dir / "region_training_data.csv"

        if not (model_path.exists() and scaler_path.exists() and csv_path.exists()):
            continue

        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            df = pd.read_csv(csv_path)
            X = tmp_trainer._build_features(df)
            y = df["发射率ε"].values.astype(float)
            Xs = scaler.transform(X)
            y_pred = np.clip(model.predict(Xs), 0.0, 1.0)
            item["mae"] = float(mean_absolute_error(y, y_pred)) if len(y) else float("nan")
            item["support_vectors"] = int(getattr(model, "support_", np.array([])).shape[0])
            item.setdefault("samples", int(len(df)))
        except Exception:
            # Keep best-effort behavior; do not break all regions on one failure.
            continue

        item["run_id"] = run_id
        item["split_protocol"] = str(item.get("split_protocol") or "group_split_by_thickness_with_random_fallback")
        item.setdefault("test_size", 0.3)
        item.setdefault("random_state", 42)

    summary["run_id"] = run_id
    summary["split_protocol"] = str(summary.get("split_protocol") or "region_train_group_split_by_thickness_with_fallback")
    summary.setdefault("meta", {})
    summary["meta"]["refreshed_by"] = "refresh_region_summary_metadata.py"

    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"Updated: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
