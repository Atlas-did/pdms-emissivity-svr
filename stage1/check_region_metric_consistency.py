#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consistency check for per-region metrics.

Goal (paper safety): distinguish statistical negative R2 (low std(y)) from run/split mismatch.

Reads per-region prediction artifacts if present under:
  <simulation_data>/region_models/<region>/

Supported files (first match is used):
  - region_test_predictions.json / region_eval_predictions.json / eval_predictions.json / predictions.json
  - region_test_predictions.csv  / region_eval_predictions.csv  / eval_predictions.csv  / predictions.csv

Expected arrays/columns:
  - JSON: y_test/y_pred or y_true/y_pred
  - CSV : y_test,y_pred or y_true,y_pred (also supports 真实值/预测值)

Outputs:
  - region_metric_consistency.json
  - region_metric_consistency.csv

This script does not retrain anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIM = ROOT / "stage1" / "simulation_data"


def _find_prediction_file_candidates(region_dir: Path) -> List[Path]:
    names = [
        "region_test_predictions.json",
        "region_eval_predictions.json",
        "eval_predictions.json",
        "predictions.json",
        "region_test_predictions.csv",
        "region_eval_predictions.csv",
        "eval_predictions.csv",
        "predictions.csv",
    ]
    out: List[Path] = []
    for n in names:
        p = region_dir / n
        if p.exists() and p.is_file():
            out.append(p)
    return out


def _metrics_from_arrays(y_true: List[float], y_pred: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[str]]:
    """Return (std_y, rmse, mae, r2, issue)."""
    if len(y_true) != len(y_pred):
        return None, None, None, None, "pred_length_mismatch"
    if not y_true:
        return None, None, None, None, "pred_empty"

    if np is not None:
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        err = yt - yp
        rmse = float(math.sqrt(float(np.mean(err * err))))
        mae = float(np.mean(np.abs(err)))
        std_y = float(np.std(yt))
        sse = float(np.sum(err * err))
        yt_mean = float(np.mean(yt))
        sst = float(np.sum((yt - yt_mean) ** 2))
    else:
        mean = sum(float(v) for v in y_true) / float(len(y_true))
        sst = 0.0
        abs_sum = 0.0
        sq_sum = 0.0
        var_sum = 0.0
        for a, b in zip(y_true, y_pred):
            a = float(a)
            b = float(b)
            e = a - b
            abs_sum += abs(e)
            sq_sum += e * e
            d = a - mean
            var_sum += d * d
        rmse = math.sqrt(sq_sum / float(len(y_true)))
        mae = abs_sum / float(len(y_true))
        std_y = math.sqrt(var_sum / float(len(y_true)))
        sse = sq_sum
        sst = var_sum

    if sst <= 0.0:
        return std_y, rmse, mae, None, "r2_undefined_zero_variance"
    r2 = float(1.0 - (sse / sst))
    return std_y, rmse, mae, r2, None


def _load_predictions_from_json(path: Path) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None, None, "pred_json_not_object"
        y_true = obj.get("y_true")
        if y_true is None:
            y_true = obj.get("y_test")
        y_pred = obj.get("y_pred")
        if not isinstance(y_true, list) or not isinstance(y_pred, list):
            return None, None, "pred_json_missing_arrays"
        return [float(v) for v in y_true], [float(v) for v in y_pred], None
    except Exception as e:
        return None, None, f"pred_json_load_failed:{type(e).__name__}"


def _load_predictions_from_csv(path: Path) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[str]]:
    try:
        if pd is not None:
            df = pd.read_csv(path)
            cols = {c.strip(): c for c in df.columns}

            def _get_col(*names: str) -> Optional[str]:
                for n in names:
                    if n in cols:
                        return cols[n]
                return None

            y_true_col = _get_col("y_true", "y_test", "真实值", "y")
            y_pred_col = _get_col("y_pred", "预测值", "yhat")
            if y_true_col is None or y_pred_col is None:
                return None, None, "pred_csv_missing_columns"
            y_true = [float(v) for v in df[y_true_col].astype(float).tolist()]
            y_pred = [float(v) for v in df[y_pred_col].astype(float).tolist()]
            return y_true, y_pred, None

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return None, None, "pred_csv_no_header"
            fields = {name.strip(): name for name in reader.fieldnames}

            def _get_field(*names: str) -> Optional[str]:
                for n in names:
                    if n in fields:
                        return fields[n]
                return None

            y_true_field = _get_field("y_true", "y_test", "真实值", "y")
            y_pred_field = _get_field("y_pred", "预测值", "yhat")
            if y_true_field is None or y_pred_field is None:
                return None, None, "pred_csv_missing_columns"
            y_true: List[float] = []
            y_pred: List[float] = []
            for row in reader:
                y_true.append(float(row[y_true_field]))
                y_pred.append(float(row[y_pred_field]))
            return y_true, y_pred, None
    except Exception as e:
        return None, None, f"pred_csv_load_failed:{type(e).__name__}"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "region",
        "predictions_path",
        "n",
        "std_y",
        "rmse",
        "mae",
        "r2",
        "rmse_gt_std_y",
        "issues",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulation-data", type=str, default=str(DEFAULT_SIM))
    ap.add_argument("--region-models-dir", type=str, default="")
    ap.add_argument("--out-json", type=str, default="")
    ap.add_argument("--out-csv", type=str, default="")
    args = ap.parse_args()

    sim_dir = Path(args.simulation_data)
    region_models_dir = Path(args.region_models_dir) if args.region_models_dir.strip() else (sim_dir / "region_models")
    out_json = Path(args.out_json) if args.out_json.strip() else (sim_dir / "region_metric_consistency.json")
    out_csv = Path(args.out_csv) if args.out_csv.strip() else (sim_dir / "region_metric_consistency.csv")

    rows: List[Dict[str, Any]] = []

    if not region_models_dir.exists() or not region_models_dir.is_dir():
        doc = {
            "meta": {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "simulation_data": str(sim_dir).replace("\\", "/"),
                "region_models_dir": str(region_models_dir).replace("\\", "/"),
                "note": "region_models_dir not found",
            },
            "rows": [],
        }
        _write_json(out_json, doc)
        _write_csv(out_csv, [])
        print(f"Wrote: {out_json}")
        print(f"Wrote: {out_csv}")
        return 0

    for region_dir in sorted([p for p in region_models_dir.iterdir() if p.is_dir()]):
        region = region_dir.name
        issues: List[str] = []
        preds_path = ""
        n: Optional[int] = None
        std_y = None
        rmse = None
        mae = None
        r2 = None
        rmse_gt_std_y: Optional[bool] = None

        cands = _find_prediction_file_candidates(region_dir)
        if not cands:
            issues.append("predictions_missing")
        else:
            p = cands[0]
            preds_path = str(p).replace("\\", "/")
            if p.suffix.lower() == ".json":
                y_true, y_pred, load_issue = _load_predictions_from_json(p)
            else:
                y_true, y_pred, load_issue = _load_predictions_from_csv(p)

            if load_issue:
                issues.append(load_issue)
            elif y_true is not None and y_pred is not None:
                n = int(len(y_true))
                std_y, rmse, mae, r2, m_issue = _metrics_from_arrays(y_true, y_pred)
                if m_issue:
                    issues.append(m_issue)
                if std_y is not None and rmse is not None and std_y > 0:
                    rmse_gt_std_y = bool(rmse > std_y)

        rows.append(
            {
                "region": region,
                "predictions_path": preds_path,
                "n": n,
                "std_y": std_y,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "rmse_gt_std_y": rmse_gt_std_y,
                "issues": ";".join(issues),
            }
        )

    doc = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "simulation_data": str(sim_dir).replace("\\", "/"),
            "region_models_dir": str(region_models_dir).replace("\\", "/"),
        },
        "rows": rows,
    }
    _write_json(out_json, doc)
    _write_csv(out_csv, rows)
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
