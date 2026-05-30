#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract per-region Support Vector counts and metrics completeness.

Purpose (Phase 3): generate a "replacement table" for the paper:
  region -> band -> SV -> MAE -> RMSE -> R2 -> run_id attribution

This tool is designed to be safe and audit-friendly:
- It never trains.
- It can run even when some artifacts are missing; it will report reasons.
- It prefers reading already-saved summaries; optional MAE computation is best-effort.

Outputs:
- JSON table (default): stage1/simulation_data/region_sv_metrics_table.json
- CSV table  (default): stage1/simulation_data/region_sv_metrics_table.csv

"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


try:
    import joblib
except Exception as e:  # pragma: no cover
    joblib = None  # type: ignore[assignment]

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]


def _load_json_best_effort(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists() or not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Write a stable header even when empty, so downstream tooling has columns.
        fieldnames = [
            "region",
            "lambda_min",
            "lambda_max",
            "th_min",
            "th_max",
            "support_vectors",
            "mae",
            "rmse",
            "r2",
            "metrics_source",
            "predictions_path",
            "predictions_rows",
            "run_id_attribution",
            "manifest_run_id",
            "issues",
            "model_path",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
        return

    # stable header
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _manifest_run_id_from_manifest_latest(manifest: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(manifest, dict):
        return None
    lineage = manifest.get("lineage")
    if isinstance(lineage, dict):
        rid = lineage.get("manifest_run_id")
        if isinstance(rid, str) and rid.strip():
            return rid.strip()
    rid2 = manifest.get("run_id")
    if isinstance(rid2, str) and rid2.strip():
        return rid2.strip()
    return None


def _find_svr_like(obj: Any) -> Optional[Any]:
    """Traverse common containers to find an estimator with support_ attribute."""
    if obj is None:
        return None

    if hasattr(obj, "support_"):
        return obj

    # sklearn Pipeline
    named_steps = getattr(obj, "named_steps", None)
    if isinstance(named_steps, dict):
        for v in named_steps.values():
            found = _find_svr_like(v)
            if found is not None:
                return found

    # custom wrappers
    for attr in ("estimator", "model", "svr", "regressor"):
        if hasattr(obj, attr):
            found = _find_svr_like(getattr(obj, attr))
            if found is not None:
                return found

    # dict container
    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_svr_like(v)
            if found is not None:
                return found

    # list/tuple container
    if isinstance(obj, (list, tuple)):
        for v in obj:
            found = _find_svr_like(v)
            if found is not None:
                return found

    return None


def _extract_support_vector_count(model_obj: Any) -> Tuple[Optional[int], Optional[str]]:
    est = _find_svr_like(model_obj)
    if est is None:
        return None, "no_svr_like_estimator_found"

    support = getattr(est, "support_", None)
    if support is None:
        return None, "svr_missing_support_"

    try:
        return int(len(support)), None
    except Exception:
        return None, "support_len_failed"


def _metrics_from_arrays(y_true: List[float], y_pred: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    """Return (mae, rmse, r2, issue)."""
    if len(y_true) != len(y_pred):
        return None, None, None, "pred_length_mismatch"
    if not y_true:
        return None, None, None, "pred_empty"

    # Use numpy if available, else Python loops.
    if np is not None:
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        err = yt - yp
        mae = float(np.mean(np.abs(err)))
        rmse = float(math.sqrt(float(np.mean(err * err))))
        sse = float(np.sum(err * err))
        yt_mean = float(np.mean(yt))
        sst = float(np.sum((yt - yt_mean) ** 2))
    else:
        abs_sum = 0.0
        sq_sum = 0.0
        mean = sum(float(v) for v in y_true) / float(len(y_true))
        sst = 0.0
        for a, b in zip(y_true, y_pred):
            e = float(a) - float(b)
            abs_sum += abs(e)
            sq_sum += e * e
        for a in y_true:
            d = float(a) - float(mean)
            sst += d * d
        mae = abs_sum / float(len(y_true))
        rmse = math.sqrt(sq_sum / float(len(y_true)))
        sse = sq_sum

    r2: Optional[float]
    if sst <= 0.0:
        r2 = None
        issue = "r2_undefined_zero_variance"
    else:
        r2 = float(1.0 - (sse / sst))
        issue = None
    return mae, rmse, r2, issue


def _load_predictions_from_json(path: Path) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None, None, "pred_json_not_object"
        # Accept y_test/y_pred OR y_true/y_pred
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
    """Load y_true/y_pred from a CSV.

    Supports column names: y_true/y_pred, y_test/y_pred, 真实值/预测值.
    """
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

        # csv module fallback
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


@dataclass
class RegionRow:
    region: str
    lambda_min: Optional[float]
    lambda_max: Optional[float]
    th_min: Optional[float]
    th_max: Optional[float]
    support_vectors: Optional[int]
    mae: Optional[float]
    rmse: Optional[float]
    r2: Optional[float]
    metrics_source: str
    predictions_path: str
    predictions_rows: Optional[int]
    run_id_attribution: str
    manifest_run_id: Optional[str]
    issues: str
    model_path: str


def _to_row_dict(r: RegionRow) -> Dict[str, Any]:
    return {
        "region": r.region,
        "lambda_min": r.lambda_min,
        "lambda_max": r.lambda_max,
        "th_min": r.th_min,
        "th_max": r.th_max,
        "support_vectors": r.support_vectors,
        "mae": r.mae,
        "rmse": r.rmse,
        "r2": r.r2,
        "metrics_source": r.metrics_source,
        "predictions_path": r.predictions_path,
        "predictions_rows": r.predictions_rows,
        "run_id_attribution": r.run_id_attribution,
        "manifest_run_id": r.manifest_run_id,
        "issues": r.issues,
        "model_path": r.model_path,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Extract region SV counts and metrics completeness")

    ap.add_argument(
        "--simulation-data",
        type=str,
        default="stage1/simulation_data",
        help="Stage1 simulation_data directory",
    )
    ap.add_argument(
        "--region-models-dir",
        type=str,
        default="",
        help="Override region_models directory (default: <simulation-data>/region_models)",
    )
    ap.add_argument(
        "--region-config",
        type=str,
        default="",
        help="Override region_config.json path (default: <simulation-data>/region_config.json)",
    )
    ap.add_argument(
        "--region-summary",
        type=str,
        default="",
        help="Override region_training_summary.json path (default: <simulation-data>/region_training_summary.json)",
    )
    ap.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Override run_manifest_latest.json path (default: <simulation-data>/run_manifest_latest.json)",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Output JSON path (default: <simulation-data>/region_sv_metrics_table.json)",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Output CSV path (default: <simulation-data>/region_sv_metrics_table.csv)",
    )

    args = ap.parse_args(argv)

    sim_dir = Path(args.simulation_data)
    region_models_dir = Path(args.region_models_dir) if args.region_models_dir.strip() else (sim_dir / "region_models")
    region_cfg_path = Path(args.region_config) if args.region_config.strip() else (sim_dir / "region_config.json")
    region_summary_path = Path(args.region_summary) if args.region_summary.strip() else (sim_dir / "region_training_summary.json")
    manifest_path = Path(args.manifest) if args.manifest.strip() else (sim_dir / "run_manifest_latest.json")

    out_json = Path(args.out_json) if args.out_json.strip() else (sim_dir / "region_sv_metrics_table.json")
    out_csv = Path(args.out_csv) if args.out_csv.strip() else (sim_dir / "region_sv_metrics_table.csv")

    manifest = _load_json_best_effort(manifest_path)
    manifest_run_id = _manifest_run_id_from_manifest_latest(manifest)

    region_cfg = _load_json_best_effort(region_cfg_path)
    regions_obj: Dict[str, Any] = {}
    if isinstance(region_cfg, dict):
        ro = region_cfg.get("regions")
        if isinstance(ro, dict):
            regions_obj = ro

    region_summary = _load_json_best_effort(region_summary_path)
    summary_regions: Dict[str, Any] = {}
    summary_run_id = None
    if isinstance(region_summary, dict):
        sr = region_summary.get("regions")
        if isinstance(sr, dict):
            summary_regions = sr
        rid = region_summary.get("run_id")
        if isinstance(rid, str) and rid.strip():
            summary_run_id = rid.strip()

    # Determine which regions to report.
    region_names: List[str] = []
    if regions_obj:
        region_names = sorted(regions_obj.keys())
    elif summary_regions:
        region_names = sorted(summary_regions.keys())
    else:
        # fallback: from model directories
        if region_models_dir.exists() and region_models_dir.is_dir():
            region_names = sorted([p.name for p in region_models_dir.iterdir() if p.is_dir()])

    rows: List[RegionRow] = []

    if joblib is None:
        # Still emit a file explaining why SV extraction is unavailable.
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "manifest_run_id": manifest_run_id,
            "error": "joblib_import_failed",
            "note": "joblib is required to load saved sklearn models. Install dependencies then re-run.",
            "rows": [],
        }
        _write_json(out_json, payload)
        _write_csv(out_csv, [])
        print(f"Wrote: {out_json}")
        print(f"Wrote: {out_csv}")
        return 2

    if not region_names:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "manifest_run_id": manifest_run_id,
            "note": "No regions discovered (missing region_config/region_summary/region_models).",
            "rows": [],
        }
        _write_json(out_json, payload)
        _write_csv(out_csv, [])
        print(f"Wrote: {out_json}")
        print(f"Wrote: {out_csv}")
        return 0

    for region_name in region_names:
        info = regions_obj.get(region_name, {}) if isinstance(regions_obj, dict) else {}

        def _f(k: str) -> Optional[float]:
            try:
                v = info.get(k)
                return float(v) if v is not None else None
            except Exception:
                return None

        lam_min = _f("lambda_min")
        lam_max = _f("lambda_max")
        th_min = _f("th_min")
        th_max = _f("th_max")

        issues: List[str] = []

        # SV count extraction
        model_path = region_models_dir / region_name / "svr_model.pkl"
        sv_count: Optional[int] = None
        if model_path.exists() and model_path.is_file():
            try:
                model_obj = joblib.load(model_path)
                sv_count, sv_err = _extract_support_vector_count(model_obj)
                if sv_err:
                    issues.append(f"sv_extract:{sv_err}")
            except Exception as e:
                issues.append(f"sv_load_failed:{type(e).__name__}")
        else:
            issues.append("missing_model_pkl")

        # Metrics extraction: prefer per-region prediction files when present.
        mae = None
        rmse = None
        r2 = None
        metrics_source = "missing"
        predictions_path = ""
        predictions_rows: Optional[int] = None

        region_dir = region_models_dir / region_name
        pred_candidates = _find_prediction_file_candidates(region_dir)
        if pred_candidates:
            pred_path = pred_candidates[0]
            predictions_path = str(pred_path).replace("\\", "/")
            y_true: Optional[List[float]] = None
            y_pred: Optional[List[float]] = None
            load_issue: Optional[str] = None
            if pred_path.suffix.lower() == ".json":
                y_true, y_pred, load_issue = _load_predictions_from_json(pred_path)
                metrics_source = "pred_json"
            elif pred_path.suffix.lower() == ".csv":
                y_true, y_pred, load_issue = _load_predictions_from_csv(pred_path)
                metrics_source = "pred_csv"
            else:
                load_issue = "pred_unsupported_suffix"

            if load_issue:
                issues.append(load_issue)
            elif y_true is not None and y_pred is not None:
                predictions_rows = int(len(y_true))
                mae, rmse, r2, m_issue = _metrics_from_arrays(y_true, y_pred)
                if m_issue:
                    issues.append(m_issue)
        
        # Fallback: read from region_training_summary.json (no recompute).
        if mae is None and region_name in summary_regions:
            rinfo = summary_regions.get(region_name)
            if isinstance(rinfo, dict):
                metrics_source = "summary"
                try:
                    mae_v = rinfo.get("mae")
                    if mae_v is not None:
                        mae = float(mae_v)
                    rmse_v = rinfo.get("rmse")
                    if rmse_v is not None:
                        rmse = float(rmse_v)
                    r2_v = rinfo.get("r2")
                    if r2_v is not None:
                        r2 = float(r2_v)
                except Exception:
                    issues.append("summary_metric_parse_failed")
            else:
                issues.append("summary_region_entry_not_dict")
        elif mae is None and region_summary_path.exists():
            issues.append("region_missing_in_summary")

        if mae is None:
            issues.append("mae_missing")

        # Attribution
        run_id_attr_parts = []
        if manifest_run_id:
            run_id_attr_parts.append(f"manifest_run_id={manifest_run_id}")
        if summary_run_id:
            run_id_attr_parts.append(f"summary_run_id={summary_run_id}")
        run_id_attribution = ";".join(run_id_attr_parts) if run_id_attr_parts else "unknown"

        rows.append(
            RegionRow(
                region=region_name,
                lambda_min=lam_min,
                lambda_max=lam_max,
                th_min=th_min,
                th_max=th_max,
                support_vectors=sv_count,
                mae=mae,
                rmse=rmse,
                r2=r2,
                metrics_source=metrics_source,
                predictions_path=predictions_path,
                predictions_rows=predictions_rows,
                run_id_attribution=run_id_attribution,
                manifest_run_id=manifest_run_id,
                issues=";".join(issues),
                model_path=str(model_path).replace("\\", "/"),
            )
        )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "manifest_run_id": manifest_run_id,
        "inputs": {
            "region_config": str(region_cfg_path).replace("\\", "/"),
            "region_summary": str(region_summary_path).replace("\\", "/"),
            "region_models_dir": str(region_models_dir).replace("\\", "/"),
            "manifest": str(manifest_path).replace("\\", "/"),
        },
        "rows": [_to_row_dict(r) for r in rows],
    }

    _write_json(out_json, payload)
    _write_csv(out_csv, [_to_row_dict(r) for r in rows])

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")

    # Small console summary
    missing_mae = sum(1 for r in rows if r.mae is None)
    missing_model = sum(1 for r in rows if "missing_model_pkl" in (r.issues or ""))
    print(f"Regions: {len(rows)} | missing_model={missing_model} | missing_mae={missing_mae}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
