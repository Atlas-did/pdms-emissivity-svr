#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(_HERE)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)


ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "stage1" / "simulation_data"
REPORTS = ROOT / "reports"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists() or not path.is_file():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _read_json_any(path: Path) -> Any:
    try:
        if not path.exists() or not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt(v: Any, *, ndigits: int = 6) -> str:
    if v is None:
        return "(missing)"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isnan(float(v)):
            return "nan"
        return f"{float(v):.{ndigits}g}"
    return str(v)


def _extract_run_context_like(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    out: Dict[str, Any] = {}
    for k in ["run_id", "manifest_run_id", "action", "split_protocol", "data_fingerprint"]:
        if k in obj:
            out[k] = obj.get(k)
        elif k in meta:
            out[k] = meta.get(k)
    return out


def _std_vs_rmse_from_test_results(test_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        y_true = test_results.get("y_test")
        y_pred = test_results.get("y_pred")
        if not isinstance(y_true, list) or not isinstance(y_pred, list) or not y_true or not y_pred:
            return None
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        std_y = float(np.std(yt))
        return {
            "scope": "global",
            "n": int(len(yt)),
            "std_y": std_y,
            "rmse": rmse,
            "rmse_over_std": (rmse / std_y) if std_y > 0 else None,
            "r2": test_results.get("r2"),
            "mae": test_results.get("mae"),
        }
    except Exception:
        return None


def _collect_region_consistency_rows() -> List[Dict[str, Any]]:
    # Prefer precomputed table if present.
    path = SIM / "region_metric_consistency.json"
    obj = _read_json(path)
    if obj and isinstance(obj.get("rows"), list):
        rows = [r for r in obj.get("rows") if isinstance(r, dict)]
        for r in rows:
            r.setdefault("source", str(path))
        return rows
    return []


def _collect_cache_evidence() -> Dict[str, Any]:
    perf = _read_json(SIM / "performance_summary_stage1_tmm.json")
    quick = _read_json(SIM / "cache_benchmark_quick.json")

    out: Dict[str, Any] = {"performance_summary_stage1_tmm": perf, "cache_benchmark_quick": quick}
    return out


def _collect_training_time_evidence() -> Dict[str, Any]:
    perf = _read_json(SIM / "performance_summary_stage1_training.json")
    out: Dict[str, Any] = {"performance_summary_stage1_training": perf}
    return out


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not headers:
        return ""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(SIM / "run_manifest_latest.json")
    test_results = _read_json(SIM / "test_results.json")
    inspect_report = _read_json(SIM / "inspect_regions_report.json")

    rc_manifest = _extract_run_context_like(manifest)
    rc_test = _extract_run_context_like(test_results)
    rc_inspect = _extract_run_context_like(inspect_report)

    std_rmse_rows: List[Dict[str, Any]] = []
    r_global = _std_vs_rmse_from_test_results(test_results) if test_results else None
    if r_global:
        r_global["source"] = str(SIM / "test_results.json")
        std_rmse_rows.append(r_global)

    for r in _collect_region_consistency_rows():
        # normalize known keys if present
        std_rmse_rows.append(r)

    cache = _collect_cache_evidence()
    train_time = _collect_training_time_evidence()

    md: List[str] = []
    md.append(f"# Paper Evidence Table\n")
    md.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n")

    md.append("## Traceability (RunContext)\n")
    md.append("This project standardizes the following audit fields on key JSON artifacts:\n")
    md.append("- run_id (forced == manifest_run_id)\n- manifest_run_id\n- action\n- split_protocol\n- data_fingerprint\n")

    md.append("**Best-effort extracted from run_manifest_latest.json:**\n")
    md.append(_md_table(
        ["field", "value"],
        [[k, _fmt(v, ndigits=10)] for k, v in rc_manifest.items()] or [["(missing)", "run_manifest_latest.json not found or has no fields"]],
    ))
    md.append("\n\n**Best-effort extracted from test_results.json:**\n")
    md.append(_md_table(
        ["field", "value"],
        [[k, _fmt(v, ndigits=10)] for k, v in rc_test.items()] or [["(missing)", "test_results.json not found or has no fields"]],
    ))

    md.append("\n\n**Best-effort extracted from inspect_regions_report.json:**\n")
    md.append(_md_table(
        ["field", "value"],
        [[k, _fmt(v, ndigits=10)] for k, v in rc_inspect.items()] or [["(missing)", "inspect_regions_report.json not found or has no fields"]],
    ))

    md.append("\n\n## Negative R² Explanation Aid (std(y) vs rmse)\n")
    md.append("Rule of thumb: if RMSE is comparable to or larger than std(y), R² may become small or negative even when MAE/RMSE look reasonable.\n")

    if std_rmse_rows:
        rows_md: List[List[str]] = []
        for r in std_rmse_rows:
            scope = str(r.get("scope") or r.get("region") or "(unknown)")
            rows_md.append([
                scope,
                _fmt(r.get("n")),
                _fmt(r.get("std_y")),
                _fmt(r.get("rmse")),
                _fmt(r.get("rmse_over_std")),
                _fmt(r.get("r2")),
                _fmt(r.get("mae")),
                str(r.get("source") or r.get("metrics_source") or ""),
            ])
        md.append(_md_table(
            ["scope", "n", "std(y)", "rmse", "rmse/std", "r2", "mae", "source"],
            rows_md,
        ))
    else:
        md.append("(missing) No std(y)/rmse evidence found yet.\n\n")
        md.append("To generate it:\n")
        md.append("- Global: run Stage1 evaluate to create stage1/simulation_data/test_results.json\n")
        md.append("- Regions: run Stage1 train_regions then run stage1/check_region_metric_consistency.py\n")

    md.append("\n\n## Cache Evidence (hit_rate=0 & duplicate=0)\n")
    perf_tmm = cache.get("performance_summary_stage1_tmm")
    if isinstance(perf_tmm, dict):
        md.append("**From performance_summary_stage1_tmm.json:**\n")
        md.append(_md_table(
            ["field", "value"],
            [
                ["tmm_cache_hit_rate", _fmt(perf_tmm.get("tmm_cache_hit_rate"))],
                ["tmm_cache_total_requests", _fmt(perf_tmm.get("tmm_cache_total_requests"))],
                ["tmm_cache_unique_key_count", _fmt(perf_tmm.get("tmm_cache_unique_key_count"))],
                ["tmm_cache_duplicate_key_count", _fmt(perf_tmm.get("tmm_cache_duplicate_key_count"))],
                ["tmm_cache_unique_key_count_capped", _fmt(perf_tmm.get("tmm_cache_unique_key_count_capped"))],
            ],
        ))
    else:
        md.append("(missing) stage1/simulation_data/performance_summary_stage1_tmm.json not found.\n")
        md.append("To generate it: run Stage1 action simulate (or full_pipeline) once.\n")

    quick_cache = cache.get("cache_benchmark_quick")
    if isinstance(quick_cache, dict):
        md.append("\n**From cache_benchmark_quick.json (CSV duplicate check):**\n")
        md.append(_md_table(
            ["field", "value"],
            [
                ["duplicate_key_count", _fmt(quick_cache.get("duplicate_key_count"))],
                ["rows_total", _fmt(quick_cache.get("rows_total"))],
                ["source_performance_file", _fmt(quick_cache.get("source_performance_file"), ndigits=10)],
            ],
        ))
    else:
        md.append("\n(missing) stage1/simulation_data/cache_benchmark_quick.json not found.\n")
        md.append("To generate it: run stage1/generate_missing_paper_artifacts_quick.py once.\n")

    md.append("\n\n## Training Time Evidence (fit-only vs end-to-end search)\n")
    perf_train = train_time.get("performance_summary_stage1_training")
    if isinstance(perf_train, dict):
        md.append("**From performance_summary_stage1_training.json:**\n")
        md.append(_md_table(
            ["field", "value"],
            [
                ["search_method", _fmt(perf_train.get("search_method"), ndigits=10)],
                ["search_elapsed_s (end-to-end search)", _fmt(perf_train.get("search_elapsed_s"))],
                ["refit_elapsed_s (SVR-fit-only best params)", _fmt(perf_train.get("refit_elapsed_s"))],
                ["train_total_elapsed_s", _fmt(perf_train.get("train_total_elapsed_s"))],
            ],
        ))
    else:
        md.append("(missing) stage1/simulation_data/performance_summary_stage1_training.json not found.\n")
        md.append("To generate it: run Stage1 action train (or full_pipeline) once.\n")

    out_path = REPORTS / "paper_evidence_table.md"
    out_path.write_text("\n".join(md).strip() + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
