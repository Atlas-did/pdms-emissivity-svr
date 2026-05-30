#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Region model diagnostics for paper traceability.

Outputs (default under stage1/simulation_data/):
- region_model_diagnostics.json
- region_model_diagnostics.csv

Each row includes:
- region
- support_vectors (len(model.support_))
- model_sha256 / size_bytes / mtime
- manifest_run_id attribution (best effort)

This script is read-only: it does not retrain anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIM = ROOT / "stage1" / "simulation_data"


def _sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    try:
        if not path.exists() or not path.is_file():
            return None
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


def _load_manifest_run_id_best_effort(path: Path) -> Optional[str]:
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


def _extract_sv_count(model_obj: Any) -> Tuple[Optional[int], Optional[str]]:
    try:
        if hasattr(model_obj, "support_"):
            sup = getattr(model_obj, "support_", None)
            if sup is None:
                return None, "svr_missing_support_"
            return int(len(sup)), None

        # Pipelines / wrappers
        for attr in ("named_steps", "steps"):
            v = getattr(model_obj, attr, None)
            if v is None:
                continue
            if isinstance(v, dict):
                for _, step in v.items():
                    if hasattr(step, "support_"):
                        return int(len(getattr(step, "support_"))), None
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, tuple) and len(item) == 2:
                        step = item[1]
                    else:
                        step = item
                    if hasattr(step, "support_"):
                        return int(len(getattr(step, "support_"))), None
        return None, "svr_not_found"
    except Exception:
        return None, "sv_extract_failed"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "region",
        "support_vectors",
        "model_path",
        "model_sha256",
        "size_bytes",
        "mtime",
        "manifest_run_id",
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
    ap.add_argument("--manifest", type=str, default="")
    ap.add_argument("--out-json", type=str, default="")
    ap.add_argument("--out-csv", type=str, default="")
    args = ap.parse_args()

    sim_dir = Path(args.simulation_data)
    region_models_dir = Path(args.region_models_dir) if args.region_models_dir.strip() else (sim_dir / "region_models")
    manifest_path = Path(args.manifest) if args.manifest.strip() else (sim_dir / "run_manifest_latest.json")

    out_json = Path(args.out_json) if args.out_json.strip() else (sim_dir / "region_model_diagnostics.json")
    out_csv = Path(args.out_csv) if args.out_csv.strip() else (sim_dir / "region_model_diagnostics.csv")

    manifest_run_id = _load_manifest_run_id_best_effort(manifest_path)

    rows: List[Dict[str, Any]] = []
    if not region_models_dir.exists() or not region_models_dir.is_dir():
        doc = {
            "meta": {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "simulation_data": str(sim_dir).replace("\\", "/"),
                "region_models_dir": str(region_models_dir).replace("\\", "/"),
                "manifest_run_id": manifest_run_id,
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
        model_path = region_dir / "svr_model.pkl"
        issues: List[str] = []

        sv = None
        if model_path.exists() and model_path.is_file() and joblib is not None:
            try:
                model_obj = joblib.load(model_path)
                sv, sv_issue = _extract_sv_count(model_obj)
                if sv_issue:
                    issues.append(sv_issue)
            except Exception as e:
                issues.append(f"model_load_failed:{type(e).__name__}")
        else:
            if joblib is None:
                issues.append("joblib_missing")
            if not model_path.exists():
                issues.append("model_missing")

        sha256 = _sha256_of_file(model_path) if model_path.exists() else None
        size_bytes = None
        mtime = None
        try:
            if model_path.exists() and model_path.is_file():
                st = model_path.stat()
                size_bytes = int(st.st_size)
                mtime = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime))
        except Exception:
            pass

        rows.append(
            {
                "region": region,
                "support_vectors": sv,
                "model_path": str(model_path).replace("\\", "/"),
                "model_sha256": sha256,
                "size_bytes": size_bytes,
                "mtime": mtime,
                "manifest_run_id": manifest_run_id,
                "issues": ";".join(issues),
            }
        )

    doc = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "simulation_data": str(sim_dir).replace("\\", "/"),
            "region_models_dir": str(region_models_dir).replace("\\", "/"),
            "manifest_run_id": manifest_run_id,
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
