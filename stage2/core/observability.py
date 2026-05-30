from __future__ import annotations

import csv
import json
import traceback
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ErrorContext:
    stage: str
    region: str = "global"
    file: str = ""
    sample_range: str = ""


def log_exception_with_context(exc: Exception, context: ErrorContext, *, print_traceback: bool = True) -> None:
    payload = asdict(context)
    print(
        f"❌ [{payload['stage']}] {type(exc).__name__}: {exc} | "
        f"region={payload['region']} | file={payload['file']} | sample_range={payload['sample_range']}"
    )
    if print_traceback:
        traceback.print_exc()


def write_lightweight_summary(outdir: Any, name: str, payload: Dict[str, Any]) -> Dict[str, str]:
    outdir = Path(outdir) if not isinstance(outdir, Path) else outdir
    outdir.mkdir(parents=True, exist_ok=True)

    json_path = outdir / f"{name}.json"
    csv_path = outdir / f"{name}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    row = {}
    for k, v in payload.items():
        if isinstance(v, (dict, list, tuple)):
            row[k] = json.dumps(v, ensure_ascii=False)
        else:
            row[k] = v

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    index_path = outdir / "performance_summary_index.json"
    index_obj: Dict[str, Any] = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "summaries": {}
    }
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                index_obj.update(loaded)
                if not isinstance(index_obj.get("summaries"), dict):
                    index_obj["summaries"] = {}
        except Exception:
            pass

    index_obj["updated_at"] = datetime.now().isoformat(timespec="seconds")
    index_obj["summaries"][name] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "stage": payload.get("stage", "stage2"),
        "module": payload.get("module", "unknown"),
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_obj, f, ensure_ascii=False, indent=2)

    print(f"✅ Stage2摘要已写入: {json_path}")
    print(f"✅ Stage2摘要已写入: {csv_path}")
    print(f"✅ Stage2摘要索引已更新: {index_path}")
    return {"json": str(json_path), "csv": str(csv_path), "index": str(index_path)}


def build_figure_batch_payload(results: List[Dict[str, Any]], key_state: Dict[str, bool]) -> Dict[str, Any]:
    total = len(results)
    success = sum(1 for r in results if r.get("ok"))
    failed = total - success
    return {
        "stage": "stage2",
        "module": "figure_batch",
        "total": int(total),
        "success": int(success),
        "failed": int(failed),
        "key_figure_state": key_state,
        "results": results,
    }
