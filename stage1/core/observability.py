from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from stage1.core.io_utils import _run_context_to_dict


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


def _normalize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, (dict, list, tuple)):
            row[k] = json.dumps(v, ensure_ascii=False)
        else:
            row[k] = v
    return row


def _update_summary_index(outdir: Path, name: str, json_path: Path, csv_path: Path, metrics: Dict[str, Any]) -> Path:
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
        "stage": metrics.get("stage", "stage1"),
        "module": metrics.get("module", "unknown"),
    }

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_obj, f, ensure_ascii=False, indent=2)
    return index_path


def write_performance_summary(outdir: Any, name: str, metrics: Dict[str, Any], run_context: Any = None) -> Dict[str, str]:
    outdir = Path(outdir) if not isinstance(outdir, Path) else outdir
    outdir.mkdir(parents=True, exist_ok=True)

    payload = dict(metrics)
    rc = _run_context_to_dict(run_context)
    if rc is not None:
        manifest_run_id = str(rc.get("manifest_run_id") or "")
        payload["manifest_run_id"] = manifest_run_id
        payload["run_id"] = manifest_run_id
        payload["action"] = str(rc.get("action") or "")
        payload["split_protocol"] = str(rc.get("split_protocol") or "")
        payload["data_fingerprint"] = rc.get("data_fingerprint") if isinstance(rc.get("data_fingerprint"), dict) else {}

    json_path = outdir / f"{name}.json"
    csv_path = outdir / f"{name}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    row = _normalize_metrics(payload)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    index_path = _update_summary_index(outdir, name, json_path, csv_path, payload)

    print(f"✅ 性能摘要已写入: {json_path}")
    print(f"✅ 性能摘要已写入: {csv_path}")
    print(f"✅ 摘要索引已更新: {index_path}")
    return {"json": str(json_path), "csv": str(csv_path), "index": str(index_path)}


def _read_json_best_effort(path: Path) -> Dict[str, Any]:
    try:
        if path.exists() and path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                return obj
    except Exception:
        pass
    return {}


def export_efficiency_decomposition(
    outdir: Any,
    *,
    config_params: Optional[Dict[str, Any]] = None,
    search_timing: Optional[Dict[str, Any]] = None,
    run_context: Any = None,
) -> Dict[str, str]:
    """统一导出效率分解摘要（JSON+CSV）。"""
    outdir_path = Path(outdir) if not isinstance(outdir, Path) else outdir
    outdir_path.mkdir(parents=True, exist_ok=True)

    cfg = config_params if isinstance(config_params, dict) else {}
    timing = search_timing if isinstance(search_timing, dict) else {}

    metric_records = cfg.get("_performance_metrics")
    phase_elapsed: Dict[str, float] = {}
    if isinstance(metric_records, list):
        for rec in metric_records:
            if not isinstance(rec, dict):
                continue
            name = str(rec.get("name") or "").strip()
            if not name:
                continue
            try:
                elapsed = float(rec.get("elapsed_s", 0.0))
            except Exception:
                elapsed = 0.0
            phase_elapsed[name] = phase_elapsed.get(name, 0.0) + max(0.0, elapsed)

    speed_results = _read_json_best_effort(outdir_path / "speed_results.json")
    tmm_summary = _read_json_best_effort(outdir_path / "performance_summary_stage1_tmm.json")

    payload: Dict[str, Any] = {
        "stage": "stage1",
        "module": "efficiency_decomposition",
        "phase_elapsed_s": phase_elapsed,
        "phase_total_elapsed_s": float(sum(phase_elapsed.values())),
        "train_search_elapsed_s": float(timing.get("search_elapsed_s", 0.0) or 0.0),
        "train_refit_elapsed_s": float(timing.get("refit_elapsed_s", 0.0) or 0.0),
        "train_mlp_fit_elapsed_s": float(timing.get("mlp_fit_elapsed_s", 0.0) or 0.0),
        "train_total_elapsed_s": float(timing.get("train_total_elapsed_s", 0.0) or 0.0),
        "tmm_total_elapsed_s": float(tmm_summary.get("tmm_elapsed_s", 0.0) or 0.0),
        "tmm_rows_valid": int(tmm_summary.get("tmm_rows_valid", 0) or 0),
        "tmm_cache_hit_rate": float(tmm_summary.get("tmm_cache_hit_rate", 0.0) or 0.0),
        "svr_time_per_spectrum_s": float(speed_results.get("svr_time_per_spectrum_s", 0.0) or 0.0),
        "tmm_time_per_spectrum_s": float(speed_results.get("tmm_time_per_spectrum_s", 0.0) or 0.0),
        "speedup": float(speed_results.get("speedup", 0.0) or 0.0),
        "benchmark_n_spectra": int(speed_results.get("benchmark_n_spectra", 0) or 0),
        "benchmark_elapsed_s": float(speed_results.get("benchmark_elapsed_s", 0.0) or 0.0),
    }

    denom = sum(
        v for v in [
            payload["tmm_total_elapsed_s"],
            payload["train_total_elapsed_s"],
            payload["benchmark_elapsed_s"],
        ] if isinstance(v, (int, float)) and v > 0
    )
    payload["decomp_share_tmm"] = float(payload["tmm_total_elapsed_s"] / denom) if denom > 0 else 0.0
    payload["decomp_share_train"] = float(payload["train_total_elapsed_s"] / denom) if denom > 0 else 0.0
    payload["decomp_share_benchmark"] = float(payload["benchmark_elapsed_s"] / denom) if denom > 0 else 0.0

    return write_performance_summary(
        outdir_path,
        "performance_summary_stage1_efficiency_decomposition",
        payload,
        run_context=run_context,
    )


def append_session_log(outdir: Any, event: str, payload: Optional[Dict[str, Any]] = None) -> Path:
    """Append a lightweight JSONL session log entry under the given outdir."""
    outdir_path = Path(outdir) if not isinstance(outdir, Path) else outdir
    outdir_path.mkdir(parents=True, exist_ok=True)
    log_path = outdir_path / "session_actions.jsonl"

    record: Dict[str, Any] = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": event,
    }
    if payload:
        record["payload"] = _normalize_metrics(payload)

    with open(log_path, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")

    return log_path


def _sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON atomically (temp file + replace) to avoid partially-written manifests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(str(tmp_path), str(path))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def _git_info_best_effort(project_root: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "commit": "",
        "branch": "",
        "is_dirty": None,
    }
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        info["commit"] = commit
        info["branch"] = branch
        info["is_dirty"] = bool(status.strip())
    except Exception:
        pass
    return info


def _collect_files_snapshot(outdir: Path, project_root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not outdir.exists():
        return out
    for p in sorted(outdir.glob("*")):
        if not p.is_file():
            continue
        stat = p.stat()
        item: Dict[str, Any] = {
            "path": _safe_relpath(p, project_root),
            "size_bytes": int(stat.st_size),
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        }
        # Hash top-level artifacts for auditability.
        item["sha256"] = _sha256_of_file(p)
        out.append(item)
    return out


def _collect_dependency_locks(project_root: Path) -> List[Dict[str, Any]]:
    candidates = [
        project_root / "requirements.lock.txt",
        project_root / "requirements.txt",
        project_root / "stage1" / "requirements.txt",
    ]
    out: List[Dict[str, Any]] = []
    for p in candidates:
        if not p.exists() or not p.is_file():
            continue
        st = p.stat()
        out.append({
            "path": _safe_relpath(p, project_root),
            "size_bytes": int(st.st_size),
            "sha256": _sha256_of_file(p),
        })
    return out


def write_run_manifest(
    outdir: Any,
    *,
    run_id: str,
    status: str,
    started_at: str,
    duration_s: Optional[float] = None,
    config_file: Optional[str] = None,
    config_snapshot: Optional[Dict[str, Any]] = None,
    input_files: Optional[List[str]] = None,
    manifest_run_id: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    action: Optional[str] = None,
    action_run_id: Optional[str] = None,
    split_protocol: Optional[str] = None,
    data_fingerprint: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write run-level reproducibility manifest under outdir.

    This function is idempotent and can be called multiple times during one run
    (e.g., status=started, then status=completed/failed).
    """
    outdir_path = Path(outdir) if not isinstance(outdir, Path) else outdir
    outdir_path.mkdir(parents=True, exist_ok=True)
    project_root = outdir_path.resolve().parents[1] if len(outdir_path.resolve().parents) >= 2 else outdir_path.resolve()

    input_meta: List[Dict[str, Any]] = []
    for s in (input_files or []):
        try:
            p = Path(s)
            if not p.is_absolute():
                p = (project_root / p).resolve()
            if not p.exists() or not p.is_file():
                continue
            st = p.stat()
            input_meta.append({
                "path": _safe_relpath(p, project_root),
                "size_bytes": int(st.st_size),
                "sha256": _sha256_of_file(p),
            })
        except Exception:
            continue

    cfg_sha = None
    if config_file:
        try:
            cfg_path = Path(config_file)
            if not cfg_path.is_absolute():
                cfg_path = (project_root / cfg_path).resolve()
            cfg_sha = _sha256_of_file(cfg_path)
            if cfg_path.exists() and cfg_path.is_file():
                st = cfg_path.stat()
                input_meta.insert(0, {
                    "path": _safe_relpath(cfg_path, project_root),
                    "size_bytes": int(st.st_size),
                    "sha256": cfg_sha,
                })
        except Exception:
            pass

    status_norm = str(status or "").strip()
    is_started = status_norm.lower() == "started"
    ended_at_value: Optional[str] = None if is_started else datetime.now().isoformat(timespec="seconds")
    duration_value: Optional[float] = None
    if not is_started:
        duration_value = float(duration_s) if duration_s is not None else None

    ppid: Optional[int]
    try:
        ppid = int(os.getppid())  # type: ignore[attr-defined]
    except Exception:
        ppid = None

    lineage_obj: Dict[str, Any] = {
        "manifest_run_id": str(manifest_run_id) if manifest_run_id else str(run_id),
        "parent_run_id": str(parent_run_id) if parent_run_id else None,
        "action": str(action) if action else None,
        "action_run_id": str(action_run_id) if action_run_id else None,
    }

    try:
        git_obj = _git_info_best_effort(project_root)
    except Exception:
        git_obj = {"commit": "", "branch": "", "is_dirty": None}

    try:
        dep_locks = _collect_dependency_locks(project_root)
    except Exception:
        dep_locks = []

    try:
        outputs_snapshot = _collect_files_snapshot(outdir_path, project_root)
    except Exception:
        outputs_snapshot = []

    if data_fingerprint is None:
        data_fingerprint = {
            "inputs": input_meta,
            "config_sha256": cfg_sha,
        }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "status": status_norm,
        "started_at": started_at,
        "ended_at": ended_at_value,
        "duration_s": duration_value,
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
        "process": {
            "pid": int(os.getpid()),
            "ppid": ppid,
            "python_executable": str(sys.executable),
            "cwd": str(Path.cwd()),
            "argv": list(sys.argv),
        },
        "lineage": lineage_obj,
        "split_protocol": str(split_protocol) if split_protocol else None,
        "data_fingerprint": data_fingerprint,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "cpu_count": int(os.cpu_count() or 0),
        },
        "git": git_obj,
        "dependency_locks": dep_locks,
        "config": {
            "config_file": str(config_file) if config_file else "",
            "config_sha256": cfg_sha,
            "snapshot": config_snapshot or {},
        },
        "inputs": input_meta,
        "outputs_top_level": outputs_snapshot,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }

    latest_path = outdir_path / "run_manifest_latest.json"
    run_path = outdir_path / f"run_manifest_{run_id}.json"
    _atomic_write_json(latest_path, manifest)
    _atomic_write_json(run_path, manifest)
    return latest_path


def repair_stale_started_manifest(outdir: Any, stale_after_s: float = 60.0) -> Optional[Path]:
    """Repair a stale 'started' manifest left by an ungraceful termination.

    If run_manifest_latest.json exists, status==started, and its age exceeds stale_after_s,
    rewrite it atomically as status='repaired_stale_started' and add a 'repair' record.
    """
    outdir_path = Path(outdir) if not isinstance(outdir, Path) else outdir
    latest_path = outdir_path / "run_manifest_latest.json"
    if not latest_path.exists() or not latest_path.is_file():
        return None

    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return None
    if not isinstance(manifest, dict):
        return None

    status = str(manifest.get("status", "") or "").strip().lower()
    if status != "started":
        return None

    now_dt = datetime.now()
    now_iso = now_dt.isoformat(timespec="seconds")

    started_dt = None
    started_at_raw = manifest.get("started_at")
    if isinstance(started_at_raw, str) and started_at_raw.strip():
        try:
            started_dt = datetime.fromisoformat(started_at_raw)
        except Exception:
            started_dt = None

    used_mtime_fallback = False
    if started_dt is None:
        try:
            started_dt = datetime.fromtimestamp(latest_path.stat().st_mtime)
            used_mtime_fallback = True
        except Exception:
            started_dt = None
    if started_dt is None:
        return None

    age_s = (now_dt - started_dt).total_seconds()
    if age_s <= float(stale_after_s):
        return None

    previous = {
        "status": manifest.get("status"),
        "ended_at": manifest.get("ended_at"),
        "duration_s": manifest.get("duration_s"),
    }
    repair_record: Dict[str, Any] = {
        "repaired_at": now_iso,
        "reason": "stale_started_manifest",
        "stale_after_s": float(stale_after_s),
        "age_s": float(age_s),
        "used_file_mtime_fallback": bool(used_mtime_fallback),
        "previous": previous,
    }

    existing = manifest.get("repair")
    if isinstance(existing, list):
        repairs = existing
    elif isinstance(existing, dict):
        repairs = [existing]
    else:
        repairs = []
    repairs.append(repair_record)
    manifest["repair"] = repairs

    manifest["status"] = "repaired_stale_started"
    manifest["ended_at"] = now_iso
    manifest["duration_s"] = None if used_mtime_fallback else float(max(0.0, age_s))
    manifest["updated_at"] = now_iso

    if not isinstance(manifest.get("process"), dict):
        try:
            ppid: Optional[int]
            try:
                ppid = int(os.getppid())  # type: ignore[attr-defined]
            except Exception:
                ppid = None
            manifest["process"] = {
                "pid": int(os.getpid()),
                "ppid": ppid,
                "python_executable": str(sys.executable),
                "cwd": str(Path.cwd()),
                "argv": list(sys.argv),
            }
        except Exception:
            pass

    if not isinstance(manifest.get("lineage"), dict):
        rid = str(manifest.get("run_id", "") or "")
        manifest["lineage"] = {
            "manifest_run_id": rid,
            "parent_run_id": None,
            "action": None,
            "action_run_id": None,
        }
    else:
        lineage = manifest["lineage"]
        lineage.setdefault("manifest_run_id", str(manifest.get("run_id", "") or ""))
        lineage.setdefault("parent_run_id", None)
        lineage.setdefault("action", None)
        lineage.setdefault("action_run_id", None)

    _atomic_write_json(latest_path, manifest)

    run_id = manifest.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        try:
            run_path = outdir_path / f"run_manifest_{run_id}.json"
            if run_path.exists() and run_path.is_file():
                _atomic_write_json(run_path, manifest)
        except Exception:
            pass

    return latest_path
