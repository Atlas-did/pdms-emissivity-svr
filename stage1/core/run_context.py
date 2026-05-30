from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class RunContext:
    """A small, stable container for traceability fields.

    Goal: every major artifact can embed the same minimal keys:
    - run_id
    - manifest_run_id
    - action
    - split_protocol
    - data_fingerprint
    """

    run_id: str
    manifest_run_id: str
    action: str
    split_protocol: str
    data_fingerprint: Dict[str, Any]


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


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def data_fingerprint_from_paths(paths: Iterable[str | Path], *, project_root: Path) -> Dict[str, Any]:
    items = []
    for p in paths:
        try:
            path = Path(p)
            if not path.is_absolute():
                path = (project_root / path).resolve()
            if not path.exists() or not path.is_file():
                continue
            st = path.stat()
            items.append(
                {
                    "path": _safe_relpath(path, project_root),
                    "size_bytes": int(st.st_size),
                    "sha256": _sha256_of_file(path),
                }
            )
        except Exception:
            continue
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": items,
    }


def split_protocol_from_config_snapshot(params: Dict[str, Any]) -> str:
    """Create a stable, human-readable split protocol string.

    Keep it conservative: only include fields that are already in config.json.
    """
    test_size = params.get("test_size")
    seed = params.get("random_seed")
    group_split = bool(params.get("group_split_by_thickness", False))
    allow_random = bool(params.get("allow_random_split", False))
    return (
        f"group_split_by_thickness={str(group_split).lower()}"
        f":allow_random_split={str(allow_random).lower()}"
        f":test_size={test_size}"
        f":random_seed={seed}"
    )


def manifest_run_id_from_manifest_latest(manifest_path: Path) -> Optional[str]:
    try:
        if not manifest_path.exists() or not manifest_path.is_file():
            return None
        with manifest_path.open("r", encoding="utf-8") as f:
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


def build_run_context(
    *,
    run_id: str,
    action: str,
    config_snapshot: Dict[str, Any],
    input_paths: Iterable[str | Path],
    project_root: Path,
    manifest_latest_path: Optional[str | Path] = None,
    split_protocol_override: Optional[str] = None,
) -> RunContext:
    manifest_run_id = None
    if manifest_latest_path is not None:
        manifest_run_id = manifest_run_id_from_manifest_latest(Path(manifest_latest_path))
    manifest_run_id = manifest_run_id or run_id

    split_protocol = (
        str(split_protocol_override).strip()
        if isinstance(split_protocol_override, str) and split_protocol_override.strip()
        else split_protocol_from_config_snapshot(config_snapshot)
    )
    data_fingerprint = data_fingerprint_from_paths(input_paths, project_root=project_root)

    return RunContext(
        run_id=run_id,
        manifest_run_id=manifest_run_id,
        action=action,
        split_protocol=split_protocol,
        data_fingerprint=data_fingerprint,
    )
