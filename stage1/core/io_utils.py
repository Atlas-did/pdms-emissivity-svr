from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def convert_numpy_types(obj: Any) -> Any:
    """递归将numpy类型转换为Python原生类型，解决JSON序列化问题。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj


def _run_context_to_dict(run_context: Any) -> Optional[Dict[str, Any]]:
    if run_context is None:
        return None
    if isinstance(run_context, dict):
        return run_context
    # Support stage1.core.run_context.RunContext dataclass-like objects
    try:
        d = {
            "run_id": getattr(run_context, "run_id"),
            "manifest_run_id": getattr(run_context, "manifest_run_id"),
            "action": getattr(run_context, "action"),
            "split_protocol": getattr(run_context, "split_protocol"),
            "data_fingerprint": getattr(run_context, "data_fingerprint"),
        }
        if isinstance(d.get("data_fingerprint"), dict):
            return d
        d["data_fingerprint"] = {}
        return d
    except Exception:
        return None


def _inject_run_context(obj: Any, run_context: Any) -> Any:
    rc = _run_context_to_dict(run_context)
    if rc is None:
        return obj
    if not isinstance(obj, dict):
        return obj

    # Prefer meta when present; else write at top level.
    meta = obj.get("meta")
    target: Dict[str, Any]
    if isinstance(meta, dict):
        target = meta
    else:
        target = obj

    # Strong constraint: force these fields from run_context.
    manifest_run_id = str(rc.get("manifest_run_id") or "")
    target["manifest_run_id"] = manifest_run_id
    # Scheme A: run_id mirrors manifest_run_id for standalone copy/self-describing join key.
    target["run_id"] = manifest_run_id
    target["action"] = str(rc.get("action") or "")
    target["split_protocol"] = str(rc.get("split_protocol") or "")
    target["data_fingerprint"] = rc.get("data_fingerprint") if isinstance(rc.get("data_fingerprint"), dict) else {}
    return obj


def _inject_schema_metadata(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj

    schema_version = str(obj.get("schema_version") or "1.0.0")
    data_version = str(obj.get("data_version") or datetime.now(timezone.utc).strftime("%Y.%m.%d"))

    meta = obj.get("meta")
    target: Dict[str, Any]
    if isinstance(meta, dict):
        target = meta
    else:
        target = obj

    target["schema_version"] = schema_version
    target["data_version"] = data_version
    return obj


def save_json(path: Any, obj: Any, run_context: Any = None) -> None:
    """保存JSON文件，带numpy类型转换；失败时抛异常。

    If run_context is provided, inject/overwrite stable traceability fields to prevent mismatches.
    """
    try:
        path = Path(path) if not isinstance(path, Path) else path
        path.parent.mkdir(parents=True, exist_ok=True)
        converted_obj = convert_numpy_types(obj)
        converted_obj = _inject_run_context(converted_obj, run_context)
        converted_obj = _inject_schema_metadata(converted_obj)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(converted_obj, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"保存JSON失败: {path}, 错误: {e}")
        traceback.print_exc()
        raise RuntimeError(f"保存JSON失败: {path}") from e


def load_json(path: Any) -> Dict:
    """加载JSON文件；失败时抛异常。"""
    try:
        path = Path(path) if not isinstance(path, Path) else path
        if not path.exists():
            raise FileNotFoundError(f"JSON文件不存在: {path}")
        # Use utf-8-sig to transparently handle UTF-8 BOM when present.
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON解析失败: {path}, 错误: {e}") from e
    except Exception as e:
        print(f"加载JSON失败: {path}, 错误: {e}")
        traceback.print_exc()
        raise RuntimeError(f"加载JSON失败: {path}: {e}") from e
