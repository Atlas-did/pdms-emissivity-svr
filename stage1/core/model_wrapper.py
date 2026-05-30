from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib

from stage1.core.io_utils import save_json


@dataclass
class ModelWrapper:
    """统一模型序列化包装器（模型 + scaler + 元数据）。"""

    model: Any
    scaler: Any
    y_scaler: Any = None
    metadata: Optional[Dict[str, Any]] = None

    def to_bundle(self) -> Dict[str, Any]:
        md = dict(self.metadata or {})
        md.setdefault("bundle_format", "model-wrapper.v1")
        md.setdefault("saved_at_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        return {
            "model": self.model,
            "scaler": self.scaler,
            "y_scaler": self.y_scaler,
            "metadata": md,
        }

    def save(self, bundle_path: Any, meta_json_path: Any, run_context: Any = None) -> Dict[str, str]:
        bundle_path = Path(bundle_path)
        meta_json_path = Path(meta_json_path)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        meta_json_path.parent.mkdir(parents=True, exist_ok=True)

        obj = self.to_bundle()
        joblib.dump(obj, bundle_path)
        save_json(meta_json_path, obj.get("metadata", {}), run_context=run_context)
        return {"bundle": str(bundle_path), "metadata": str(meta_json_path)}

    @staticmethod
    def load(bundle_path: Any) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        bundle_path = Path(bundle_path)
        obj = joblib.load(bundle_path)
        if not isinstance(obj, dict):
            raise TypeError(f"Invalid model bundle payload type: {type(obj).__name__}")
        model = obj.get("model")
        scaler = obj.get("scaler")
        y_scaler = obj.get("y_scaler")
        metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
        return model, scaler, y_scaler, metadata
