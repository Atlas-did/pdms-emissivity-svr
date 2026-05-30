from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentTracker:
    """实验追踪器：优先MLflow，缺失时回退JSONL。"""

    def __init__(self, enabled: bool, experiment_name: str, tracking_uri: str, outdir: Path):
        self.enabled = bool(enabled)
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.outdir = Path(outdir)
        self._mlflow = None
        self._fallback_file = self.outdir / "experiment_tracking.jsonl"

        if not self.enabled:
            return

        try:
            import mlflow  # type: ignore
            self._mlflow = mlflow
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
        except Exception:
            self._mlflow = None

    def _append_fallback(self, kind: str, payload: Dict[str, Any]) -> None:
        self.outdir.mkdir(parents=True, exist_ok=True)
        line = {
            "ts": time.time(),
            "kind": kind,
            "experiment": self.experiment_name,
            "payload": payload,
        }
        with open(self._fallback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self._mlflow is not None:
            with self._mlflow.start_run(nested=True):
                self._mlflow.log_params(params)
        else:
            self._append_fallback("params", params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        if self._mlflow is not None:
            with self._mlflow.start_run(nested=True):
                self._mlflow.log_metrics(metrics, step=step)
        else:
            payload: Dict[str, Any] = {"metrics": metrics}
            if step is not None:
                payload["step"] = int(step)
            self._append_fallback("metrics", payload)
