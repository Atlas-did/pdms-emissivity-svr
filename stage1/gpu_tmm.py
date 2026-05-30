from __future__ import annotations

"""Compatibility shim for historical `gpu_tmm` imports.

迁移说明：
- 真实实现已迁移到 `stage1.tmm_cpu_threaded`。
- 本模块仅保留兼容转发，避免旧脚本导入失败。
- 计划在 1~2 个版本迁移期后，评估是否移除该兼容壳。
"""

import warnings
from typing import Any, Dict, Iterable, List, Tuple

from stage1.tmm_cpu_threaded import (
    batch_simulate_tmm as _batch_simulate_tmm_cpu_threaded,
    batch_simulate_tmm_gpu as _batch_simulate_tmm_gpu_stub,
)


def _warn_deprecated() -> None:
    warnings.warn(
        "stage1.gpu_tmm is a compatibility shim (CPU fallback only). "
        "Please migrate to stage1.tmm_cpu_threaded within 1-2 versions.",
        DeprecationWarning,
        stacklevel=2,
    )


def batch_simulate_tmm(
    simulator: Any,
    tasks: Iterable[Tuple[float, float, float]],
    backend: str = "auto"
) -> List[Dict[str, float]]:
    """兼容转发：调用 `stage1.tmm_cpu_threaded.batch_simulate_tmm`。"""
    _warn_deprecated()
    print("[gpu_tmm] compatibility shim active: CPU fallback only, forwarding to stage1.tmm_cpu_threaded")
    return _batch_simulate_tmm_cpu_threaded(simulator, tasks, backend=backend)


def batch_simulate_tmm_gpu(*args, **kwargs):
    """兼容转发：未来GPU接口占位（当前未实现）。"""
    _warn_deprecated()
    return _batch_simulate_tmm_gpu_stub(*args, **kwargs)


__all__ = ["batch_simulate_tmm", "batch_simulate_tmm_gpu"]
