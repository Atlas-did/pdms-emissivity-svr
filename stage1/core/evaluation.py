from __future__ import annotations

"""Evaluation compatibility layer for Phase 2.

评估/测速/外推能力当前由 `stage1.core.training` 承载。
本模块提供稳定导出，便于后续独立拆分实现而不影响调用方。
"""

from stage1.core.training import (
    SVRExtrapolationEvaluator,
    SVRModelEvaluator,
    SVRSpeedBenchmarker,
)

__all__ = [
    "SVRModelEvaluator",
    "SVRSpeedBenchmarker",
    "SVRExtrapolationEvaluator",
]
