from __future__ import annotations

from typing import Any

import pandas as pd


def clear_loader_state(loader: Any) -> None:
    """加载失败时清理DataLoader关键状态，避免脏状态残留。"""
    loader.config = {}
    loader.material_info = {}
    loader.material_data = {}
    loader.tmm_data = pd.DataFrame()
    loader.processed_data = pd.DataFrame()
    loader.model_router = None
    loader.test_results = {}
    loader.speed_results = {}
    loader.model_info = {}
    loader.region_summary = {}
    loader.scaler = None


def handle_load_failure(loader: Any, exc: Exception, traceback_module: Any) -> None:
    """统一处理加载异常：打印并回滚。"""
    print(f"[FAIL] 数据加载失败: {exc}")
    traceback_module.print_exc()
    clear_loader_state(loader)
