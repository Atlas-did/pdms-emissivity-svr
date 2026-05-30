from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def apply_data_bundle(loader: Any, bundle: Dict[str, Any]) -> None:
    """将bundle字段批量写回DataLoader实例。"""
    loader.config = bundle["config"]
    loader.material_info = bundle["material_info"]
    loader.material_data = bundle["material_data"]
    loader.tmm_data = bundle["tmm_data"]
    loader.processed_data = bundle["processed_data"]
    loader.model_router = bundle["model_router"]
    loader.test_results = bundle["test_results"]
    loader.speed_results = bundle["speed_results"]
    loader.model_info = bundle["model_info"]
    loader.region_summary = bundle["region_summary"]
    loader.scaler = bundle["scaler"]


def validate_router_state(model_router: Any) -> None:
    """校验router必要状态，失败时降级为警告（不再阻断加载流程）。

    当 model_router 为空时仅打印警告并返回，
    交由 Visualization._ensure_minimal_model_router() 在需要时动态构建。
    """
    if not model_router or not getattr(model_router, "region_models", None):
        print("[WARN]  未加载region模型（将在figure方法中构建fallback模型）")
        return
    if not hasattr(model_router, "region_configs"):
        raise AttributeError("model_router缺少region_configs属性")


def print_load_overview(datadir: Any, tmm_rows: int, processed_rows: int, material_data: Dict[str, Any]) -> None:
    """打印核心加载概况。"""
    datadir = Path(datadir) if not isinstance(datadir, Path) else datadir
    print(f"[OK] 加载配置: {datadir / 'config.json'}")
    print(f"[OK] 加载TMM数据: {tmm_rows} 行")

    if processed_rows == tmm_rows:
        print("[WARN]  使用TMM数据作为处理数据")
    else:
        print(f"[OK] 加载处理数据: {processed_rows} 行")

    if material_data:
        print(
            f"[OK] 加载材料数据: PDMS {material_data.get('pdms_count', 0)}点, "
            f"SiO2 {material_data.get('sio2_count', 0)}点"
        )


def print_optional_status(test_results: Dict[str, Any], speed_results: Dict[str, Any], model_info: Dict[str, Any],
                          region_summary: Dict[str, Any], scaler: Any) -> None:
    """打印可选资源加载状态。"""
    if test_results:
        print("[OK] 加载测试结果")
    else:
        print("[WARN]  未找到测试结果文件")

    if speed_results:
        print("[OK] 加载速度结果")
    if model_info:
        print("[OK] 加载模型信息")
    if region_summary:
        print(f"[OK] 加载region训练总结: {len(region_summary.get('results', []))} 个region结果")
    if scaler is not None:
        print("[OK] 加载全局scaler")
