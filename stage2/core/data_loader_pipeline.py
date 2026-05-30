from __future__ import annotations

from pathlib import Path
from typing import Any

from .data_loader_io import load_data_bundle
from .data_loader_reporting import (
    apply_data_bundle,
    validate_router_state,
    print_load_overview,
    print_optional_status,
)


def execute_data_loading_pipeline(loader: Any, *, datadir: Any, region_models_dir: Any, load_json_fn, router_cls, root: Any) -> None:
    """DataLoader成功路径统一入口：加载->赋值->校验->日志。"""
    datadir_path = Path(datadir) if not isinstance(datadir, Path) else datadir
    root_path = Path(root) if not isinstance(root, Path) else root

    bundle = load_data_bundle(
        datadir=datadir_path,
        region_models_dir=region_models_dir,
        load_json_fn=load_json_fn,
        router_cls=router_cls,
        root=root_path,
    )

    apply_data_bundle(loader, bundle)
    validate_router_state(loader.model_router)

    print_load_overview(
        datadir=datadir_path,
        tmm_rows=len(loader.tmm_data),
        processed_rows=len(loader.processed_data),
        material_data=loader.material_data,
    )
    print_optional_status(
        test_results=loader.test_results,
        speed_results=loader.speed_results,
        model_info=loader.model_info,
        region_summary=loader.region_summary,
        scaler=loader.scaler,
    )
