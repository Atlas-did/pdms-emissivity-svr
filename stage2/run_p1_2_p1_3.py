#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1-2/P1-3 专用执行入口：生成物理约束与消融摘要。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


STAGE2_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STAGE2_DIR.parent
MAIN_FILE = STAGE2_DIR / "分区图片1.0.py"
FIG_DIR = STAGE2_DIR / "figures"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_stage2_module():
    spec = importlib.util.spec_from_file_location("stage2_main_cn", str(MAIN_FILE))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载主脚本: {MAIN_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    try:
        mod = _load_stage2_module()

        loader = mod.DataLoader(mod.DATADIR)
        loader.load_all()

        viz = mod.Visualization(loader, FIG_DIR)
        viz.figure17_physical_constraints_effect()
        viz.figure5_ablation_study()

        print("\n✅ P1-2/P1-3 已完成（share_clean_bundle/stage2 内闭环）")
        print(f"📁 输出目录: {FIG_DIR}")
        print(f"  - {FIG_DIR / 'figure17_physical_constraints_effect.png'}")
        print(f"  - {FIG_DIR / 'physical_constraint_summary.json'}")
        print(f"  - {FIG_DIR / 'physical_constraint_summary.csv'}")
        print(f"  - {FIG_DIR / 'figure5_ablation_study.png'}")
        print(f"  - {FIG_DIR / 'ablation_summary.json'}")
        print(f"  - {FIG_DIR / 'ablation_summary.csv'}")
        print(f"  - {FIG_DIR / 'ablation_table.md'}")
        return 0
    except Exception as exc:
        print(f"❌ P1-2/P1-3 执行失败: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
