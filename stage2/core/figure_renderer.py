from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .figure_registry import FigureTask
from .observability import (
    ErrorContext,
    build_figure_batch_payload,
    log_exception_with_context,
    write_lightweight_summary,
)


def run_figure_task(viz: Any, task: FigureTask) -> Dict[str, Any]:
    method = getattr(viz, task.method_name)
    output_path = Path(viz.figdir) / task.output_name
    print(f"▶ {task.figure_id}: source={task.data_source}, output={output_path}")
    try:
        method()
        exists = output_path.exists()
        if exists:
            print(f"✅ {task.figure_id} 完成")
        else:
            print(f"⚠️ {task.figure_id} 未检测到输出文件: {output_path}")
        return {
            "figure_id": task.figure_id,
            "ok": bool(exists),
            "output": str(output_path),
            "error": None,
        }
    except Exception as e:
        log_exception_with_context(
            e,
            ErrorContext(
                stage="stage2.render.figure_task",
                region="global",
                file=str(output_path),
                sample_range=task.figure_id,
            ),
            print_traceback=True,
        )
        return {
            "figure_id": task.figure_id,
            "ok": False,
            "output": str(output_path),
            "error": f"{type(e).__name__}: {e}",
        }


def run_figure_batch(viz: Any, tasks: Iterable[FigureTask]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for task in tasks:
        results.append(run_figure_task(viz, task))
    return results


def summarize_figure_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    success = sum(1 for r in results if r.get("ok"))
    failed = total - success
    failed_items = [r for r in results if not r.get("ok")]

    print("\n📊 图表批处理总结")
    print(f"   - Total: {total}")
    print(f"   - Success: {success}")
    print(f"   - Failed: {failed}")
    if failed_items:
        print("   - Failures:")
        for item in failed_items:
            print(f"     * {item['figure_id']} -> {item['error'] or 'output missing'}")

    return {
        "total": total,
        "success": success,
        "failed": failed,
        "failed_items": failed_items,
    }


def check_key_figure_outputs(results: List[Dict[str, Any]], key_ids: List[str]) -> Dict[str, bool]:
    state = {fid: False for fid in key_ids}
    by_id = {r["figure_id"]: bool(r.get("ok")) for r in results}
    for fid in key_ids:
        state[fid] = by_id.get(fid, False)
    print("\n🎯 关键图输出检查")
    for fid, ok in state.items():
        print(f"   - {fid}: {'OK' if ok else 'MISSING'}")
    return state


def persist_figure_batch_summary(figdir: Any, results: List[Dict[str, Any]], key_state: Dict[str, bool]) -> Dict[str, str]:
    payload = build_figure_batch_payload(results, key_state)
    return write_lightweight_summary(figdir, "performance_summary_stage2_figures", payload)
