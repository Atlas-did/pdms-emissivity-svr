from __future__ import annotations

import functools
import time
import tracemalloc
from typing import Any, Callable, Dict


def _ensure_tracemalloc_started() -> None:
    try:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
    except Exception:
        pass


def _get_memory_mb_best_effort() -> float:
    """Best-effort memory snapshot (MB).

    Uses tracemalloc current tracked memory as a portable fallback.
    """
    try:
        _ensure_tracemalloc_started()
        current, _ = tracemalloc.get_traced_memory()
        return float(current) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def monitor_performance(metric_name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for elapsed time / memory delta tracking.

    If `self.config.params` exists, metrics are appended into
    `self.config.params['_performance_metrics']`.
    """

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        name = metric_name or func.__name__

        @functools.wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            mem0 = _get_memory_mb_best_effort()
            t0 = time.perf_counter()
            ok = True
            try:
                return func(*args, **kwargs)
            except Exception:
                ok = False
                raise
            finally:
                elapsed = time.perf_counter() - t0
                mem1 = _get_memory_mb_best_effort()
                rec: Dict[str, Any] = {
                    "name": str(name),
                    "elapsed_s": float(elapsed),
                    "memory_delta_mb": float(mem1 - mem0),
                    "ok": bool(ok),
                }

                # Best-effort persistence to config params.
                try:
                    self_obj = args[0] if args else None
                    cfg = getattr(self_obj, "config", None)
                    params = getattr(cfg, "params", None)
                    if isinstance(params, dict):
                        bucket = params.get("_performance_metrics")
                        if not isinstance(bucket, list):
                            bucket = []
                            params["_performance_metrics"] = bucket
                        bucket.append(rec)
                        # keep bounded
                        if len(bucket) > 200:
                            del bucket[:-200]
                except Exception:
                    pass

                print(
                    f"[PERF] {name}: {elapsed:.3f}s, "
                    f"Δmem={rec['memory_delta_mb']:.3f}MB, ok={ok}"
                )

        return _wrapper

    return _decorator
