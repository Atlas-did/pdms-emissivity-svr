from __future__ import annotations

"""CPU-threaded TMM batch backend.

说明：
- 该模块是 Stage1 批量TMM计算的真实实现位置。
- 历史上的 `stage1.gpu_tmm` 名称保留为兼容转发壳。
- 当前版本仅提供 CPU 并行能力；未来如需真GPU实现，请在
  `batch_simulate_tmm_gpu()` 中引入实际GPU核函数。
"""

from typing import Any, Dict, Iterable, List, Tuple

from joblib import Parallel, delayed


def batch_simulate_tmm(
    simulator: Any,
    tasks: Iterable[Tuple[float, float, float]],
    backend: str = "auto"
) -> List[Dict[str, float]]:
    """批量TMM接口（CPU线程并行实现）。

    当前版本：
    - backend='cpu'：CPU并行（joblib）
    - backend='auto'：自动探测torch；若未实现GPU核函数则显式回退CPU并行
    - backend='torch'：当前仅做可用性探测并显式回退CPU并行（不伪装GPU加速）
    """
    tasks = list(tasks)
    if not tasks:
        return []

    if backend not in {"auto", "cpu", "torch"}:
        raise ValueError(f"unsupported backend: {backend}")

    torch_available = False
    cuda_available = False
    if backend in {"auto", "torch"}:
        try:
            import torch  # type: ignore[import-not-found]
            torch_available = True
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            torch_available = False
            cuda_available = False

    if backend == "auto":
        if torch_available:
            state = "CUDA可用" if cuda_available else "CUDA不可用"
            print(f"[tmm_cpu_threaded] backend=auto检测到torch({state})，GPU核函数未实现，回退CPU并行。")
        backend = "cpu"
    elif backend == "torch":
        state = "CUDA可用" if cuda_available else "CUDA不可用"
        if torch_available:
            print(f"[tmm_cpu_threaded] backend=torch请求已接收，torch已安装({state})；当前版本回退CPU并行。")
        else:
            print("[tmm_cpu_threaded] backend=torch请求已接收，但未检测到torch，回退CPU并行。")
        backend = "cpu"

    n_jobs = getattr(simulator, "n_jobs", -1)
    if n_jobs is None:
        n_jobs = -1
    joblib_backend = "threading"
    try:
        joblib_backend = str(getattr(simulator, "cfg", {}).params.get("joblib_backend", "threading"))
    except Exception:
        pass

    return Parallel(n_jobs=int(n_jobs), backend=joblib_backend)(
        delayed(simulator.simulate_point)(lam, pdms, sio2) for lam, pdms, sio2 in tasks
    )


def batch_simulate_tmm_gpu(*args, **kwargs):
    """未来真GPU实现预留接口。

    TODO:
    - 对接真实GPU内核（CUDA/torch/jax等）
    - 保持与 `batch_simulate_tmm()` 的输入输出协议一致
    """
    raise NotImplementedError(
        "batch_simulate_tmm_gpu() is a reserved stub; true GPU kernel is not implemented yet."
    )
