# GPU Fallback Migration (Phase 5)

## 变更摘要
- 真实批量TMM实现已迁移到 [stage1/tmm_cpu_threaded.py](stage1/tmm_cpu_threaded.py)。
- [stage1/gpu_tmm.py](stage1/gpu_tmm.py) 现为兼容转发壳（CPU fallback only）。
- 预留未来接口：`batch_simulate_tmm_gpu()`（当前抛 `NotImplementedError`）。

## 迁移期策略（1~2个版本）
- 在迁移期内：
  - 旧导入 `from stage1.gpu_tmm import batch_simulate_tmm` 仍可用。
  - 会触发 `DeprecationWarning` 并打印兼容转发日志。
- 迁移期结束后：
  - 评估项目内外部脚本使用情况。
  - 若无风险，可移除 `stage1.gpu_tmm` 兼容壳。

## 推荐导入
```python
from stage1.tmm_cpu_threaded import batch_simulate_tmm
```

## 现阶段语义
- `use_gpu_tmm=True` 仅表示“启用GPU后端探测流程”，当前仍回退CPU并行。
- 尚未提供真GPU内核实现。
