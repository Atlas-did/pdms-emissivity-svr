# 便携版开箱即用（只拷贝 stage1 + stage2）

目标：你只需要把 **stage1/** 和 **stage2/** 两个文件夹复制到同一个新目录里，对方就能按命令从零跑通（生成数据→训练→评估→出图）。

## 目录结构要求
复制后目录应长这样（stage1 与 stage2 必须是同级目录）：

```
<YOUR_ROOT>/
  stage1/
  stage2/
```

> 不需要复制本仓库的其他目录（tools/、tests/、reports/ 等）。脚本会自动创建 figures/、reports/ 目录。

## 0) 前置条件
- Windows + Python（建议 3.10/3.11）
- 能联网安装 pip 依赖（或已准备好离线 wheel）

## 1) 一键创建虚拟环境 + 安装依赖
在 <YOUR_ROOT> 目录打开 PowerShell，运行：

```powershell
.\stage1\portable_setup.ps1
```

如果你机器上 python 不是默认命令，可指定解释器：

```powershell
.\stage1\portable_setup.ps1 -PythonExe "C:\\Path\\To\\python.exe"
```

## 2) 快速验证（不需要已有数据）
只跑一个最轻量动作（会产出审计 JSON）：

```powershell
.\stage1\portable_run.ps1 -Mode inspect
```

成功后你会看到：
- stage1/simulation_data/run_manifest_latest.json
- stage1/simulation_data/inspect_regions_report.json

## 3) 从零跑通（生成数据→训练→评估→出图）

```powershell
.\stage1\portable_run.ps1 -Mode full
```

主要输出：
- Stage1: stage1/simulation_data/（csv/json/pkl 等）
- Stage2: figures/（figure*.png 等）

## 4) 常见问题
### Q1: 为什么 Stage2 没图/只有缺失提示？
Stage2 需要 Stage1 的产物（例如 test_results.json、模型 pkl、region_models/ 等）。请先运行 `-Mode full`，或者至少跑 `simulate/train/evaluate`。

### Q2: Excel 输入文件要带吗？
要。stage1/ 里自带：
- stage1/pdms_nk.xlsx
- stage1/sio2_nk.xlsx

### Q3: 我只想跑 Stage2 出图，不想重训？
把已有的 stage1/simulation_data/（包含模型和 test_results）一并复制过去，然后直接运行：

```powershell
.\stage1\portable_run.ps1 -Mode inspect
.\stage1\portable_run.ps1 -Mode evaluate
.\venv\Scripts\python.exe .\stage2\分区图片1.0.py
```

（如果没有 .venv，用 portable_setup 先建环境。）
