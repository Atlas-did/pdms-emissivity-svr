# 使用流程（精简交付版）

> 目标：让接收方在最少步骤下完成“运行 Stage1/Stage2 + 生成图像”。

---

## 1) 环境准备

1. 使用 Python 3.10+。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 确认材料文件存在：

- stage1/pdms_nk.xlsx
- stage1/sio2_nk.xlsx

---

## 2) 快速运行

### 2.1 运行 Stage1（生成/更新模型与数据）

```bash
python stage1/main_pdms_svr_bandscan_full.py
```

主要输出：

- stage1/simulation_data/tmm_emissivity_data.csv
- stage1/simulation_data/svr_emissivity_model.pkl
- stage1/simulation_data/scaler.pkl
- stage1/simulation_data/test_results.json

### 2.2 运行 Stage2（生成图像）

```bash
python stage2/分区图片1.0.py
```

主要输出：

- stage2/figures/figure*.png

### 2.3 运行 P1-1 专用入口（推荐）

```bash
python stage2/run_p1_1.py
```

预期输出：

- stage2/figures/figure15_direct_validation.png
- stage2/figures/direct_validation_summary.json
- stage2/figures/direct_validation_summary.csv
- stage2/figures/paper_citation_table_p1_1.md（运行后回填 run_id 与阈值结论）

---

## 3) 常用配置（建议显式设置）

配置文件：stage1/simulation_data/config.json

建议重点检查：

- substrate_n_real / substrate_n_imag
- assume_opaque_substrate
- lambda_min / lambda_max / lambda_step
- min_thickness_nm / max_thickness / thickness_step
- strict_energy_check / energy_tol

---

## 4) 常见问题

1. 模型维度不匹配（如 3 vs 13）
   - 原因：训练与推理特征空间不一致。
   - 处理：重新运行 Stage1，确保 Stage2 使用同一批模型工件。

2. Figure 6 缺少文献点
   - 检查：stage2/simulation_data/mandal2018_experimental.csv 是否存在。

3. 发射率异常越界
   - 检查：是否开启 strict_energy_check；核对基底参数与波长范围。

---

## 5) 新窗口续改顺序（P1-1 → P1-2 → P1-3）

1. **P1-1（先做）**
   - 执行：`python stage2/run_p1_1.py`
   - 核对输出：
     - `figure15_direct_validation.png`
     - `direct_validation_summary.json`
     - `direct_validation_summary.csv`
2. **P1-2（其次）**
   - 执行：`python stage2/run_p1_2_p1_3.py`
   - 目标输出（约定文件名）：
     - `physical_constraint_summary.json`
     - `physical_constraint_summary.csv`
     - `figure17_physical_constraints_effect.png`
3. **P1-3（最后）**
   - 执行：`python stage2/run_p1_2_p1_3.py`（与 P1-2 同入口）
   - 目标输出（约定文件名）：
     - `ablation_summary.json`
     - `ablation_summary.csv`
     - `figure5_ablation_study.png`

