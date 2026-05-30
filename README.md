# PDMS 发射率项目（精简交付版）

本目录已精简为“可复现运行 + 出图”最小交付结构，重点保留：

- Stage1：数据生成与训练
- Stage2：分区路由与图像生成
- 必要依赖与少量说明文档

---

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 运行 Stage1（可选，若你只复用现成结果可跳过）

```bash
python stage1/main_pdms_svr_bandscan_full.py
```

3. 运行 Stage2 生成图像

```bash
python stage2/分区图片1.0.py
```

4. 运行 P1-1 专用入口（交接后首选）

```bash
python stage2/run_p1_1.py
```

---

## 关键目录

- [stage1](stage1)：TMM + SVR 主流程
- [stage2](stage2)：分区与绘图主流程

---

## 文档（仅保留 4 份）

- [README.md](README.md)：项目总览（本文件）
- [使用流程.md](使用流程.md)：执行步骤与常见问题
- [技术与物理说明.md](技术与物理说明.md)：数学/物理核心机制与配置口径
- [论文与版本说明.md](论文与版本说明.md)：论文大纲建议 + 版本更新说明

论文专项整改计划：

- [整改总计划.md](整改总计划.md)

---

## 交接入口导航（新窗口先看这里）

1. 总览与矩阵： [整改总计划.md](整改总计划.md)
2. 30轮口径： [论文与版本说明.md](论文与版本说明.md)
3. 操作顺序： [使用流程.md](使用流程.md)
4. P1-1 执行入口： [stage2/run_p1_1.py](stage2/run_p1_1.py)
5. P1-2/P1-3 执行入口： [stage2/run_p1_2_p1_3.py](stage2/run_p1_2_p1_3.py)
6. P1 系列输出目录： [stage2/figures](stage2/figures)
7. 论文可引用表： [stage2/figures/paper_citation_table_p1_1.md](stage2/figures/paper_citation_table_p1_1.md)
