# mandal2018_experimental.csv 说明

当前仓库已放入一个可运行的示例 CSV（用于让 Figure 6 直接出图）。

请在正式分析/论文前替换为真实文献数据，字段保持：

- `wavelength_um`
- `emissivity`
- `error`（可选；缺省会用默认误差）

---

## 推荐数据来源链接（你可自行下载）

1. Google Scholar 关键词检索（建议）  
	https://scholar.google.com/scholar?q=Mandal+2018+radiative+cooling+emissivity

2. Crossref 关键词检索  
	https://search.crossref.org/?q=Mandal%202018%20radiative%20cooling

3. Web of Science 检索（机构权限）  
	https://www.webofscience.com/

4. 若论文只给图不给表，推荐用 WebPlotDigitizer 提取曲线点  
	https://automeris.io/WebPlotDigitizer/

> 说明：本仓库不会自动爬取受版权保护的整篇论文 PDF 内容；
> 但你可以把合法获取到的 CSV/表格数据放入本目录，然后直接出图。

---

## 必需格式（最小）

CSV UTF-8，至少三列（前两列必须）：

```csv
wavelength_um,emissivity,error
8.0,0.93,0.02
9.0,0.95,0.02
...
```

- `wavelength_um`：单位必须是 μm（若原始是 nm，请先除以 1000）
- `emissivity`：建议范围 [0,1]
- `error`：可选；若缺失，Figure 6 会用默认误差

可选附加列（不影响现有读取）：

- `source`：文献来源（例如 DOI 或论文名）
- `condition`：实验条件（厚度、角度、温度等）

---

## 一键规范化脚本（已提供）

可使用：

- `tools/prepare_literature_csv.py`

把下载/数字化后的 CSV 统一转换为本项目所需格式，输出到：

- `stage2/simulation_data/mandal2018_experimental.csv`
