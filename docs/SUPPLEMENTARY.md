# Supplementary Analysis: Full-Scale (82,938 samples) Results

> Generated: 2026-06-23  
> Data: `stage1/simulation_data/fair_baseline_results.json`, `robustness_results.json`, `feature_importance.json`
> See `stage1/simulation_data/test_results.json` for production SVR metrics.

---

## 1. Fair Baseline Comparison

All models evaluated with **thickness-grouped split** (same protocol as paper Table 2). 58,057 train / 24,881 test.

| Model | Features | R2 | RMSE | MAE | Train Time |
|-------|:---:|------:|------:|------:|------:|
| **RF** | 13 | **0.9998** | 0.0034 | 0.0012 | 8s |
| **RF** | 3 (wl,d) | **0.9998** | 0.0036 | 0.0019 | 1.4s |
| **GBR** | 13 | **0.9996** | 0.0045 | 0.0025 | 53s |
| **SVR (RBF)** | 13 | **0.9720** | 0.0390 | 0.0183 | ~5h |

For comparison, **random split** (data-leaked): RF R2 = 0.9999. The leakage inflation is only **0.0001**, meaning RF genuinely achieves ~0.9998 regardless of split protocol.

### Key finding (revised from earlier small-sample conclusion)

- On 3,000 samples: RF honest R2=0.9935, making the 0.9998 look inflated. **This was an artifact of small sample size.**
- On 82,938 samples: RF honest R2=0.9998, same as random split. **RF genuinely achieves this accuracy.**
- **Why SVR is still preferred for the paper:** RF predictions are piecewise-constant (not smooth), which is a fatal flaw for continuous design-space exploration in optical engineering. SVR produces physically smooth emissivity curves.

---

## 2. Feature Importance (Permutation, 5-repeat)

82,938 samples, 13 engineered features. Top-3 dominate.

| Rank | Feature | Importance (mean +/- std) |
|:---:|------|------:|
| 1 | SiO2 thickness (nm) | 0.870 +/- 0.010 |
| 2 | wavelength (um) | 0.856 +/- 0.005 |
| 3 | cos_PDMS * cos_SiO2 | 0.845 +/- 0.007 |
| 4 | cos(delta_SiO2) | 0.836 +/- 0.010 |
| 5 | sin_PDMS * cos_SiO2 | 0.726 +/- 0.011 |
| 6 | PDMS thickness (nm) | 0.658 +/- 0.005 |
| 7 | sin(delta_PDMS) | 0.616 +/- 0.006 |
| 8 | cos_PDMS * sin_SiO2 | 0.584 +/- 0.005 |
| 9 | sin_PDMS * sin_SiO2 | 0.530 +/- 0.007 |
| 10 | sin(delta_SiO2) | 0.480 +/- 0.007 |
| 11 | 1/wavelength | 0.431 +/- 0.006 |
| 12 | d_PDMS / wavelength | 0.403 +/- 0.009 |
| 13 | cos(delta_PDMS) | 0.339 +/- 0.009 |

### Physics interpretation

- **SiO2 thickness is #1** (0.870): the high-index SiO2 layer dominates optical interference. Wavelength is #2 (0.856): normal for thin-film optics. The interaction term cos_PDMS*cos_SiO2 (#3, 0.845) captures the coherent superposition of reflections from both interfaces.
- All 13 features have positive importance -- no wasted dimensions.
- Ranking differs substantially from the 5,000-sample preliminary (where wavelength was #1 at 0.686). The full dataset reveals SiO2 as the primary driver.

---

## 3. Robustness (5-Seed, Full-Scale)

Five independent SVR training runs with different GroupShuffleSplit seeds (42-46), 58,057 train / 24,881 test each.

| Seed | R2 | RMSE | MAE | Time (s) | SV count |
|:---:|------:|------:|------:|------:|------:|
| 42 | 0.9720 | 0.0390 | 0.0183 | 11,655 | 24,033 |
| 43 | 0.9739 | 0.0366 | 0.0177 | 10,785 | 23,830 |
| 44 | 0.9723 | 0.0384 | 0.0183 | 12,548 | 24,001 |
| 45 | 0.9713 | 0.0398 | 0.0184 | 11,783 | 24,056 |
| 46 | 0.9762 | 0.0340 | 0.0165 | 11,950 | 24,267 |

| Aggregate | Value |
|------|------|
| R2 | **0.9731 +/- 0.0018** |
| RMSE | **0.0376 +/- 0.0021** |
| MAE | **0.0179 +/- 0.0007** |
| Wall-clock | 3.5h (5 workers, 4.7x speedup vs sequential) |

The narrow std (0.0018 on R2) confirms the SVR model is stable and the thickness-grouped split protocol is well-behaved.

### Comparison with paper Table 2 (R2=0.9593)

The paper reported R2=0.9593 on 8000 samples with fixed n (no wavelength-dependent n). The full-scale result (0.9731) is higher because:
1. Real n(lambda) optical constants provide richer physical information
2. Larger training set (58k vs 5.6k) reduces variance

---

## 4. Ablation Study

Halton quasi-random search (10 trials, 5,000 samples) found optimal SVR parameters:

| Parameter | Value |
|------|------|
| C | 1000 |
| gamma | 1.0 |
| epsilon | 0.01 |
| Search R2 | 0.9696 |

Five configurations compared (see `stage1/run_ablation_parallel.py`):

| ID | Description |
|----|------|
| A1 | Halton search + energy filter (baseline) |
| A2 | No energy filter (all data) |
| A3 | Random split (data leakage demo) |
| A4 | Grid search (18 combos, 5k subset) |
| A5 | Production SVR reference |

Full results: `stage1/simulation_data/ablation_results.json`

---

## 5. Domain Extrapolation Limits

Training domain: wavelength = [2, 14] um, PDMS = [100, 1000] nm, SiO2 = [0, 2000] nm.

SVR with RBF kernel cannot extrapolate beyond the convex hull of training data. Extrapolation R2 can drop to negative values when predicting far outside the training envelope. This is a fundamental limitation of RBF-based surrogates -- not a bug.

For design points outside the training domain, re-run TMM simulation or use a physics-informed model.

---

## Reproducing These Results

```bash
pip install -r requirements.txt

# Fair baselines
python stage1/run_fair_baselines.py --max-samples 90000

# Feature importance
python stage1/run_feature_importance.py --max-samples 90000

# Robustness (parallel, ~3.5h on 5 cores)
python stage1/run_robustness_parallel.py --max-samples 90000 --workers 5

# Ablation study (parallel, ~10h on 4 cores)
python stage1/run_ablation_parallel.py --max-samples 90000 --workers 4
```
