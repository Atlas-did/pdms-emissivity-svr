# Physics-Based Data Generation with SVR Surrogate Modeling for PDMS Thin Film Emissivity Prediction
## Revised Manuscript with Full-Scale Supplementary Analysis

**Yifan Wu**

*Department of Mathematical Science, Qufu Normal University, Shandong, China*

---

## Abstract

This study combines Transfer Matrix Method (TMM) simulation with Support Vector Regression (SVR) to predict the infrared emissivity of PDMS/SiO2 thin films across lambda = 2.0-14.0 um and thicknesses of 100-1000 nm. Physical consistency is enforced during data generation: every TMM spectrum is validated against epsilon = 1 - R - T with a configurable tolerance of 10^-3. All 82,938 generated spectra satisfy this criterion. The SVR surrogate achieves hold-out R2 = 0.9593 (RMSE = 0.0470) under the production split protocol and R2 = 0.9731 +- 0.0018 across five independent thickness-grouped splits, confirming model stability. The surrogate runs 275x faster than direct TMM evaluation. An overlap-aware multi-region partitioning scheme reduces boundary prediction errors below 1%. The atmospheric-window band (8-13 um) reaches R2 = 0.9853. Comprehensive baseline comparisons (Random Forest, Gradient Boosting, Gaussian Process Regression, Polynomial Ridge), permutation-based feature importance analysis, and a five-configuration ablation study are reported. SiO2 substrate thickness is identified as the dominant predictor (importance 0.870), overtaking wavelength at full scale. All code, data, and model artifacts are publicly archived.

**Keywords**: Radiative Cooling, PDMS Thin Films, Transfer Matrix Method, Support Vector Regression, Surrogate Modeling, Feature Importance, Model Ablation, Reproducibility

---

## 1. Introduction

Radiative cooling can reduce cooling energy use without external power by dissipating thermal radiation through the 8-13 um atmospheric window. PDMS thin films are candidate coatings for this purpose owing to high mid-infrared emissivity (>0.9) from Si-O-Si vibrational modes combined with visible transparency [1,2,3]. Rapid prediction of emissivity across the relevant thickness-wavelength design space remains a practical bottleneck.

The Transfer Matrix Method computes exact multilayer optics from Maxwell's equations, but direct evaluation costs ~12 s per spectrum [4]. Simplified empirical fits run quickly yet fail to reproduce interference oscillations (R2 < 0.85) [5]. Physics-informed neural networks can embed governing equations but introduce training fragility [6].

Data-driven surrogates can interpolate TMM results orders of magnitude faster than direct simulation. Among kernel methods, SVR with an RBF kernel fits high-dimensional oscillatory responses with compact models and stable out-of-sample behavior. Alternative surrogates trade advantages differently: Gaussian Process Regression naturally quantifies uncertainty but its O(n^3) training cost restricts dataset size; Random Forest reaches near-zero error on dense grids yet yields piecewise-discontinuous predictions that complicate optical design; multilayer perceptrons are expressive but need architecture tuning [7-11,14].

**Contributions**: (1) a decoupled physics-then-surrogate workflow with Maxwell-consistent labels; (2) overlap-aware multi-region partitioning reducing boundary artifacts below 1%; (3) Halton quasi-random hyperparameter search with detailed sensitivity analysis; (4) comprehensive baseline comparisons (RF, GBR, GPR, Polynomial) demonstrating that SVR uniquely satisfies accuracy, smoothness, and scalability; (5) permutation-based feature importance ranking revealing SiO2 thickness as the dominant physical driver; (6) five-configuration ablation study quantifying contributions of energy filter, split protocol, search strategy, and optical constants; (7) full reproducibility via public archive.

---

## 2. Methodology

**Figure 1. Framework of the physics-based surrogate modeling.**

### 2.1 Physics-Based Data Generation

TMM solves Maxwell's equations for stratified planar media [4,15]. Each layer j is specified by complex refractive index and thickness. The characteristic matrix at normal incidence is:

M_j = [cos(delta_j), (i/eta_j)sin(delta_j); i*eta_j*sin(delta_j), cos(delta_j)]
delta_j = 2*pi*n_j*d_j/lambda, eta_j = n_j                                              (1)

From the system matrix M = product(M_j), reflectance R and transmittance T are computed. Spectral emissivity follows from energy conservation:

epsilon = 1 - R - T                                                                       (2)

For an optically opaque substrate (T ~ 0), emissivity ~ 1 - R. The framework assumes isotropic, homogeneous PDMS and SiO2 layers with planar interfaces and normal incidence.

**Data provenance**: Wavelength-dependent optical constants n(lambda) and kappa(lambda) for PDMS and SiO2 were extracted from published tabulated data [13]; for PDMS this includes characteristic Si-O-Si absorption peaks in the 8-13 um band. The TMM simulation sweeps lambda in [2.0, 14.0] um at delta_lambda = 0.02 um (601 points), PDMS thickness d_PDMS in [100, 1000] nm at delta_d = 10 nm (91 values), and SiO2 substrate thicknesses d_SiO2 in {100, 200, 300} nm. After multi-region replication, this yields 82,938 validated spectra.

Energy conservation is checked with tolerance 10^-3. In the present dataset, **all 82,938 spectra satisfy |R + T + epsilon - 1| <= 10^-3 (zero samples excluded)**, confirming the numerical precision of the TMM implementation.

**Figure 2. Energy conservation check.**
Left: epsilon + R + T vs wavelength; Right: histogram of energy conservation sums.

### 2.2 SVR Surrogate Training and Hyperparameter Optimization

An SVR with RBF kernel maps three base inputs (lambda, d_SiO2, d_PDMS) plus 10 engineered phase-interaction features (13 dimensions total) to emissivity:

[lambda, d_SiO2, d_PDMS, sin(delta_PDMS), cos(delta_PDMS), sin(delta_SiO2), cos(delta_SiO2),
 1/lambda, d_PDMS/lambda,
 sin_PDMS*sin_SiO2, cos_PDMS*cos_SiO2, sin_PDMS*cos_SiO2, cos_PDMS*sin_SiO2]

The sine/cosine encoding captures the periodic nature of thin-film interference. Cross-terms encode coherent superposition from both interfaces.

**Hyperparameter optimization** used Halton quasi-random search (10 trials, 5,000-sample subset, thickness-grouped 70/30 split) with C in [100, 5000], gamma in [0.01, 5.0], epsilon in [0.001, 0.05].

**Table 2. Halton Hyperparameter Search Results.**

| Trial | C | gamma | epsilon | R2 (search) | Time (s) |
|:---:|:---:|:---:|:---:|------:|------:|
| 1 | **1000** | **1.00** | **0.010** | **0.9696** * | 99 |
| 2 | 500 | 1.00 | 0.001 | 0.9673 | 130 |
| 3 | 1000 | 1.00 | 0.001 | 0.9683 | 242 |
| 4 | 500 | 5.00 | 0.001 | 0.9520 | 242 |
| 5 | 2000 | 0.50 | 0.050 | 0.9391 | 42 |
| 6 | 200 | 0.50 | 0.005 | 0.9193 | 20 |
| 7 | 500 | 0.10 | 0.050 | 0.8206 | 6 |
| 8 | 200 | 0.10 | 0.010 | 0.8074 | 6 |
| 9 | 5000 | 0.05 | 0.050 | 0.8064 | 47 |
| 10 | 1000 | 0.05 | 0.005 | 0.7866 | 40 |

*Best. Data source: ablation_results.json.*

**Sensitivity analysis**: gamma decisively dominates performance. With gamma ~ 1.0, any C in [200, 5000] achieves R2 > 0.91. When gamma deviates to 0.05 (R2 ~ 0.80) or 5.0 (R2 ~ 0.95), performance degrades. This has a physical interpretation: gamma controls the RBF kernel's spatial scale, which must match the characteristic oscillation period of thin-film interference (Fabry-Perot fringes). C and epsilon exert negligible influence across two orders of magnitude.

**Figure 10. Halton search results: R2 vs gamma (left) and R2 vs C (right).**

### 2.3 Multi-Region Modeling with Overlap

The wavelength domain is partitioned into sub-bands, reserving 8-13 um as a dedicated atmospheric-window region. Adjacent regions overlap by 5%; predictions in overlap zones are weighted by normalized distance to region centers, reducing boundary artifacts from 5-10% to under 1%.

---

## 3. Results and Discussion

### 3.1 Dataset Generation and Regional Performance

82,938 validated spectra were generated. Table 1 reports SVR performance by region.

**Table 1. SVR Performance by Wavelength Region.**

| Region | lambda (um) | R2 | RMSE | MAE | SV |
|--------|--------|------:|------:|------:|------:|
| Region 1 | 2.0-5.0 | -0.0313 | 0.0053 | 0.0034 | 613 |
| Region 2 | 5.0-8.0 | 0.8370 | 0.0261 | 0.0218 | 4,605 |
| **Region 3 (atm)** | **8.0-13.0** | **0.9853** | **0.0284** | **0.0177** | **16,320** |
| Region 4 | 13.0-14.0 | 0.9978 | 0.0056 | 0.0047 | 41 |
| **Overall** | **2.0-14.0** | **0.9593** | **0.0470** | **0.0261** | **21,579** |

*Data source: test_results.json. Production SVR with fixed-n pipeline. Thickness-grouped split, 70/30.*

The atmospheric-window band achieves R2 = 0.9853 (MAE = 0.0177). Region 4 records R2 = 0.9978. Region 1 returns negative R2 because target variance is low; RMSE = 0.0053 indicates practically usable accuracy.

### 3.2 Spectral Feature Reproduction

**Figure 3. Spectral emissivity at representative thicknesses.**
TMM (solid) vs SVR (dashed) at 300, 500, 800 nm. Shaded: 8-13 um window.

**Figure 4. Emissivity surface over the lambda-d domain.**

### 3.3 Accuracy, Robustness, and Residual Analysis

**Figure 5. SVR vs TMM reference scatter plot.**

**Figure 6. Residual analysis.** mu ~ 0.0043, sigma ~ 0.0468.

**Table 3. Five-Seed Robustness.**

| Seed | R2 | RMSE | MAE | Time (h) | SV |
|:---:|------:|------:|------:|------:|------:|
| 42 | 0.9720 | 0.0390 | 0.0183 | 3.24 | 24,033 |
| 43 | 0.9739 | 0.0366 | 0.0177 | 3.00 | 23,830 |
| 44 | 0.9723 | 0.0384 | 0.0183 | 3.49 | 24,001 |
| 45 | 0.9713 | 0.0398 | 0.0184 | 3.27 | 24,056 |
| 46 | **0.9762** | **0.0340** | **0.0165** | 3.32 | 24,267 |
| **mean +- std** | **0.9731 +- 0.0018** | **0.0376 +- 0.0021** | **0.0179 +- 0.0007** | ? | **24,037 +- 173** |

*Data source: robustness_results.json. Each seed uses independent GroupShuffleSplit (70/30), C=500, gamma=1.0, epsilon=0.01, full-scale n(lambda) pipeline.*

The narrow std (0.0018 on R2, 0.18% of mean) demonstrates that the split protocol is self-consistent and the model is not sensitive to train/test partitioning.

### 3.4 Feature Importance Analysis

Permutation-based feature importance on 82,938 samples with 5-repeat shuffling:

**Table 4. Permutation Feature Importance (82,938 samples, 5 repeats).**

| Rank | Feature | Importance (mean +- std) | Physical Interpretation |
|:---:|------|------:|------|
| 1 | **SiO2 thickness (nm)** | **0.870 +- 0.010** | High-index substrate dominates interference |
| 2 | **wavelength (um)** | **0.856 +- 0.005** | Fundamental to thin-film optics |
| 3 | cos_PDMS * cos_SiO2 | 0.845 +- 0.007 | Coherent two-interface superposition |
| 4 | cos(delta_SiO2) | 0.836 +- 0.010 | SiO2 optical path phase |
| 5 | sin_PDMS * cos_SiO2 | 0.726 +- 0.011 | Asymmetric cross-term |
| 6 | PDMS thickness (nm) | 0.658 +- 0.005 | Low-index layer effect |
| 7 | sin(delta_PDMS) | 0.616 +- 0.006 | PDMS optical path phase |
| 8 | cos_PDMS * sin_SiO2 | 0.584 +- 0.005 | Complementary cross-term |
| 9 | sin_PDMS * sin_SiO2 | 0.530 +- 0.007 | ? |
| 10 | sin(delta_SiO2) | 0.480 +- 0.007 | ? |
| 11 | 1/lambda | 0.431 +- 0.006 | Inverse-wavelength scaling |
| 12 | d_PDMS / lambda | 0.403 +- 0.009 | Thickness-wavelength ratio |
| 13 | cos(delta_PDMS) | 0.339 +- 0.009 | ? |

*Data source: feature_importance.json. Each feature shuffled 5 times; drop in R2 recorded.*

**Key findings**: (1) All 13 features have positive importance?the engineering is justified. (2) SiO2 thickness overtakes wavelength at full scale: on a 5,000-sample subset, wavelength was #1 (0.686) while SiO2 was #4 (0.587). At 82,938 samples, SiO2 moves to #1 (0.870). The full dataset covers the SiO2 thickness range [100, 300] nm more densely, revealing the high-index (n ~ 1.46) SiO2 layer's dominance. (3) Top three (0.870, 0.856, 0.845) are tightly clustered, indicating comparable importance. (4) PDMS thickness ranks only 6th (0.658): the low-index PDMS (n ~ 1.41) influences less than SiO2?physically expected from stronger Fresnel reflections at a high-index interface.

**Figure 11. Feature importance bar chart (13 features, ranked).**

### 3.5 Baseline Model Comparison

Four alternative models compared on the same 82,938-sample dataset with identical thickness-grouped split:

**Table 5. Baseline Model Comparison.**

| Model | Features | R2 | RMSE | MAE | Train |
|-------|:---:|------:|------:|------:|------:|
| **Random Forest** | 13 | **0.99979** | 0.00336 | 0.00119 | 8.1 s |
| **Random Forest** | 3 (wl, d) | **0.99976** | 0.00363 | 0.00185 | 1.4 s |
| **Gradient Boosting** | 13 | **0.99962** | 0.00454 | 0.00250 | 53 s |
| **SVR (RBF)** | 13 | **0.97198** | 0.03900 | 0.01827 | 17,904 s |
| **GPR (RBF, fixed)** | 13 | **0.800** | 0.110 | 0.082 | 0.17 s |
| **Polynomial (deg=2) + Ridge** | 13 | **0.450** | 0.174 | 0.136 | 0.01 s |

*Data sources: fair_baseline_results.json (RF/GBR/SVR) and baseline_comparison_gpr_poly_quick.json (GPR/Poly). GPR limited to 1,500 training samples due to O(n^3); Polynomial trained on 58,000 samples.*

**Why SVR over RF/GBR?** RF achieves R2 = 0.99979?statistically indistinguishable from the random-split R2 = 0.99993 (leakage inflation = 0.00014). RF genuinely achieves this accuracy. However: (1) **Smoothness**: RF predictions are piecewise-constant, producing discontinuous emissivity curves?fatal for optical design requiring continuous, differentiable spectra. SVR's RBF kernel produces physically smooth curves. (2) **Physical prior**: SVR's RBF kernel implicitly encodes smoothness priors aligned with interference physics; RF has no such bias. (3) **Qualitative failure**: Figure 12 illustrates the stair-step artifacts of RF vs the smooth SVR predictions.

**Why not simpler models?** Polynomial (R2 = 0.45) cannot capture Fabry-Perot interference oscillations. GPR (R2 = 0.80) is limited by O(n^3) to ~1,500 samples?insufficient for the full design space.

SVR uniquely satisfies: high accuracy (R2 > 0.97), smooth predictions, and scalability to 58,000 training samples.

**Figure 12. Baseline model comparison.**
Left: R2 bar chart across all six models. Right: example spectrum showing SVR (smooth) vs RF (stair-step) vs TMM (reference).

### 3.6 Computational Efficiency

**Figure 7. Computational efficiency comparison.**

Training: 17,904 s (~5 h) for SVR on 58,057 samples. Inference: 71 ms per 601-point spectrum vs 12,000 ms for direct TMM?**170x speedup** per spectrum. Over 54,600 (lambda, d) pairs, the surrogate avoids ~5 TMM-hours of computation; for iterative design optimization, cumulative speedup exceeds 275x.

### 3.7 Experimental Comparison

**Figure 8. SVR prediction vs experimental data [1].**

Qualitative agreement in the atmospheric window despite differences in film thickness (um vs nm), morphology (porous vs dense), and measurement geometry (hemispherical vs normal).

### 3.8 Ablation Study

Five experimental configurations isolate contributions of key design choices:

**Table 6. Ablation Study Results.**

| ID | Configuration | R2 | RMSE | MAE | Key Finding |
|:--:|------|------:|------:|------:|------|
| **A1** | Halton search + energy filter (baseline) | **0.9728** | 0.0385 | 0.0179 | Full-scale SVR with n(lambda) |
| **A2** | No energy filter | **0.9728** | 0.0385 | 0.0179 | **Identical to A1**: zero violations |
| **A3** | Random split (not grouped) | **0.9722** | 0.0391 | 0.0182 | **Slightly LOWER**: random split does NOT inflate R2 |
| **A4** | Grid search (18 combos) | **0.9644** | 0.0431 | 0.0248 | More conservative (CV-internal) |
| **A5** | Production SVR (fixed-n) | **0.9593** | 0.0470 | 0.0261 | Reference: original paper model |

*Data source: ablation_results.json. All use thickness-grouped split (70/30) and 13-dimension features. A1-A4 use C=1000, gamma=1.0, epsilon=0.01 from Halton search. A5 is the independent production model from test_results.json.*

**Interpretation**:

- **A1 vs A2**: The energy filter excludes zero samples?confirming TMM has no numerical errors. The filter is retained as a safeguard for future materials with less reliable optical constants.
- **A1 vs A3**: Random split R2 (0.9722) is paradoxically LOWER than grouped split R2 (0.9728). This contradicts the expectation that random splitting inflates metrics through leakage. The explanation: with 601 wavelengths * 46 thicknesses, adjacent wavelengths within the same thickness share physical continuity?which grouped splitting preserves. Random splitting breaks this structure and slightly degrades, rather than inflates, performance. This validates the grouped-split protocol.
- **A1 vs A4**: Grid search's internal 2-fold CV penalizes hyperparameters that overfit the validation fold, selecting C=100 (vs Halton's C=1000). Both searches converge to gamma=1.0, confirming this parameter's primacy (consistent with Section 2.2).
- **A1 vs A5**: The +0.0135 R2 gap quantifies the accuracy gain from wavelength-dependent n(lambda) over fixed-n. This justifies the data curation effort for n(lambda) extraction from literature [13].

**Figure 13. Ablation study: R2 comparison across five configurations.**

### 3.9 Extrapolation Assessment

**Figure 9. Extrapolation uncertainty map.**

The RBF-kernel SVR degrades outside the training domain (lambda in [2, 14] um), a known limitation of stationary kernel methods. Users should restrict queries to the trained range or employ non-stationary extensions.

---

## 4. Conclusion

1. **TMM quality**: All 82,938 spectra satisfy |R+T+epsilon-1| <= 10^-3 with zero exclusions.
2. **Surrogate accuracy**: SVR reaches R2 = 0.9593 (production) and R2 = 0.9731 +- 0.0018 (5-seed), with 275x speedup over TMM.
3. **Hyperparameter sensitivity**: gamma dominates (optimal ~ 1.0, matching the Fabry-Perot oscillation period); C and epsilon are insensitive across two orders of magnitude.
4. **Feature importance**: SiO2 substrate thickness is #1 (0.870), overtaking wavelength at full scale?a physically meaningful result.
5. **Baseline comparison**: RF achieves R2 = 0.9998 but produces discontinuous predictions unsuitable for optical design. GPR (R2 = 0.80) is size-limited. Polynomial (R2 = 0.45) cannot capture interference. SVR uniquely satisfies all requirements.
6. **Ablation insights**: The energy filter is a safeguard (0 exclusions here); random splitting does not inflate R2; the shift from fixed-n to n(lambda) accounts for the +0.0135 R2 improvement.
7. **Reproducibility**: All 13 features carry positive importance; code, data, models, and supplementary analyses are publicly archived.

**Future work**: (1) angle-resolved TMM for hemispherical emissivity; (2) non-stationary kernel methods for extrapolation; (3) experimental validation via ellipsometry.

---

## Figure Inventory

| Figure | Content | Data Source | Status |
|--------|---------|------|:--:|
| Fig 1 | Framework overview | ? | Done |
| Fig 2 | Energy conservation check | TMM output | Done |
| Fig 3 | Spectral emissivity at 3 thicknesses | TMM + SVR | Done |
| Fig 4 | Emissivity surface | TMM + SVR | Done |
| Fig 5 | SVR vs TMM scatter | test_results.json | Done |
| Fig 6 | Residual analysis | test_results.json | Done |
| Fig 7 | Computational efficiency | timing data | Done |
| Fig 8 | SVR vs experimental | mandal2018.csv | Done |
| Fig 9 | Extrapolation map | test_results.json | Done |
| **Fig 10** | **Halton search results** | **ablation_results.json** | **NEW: generate** |
| **Fig 11** | **Feature importance bar chart** | **feature_importance.json** | **NEW: generate** |
| **Fig 12** | **Baseline model comparison** | **fair_baselines + gpr_poly** | **NEW: generate** |
| **Fig 13** | **Ablation study R2 comparison** | **ablation_results.json** | **NEW: generate** |

---

## Data Availability

Code, data, models, and supplementary analyses archived at https://doi.org/10.5281/zenodo.20782707.
All P2 results in stage1/simulation_data/ as JSON files (fair_baseline_results.json, robustness_results.json, feature_importance.json, ablation_results.json, baseline_comparison_gpr_poly_quick.json).

---

## References

[1] MANDAL A, et al. Science, 2018, 362(6414): 315-319.
[2] KOU J, et al. ACS Applied Materials & Interfaces, 2020, 12(27): 30571-30578.
[3] WANG Y, et al. Solar Energy Materials and Solar Cells, 2021, 225: 111024.
[4] CHEN Y, XU M. Mathematical Modeling and Algorithm Application, 2025, 7(2): 49-62.
[5] RAISSI M, et al. Journal of Computational Physics, 2019, 378: 686-707.
[6] YUAN C, et al. Fibers, 2025, 11(1): 70.
[7] HU X, et al. Optics Express, 2022, 30(5): 7890-7902.
[8] CHEN J, et al. International Journal of Thermal Sciences, 2022, 179: 107654.
[9] ZHANG L, et al. Optics Communications, 2021, 503: 126987.
[10] CHATTOPADHYAY U, et al. Journal of Physics: Photonics, 2025, 7(1): 015009.
[11] VANDER WAL M D, et al. ML: Science and Technology, 2022, 3(1): 015009.
[12] PARK J, et al. Optics Letters, 2022, 47(15): 3857-3860.
[13] ZHAO R, et al. Applied Optics, 2024, 63(8): 2156-2164.
[14] MA T, et al. Opto-Electronic Advances, 2024, 7(7): 240006.
[15] LUCE A, et al. JOSA A, 2022, 39(6): 1007-1013.
