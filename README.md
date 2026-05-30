# PDMS Thin-Film Emissivity: TMM-SVR Surrogate Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Physics-constrained surrogate modeling pipeline that combines **Transfer Matrix Method (TMM)**
simulation with **Support Vector Regression (SVR)** for rapid prediction of PDMS/SiO₂ thin-film
infrared emissivity across λ = 2.0–14.0 μm and d = 100–1000 nm.

**Key result:** Hold-out R² = 0.959, RMSE = 0.047, MAE = 0.026 — **275× faster** than direct TMM evaluation.

---

## Quick Start

```bash
pip install -r requirements.txt

# Generate TMM training data & train SVR surrogate (~2 min)
python stage1/main_pdms_svr_bandscan_full.py --action full_pipeline

# Generate all paper figures
python stage2/分区图片1.0.py
```

For detailed instructions in Chinese, see [docs/usage_zh.md](docs/usage_zh.md).

---

## Repository Structure

```
├── stage1/                      # Data generation & model training
│   ├── core/                    #   TMM simulator, SVR trainer, config, I/O
│   ├── main_pdms_svr_bandscan_full.py  # Main entry (18-menu interactive CLI)
│   ├── train_gpr_uncertainty.py #   GPR uncertainty training (optional)
│   ├── pdms_nk.xlsx             #   PDMS optical constants (n, k)
│   └── sio2_nk.xlsx             #   SiO₂ optical constants (n, k)
├── stage2/                      # Figure generation & visualization
│   ├── core/                    #   Region router, figure registry, data loader
│   ├── 分区图片1.0.py            #   Main figure engine (22 figure methods)
│   └── run_p1_p2_p3_all.py      #   Unified runner for all paper figures
├── docs/                        # Documentation
│   ├── usage_zh.md              #   Usage guide (Chinese)
│   └── physics_zh.md            #   Physical & mathematical background (Chinese)
├── requirements.txt             # Python dependencies
├── requirements.lock.txt        # Pinned dependency versions (exact reproducibility)
├── LICENSE                      # MIT
└── CITATION.cff                 # Citation metadata
```

---

## Pipeline Overview

```
Optical Constants (n,k) + Structural Params (λ, d)
        │
        ▼
Phase 1: TMM Simulation
  • Transfer Matrix Method (Maxwell-consistent)
  • Energy conservation check: |R+T+ε−1| < 10⁻³
  • TTL caching for repeated evaluations
        │
        ▼
Phase 2: SVR Surrogate Training
  • RBF kernel with 13-dim engineered features
  • Halton quasi-random hyperparameter search
  • Overlap-aware multi-region partitioning
        │
        ▼
Fast Inference: ε(λ, d) in milliseconds
  • 275× speedup over direct TMM
  • Multi-region routing with weighted blending
```

---

## Key Features

- **Physics-constrained data generation**: TMM enforces Maxwell's equations; energy conservation validates every training point
- **Multi-region surrogate**: Wavelength-domain partitioning with 5% overlap reduces boundary errors from 5–10% to < 1%
- **Reproducible**: Full run manifest, data fingerprint, and exported configuration for every experiment
- **Uncertainty quantification**: Optional GPR surrogate provides prediction intervals alongside SVR point estimates
- **Extensible**: Replace optical constants to model other material systems or multilayer stacks

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wu2026pdms,
  title   = {Physics-Based Data Generation with SVR Surrogate Modeling
             for PDMS Thin Film Emissivity Prediction},
  author  = {Wu, Yifan},
  journal = {arXiv preprint},
  year    = {2026}
}
```

You can also use the **"Cite this repository"** button on the right sidebar (powered by [CITATION.cff](CITATION.cff)).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
