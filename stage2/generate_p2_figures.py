#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""generate_p2_figures.py ? Generate 4 supplementary figures from P2 JSON results.

Usage:
    python stage2/generate_p2_figures.py
    python stage2/generate_p2_figures.py --output figures/

Output:
    fig10_halton_search.png     Halton hyperparameter search (gamma & C sensitivity)
    fig11_feature_importance.png  Permutation feature importance bar chart
    fig12_baseline_comparison.png Baseline models: R2 bar chart + RF-stair-step demo
    fig13_ablation_study.png    Five-configuration ablation R2 comparison
"""

from __future__ import annotations
import json, sys, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load_json(path):
    """Load JSON with encoding fallback."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
DATA = ROOT / "stage1" / "simulation_data"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--output" else (ROOT / "stage2" / "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# ============================================================
# Fig 10: Halton Search ? gamma sensitivity + C insensitivity
# ============================================================
def fig10_halton_search():
    abl = load_json(DATA / "ablation_results.json")
    # Halton trials are embedded in A1; we need to reconstruct from the script output
    # Hardcode from the known search results
    trials = [
        {"C": 1000, "gamma": 1.00, "eps": 0.010, "r2": 0.9696},
        {"C": 500,  "gamma": 1.00, "eps": 0.001, "r2": 0.9673},
        {"C": 1000, "gamma": 1.00, "eps": 0.001, "r2": 0.9683},
        {"C": 500,  "gamma": 5.00, "eps": 0.001, "r2": 0.9520},
        {"C": 2000, "gamma": 0.50, "eps": 0.050, "r2": 0.9391},
        {"C": 200,  "gamma": 0.50, "eps": 0.005, "r2": 0.9193},
        {"C": 500,  "gamma": 0.10, "eps": 0.050, "r2": 0.8206},
        {"C": 200,  "gamma": 0.10, "eps": 0.010, "r2": 0.8074},
        {"C": 5000, "gamma": 0.05, "eps": 0.050, "r2": 0.8064},
        {"C": 1000, "gamma": 0.05, "eps": 0.005, "r2": 0.7866},
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: R2 vs gamma
    gammas = [t["gamma"] for t in trials]
    r2s = [t["r2"] for t in trials]
    Cs = [t["C"] for t in trials]
    colors = plt.cm.viridis(np.log10(Cs) / np.log10(5000))

    scatter1 = ax1.scatter(gammas, r2s, c=Cs, cmap="viridis", norm=matplotlib.colors.LogNorm(vmin=100, vmax=5000),
                           s=100, edgecolors="black", linewidth=0.5, zorder=5)
    ax1.set_xscale("log")
    ax1.set_xlabel("gamma (RBF kernel width)")
    ax1.set_ylabel("R^2 (search set)")
    ax1.set_title("Sensitivity to gamma")
    ax1.axvline(1.0, color="red", linestyle="--", alpha=0.5, label="optimal gamma=1.0")
    ax1.axhline(0.9696, color="green", linestyle=":", alpha=0.5)
    ax1.set_ylim(0.75, 1.0)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1, label="C (penalty)")
    ax1.annotate("R^2 collapses below gamma=0.5", xy=(0.06, 0.80), fontsize=9, color="red",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Right: R2 vs C (only for gamma near optimal: gamma in [0.5, 5.0])
    near_opt = [t for t in trials if 0.3 <= t["gamma"] <= 5.5]
    Cs_opt = [t["C"] for t in near_opt]
    r2s_opt = [t["r2"] for t in near_opt]
    ax2.scatter(Cs_opt, r2s_opt, c=[t["gamma"] for t in near_opt], cmap="plasma",
                s=120, edgecolors="black", linewidth=0.5, zorder=5)
    ax2.set_xscale("log")
    ax2.set_xlabel("C (penalty)")
    ax2.set_ylabel("R^2 (search set)")
    ax2.set_title("Insensitivity to C (gamma >= 0.5)")
    ax2.set_ylim(0.75, 1.0)
    ax2.grid(True, alpha=0.3)
    # Add horizontal band
    ax2.axhline(0.9696, color="green", linestyle=":", alpha=0.5)
    ax2.axhspan(0.91, 0.97, alpha=0.1, color="green", label="C-range stability")
    # Annotate
    for t in near_opt:
        ax2.annotate(f"gamma={t['gamma']}", (t["C"], t["r2"]), fontsize=8, alpha=0.7,
                     textcoords="offset points", xytext=(0, 8), ha="center")
    ax2.legend(fontsize=9)

    fig.suptitle("Figure 10: Halton Hyperparameter Search ? gamma Dominates, C is Insensitive",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig10_halton_search.png")
    plt.close(fig)
    print(f"  [OK] fig10_halton_search.png")


# ============================================================
# Fig 11: Feature Importance Bar Chart
# ============================================================
def fig11_feature_importance():
    fi = load_json(DATA / "feature_importance.json")
    features = fi["features"]

    names = [f["feature"] for f in features][::-1]  # reverse for horizontal bar
    means = [f["importance_mean"] for f in features][::-1]
    stds = [f["importance_std"] for f in features][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color gradient: top-3 dark, rest light
    colors = ["#2166AC"] * 3 + ["#92C5DE"] * 7 + ["#D1E5F0"] * 3
    colors = colors[::-1]

    bars = ax.barh(range(len(names)), means, xerr=stds, color=colors, edgecolor="black",
                   linewidth=0.5, capsize=2, height=0.7)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Permutation Importance (drop in R^2)")
    ax.set_title("Figure 11: Feature Importance Ranking (82,938 samples, 5 repeats)", fontweight="bold")

    # Add value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.01, i, f"{m:.3f}", va="center", fontsize=8)

    ax.set_xlim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="x")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2166AC", label="Top-3 (dominate)"),
        Patch(facecolor="#92C5DE", label="Mid (contribute)"),
        Patch(facecolor="#D1E5F0", label="Bottom-3 (auxiliary)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Annotation
    ax.annotate("SiO2 thickness #1\n(0.870) ? high-index\nsubstrate dominates\ninterference pattern",
                xy=(0.870, 12), fontsize=8, color="#2166AC",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
                xytext=(0.60, 10), arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5))

    fig.tight_layout()
    fig.savefig(OUT / "fig11_feature_importance.png")
    plt.close(fig)
    print(f"  [OK] fig11_feature_importance.png")


# ============================================================
# Fig 12: Baseline Model Comparison
# ============================================================
def fig12_baseline_comparison():
    fb = load_json(DATA / "fair_baseline_results.json")
    gpr = load_json(DATA / "baseline_comparison_gpr_poly_quick.json")

    models = [
        ("RF (13 feat)", fb["results"]["RF_13feat_grouped"]["r2"], "#E74C3C"),
        ("RF (3 feat)", fb["results"]["RF_3feat_grouped"]["r2"], "#E74C3C"),
        ("GBR (13 feat)", fb["results"]["GBR_13feat_grouped"]["r2"], "#F39C12"),
        ("SVR (RBF)", fb["results"]["SVR_13feat_grouped"]["r2"], "#2ECC71"),
        ("GPR (RBF)", gpr["results"]["GPR (fixed kernel, quick)"]["r2"], "#9B59B6"),
        ("Polynomial (deg=2)", gpr["results"]["Polynomial (deg=2) + Ridge"]["r2"], "#95A5A6"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: R2 bar chart
    names = [m[0] for m in models]
    r2s = [m[1] for m in models]
    colors = [m[2] for m in models]

    bars = ax1.bar(range(len(names)), r2s, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("R^2")
    ax1.set_title("Model Accuracy (82,938 samples, grouped split)", fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, axis="y")

    # Value labels
    for bar, r2 in zip(bars, r2s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{r2:.4f}", ha="center", fontsize=9, fontweight="bold")

    # Highlight SVR
    ax1.annotate("SVR: best balance of\naccuracy + smoothness + scalability",
                xy=(3, 0.972), fontsize=8, color="#2ECC71", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                xytext=(1.5, 0.65), arrowprops=dict(arrowstyle="->", color="#2ECC71"))

    # Right: RF stair-step vs SVR smooth (schematic)
    x = np.linspace(0, 10, 200)
    y_smooth = 0.5 + 0.4 * np.sin(x) * np.exp(-x/5)
    # Simulate RF piecewise: average within each bin
    bin_edges = np.linspace(0, 10, 25)
    y_rf = np.zeros_like(x)
    for i in range(len(bin_edges)-1):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
        if mask.any():
            y_rf[mask] = np.mean(y_smooth[mask])

    ax2.plot(x, y_smooth, "b-", linewidth=2, label="SVR (smooth, continuous)", alpha=0.8)
    ax2.plot(x, y_rf, "r-", linewidth=2, label="RF (piecewise-constant, discontinuous)", alpha=0.8)
    ax2.set_xlabel("Wavelength (schematic)")
    ax2.set_ylabel("Emissivity (schematic)")
    ax2.set_title("Prediction Smoothness Comparison", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Annotate stair-step artifacts
    for i in range(5, len(bin_edges)-1, 5):
        ax2.axvline(bin_edges[i], color="red", linestyle=":", alpha=0.3, linewidth=0.5)

    ax2.annotate("Stair-step artifacts\n(unacceptable for\noptical design)",
                xy=(bin_edges[10], y_rf[50]), fontsize=8, color="red",
                xytext=(7, 0.7),
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.5))

    fig.suptitle("Figure 12: Baseline Model Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig12_baseline_comparison.png")
    plt.close(fig)
    print(f"  [OK] fig12_baseline_comparison.png")


# ============================================================
# Fig 13: Ablation Study
# ============================================================
def fig13_ablation_study():
    abl = load_json(DATA / "ablation_results.json")

    configs = [
        ("A1", "Halton search\n+ energy filter", abl["A1_full_model"]["r2"], "#2ECC71"),
        ("A2", "No energy\nfilter", abl["A2_no_energy_filter"]["r2"], "#3498DB"),
        ("A3", "Random split\n(not grouped)", abl["A3_random_split"]["r2"], "#E74C3C"),
        ("A4", "Grid search\n(18 combos)", abl["A4_grid_search"]["r2"], "#F39C12"),
        ("A5", "Production SVR\n(fixed-n)", abl["A5_global_svr_reference"]["r2"], "#95A5A6"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    names = [c[1] for c in configs]
    r2s = [c[2] for c in configs]
    colors = [c[3] for c in configs]

    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, r2s, color=colors, edgecolor="black", linewidth=0.8, width=0.6)

    # A5 reference line
    ax.axhline(y=abl["A5_global_svr_reference"]["r2"], color="#95A5A6", linestyle="--",
               linewidth=1.5, alpha=0.7, label=f"Production baseline (R^2={r2s[-1]:.4f})")

    # Value labels
    for bar, r2, cid in zip(bars, r2s, [c[0] for c in configs]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.015,
                f"{r2:.4f}", ha="center", fontsize=12, fontweight="bold", color="white")
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                cid, ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("R^2")
    ax.set_title("Figure 13: Ablation Study ? Isolating Design Choices", fontweight="bold")
    ax.set_ylim(0.94, 0.99)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    # Annotation arrows
    ax.annotate("", xy=(0, r2s[0]), xytext=(1, r2s[0]),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1))
    ax.text(0.5, r2s[0] + 0.005, "identical\n(0 energy\nviolations)", ha="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))

    ax.annotate("Random split\nslightly LOWER\n(no leakage!)", xy=(2, r2s[2]),
                fontsize=8, color="#E74C3C",
                xytext=(2.5, 0.965), arrowprops=dict(arrowstyle="->", color="#E74C3C", alpha=0.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    ax.annotate(f"+0.0135 from\nn(lambda)", xy=(0, r2s[0]), fontsize=8, color="#2ECC71",
                xytext=(3.5, 0.985),
                arrowprops=dict(arrowstyle="->", color="#2ECC71", alpha=0.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    fig.tight_layout()
    fig.savefig(OUT / "fig13_ablation_study.png")
    plt.close(fig)
    print(f"  [OK] fig13_ablation_study.png")


# ============================================================
if __name__ == "__main__":
    print(f"Output directory: {OUT}")
    print()
    fig10_halton_search()
    fig11_feature_importance()
    fig12_baseline_comparison()
    fig13_ablation_study()
    print(f"\n[DONE] 4 figures saved to {OUT}")
