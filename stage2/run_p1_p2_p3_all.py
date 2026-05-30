#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified entry point: Run P1, P2, P3 paper figures.

Usage:
    python stage2/run_p1_p2_p3_all.py                  # All P1+P2+P3 figures
    python stage2/run_p1_p2_p3_all.py --only p1         # Only P1 figures
    python stage2/run_p1_p2_p3_all.py --only p2         # Only P2 figures
    python stage2/run_p1_p2_p3_all.py --only p3         # Only P3 figures
    python stage2/run_p1_p2_p3_all.py --skip-p3         # Skip P3
    python stage2/run_p1_p2_p3_all.py --include-standalone  # Include standalone figs
    python stage2/run_p1_p2_p3_all.py --list            # List all figure methods
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import List, Tuple

STAGE2_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STAGE2_DIR.parent
MAIN_FILE = STAGE2_DIR / "分区图片1.0.py"
FIG_DIR = STAGE2_DIR / "figures"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Figure groups ---
P1_FIGURES: List[Tuple[str, str]] = [
    ("figure15_direct_validation", "P1-1: Direct validation (scatter + error + heatmap)"),
    ("figure17_physical_constraints_effect", "P1-2: Physical constraints effect"),
    ("figure5_ablation_study", "P1-3: Multi-region ablation study"),
]

P2_FIGURES: List[Tuple[str, str]] = [
    ("figure19_gpr_uncertainty", "P2.1: GPR uncertainty quantification"),
    ("figure16_extrapolation", "P2.2: Extrapolation capability assessment"),
    ("figure8_efficiency", "P2.3: Computational efficiency breakdown"),
]

P3_FIGURES: List[Tuple[str, str]] = [
    ("figure18_ellipsometry_validation", "P3.1: Ellipsometry validation"),
    ("figure20_temperature_dependence", "P3.2: Temperature dependence"),
]

STANDALONE_METHODS = [
    "figure1_optical_constants",
    "figure2_key_thickness_spectra",
    "figure3_3d_surface",
    "figure4_scatter_plot",
    "figure5_residual_analysis",
    "figure6_literature_comparison",
    "figure7_region_metrics",
    "figure8_learning_curves",
    "figure9_energy_conservation",
    "figure14_model_comparison",
    "figure6_boundary_error",
    "figure3_process_flow",
    "figure4_sampling_strategy",
]


def _load_module():
    """Load stage2 main module via importlib (matches run_p1_1.py pattern)."""
    spec = importlib.util.spec_from_file_location("stage2_main_cn", str(MAIN_FILE))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load main script: {MAIN_FILE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_figures(viz, figure_list, label):
    """Run a list of (method_name, description) tuples.
    Returns list of (method_name, description, status, elapsed_seconds).
    """
    results = []
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    for method_name, desc in figure_list:
        print(f"\n> {desc}")
        print(f"  method: {method_name}")
        t0 = time.time()
        try:
            method = getattr(viz, method_name, None)
            if method is None:
                print(f"  [WARN]  Method '{method_name}' not found on Visualization")
                results.append((method_name, desc, "MISSING_METHOD", 0.0))
                continue
            method()
            elapsed = time.time() - t0
            print(f"  [OK] Done in {elapsed:.1f}s")
            results.append((method_name, desc, "OK", elapsed))
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [FAIL] Failed ({elapsed:.1f}s): {e}")
            import traceback
            traceback.print_exc()
            results.append((method_name, desc, f"ERROR: {type(e).__name__}", elapsed))
    return results


def print_summary(all_results):
    """Print comprehensive summary and write citation table."""
    print("\n\n" + "=" * 90)
    print("  PAPER FIGURE CITATION TABLE")
    print("=" * 90)
    print(f"  {'#':<4} {'Figure':<52} {'Method':<48} {'Status':<18} {'Time':>8}")
    print(f"  {'-'*4} {'-'*52} {'-'*48} {'-'*18} {'-'*8}")

    for i, (method_name, desc, status, elapsed) in enumerate(all_results, 1):
        sd = desc[:51] if len(desc) > 51 else desc
        sm = method_name[:47] if len(method_name) > 47 else method_name
        print(f"  {i:<4} {sd:<52} {sm:<48} {status:<18} {elapsed:>7.1f}s")

    ok = sum(1 for _, _, s, _ in all_results if s == "OK")
    fail = sum(1 for _, _, s, _ in all_results if s.startswith("ERROR"))
    missing = sum(1 for _, _, s, _ in all_results if s == "MISSING_METHOD")
    print(f"\n  Total: {len(all_results)}  |  OK: {ok}  |  Failed: {fail}  |  Missing: {missing}")

    # Write citation table
    priority_map = {}
    for lst, pri in [(P1_FIGURES, "P1"), (P2_FIGURES, "P2"), (P3_FIGURES, "P3")]:
        for m, d in lst:
            priority_map[m] = pri

    table_path = FIG_DIR / "paper_citation_table_p1_p2_p3.md"
    lines = [
        "# Paper Figure Citation Table (P1 / P2 / P3)",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| # | Priority | Figure Description | Method | Status | Time (s) |",
        "|---|:---:|---|---|---:|---:|",
    ]
    for i, (method_name, desc, status, elapsed) in enumerate(all_results, 1):
        pri = priority_map.get(method_name, "-")
        emoji = "[OK]" if status == "OK" else ("[FAIL]" if status.startswith("ERROR") else "[WARN]")
        lines.append(
            f"| {i} | {pri} | {desc[:60]} | `{method_name}` | {emoji} {status} | {elapsed:.1f} |"
        )
    lines.extend([
        "",
        f"**Summary:** {ok} succeeded, {fail} failed, {missing} missing (out of {len(all_results)} total)",
        "",
        "---",
        "",
        "## Figure Output Files",
        "",
        "| File | Corresponding Figure |",
        "|---|---|",
    ])
    for method_name, desc, status, _ in all_results:
        if status == "OK":
            lines.append(f"| `{method_name}.png` | {desc[:60]} |")
    table_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Citation table written to: {table_path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run P1/P2/P3 paper figures for PDMS emissivity project"
    )
    ap.add_argument("--only", choices=["p1", "p2", "p3", "all"], default="all")
    ap.add_argument("--skip-p3", action="store_true", help="Skip P3 figures")
    ap.add_argument("--skip-p2", action="store_true", help="Skip P2 figures")
    ap.add_argument("--skip-p1", action="store_true", help="Skip P1 figures")
    ap.add_argument("--include-standalone", action="store_true",
                    help="Also generate standalone figures (Figures 1-9, 14)")
    ap.add_argument("--list", action="store_true", help="List all figure methods and exit")
    args = ap.parse_args()

    if args.list:
        print("\nP1 Figures (must-have):")
        for m, d in P1_FIGURES:
            print(f"  {m:<45} -> {d}")
        print("\nP2 Figures (strongly suggested):")
        for m, d in P2_FIGURES:
            print(f"  {m:<45} -> {d}")
        print("\nP3 Figures (suggested):")
        for m, d in P3_FIGURES:
            print(f"  {m:<45} -> {d}")
        print(f"\nStandalone methods ({len(STANDALONE_METHODS)}):")
        for m in STANDALONE_METHODS:
            print(f"  {m}")
        return 0

    print("=" * 70)
    print("  PDMS Emissivity Paper - P1/P2/P3 Figure Generator")
    print("=" * 70)
    print(f"  Stage2 entry: {MAIN_FILE}")
    print(f"  Output dir:   {FIG_DIR}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load the stage2 module
    print("\n[1/3] Loading stage2 module...")
    mod = _load_module()

    # Initialize data loader
    print("[2/3] Loading data...")
    loader = mod.DataLoader(mod.DATADIR)
    loader.load_all()

    # Initialize visualization
    print("[3/3] Initializing Visualization...")
    viz = mod.Visualization(loader, FIG_DIR)

    # Collect figures to run
    tasks_to_run: List[Tuple[str, str]] = []

    if not args.skip_p1 and args.only in ("p1", "all"):
        tasks_to_run.extend(P1_FIGURES)

    if not args.skip_p2 and args.only in ("p2", "all"):
        tasks_to_run.extend(P2_FIGURES)

    if not args.skip_p3 and args.only in ("p3", "all"):
        tasks_to_run.extend(P3_FIGURES)

    if not tasks_to_run:
        print("[WARN]  No figures selected to run. Check your --only / --skip flags.")
        return 0

    print(f"\nWill generate {len(tasks_to_run)} figure(s):")
    for m, d in tasks_to_run:
        print(f"  - {d}")

    # Run
    all_results = run_figures(viz, tasks_to_run, "Paper Figures (P1/P2/P3)")

    # Optional standalone figures
    if args.include_standalone:
        standalone_tasks = [(m, f"Standalone: {m}") for m in STANDALONE_METHODS]
        standalone_results = run_figures(viz, standalone_tasks, "Standalone Figures (1-9, 14)")
        all_results.extend(standalone_results)

    print_summary(all_results)

    failed = sum(1 for _, _, s, _ in all_results if s.startswith("ERROR"))
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
