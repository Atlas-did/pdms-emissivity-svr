#!/usr/bin/env python3
"""Batch runner for all remaining Stage 1 experiments.
Skips tasks whose JSON output already exists.
Usage:
    python stage1/run_all_remaining.py              # run all pending
    python stage1/run_all_remaining.py --dry-run    # show status only
    python stage1/run_all_remaining.py --only dkl_5000,svgp
"""

from __future__ import annotations
import sys, json, subprocess, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "stage1" / "simulation_data"
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")

TASKS = {
    "dkl_5000": {
        "script": "stage1/train_dkl.py",
        "args": ["--max-samples", "5000", "--n-iter", "200"],
        "output": OUT_DIR / "dkl_results.json",
        "desc": "DKL ExactGP 5000 samples, 200 iter (~2-5 min)",
        "timeout_s": 600,
    },
    "dkl_8000": {
        "script": "stage1/train_dkl.py",
        "args": ["--max-samples", "8000", "--n-iter", "300",
                  "--output", str(OUT_DIR / "dkl_results_8k.json")],
        "output": OUT_DIR / "dkl_results_8k.json",
        "desc": "DKL ExactGP 8000 samples, 300 iter (~5-10 min)",
        "timeout_s": 1200,
    },
    "mlp_distill": {
        "script": "stage1/train_distillation_mlp.py",
        "args": ["--max-samples", "90000"],
        "output": OUT_DIR / "distillation_mlp_results.json",
        "desc": "RF->MLP distillation, 82K samples (~10 min)",
        "timeout_s": 1800,
    },
    "inverse_design": {
        "script": "stage1/inverse_design.py",
        "args": [],
        "output": OUT_DIR / "inverse_design_results.json",
        "desc": "Inverse design: L-BFGS-B optimize d_PDMS (~1 min)",
        "timeout_s": 120,
    },
    "svgp": {
        "script": "stage1/train_svgp.py",
        "args": ["--max-samples", "10000", "--n-iter", "200"],
        "output": OUT_DIR / "svgp_results.json",
        "desc": "SVGP sparse GP, 10K samples (~5 min)",
        "timeout_s": 600,
    },
    "svgp_full": {
        "script": "stage1/train_svgp.py",
        "args": ["--max-samples", "90000", "--n-iter", "200",
                  "--output", str(OUT_DIR / "svgp_results_full.json")],
        "output": OUT_DIR / "svgp_results_full.json",
        "desc": "SVGP sparse GP, 82K samples (~30 min)",
        "timeout_s": 3600,
    },
}


def check_done(task_id):
    info = TASKS[task_id]
    out = info["output"]
    if not out.exists():
        return False
    try:
        data = json.loads(out.read_text(encoding="utf-8"))
        if "results" in data and "r2" in data.get("results", {}):
            return True
        if "models" in data or "optimization" in data:
            return True
    except Exception:
        pass
    return out.stat().st_size > 100


def run_task(task_id):
    info = TASKS[task_id]
    cmd = [PYTHON, str(ROOT / info["script"])] + [str(a) for a in info["args"]]
    sep = "=" * 60
    print()
    print(sep)
    txt = "  [" + task_id + "] " + info["desc"]
    print(txt)
    print(sep)
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), timeout=info["timeout_s"])
        if result.returncode != 0:
            print("  [" + task_id + "] FAILED (exit=" + str(result.returncode) + ")")
            return False
        if check_done(task_id):
            print("  [" + task_id + "] DONE")
            return True
        else:
            print("  [" + task_id + "] WARNING: no output file")
            return False
    except subprocess.TimeoutExpired:
        print("  [" + task_id + "] TIMEOUT after " + str(info["timeout_s"]) + "s")
        return False


def status_table():
    print()
    for tid, info in TASKS.items():
        done = check_done(tid)
        status = "DONE" if done else "PENDING"
        print("  " + tid.ljust(20) + " " + status.ljust(12) + " " + info["output"].name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--skip", type=str, default="")
    args = ap.parse_args()

    status_table()
    if args.dry_run:
        return 0

    skip = set(s.strip() for s in args.skip.split(",") if s.strip())
    only = set(s.strip() for s in args.only.split(",") if s.strip())

    completed = 0
    failed = []
    for tid in TASKS:
        if only and tid not in only:
            continue
        if tid in skip:
            print("  [" + tid + "] SKIPPED")
            continue
        if check_done(tid):
            print("  [" + tid + "] already done, skip")
            completed += 1
            continue
        ok = run_task(tid)
        if ok:
            completed += 1
        else:
            failed.append(tid)

    sep = "=" * 60
    print()
    print(sep)
    print("  COMPLETED: " + str(completed) + "/" + str(len(TASKS)))
    if failed:
        print("  FAILED: " + ", ".join(failed))
    print(sep)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())