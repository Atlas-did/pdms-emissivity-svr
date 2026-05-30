#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

if __package__ is None or __package__ == "":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(_HERE)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from stage1.main_pdms_svr_bandscan_full import Config, SVRTrainer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "stage1" / "simulation_data"
CSV = OUT / "tmm_emissivity_data.csv"
CONFIG = OUT / "config.json"
MODEL_INFO = OUT / "model_info.json"


def _save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=4), encoding="utf-8")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _fit_eval(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, params: dict) -> dict:
    scaler = StandardScaler()
    t0 = time.time()
    X_train_s = scaler.fit_transform(X_train)
    fit_prep_s = time.time() - t0

    model = SVR(kernel="rbf", C=float(params["C"]), gamma=float(params["gamma"]), epsilon=float(params["epsilon"]))
    t1 = time.time()
    model.fit(X_train_s, y_train)
    fit_s = time.time() - t1

    t2 = time.time()
    y_pred = np.clip(model.predict(scaler.transform(X_test)), 0.0, 1.0)
    pred_s = time.time() - t2

    m = _metrics(y_test, y_pred)
    m.update({
        "fit_prep_s": float(fit_prep_s),
        "fit_s": float(fit_s),
        "pred_s": float(pred_s),
        "pred_time_ms_per_point": float((pred_s / max(1, len(X_test))) * 1000.0),
        "n_support_vectors": int(getattr(model, "support_", np.array([])).shape[0]),
    })
    return m


def main() -> int:
    if not CSV.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV}")
    if not MODEL_INFO.exists():
        raise FileNotFoundError(f"Missing model_info: {MODEL_INFO}")

    run_id = datetime.now().strftime("quick_missing_%Y%m%d_%H%M%S")

    cfg = Config()
    if CONFIG.exists():
        cfg.config_file = str(CONFIG)
    cfg.load()

    model_info = json.loads(MODEL_INFO.read_text(encoding="utf-8"))
    params = model_info.get("best_params") or {"C": 500.0, "gamma": 1.0, "epsilon": 0.01}

    df = pd.read_csv(CSV)
    required = {"波长_μm", "基底厚度_nm", "PDMS厚度_nm", "发射率ε"}
    if not required.issubset(df.columns):
        raise KeyError(f"Missing required columns: {sorted(required - set(df.columns))}")

    trainer = SVRTrainer(cfg)
    X_all = trainer._build_features(df)
    y_all = df["发射率ε"].values.astype(float)

    # ---------------- Extrapolation artifact (fixed-best-params quick) ----------------
    tr_min = int(cfg.params.get("extrap_train_thickness_min", 100))
    tr_max = int(cfg.params.get("extrap_train_thickness_max", 700))
    te_min = int(cfg.params.get("extrap_test_thickness_min", 750))
    te_max = int(cfg.params.get("extrap_test_thickness_max", 1000))

    train_mask = (df["PDMS厚度_nm"] >= tr_min) & (df["PDMS厚度_nm"] <= tr_max)
    test_mask = (df["PDMS厚度_nm"] >= te_min) & (df["PDMS厚度_nm"] <= te_max)

    X_train = X_all[train_mask.values]
    y_train = y_all[train_mask.values]
    X_test = X_all[test_mask.values]
    y_test = y_all[test_mask.values]

    rng = np.random.RandomState(int(cfg.params.get("random_seed", 42)))
    max_train = 12000
    max_test = 8000
    if len(X_train) > max_train:
        idx = rng.choice(len(X_train), size=max_train, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    if len(X_test) > max_test:
        idx = rng.choice(len(X_test), size=max_test, replace=False)
        X_test, y_test = X_test[idx], y_test[idx]

    extrap = _fit_eval(X_train, y_train, X_test, y_test, params)
    extrap.update({
        "run_id": run_id,
        "mode": "quick_fixed_best_params",
        "best_params_source": "model_info.json",
        "train_range_nm": [tr_min, tr_max],
        "test_range_nm": [te_min, te_max],
        "n_train_used": int(len(X_train)),
        "n_test_used": int(len(X_test)),
    })

    extrap_path = OUT / f"extrap_thickness_{tr_min}-{tr_max}__{te_min}-{te_max}.json"
    _save(extrap_path, extrap)

    # ---------------- Split protocol comparison artifact ----------------
    max_total = 30000
    if len(X_all) > max_total:
        idx = rng.choice(len(X_all), size=max_total, replace=False)
        X_cmp = X_all[idx]
        y_cmp = y_all[idx]
        g_cmp = df["PDMS厚度_nm"].values[idx]
    else:
        X_cmp, y_cmp, g_cmp = X_all, y_all, df["PDMS厚度_nm"].values

    test_size = float(cfg.params.get("test_size", 0.3))
    rs = int(cfg.params.get("random_seed", 42))

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
    tr_i, te_i = next(gss.split(np.arange(len(X_cmp)), y_cmp, g_cmp))
    group_res = _fit_eval(X_cmp[tr_i], y_cmp[tr_i], X_cmp[te_i], y_cmp[te_i], params)
    group_res.update({"n_train": int(len(tr_i)), "n_test": int(len(te_i))})

    tr_i2, te_i2 = train_test_split(np.arange(len(X_cmp)), test_size=test_size, random_state=rs)
    random_res = _fit_eval(X_cmp[tr_i2], y_cmp[tr_i2], X_cmp[te_i2], y_cmp[te_i2], params)
    random_res.update({"n_train": int(len(tr_i2)), "n_test": int(len(te_i2))})

    cmp_obj = {
        "run_id": run_id,
        "mode": "quick_fixed_best_params",
        "best_params_source": "model_info.json",
        "best_params": params,
        "test_size": test_size,
        "random_state": rs,
        "n_total_used": int(len(X_cmp)),
        "group_split_by_thickness": group_res,
        "random_split": random_res,
    }
    cmp_path = OUT / "split_protocol_comparison.json"
    _save(cmp_path, cmp_obj)

    # ---------------- Cache quick evidence artifact ----------------
    perf_tmm = OUT / "performance_summary_stage1_tmm.json"
    cache_obj = {
        "run_id": run_id,
        "mode": "quick_static_plus_duplicate_check",
        "source_performance_file": str(perf_tmm),
        "rows_total": int(len(df)),
        "duplicate_key_count": int(df.duplicated(subset=["波长_μm", "基底厚度_nm", "PDMS厚度_nm"]).sum()),
    }
    if perf_tmm.exists():
        try:
            cache_obj["performance_summary"] = json.loads(perf_tmm.read_text(encoding="utf-8"))
        except Exception:
            pass
    _save(OUT / "cache_benchmark_quick.json", cache_obj)

    print(f"Generated: {extrap_path}")
    print(f"Generated: {cmp_path}")
    print(f"Generated: {OUT / 'cache_benchmark_quick.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
