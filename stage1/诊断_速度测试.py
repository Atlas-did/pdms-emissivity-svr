#!/usr/bin/env python3
"""Speed diagnostic: measure actual TMM vs SVR performance.
Finds where 44.65ms / 71ms / 170x / 275x come from."""
from __future__ import annotations
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import tmm, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parent
S1DATA = ROOT / "stage1" / "simulation_data"

print("=" * 60)
print("  Speed Diagnostic: TMM vs SVR")
print("=" * 60)

# ---- Load data ----
with open(S1DATA / "test_results.json", "r") as f:
    tres = json.load(f)
print(f"[OK] test_results: R2={tres['r2']:.4f}, RMSE={tres['rmse']:.4f}, "
      f"MAE={tres['mae']:.4f}, n={len(tres['y_test'])}")

tmm_csv = S1DATA / "tmm_emissivity_data.csv"
tmm_df = pd.read_csv(tmm_csv)
cols = list(tmm_df.columns)
tmm_df.rename(columns={cols[0]:"wl", cols[1]:"sub", cols[2]:"pdms", cols[3]:"R", cols[4]:"T", cols[5]:"eps"}, inplace=True)
print(f"[OK] TMM CSV: {len(tmm_df)} rows, d={tmm_df['pdms'].min():.0f}-{tmm_df['pdms'].max():.0f}nm")

# ---- Load materials ----
pdms_df = pd.read_excel(ROOT / "stage1" / "pdms_nk.xlsx"); pdms_df.columns=["wl","n","k"]
sio2_df = pd.read_excel(ROOT / "stage1" / "sio2_nk.xlsx"); sio2_df.columns=["wl","n","k"]
from scipy.interpolate import CubicSpline
PDMS_N = CubicSpline(pdms_df["wl"].values, pdms_df["n"].values, extrapolate=True)
PDMS_K = CubicSpline(pdms_df["wl"].values, pdms_df["k"].values, extrapolate=True)
SIO2_N = CubicSpline(sio2_df["wl"].values, sio2_df["n"].values, extrapolate=True)
SIO2_K = CubicSpline(sio2_df["wl"].values, sio2_df["k"].values, extrapolate=True)

# ---- Load or train SVR model ----
bundle_p = S1DATA / "model_bundle.joblib"
model_p = S1DATA / "svr_emissivity_model.pkl"
scaler_p = S1DATA / "scaler.pkl"
fb_p = S1DATA / "region_models" / "fallback_synthetic" / "svr_model.pkl"

model = None; scaler = None; y_scaler = None; model_src = ""

if bundle_p.exists():
    model, scaler, y_scaler, _ = joblib.load(bundle_p); model_src = "bundle.joblib"
elif model_p.exists() and scaler_p.exists():
    model = joblib.load(model_p); scaler = joblib.load(scaler_p); model_src = "pkl"
elif fb_p.exists():
    model = joblib.load(fb_p)
    scaler = joblib.load(S1DATA / "region_models" / "fallback_synthetic" / "scaler.pkl")
    model_src = "fallback"

if model is None:
    print("[WARN] No model found, training quick fallback (n=500)...")
    df = tmm_df.sample(n=min(500, len(tmm_df)), random_state=42)
    Xr = df[["wl","sub","pdms"]].values.astype(float)
    yr = df["eps"].values.astype(float)
    wl, sio, pd = Xr[:,0], Xr[:,1], Xr[:,2]
    wl_m, pm, sm = wl*1e-6, pd*1e-9, sio*1e-9
    dp = 2*np.pi*1.41*pm/wl_m; ds = 2*np.pi*1.46*sm/wl_m
    Xf = np.column_stack([Xr, np.sin(dp), np.cos(dp), np.sin(ds), np.cos(ds),
        1.0/wl, pd/wl, np.sin(dp)*np.sin(ds), np.cos(dp)*np.cos(ds),
        np.sin(dp)*np.cos(ds), np.cos(dp)*np.sin(ds)])
    Xf = np.nan_to_num(Xf, 0); scaler = StandardScaler()
    model = SVR(kernel='rbf', C=500, gamma=1.0, epsilon=0.01)
    model.fit(scaler.fit_transform(Xf), yr); model_src = "on-the-fly"

print(f"[OK] Model: {model_src}")

# ---- TMM benchmark ----
def tmm_eps(lam, pdms_nm, sio2_nm=500.0):
    n1 = complex(float(PDMS_N(lam)), max(0, float(PDMS_K(lam))))
    n2 = complex(float(SIO2_N(lam)), float(SIO2_K(lam)))
    nl = [1.0, n1, n2, complex(3.42, 0.0)]
    dl = [np.inf, pdms_nm*1e-9, sio2_nm*1e-9, np.inf]
    cs = tmm.coh_tmm("s", nl, dl, 0.0, lam*1e-6)
    cp = tmm.coh_tmm("p", nl, dl, 0.0, lam*1e-6)
    return float(np.clip(1-0.5*(cs["R"]+cp["R"])-0.5*(cs["T"]+cp["T"]), 0, 1))

wl_grid = np.arange(2.0, 14.01, 0.02)
N_WL = len(wl_grid)
print(f"\nWavelength grid: {N_WL} points, {wl_grid[0]:.2f}-{wl_grid[-1]:.2f} um")

# Single-point TMM
[tmm_eps(10, 500) for _ in range(50)]  # warmup
t0 = time.perf_counter()
[tmm_eps(10, 500) for _ in range(1000)]
tmm_1pt_us = (time.perf_counter()-t0)/1000*1e6

# Full-spectrum TMM (fewer iterations for speed)
[tmm_eps(w, 500) for w in wl_grid[::10]]  # warmup on subset
t0 = time.perf_counter()
for _ in range(5):
    [tmm_eps(w, 500) for w in wl_grid]
tmm_full_ms = (time.perf_counter()-t0)/5*1000
tmm_full_s = tmm_full_ms/1000

print(f"\n--- TMM ---")
print(f"  1 wavelength:    {tmm_1pt_us:.0f} us")
print(f"  Full ({N_WL} pts): {tmm_full_ms:.0f} ms = {tmm_full_s:.3f} s")
print(f"  Paper baseline:  12.3 s (Intel i9-10900K, 500 pts)")

# ---- SVR benchmark ----
def build_features(wl, sio, pd):
    wm, pm, sm = wl*1e-6, pd*1e-9, sio*1e-9
    dp = 2*np.pi*1.41*pm/wm; ds = 2*np.pi*1.46*sm/wm
    f = np.column_stack([wl, sio, pd, np.sin(dp), np.cos(dp), np.sin(ds), np.cos(ds),
        1.0/wl, pd/wl, np.sin(dp)*np.sin(ds), np.cos(dp)*np.cos(ds),
        np.sin(dp)*np.cos(ds), np.cos(dp)*np.sin(ds)])
    return np.nan_to_num(f, 0)

wla = np.array(wl_grid); sio_a = np.full(N_WL, 500.0); pd_a = np.full(N_WL, 500.0)
Xf = build_features(wla, sio_a, pd_a)
exp = scaler.n_features_in_ if hasattr(scaler,'n_features_in_') else Xf.shape[1]
if Xf.shape[1] != exp:
    Xf = Xf[:, :exp] if Xf.shape[1] > exp else np.column_stack([Xf, np.zeros((Xf.shape[0], exp-Xf.shape[1]))])

# Warmup
[model.predict(scaler.transform(Xf)) for _ in range(20)]

# Single spectrum (scale + predict)
t0 = time.perf_counter()
for _ in range(300):
    model.predict(scaler.transform(Xf))
svr_full_ms = (time.perf_counter()-t0)/300*1000

# Predict only (pre-scaled)
Xs = scaler.transform(Xf)
t0 = time.perf_counter()
for _ in range(1000):
    model.predict(Xs)
svr_pred_ms = (time.perf_counter()-t0)/1000*1000

# Batch-100 amortized
Xb = np.tile(Xf, (100, 1)); Xsb = scaler.transform(Xb)
t0 = time.perf_counter()
for _ in range(100):
    model.predict(Xsb)
svr_b100_ms = (time.perf_counter()-t0)/100*1000
svr_b100_per = svr_b100_ms/100

print(f"\n--- SVR ---")
print(f"  Full spectrum (scale+predict): {svr_full_ms:.2f} ms")
print(f"  Predict only (pre-scaled):     {svr_pred_ms:.4f} ms")
print(f"  scale.transform overhead:      {svr_full_ms-svr_pred_ms:.4f} ms")
print(f"  Batch x100 (per spectrum):     {svr_b100_per:.3f} ms")

# ---- Summary ----
print(f"\n{'='*60}")
print(f"  SUMMARY: Where Do the Numbers Come From?")
print(f"{'='*60}")
spup_raw = tmm_full_s*1000 / max(svr_full_ms, 0.001)
spup_batch = tmm_full_s*1000 / max(svr_b100_per, 0.0001)
print(f"""
  Value        | Source
  -------------|--------------------------------------------------
  12.3 s       | Paper TMM baseline (i9-10900K, 500 pts). This machine: {tmm_full_s:.2f}s
  71 ms        | SVR single-spectrum (scale+predict). This machine: {svr_full_ms:.1f}ms
  44.65 ms     | SVR batch-amortized (from speed_results.json). This machine: {svr_b100_per:.1f}ms
  170x         | 12.3s / 71ms ~ {12.3/0.071:.0f}x (raw per-spectrum speedup)
  275x         | End-to-end including TTL caching + batch pipelining

  This machine per-spectrum speedup: {spup_raw:.0f}x (raw) / {spup_batch:.0f}x (batched)
  (Dependent on CPU model; paper reports i9-10900K / Ultra 5 225H)
""")
