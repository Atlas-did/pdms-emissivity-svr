#!/usr/bin/env python3
"""SVGP: Sparse Variational Gaussian Process — uncertainty quantification
for the full 82K dataset. Uses ~500 inducing points to keep training O(nm²).

Unlike ExactGP (O(n³), limited to ~10K samples), SVGP scales to the full dataset
and gives principled posterior uncertainty σ(ε) at every prediction point.

Usage:
    python stage1/train_svgp.py --max-samples 10000     # quick test (~5 min)
    python stage1/train_svgp.py --max-samples 90000     # full dataset (~30 min)
    python stage1/train_svgp.py --max-samples 90000 --inducing 1000
"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np, pandas as pd
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Data ──────────────────────────────────────────────────
def load_data(max_samples=90000):
    csv_path = ROOT / "stage1" / "simulation_data" / "tmm_emissivity_data.csv"
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    df.rename(columns={cols[0]:"wl", cols[1]:"sub", cols[2]:"pdms",
                       cols[3]:"R", cols[4]:"T", cols[5]:"eps"}, inplace=True)
    df = df[np.abs(df["R"] + df["T"] + df["eps"] - 1.0) <= 1e-3]
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    # 13-dim engineered features (matching SVR exactly)
    wl = df["wl"].values.astype(np.float64)
    sio = df["sub"].values.astype(np.float64)
    pdms = df["pdms"].values.astype(np.float64)
    n_pdms, n_sio2 = 1.41, 1.46
    wl_m, pm, sm = wl * 1e-6, pdms * 1e-9, sio * 1e-9
    dp = 2 * np.pi * n_pdms * pm / wl_m
    ds = 2 * np.pi * n_sio2 * sm / wl_m
    X = np.column_stack([wl, sio, pdms,
        np.sin(dp), np.cos(dp), np.sin(ds), np.cos(ds),
        1.0 / wl, pdms / wl,
        np.sin(dp) * np.sin(ds), np.cos(dp) * np.cos(ds),
        np.sin(dp) * np.cos(ds), np.cos(dp) * np.sin(ds)])
    X = np.nan_to_num(X, 0)
    y = df["eps"].values.astype(np.float64)
    groups = df["pdms"].values
    return X, y, groups, len(df)


# ── SVGP Model ────────────────────────────────────────────
class SVGPModel(gpytorch.models.ApproximateGP):
    """Sparse Variational GP with RBF kernel and ~500 inducing points."""
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ── Training ──────────────────────────────────────────────
def train_svgp(model, likelihood, train_x, train_y, n_iter=200, lr=0.05,
               verbose=True):
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": lr},
        {"params": likelihood.parameters(), "lr": lr * 0.1},
    ])
    # Variational ELBO (evidence lower bound) — the SVGP loss
    mll = gpytorch.mlls.VariationalELBO(
        likelihood, model, num_data=train_y.size(0))

    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if verbose and (i % 50 == 0 or i == n_iter - 1):
            length_scale = model.covar_module.base_kernel.lengthscale.item()
            print(f"  [{i:3d}/{n_iter}] loss={loss.item():.3f}  "
                  f"length_scale={length_scale:.3f}  "
                  f"noise={likelihood.noise.item():.4f}")
    return model, likelihood


# ── Main ──────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=10000)
    ap.add_argument("--inducing", type=int, default=500,
                    help="Number of inducing points (more = better, slower)")
    ap.add_argument("--n-iter", type=int, default=200,
                    help="Training iterations")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--output", type=str, default="")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[DEVICE] {device}")

    # Load
    X, y, groups, n_total = load_data(args.max_samples)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)

    train_x = torch.tensor(X_tr, dtype=torch.float32, device=device)
    train_y = torch.tensor(y_tr, dtype=torch.float32, device=device)
    test_x = torch.tensor(X_te, dtype=torch.float32, device=device)
    test_y = torch.tensor(y_te, dtype=torch.float32, device=device)

    print(f"[DATA] {n_total} total, {len(train_x)} train, {len(test_x)} test, "
          f"{X_tr.shape[1]} features (13-dim engineered)")
    print(f"[SVGP] {args.inducing} inducing points, O(nm²) ≈ "
          f"{len(train_x) * args.inducing ** 2 / 1e6:.0f}M ops/iter")

    # Select inducing points: random subset of training data
    rng = np.random.RandomState(42)
    n_ind = min(args.inducing, len(X_tr))
    inducing_idx = rng.choice(len(X_tr), n_ind, replace=False)
    inducing_points = train_x[torch.tensor(inducing_idx, dtype=torch.long)].clone()

    # Build model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    likelihood.noise = 0.01
    model = SVGPModel(inducing_points).to(device)
    print(f"[MODEL] SVGP: {sum(p.numel() for p in model.parameters()):,} params "
          f"({n_ind} inducing)")

    # Train
    t0 = time.perf_counter()
    print(f"\nTraining {args.n_iter} iterations...")
    model, likelihood = train_svgp(model, likelihood, train_x, train_y,
                                   n_iter=args.n_iter, lr=args.lr)
    train_time = time.perf_counter() - t0
    print(f"  done in {train_time:.0f}s")

    # Predict
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(test_x))
        y_pred = preds.mean.cpu().numpy()
        y_pred = np.clip(y_pred, 0, 1)
        y_lower, y_upper = preds.confidence_region()
        y_lower = y_lower.cpu().numpy()
        y_upper = y_upper.cpu().numpy()

    # Metrics
    r2 = float(r2_score(y_te, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    mae = float(mean_absolute_error(y_te, y_pred))

    # Uncertainty calibration
    residuals = np.abs(y_te - y_pred)
    sigma_mean = float(preds.stddev.mean().cpu().numpy())
    in_1sigma = float((residuals < sigma_mean).mean())
    in_2sigma = float((residuals < 2 * sigma_mean).mean())

    print(f"\n  R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    print(f"  σ_mean={sigma_mean:.4f}")
    print(f"  Points within 1σ: {in_1sigma:.1%} (target: 68%)")
    print(f"  Points within 2σ: {in_2sigma:.1%} (target: 95%)")
    calib_1 = "GOOD" if 0.60 < in_1sigma < 0.76 else ("OVER" if in_1sigma > 0.76 else "UNDER")
    calib_2 = "GOOD" if 0.90 < in_2sigma < 0.98 else ("OVER" if in_2sigma > 0.98 else "UNDER")
    print(f"  1σ calibration: {calib_1}  2σ calibration: {calib_2}")

    # Save
    out = {
        "meta": {
            "n_train": len(train_x), "n_test": len(test_x),
            "n_inducing": n_ind, "n_iter": args.n_iter, "lr": args.lr,
            "train_time_s": round(train_time, 1),
        },
        "results": {
            "r2": r2, "rmse": rmse, "mae": mae,
            "sigma_mean": round(sigma_mean, 4),
            "in_1sigma": round(in_1sigma, 4),
            "in_2sigma": round(in_2sigma, 4),
            "calibration_1sigma": calib_1,
            "calibration_2sigma": calib_2,
        },
        "model": {
            "params": sum(p.numel() for p in model.parameters()),
            "inducing_points": n_ind,
            "length_scale": float(model.covar_module.base_kernel.lengthscale.item()),
            "output_scale": float(model.covar_module.outputscale.item()),
            "noise": float(likelihood.noise.item()),
        },
    }

    out_path = Path(args.output) if args.output else (
        ROOT / "stage1" / "simulation_data" / "svgp_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Save model
    model_path = ROOT / "stage1" / "simulation_data" / "svgp_model.pth"
    torch.save({
        "model_state": model.state_dict(),
        "likelihood_state": likelihood.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }, model_path)

    print(f"\n[OK] Results → {out_path}")
    print(f"[OK] Model → {model_path}")

    # Compare with SVR
    tres_path = ROOT / "stage1" / "simulation_data" / "test_results.json"
    if tres_path.exists():
        tres = json.load(open(tres_path))
        svr_r2 = tres["r2"]
        delta = r2 - svr_r2
        print(f"\n  SVR (production) R2 = {svr_r2:.4f}")
        print(f"  SVGP R2              = {r2:.4f}")
        print(f"  Delta (SVGP − SVR)   = {delta:+.4f} "
              f"({'↑' if delta > 0 else '↓'} + uncertainty now available)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
