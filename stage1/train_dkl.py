#!/usr/bin/env python3
"""DKL: Deep Kernel Learning — replace hand-crafted 13 features + stationary RBF
with a NN feature extractor + learnable RBF kernel.

Key advantages over SVR:
- Non-stationary kernel: NN automatically learns different length scales for
  short-wavelength (rapid oscillations) vs long-wavelength (smooth) regions
- Automatic feature learning: no need for hand-crafted sin/cos/phase features
- Built-in uncertainty: GP posterior gives σ(ε) alongside ε̂

Usage:
    python stage1/train_dkl.py --max-samples 8000          # quick test
    python stage1/train_dkl.py --max-samples 5000          # medium (ExactGP O(n^3), keep <= 8000)
    # WARNING: >8000 samples will OOM with ExactGP. Use SVGP for large data.
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
    # Raw 3D input only — DKL learns the phase encoding automatically
    X_raw = df[["wl", "sub", "pdms"]].values.astype(np.float64)
    y = df["eps"].values.astype(np.float64)
    groups = df["pdms"].values
    return X_raw, y, groups, len(df)

# ── DKL Model ─────────────────────────────────────────────
class FeatureExtractor(torch.nn.Module):
    """Small MLP that maps 3D raw input → 32D latent feature space."""
    def __init__(self, input_dim=3, hidden_dims=(64, 128, 64), output_dim=32):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(h))
            prev = h
        layers.append(torch.nn.Linear(prev, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DKLModel(gpytorch.models.ExactGP):
    """Deep Kernel Learning: NN feature map + RBF kernel GP."""
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        # RBF kernel on the NN-transformed features
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feature_extractor.net[-1].out_features)
        )
        self.mean_module = gpytorch.means.ConstantMean()

    def forward(self, x):
        z = self.feature_extractor(x)  # NN feature transform
        mean_x = self.mean_module(z)
        covar_x = self.covar_module(z)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ── Training ──────────────────────────────────────────────
def train_dkl(model, likelihood, train_x, train_y, n_iter=200, lr=0.01, verbose=True):
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
        {"params": model.feature_extractor.parameters(), "lr": lr},
        {"params": model.covar_module.parameters(), "lr": lr * 0.1},
        {"params": model.mean_module.parameters(), "lr": lr * 0.1},
        {"params": likelihood.parameters(), "lr": lr * 0.1},
    ])
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if verbose and (i % 50 == 0 or i == n_iter - 1):
            length_scales = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
            print(f"  [{i:3d}/{n_iter}] loss={loss.item():.3f}  "
                  f"length_scale=[{length_scales.min():.2f}, {length_scales.max():.2f}]  "
                  f"noise={likelihood.noise.item():.4f}")
    return model, likelihood

# ── Main ──────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=8000,
                    help="Training samples (90000 for full)")
    ap.add_argument("--n-iter", type=int, default=200,
                    help="Training iterations")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--output", type=str, default="")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[DEVICE] {device}")

    # Load and split
    X_raw, y, groups, n_total = load_data(args.max_samples)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X_raw, y, groups))
    X_tr_raw, X_te_raw = X_raw[train_idx], X_raw[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)

    # Convert to torch
    train_x = torch.tensor(X_tr, dtype=torch.float32, device=device)
    train_y = torch.tensor(y_tr, dtype=torch.float32, device=device)
    test_x = torch.tensor(X_te, dtype=torch.float32, device=device)
    test_y = torch.tensor(y_te, dtype=torch.float32, device=device)

    print(f"[DATA] {n_total} total, {len(train_x)} train, {len(test_x)} test, "
          f"{X_tr.shape[1]} features (raw 3D input)")

    # Build model
    feature_extractor = FeatureExtractor(input_dim=3).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    likelihood.noise = 0.01  # initial guess

    model = DKLModel(train_x, train_y, likelihood, feature_extractor).to(device)
    print(f"[MODEL] Feature extractor: {sum(p.numel() for p in feature_extractor.parameters()):,} params")
    print(f"[MODEL] DKL: {sum(p.numel() for p in model.parameters()):,} total params")

    # Train
    t0 = time.perf_counter()
    print(f"\nTraining {args.n_iter} iterations...")
    model, likelihood = train_dkl(model, likelihood, train_x, train_y,
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
        y_std = preds.stddev.cpu().numpy()

    # Metrics
    r2 = float(r2_score(y_te, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    mae = float(mean_absolute_error(y_te, y_pred))
    print(f"\n  R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  "
          f"mean_uncertainty={y_std.mean():.4f}")

    # Get learned length scales
    length_scales = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
    print(f"  Learned length scales (per latent dim): "
          f"min={length_scales.min():.3f} max={length_scales.max():.3f} "
          f"mean={length_scales.mean():.3f}")
    print(f"  → Non-stationary: different latent dims use different length scales")

    # Save
    out = {
        "meta": {
            "n_train": len(train_x), "n_test": len(test_x),
            "n_iter": args.n_iter, "lr": args.lr,
            "train_time_s": round(train_time, 1),
            "input_features": ["wl_um", "sub_nm", "pdms_nm"],
        },
        "architecture": {
            "feature_extractor": "MLP(3→64→128→64→32)+ReLU+BatchNorm",
            "kernel": "RBF(ARD, 32 dims) × ScaleKernel",
            "params": sum(p.numel() for p in model.parameters()),
        },
        "results": {"r2": r2, "rmse": rmse, "mae": mae,
                    "mean_uncertainty": float(y_std.mean())},
        "length_scales": length_scales.tolist(),
    }

    out_path = Path(args.output) if args.output else (
        ROOT / "stage1" / "simulation_data" / "dkl_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    # Save model
    model_path = out_path.parent / "dkl_model.pth"
    torch.save({"model_state": model.state_dict(),
                "feature_extractor_state": feature_extractor.state_dict(),
                "likelihood_state": likelihood.state_dict(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist()}, model_path)
    print(f"\n[OK] Results -> {out_path}")
    print(f"[OK] Model -> {model_path}")

    # Compare with SVR reference
    tres_path = ROOT / "stage1" / "simulation_data" / "test_results.json"
    if tres_path.exists():
        tres = json.load(open(tres_path))
        svr_r2 = tres["r2"]
        delta = r2 - svr_r2
        print(f"\n  SVR (production) R2 = {svr_r2:.4f}")
        print(f"  DKL R2               = {r2:.4f}")
        print(f"  Delta (DKL − SVR)    = {delta:+.4f} "
              f"({'↑ improvement' if delta > 0 else '↓ no improvement'})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
