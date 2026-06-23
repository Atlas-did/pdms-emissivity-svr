#!/usr/bin/env python3
"""Ablation study: compare model variants. With progress indicators.

Key flags:
  --max-samples N      : total data for final evaluation (default 8000)
  --search-samples N   : subset for Halton search ONLY (default 5000)
                          Halton search is O(n^2) - NEVER use full data!
  --n-iter N           : Halton search trials (default 10)
  --quick              : ultra-fast: n_iter=5, search_samples=2000, small C range

Why --search-samples exists:
  SVR.fit scales ~O(n^2) with training samples. 58,000 samples = ~5 hours/fit.
  Hyperparameter search on 3,500 training samples takes ~30s/fit instead.
  Best params found on subset are then evaluated on full data.
"""
from __future__ import annotations
import sys, os, json, time, argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def build_features(df):
    wl=df["wl"].values.astype(float); sio=df["sub"].values.astype(float)
    pdms=df["pdms"].values.astype(float)
    n_pdms,n_sio2=1.41,1.46; wl_m,pm,sm=wl*1e-6,pdms*1e-9,sio*1e-9
    dp=2*np.pi*n_pdms*pm/wl_m; ds=2*np.pi*n_sio2*sm/wl_m
    X=np.column_stack([wl,sio,pdms,
        np.sin(dp),np.cos(dp),np.sin(ds),np.cos(ds),
        1.0/wl,pdms/wl,
        np.sin(dp)*np.sin(ds),np.cos(dp)*np.cos(ds),
        np.sin(dp)*np.cos(ds),np.cos(dp)*np.sin(ds)])
    return np.nan_to_num(X,0)

def train_eval(X, y, groups, C=500, gamma=1.0, eps=0.01, label=""):
    gss=GroupShuffleSplit(n_splits=1,test_size=0.3,random_state=42)
    train_idx,test_idx=next(gss.split(X,y,groups))
    X_tr,X_te=X[train_idx],X[test_idx]; y_tr,y_te=y[train_idx],y[test_idx]
    scaler=StandardScaler()
    svr=SVR(kernel='rbf',C=C,gamma=gamma,epsilon=eps)
    t0=time.perf_counter()
    svr.fit(scaler.fit_transform(X_tr),y_tr)
    ts=time.perf_counter()-t0
    yp=np.clip(svr.predict(scaler.transform(X_te)),0,1)
    r2v=float(r2_score(y_te,yp))
    if label:
        print(f"  {label}: n_train={len(X_tr)}, n_test={len(X_te)}, "
              f"R2={r2v:.4f}, time={ts:.1f}s, SV={svr.support_.shape[0]}")
    return {"r2":r2v,"rmse":float(np.sqrt(mean_squared_error(y_te,yp))),
            "mae":float(mean_absolute_error(y_te,yp)),"train_time_s":round(ts,2),
            "n_train":len(X_tr),"n_test":len(X_te),
            "n_support_vectors":int(svr.support_.shape[0])}

def halton_search(X, y, groups, n_iter=10, quick=False):
    """Random search over C/gamma/eps grid. Use SUBSET data only!"""
    if quick:
        C_vals=[100,200,500]; gamma_vals=[0.05,0.1,0.5]; eps_vals=[0.001,0.01]
    else:
        C_vals=[100,200,500,1000,2000]; gamma_vals=[0.05,0.1,0.5,1.0]
        eps_vals=[0.001,0.005,0.01,0.05]

    n_train = int(len(y)*0.7)
    print(f"  Halton search: {n_iter} trials "
          f"(C={C_vals}, gamma={gamma_vals}, eps={eps_vals})...")
    print(f"  Training on {n_train} samples, testing on {len(y)-n_train}")
    if n_train > 10000:
        print(f"  WARNING: {n_train} training samples is SLOW for SVR fit.")
        print(f"  Consider --search-samples 5000 for ~30s/iteration instead.")

    gss=GroupShuffleSplit(n_splits=1,test_size=0.3,random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_tr_h, X_te_h = X[train_idx], X[test_idx]
    y_tr_h, y_te_h = y[train_idx], y[test_idx]
    scaler_h = StandardScaler()
    X_tr_hs = scaler_h.fit_transform(X_tr_h)
    X_te_hs = scaler_h.transform(X_te_h)

    rng=np.random.RandomState(42)
    best_score=-np.inf; best_params=None
    t0=time.perf_counter()
    for i in range(n_iter):
        C=rng.choice(C_vals); gamma=rng.choice(gamma_vals); eps=rng.choice(eps_vals)
        tit=time.perf_counter()
        svr=SVR(kernel='rbf',C=C,gamma=gamma,epsilon=eps)
        svr.fit(X_tr_hs, y_tr_h)
        score=r2_score(y_te_h, np.clip(svr.predict(X_te_hs), 0, 1))
        ei=time.perf_counter()-tit
        mark=" *" if score>best_score else ""
        if score>best_score: best_score=score; best_params={"C":C,"gamma":gamma,"eps":eps}
        elapsed_i = time.perf_counter()-t0
        eta=(n_iter-i-1)*(elapsed_i/(i+1))
        elapsed_str=time.strftime("%M:%S",time.gmtime(elapsed_i)) if elapsed_i<3600 else time.strftime("%H:%M:%S",time.gmtime(elapsed_i))
        print(f"  [{i+1:2d}/{n_iter}] C={C:4d} gamma={gamma:.3f} eps={eps:.4f} "
              f"-> R2={score:.4f}{mark} ({ei:.0f}s, wall={elapsed_str}, ETA {eta:.0f}s)")
    return best_params,round(time.perf_counter()-t0,2),round(best_score,4)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-samples",type=int,default=8000,
                    help="Total samples for final evaluation (default 8000)")
    ap.add_argument("--search-samples",type=int,default=5000,
                    help="Subset for Halton search ONLY (default 5000, keep <10000)")
    ap.add_argument("--n-iter",type=int,default=10,
                    help="Halton search iterations (default 10)")
    ap.add_argument("--quick",action="store_true",
                    help="Ultra-fast: n_iter=5, search_samples=2000, small C range")
    ap.add_argument("--output",type=str,default="")
    args=ap.parse_args()

    n_iter = 5 if args.quick else args.n_iter
    search_n = 2000 if args.quick else args.search_samples

    csv_path=ROOT/"stage1"/"simulation_data"/"tmm_emissivity_data.csv"
    if not csv_path.exists():
        print(f"[FAIL] {csv_path}"); return 1
    df=pd.read_csv(csv_path)
    cols=list(df.columns)
    df.rename(columns={cols[0]:"wl",cols[1]:"sub",cols[2]:"pdms",cols[3]:"R",cols[4]:"T",cols[5]:"eps"},inplace=True)
    df["sum"]=df["R"]+df["T"]+df["eps"]
    df_clean=df[np.abs(df["sum"]-1.0)<=1e-3].copy()
    n_filtered=len(df)-len(df_clean)
    print(f"[LOAD] {len(df)} rows, {n_filtered} failed energy check, using {len(df_clean)}")

    # ---- Eval data (full) ----
    df_eval = df_clean
    if len(df_eval) > args.max_samples:
        df_eval = df_eval.sample(n=args.max_samples, random_state=42)
    X_eval=build_features(df_eval); y_eval=df_eval["eps"].values.astype(float)
    groups_eval=df_eval["pdms"].values
    print(f"  Eval set: {len(df_eval)} samples")

    # ---- Search data (subset for Halton) ----
    search_n = min(search_n, len(df_clean))
    if search_n < len(df_clean):
        df_search = df_clean.sample(n=search_n, random_state=123)
    else:
        df_search = df_clean
    X_search=build_features(df_search); y_search=df_search["eps"].values.astype(float)
    groups_search=df_search["pdms"].values
    print(f"  Search set: {len(df_search)} samples (Halton search only)")

    results={}
    print(f"\n{'='*60}\n  Ablation Study\n  Eval: {len(df_eval)} samples | Search: {len(df_search)} samples\n{'='*60}")

    # A1: Full model
    print("\n[A1] Full model (Halton search on SEARCH set, eval on EVAL set)...")
    bp,ss,bs=halton_search(X_search, y_search, groups_search, n_iter, quick=args.quick)
    C_bp=bp.get("C",500); g_bp=bp.get("gamma",1.0); e_bp=bp.get("eps",0.01)
    m1=train_eval(X_eval, y_eval, groups_eval, C=C_bp, gamma=g_bp, eps=e_bp, label="  final eval")
    results["A1_full_model"]={"description":"Halton search + energy filter (baseline)",
        "hyperparams":bp,"search_time_s":ss,"search_samples":search_n,**m1}
    print(f"  => R2={m1['r2']:.4f}, RMSE={m1['rmse']:.4f}, search={ss}s")

    # A2: No energy filter
    print("\n[A2] No energy filter (all data, same params)...")
    df_all=df.copy()
    if len(df_all)>args.max_samples: df_all=df_all.sample(n=args.max_samples,random_state=42)
    X_all=build_features(df_all); y_all=df_all["eps"].values.astype(float)
    groups_all=df_all["pdms"].values
    m2=train_eval(X_all,y_all,groups_all,C=C_bp,gamma=g_bp,eps=e_bp,label="  final eval")
    results["A2_no_energy_filter"]={"description":f"No energy filter ({n_filtered} extra pts)",
        "extra_samples":n_filtered,**m2}
    print(f"  => R2={m2['r2']:.4f} (delta vs A1: {m2['r2']-m1['r2']:+.4f})")

    # A3: Random split
    print("\n[A3] Random split (NOT thickness-grouped) -- data leakage demo...")
    from sklearn.model_selection import train_test_split
    X_tr3,X_te3,y_tr3,y_te3=train_test_split(X_eval,y_eval,test_size=0.3,random_state=42)
    sc3=StandardScaler()
    svr3=SVR(kernel="rbf",C=C_bp,gamma=g_bp,epsilon=e_bp)
    t0=time.perf_counter()
    svr3.fit(sc3.fit_transform(X_tr3),y_tr3)
    ts3=time.perf_counter()-t0
    yp3=np.clip(svr3.predict(sc3.transform(X_te3)),0,1)
    m3={"r2":float(r2_score(y_te3,yp3)),
        "rmse":float(np.sqrt(mean_squared_error(y_te3,yp3))),
        "mae":float(mean_absolute_error(y_te3,yp3)),
        "train_time_s":round(ts3,2)}
    results["A3_random_split"]={"description":"Random split (inflated R2, NOT comparable)",
        "note":"Data leakage from adjacent wavelengths",**m3}
    print(f"  => R2={m3['r2']:.4f} (inflated by data leakage vs A1={m1['r2']:.4f})")

    # A4: Grid search
    print("\n[A4] Grid search (3x3x2=18 combos)...")
    t0=time.perf_counter()
    pg={"C":[100,500,1000],"gamma":[0.1,0.5,1.0],"epsilon":[0.001,0.01]}
    gss4=GroupShuffleSplit(n_splits=1,test_size=0.3,random_state=42)
    train_idx4,test_idx4=next(gss4.split(X_eval,y_eval,groups_eval))
    X_tr4,X_te4=X_eval[train_idx4],X_eval[test_idx4]; y_tr4,y_te4=y_eval[train_idx4],y_eval[test_idx4]
    scaler4=StandardScaler()
    X_tr4s=scaler4.fit_transform(X_tr4)
    gs=GridSearchCV(SVR(kernel='rbf'),pg,cv=2,scoring='r2',n_jobs=1,verbose=1)
    print(f"  Fitting {len(pg['C'])}x{len(pg['gamma'])}x{len(pg['epsilon'])}=18 combos x 2-fold CV = 36 fits...")
    gs.fit(X_tr4s,y_tr4)
    gs_time=round(time.perf_counter()-t0,2)
    yp4=np.clip(gs.best_estimator_.predict(scaler4.transform(X_te4)),0,1)
    m4={"best_params":gs.best_params_,"search_time_s":gs_time,
        "r2":float(r2_score(y_te4,yp4)),
        "rmse":float(np.sqrt(mean_squared_error(y_te4,yp4))),
        "mae":float(mean_absolute_error(y_te4,yp4))}
    results["A4_grid_search"]={"description":"Grid search (18 combos) instead of Halton",**m4}
    print(f"  => R2={m4['r2']:.4f}, search={gs_time}s, params={gs.best_params_}")

    # A5: Reference from test_results
    print("\n[A5] Reference: production SVR from test_results.json...")
    tres_path=ROOT/"stage1"/"simulation_data"/"test_results.json"
    if tres_path.exists():
        tres=json.load(open(tres_path))
        results["A5_global_svr_reference"]={
            "description":"Global SVR (paper Table 2, production model)",
            "r2":tres["r2"],"rmse":tres["rmse"],"mae":tres["mae"],
            "split_protocol":tres["split_protocol"]}
        print(f"  => R2={tres['r2']:.4f}, RMSE={tres['rmse']:.4f}, MAE={tres['mae']:.4f}")
    else:
        print(f"  [SKIP] {tres_path} not found")

    # Save
    out_path=Path(args.output) if args.output else ROOT/"stage1"/"simulation_data"/"ablation_results.json"
    out_path.parent.mkdir(parents=True,exist_ok=True)
    with open(out_path,'w',encoding='utf-8') as f:
        json.dump(results,f,indent=2,ensure_ascii=False)
    print(f"\n[OK] Saved to {out_path}")

    print(f"\n{'='*60}\n  Summary\n{'='*60}")
    print(f"  {'Configuration':<35} {'R2':>7}  {'RMSE':>7}  {'MAE':>7}")
    print(f"  {'-'*35} {'-'*7}  {'-'*7}  {'-'*7}")
    for k,r in results.items():
        if "r2" in r:
            print(f"  {r['description'][:35]:<35} {r['r2']:7.4f}  {r.get('rmse',0):7.4f}  {r.get('mae',0):7.4f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
