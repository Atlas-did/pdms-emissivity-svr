from __future__ import annotations

import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from stage1.core.io_utils import save_json
from stage1.core.model_wrapper import ModelWrapper
from stage1.core.observability import export_efficiency_decomposition, write_performance_summary
from stage1.core.performance import monitor_performance
from stage1.core.paths import (
    MODEL_BUNDLE_META_JSON,
    MODEL_BUNDLE_PKL,
    MODEL_INFO_JSON,
    MODEL_PKL,
    OUTDIR,
    SCALER_PKL,
    SPEED_RESULTS_JSON,
    TEST_RESULTS_JSON,
    Y_SCALER_PKL,
)
from stage1.experiment_tracker import ExperimentTracker


class SVRModelEvaluator:
    """模型评估与结果落盘。"""

    @staticmethod
    def _get_run_context_best_effort(trainer: "SVRTrainer") -> Optional[Dict[str, Any]]:
        rc = trainer.config.params.get("_run_context")
        return rc if isinstance(rc, dict) else None

    @staticmethod
    def evaluate(trainer: "SVRTrainer", X_test: np.ndarray, y_test: np.ndarray, data: pd.DataFrame):
        rc = SVRModelEvaluator._get_run_context_best_effort(trainer)
        run_id = str((rc or {}).get("manifest_run_id") or trainer.config.params.get("_run_id", ""))
        y_pred_model = trainer.model.predict(X_test)
        if trainer.y_scaler is not None:
            y_pred = trainer.y_scaler.inverse_transform(y_pred_model.reshape(-1, 1)).ravel()
        else:
            y_pred = y_pred_model
        y_pred = np.clip(y_pred, 0.0, 1.0)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        test_results = {
            "run_id": run_id,
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        }
        save_json(TEST_RESULTS_JSON, test_results, run_context=rc)

        model_info = {
            "run_id": run_id,
            "best_params": trainer.best_params_,
            "best_score": float(trainer.best_score_ if trainer.best_score_ is not None else np.nan),
            "used_y_scaler": bool(trainer.y_scaler is not None),
            "y_scaler_file": os.path.basename(Y_SCALER_PKL) if trainer.y_scaler is not None else "",
            "param_importance": trainer.param_importance_ or {},
            "model_version": trainer.config.params.get("model_version", "svr-v1"),
        }
        save_json(MODEL_INFO_JSON, model_info, run_context=rc)

        if trainer.tracker is not None:
            trainer.tracker.log_metrics({
                "r2": float(r2),
                "rmse": float(rmse),
                "mae": float(mae),
                "best_cv": float(trainer.best_score_ if trainer.best_score_ is not None else np.nan),
            })

        print(f"Evaluation done: R2={r2:.4f}, RMSE={rmse:.6f}, MAE={mae:.6f}")
        return r2, rmse, mae


class SVRSpeedBenchmarker:
    """推理速度基准测试。"""

    @staticmethod
    def _get_cpu_name_best_effort() -> str:
        """Return a human-readable CPU model name when available.

        On Windows, prefer the registry ProcessorNameString. Else fall back to
        platform.processor(), which may be less descriptive.
        """
        try:
            if os.name == "nt":
                import winreg  # type: ignore

                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                ) as key:
                    value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                    if isinstance(value, str):
                        return value.strip()
        except Exception:
            pass
        try:
            return (platform.processor() or "").strip()
        except Exception:
            return ""

    @staticmethod
    def benchmark(trainer: "SVRTrainer", n_spectra: int = 100):
        print("\n[SVRTrainer] Benchmarking speed...")
        rc = trainer.config.params.get("_run_context") if isinstance(trainer.config.params.get("_run_context"), dict) else None
        run_id = str((rc or {}).get("manifest_run_id") or trainer.config.params.get("_run_id", ""))
        rng = np.random.RandomState(trainer.config.params.get("random_seed", 42))
        lam_min = trainer.config.params.get("lambda_min", 0.3)
        lam_max = trainer.config.params.get("lambda_max", 25.0)
        lam_step = trainer.config.params.get("lambda_step", 0.1)
        wavelengths = np.arange(lam_min, lam_max + lam_step, lam_step)
        n_wl = len(wavelengths)
        pdms_thicknesses = rng.uniform(100.0, trainer.config.params.get("max_thickness", 1000.0), size=n_spectra)
        sio2_fixed = 1000.0

        start = time.time()
        for d in pdms_thicknesses:
            df = pd.DataFrame({
                "波长_μm": wavelengths,
                "基底厚度_nm": np.full(n_wl, sio2_fixed),
                "PDMS厚度_nm": np.full(n_wl, d),
            })
            X = trainer._build_features(df)
            Xs = trainer.scaler.transform(X)
            _ = trainer.model.predict(Xs)
        elapsed = time.time() - start
        svr_time_per_spectrum = elapsed / n_spectra
        tmm_time_per_spectrum = float(trainer.config.params.get("tmm_baseline_time_s", 12.3))
        speedup = tmm_time_per_spectrum / (svr_time_per_spectrum + 1e-12)

        baseline_points = trainer.config.params.get("tmm_baseline_wavelength_points", None)
        baseline_hw = trainer.config.params.get("tmm_baseline_hardware", "")
        speed_results = {
            "run_id": run_id,
            "svr_time_per_spectrum_s": float(svr_time_per_spectrum),
            "tmm_time_per_spectrum_s": float(tmm_time_per_spectrum),
            "speedup": float(speedup),
            "benchmark_n_spectra": int(n_spectra),
            "benchmark_elapsed_s": float(elapsed),
            "benchmark_lambda_min_um": float(lam_min),
            "benchmark_lambda_max_um": float(lam_max),
            "benchmark_lambda_step_um": float(lam_step),
            "benchmark_n_wavelength_points": int(n_wl),
            "tmm_baseline_wavelength_points": int(baseline_points) if baseline_points is not None else None,
            "tmm_baseline_hardware": str(baseline_hw),
            "system_cpu_name": SVRSpeedBenchmarker._get_cpu_name_best_effort(),
            "system_cpu_count": int(os.cpu_count() or 0),
            "system_platform": platform.platform(),
            "system_python": sys.version.split()[0],
        }
        save_json(SPEED_RESULTS_JSON, speed_results, run_context=rc)
        if trainer.tracker is not None:
            trainer.tracker.log_metrics({
                "svr_time_per_spectrum_s": float(svr_time_per_spectrum),
                "speedup": float(speedup)
            })
        print(f"[SVRTrainer] SVR {svr_time_per_spectrum:.4f}s/spectrum, speedup {speedup:.1f}x")
        return speedup, svr_time_per_spectrum, tmm_time_per_spectrum


class SVRExtrapolationEvaluator:
    """厚度外推测试。"""

    @staticmethod
    def run(trainer: "SVRTrainer", df: pd.DataFrame,
            train_min_nm: int = None, train_max_nm: int = None,
            test_min_nm: int = None, test_max_nm: int = None):
        p = trainer.config.params
        if train_min_nm is None:
            train_min_nm = p.get("extrap_train_thickness_min", 100)
        if train_max_nm is None:
            train_max_nm = p.get("extrap_train_thickness_max", 700)
        if test_min_nm is None:
            test_min_nm = p.get("extrap_test_thickness_min", 750)
        if test_max_nm is None:
            test_max_nm = p.get("extrap_test_thickness_max", 1000)

        required_cols = {"波长_μm", "基底厚度_nm", "PDMS厚度_nm", "发射率ε"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise KeyError(f"DataFrame missing required columns for extrapolation: {missing}")

        train_mask = (df["PDMS厚度_nm"] >= train_min_nm) & (df["PDMS厚度_nm"] <= train_max_nm)
        test_mask = (df["PDMS厚度_nm"] >= test_min_nm) & (df["PDMS厚度_nm"] <= test_max_nm)
        df_train = df.loc[train_mask].copy()
        df_test = df.loc[test_mask].copy()

        if len(df_train) < 50 or len(df_test) < 50:
            raise ValueError(f"Training or test subset too small: train={len(df_train)}, test={len(df_test)}")

        X_train = df_train[["波长_μm", "基底厚度_nm", "PDMS厚度_nm"]].values.astype(float)
        y_train = df_train["发射率ε"].values.astype(float)
        X_test = df_test[["波长_μm", "基底厚度_nm", "PDMS厚度_nm"]].values.astype(float)
        y_test = df_test["发射率ε"].values.astype(float)

        scaler_local = StandardScaler().fit(X_train)
        X_train_s = scaler_local.transform(X_train)
        X_test_s = scaler_local.transform(X_test)

        svr = SVR(kernel="rbf")
        search = RandomizedSearchCV(
            svr,
            {"C": p["C_values"], "gamma": p["gamma_values"], "epsilon": p["epsilon_values"]},
            n_iter=min(20, p.get("random_search_iter", 10)),
            cv=min(5, p.get("cv_folds", 3)),
            scoring="r2",
            n_jobs=trainer._safe_parallel_n_jobs(p.get("n_jobs", -1)),
            random_state=p.get("random_seed", 42)
        )
        search.fit(X_train_s, y_train)
        model = search.best_estimator_
        y_pred = model.predict(X_test_s)
        y_pred = np.clip(y_pred, 0.0, 1.0)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        out = {
            "run_id": str((trainer.config.params.get("_run_context") or {}).get("manifest_run_id") or trainer.config.params.get("_run_id", ""))
            if isinstance(trainer.config.params.get("_run_context"), dict) else str(trainer.config.params.get("_run_id", "")),
            "train_range_nm": [train_min_nm, train_max_nm],
            "test_range_nm": [test_min_nm, test_max_nm],
            "r2": float(r2), "rmse": float(rmse), "mae": float(mae),
            "best_params": search.best_params_
        }
        fname = os.path.join(
            OUTDIR,
            f"extrap_thickness_{train_min_nm}-{train_max_nm}__{test_min_nm}-{test_max_nm}.json"
        )
        rc = trainer.config.params.get("_run_context") if isinstance(trainer.config.params.get("_run_context"), dict) else None
        save_json(fname, out, run_context=rc)
        if trainer.tracker is not None:
            trainer.tracker.log_metrics({
                "extrap_r2": float(r2),
                "extrap_rmse": float(rmse),
                "extrap_mae": float(mae),
            })
        print(f"[SVRTrainer] Thickness extrapolation done: R2={r2:.4f}")
        return out


class SVRTrainer:
    """SVR trainer with optional target scaling, compatible aliases and utility methods."""

    def __init__(self, config: Any):
        self.config = config
        self.scaler: StandardScaler = StandardScaler()
        self.y_scaler: Optional[StandardScaler] = None
        self.model: Optional[SVR] = None
        self.mlp: Optional[MLPRegressor] = None
        self.best_params_: Optional[Dict] = None
        self.best_score_: Optional[float] = None
        self.is_trained: bool = False
        self.param_importance_: Optional[Dict[str, float]] = None
        self.evaluator = SVRModelEvaluator()
        self.speed_benchmarker = SVRSpeedBenchmarker()
        self.extrapolation_evaluator = SVRExtrapolationEvaluator()
        self.tracker = ExperimentTracker(
            enabled=bool(self.config.params.get("enable_experiment_tracking", False)),
            experiment_name=str(self.config.params.get("experiment_name", "pdms_emissivity")),
            tracking_uri=str(self.config.params.get("experiment_tracking_uri", "")),
            outdir=OUTDIR,
        )
        self.search_timing_: Dict[str, Any] = {}
        self._io_write_count: int = 0
        self._model_already_refit_full: bool = False

    def _safe_parallel_n_jobs(self, requested_n_jobs: Any) -> int:
        """在Windows下若venv基解释器失效，自动降级为单进程，避免WinError 3。"""
        try:
            n_jobs = int(requested_n_jobs)
        except Exception:
            n_jobs = 1
        if n_jobs == 1:
            return 1

        if os.name == "nt":
            base_exe = getattr(sys, "_base_executable", "") or ""
            if base_exe and not os.path.exists(base_exe):
                print(
                    "⚠️  检测到失效的基础解释器路径，已自动将n_jobs降为1。"
                    f" missing={base_exe}"
                )
                return 1
        return n_jobs

    @monitor_performance("svr.preprocess")
    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """清洗数据、构造特征并划分训练/测试集。"""
        df = df.dropna(subset=['发射率ε']).copy()
        df = df[(df['发射率ε'] >= 0) & (df['发射率ε'] <= 1)].reset_index(drop=True)

        X = self._build_features(df)
        y = df['发射率ε'].values.astype(float)
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test, split_note = self._split_train_test(X_scaled, y, df)
        print(f"[SVRTrainer] 数据划分策略: {split_note}")

        return X_train, X_test, y_train, y_test, df

    preprocess_data = preprocess

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """构造基础特征与物理相位特征。"""
        wl = df['波长_μm'].values.astype(float)
        d_sio2 = df['基底厚度_nm'].values.astype(float)
        d_pdms = df['PDMS厚度_nm'].values.astype(float)

        base = np.column_stack([wl, d_sio2, d_pdms])

        if not self.config.params.get("feature_engineering", True):
            return base

        n_pdms = float(self.config.params.get("n_pdms_eff", 1.41))
        n_sio2 = float(self.config.params.get("n_sio2_eff", 1.46))

        wl_m = wl * 1e-6
        pdms_m = d_pdms * 1e-9
        sio2_m = d_sio2 * 1e-9

        with np.errstate(divide='ignore', invalid='ignore'):
            delta_pdms = 2 * np.pi * n_pdms * pdms_m / wl_m
            delta_sio2 = 2 * np.pi * n_sio2 * sio2_m / wl_m
            inv_wl = 1.0 / wl
            d_over_wl = d_pdms / wl

        fea = np.column_stack([
            base,
            np.sin(delta_pdms), np.cos(delta_pdms),
            np.sin(delta_sio2), np.cos(delta_sio2),
            inv_wl, d_over_wl
        ])

        if self.config.params.get("use_interaction_features", True):
            inter = np.column_stack([
                np.sin(delta_pdms) * np.sin(delta_sio2),
                np.cos(delta_pdms) * np.cos(delta_sio2),
                np.sin(delta_pdms) * np.cos(delta_sio2),
                np.cos(delta_pdms) * np.sin(delta_sio2)
            ])
            fea = np.column_stack([fea, inter])

        fea = np.nan_to_num(fea, nan=0.0, posinf=0.0, neginf=0.0)
        return fea

    def _make_stratify_labels(self, df: pd.DataFrame) -> Optional[pd.Series]:
        try:
            n_bins_th = min(8, max(2, int(np.sqrt(df['PDMS厚度_nm'].nunique()))))
            n_bins_wl = min(8, max(2, int(np.sqrt(df['波长_μm'].nunique()))))
            th_bins = pd.qcut(df['PDMS厚度_nm'], q=n_bins_th, duplicates='drop')
            wl_bins = pd.qcut(df['波长_μm'], q=n_bins_wl, duplicates='drop')
            labels = th_bins.astype(str) + "|" + wl_bins.astype(str)
            if labels.value_counts().min() >= 2:
                return labels
        except Exception:
            return None
        return None

    def _split_train_test(self, X: np.ndarray, y: np.ndarray, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        test_size = self.config.params['test_size']
        random_seed = self.config.params.get("random_seed", 42)

        idx_all = np.arange(len(df))

        def _has_atm_window(test_indices: np.ndarray) -> bool:
            wl = df.iloc[test_indices]['波长_μm'].values
            return np.any((wl >= 8.0) & (wl <= 13.0))

        if self.config.params.get("group_split_by_thickness", True):
            groups = df['PDMS厚度_nm']
            if groups.nunique() >= 2:
                for k in range(6):
                    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed + k)
                    train_idx, test_idx = next(gss.split(idx_all, y, groups))
                    if _has_atm_window(test_idx):
                        return X[train_idx], X[test_idx], y[train_idx], y[test_idx], "GroupShuffleSplit by PDMS thickness"
                raise ValueError("GroupShuffleSplit未能覆盖大气窗口样本，请调整test_size或数据分布。")

        stratify_labels = self._make_stratify_labels(df)
        if stratify_labels is not None:
            try:
                train_idx, test_idx = train_test_split(
                    idx_all, test_size=test_size, random_state=random_seed, stratify=stratify_labels
                )
                if _has_atm_window(test_idx):
                    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], "Stratified split (binned thickness & wavelength)"
            except Exception:
                pass

        if self.config.params.get("allow_random_split", False):
            train_idx, test_idx = train_test_split(
                idx_all, test_size=test_size, random_state=random_seed
            )
            return X[train_idx], X[test_idx], y[train_idx], y[test_idx], "Random split (explicit fallback)"

        raise ValueError("无法安全划分训练/测试集（禁止随机划分）。请检查分布或降低约束。")

    @monitor_performance("svr.train")
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        train_t0 = time.time()
        print("[SVRTrainer][1/5] 准备训练输入...")
        if self.tracker is not None:
            self.tracker.log_params({
                "search_method": "optuna" if self.config.params.get('use_bayesian_opt', False) else "traditional",
                "scale_y": bool(self.config.params.get('scale_y', False)),
                "cv_folds": int(self.config.params.get('cv_folds', 3)),
                "model_version": str(self.config.params.get("model_version", "svr-v1")),
            })

        if self.config.params.get('scale_y', False):
            from sklearn.preprocessing import StandardScaler as _SS
            self.y_scaler = _SS()
            y_for_fit = self.y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        else:
            self.y_scaler = None
            y_for_fit = y_train

        p = self.config.params
        if p.get("search_use_subset", False) and len(X_train) > int(p.get("search_max_samples", 8000)):
            rng = np.random.RandomState(p.get("random_seed", 42))
            idx = rng.choice(len(X_train), size=int(p.get("search_max_samples", 8000)), replace=False)
            X_search = X_train[idx]
            y_search = y_for_fit[idx]
        else:
            X_search = X_train
            y_search = y_for_fit

        search_t0 = time.time()
        method_name = "optuna" if self.config.params.get('use_bayesian_opt', False) else (
            "halton" if self.config.params.get("use_halton_search", False) and self.config.params.get("use_random_search", True) else (
                "random" if self.config.params.get('use_random_search', True) else "grid"
            )
        )
        print(f"[SVRTrainer][2/5] 开始超参数搜索... method={method_name} | search_samples={len(X_search)}")
        if self.config.params.get('use_bayesian_opt', False):
            search = self._train_with_optuna(X_search, y_search, X_train, y_for_fit)
        else:
            search = self._train_with_traditional_search(X_search, y_search)
        search_elapsed = time.time() - search_t0
        print(f"[SVRTrainer][2/5] 搜索完成，用时 {search_elapsed:.1f}s")

        refit_elapsed = 0.0
        print("[SVRTrainer][3/5] 模型定型阶段...")
        if self.best_params_ is not None and not self._model_already_refit_full:
            refit_t0 = time.time()
            print("[SVRTrainer] 正在用全量训练集重训最佳参数模型...")
            try:
                self.model = SVR(kernel='rbf', **self.best_params_)
                self.model.fit(X_train, y_for_fit)
            except Exception as e:
                print(f"⚠️  使用全量数据重训失败: {e}")
            refit_elapsed = time.time() - refit_t0
            print(f"[SVRTrainer] 全量重训完成，用时 {refit_elapsed:.1f}s")
        elif self._model_already_refit_full:
            print("[SVRTrainer] 当前搜索流程已完成全量拟合，跳过重复重训。")

        self.is_trained = True

        mlp_elapsed = 0.0
        print("[SVRTrainer][4/5] 可选基线模型阶段...")
        if self.config.params.get("train_mlp_baseline", False):
            mlp_t0 = time.time()
            self.mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                                    random_state=self.config.params.get("random_seed", 42))
            self.mlp.fit(X_train, y_train)
            mlp_elapsed = time.time() - mlp_t0

        self.search_timing_.update({
            "search_method": "optuna" if self.config.params.get('use_bayesian_opt', False) else (
                "halton" if self.config.params.get("use_halton_search", False) else (
                    "random" if self.config.params.get('use_random_search', True) else "grid"
                )
            ),
            "search_elapsed_s": float(search_elapsed),
            "refit_elapsed_s": float(refit_elapsed),
            "mlp_fit_elapsed_s": float(mlp_elapsed),
            "train_total_elapsed_s": float(time.time() - train_t0),
        })
        print(f"[SVRTrainer][5/5] 训练阶段完成，总耗时 {self.search_timing_['train_total_elapsed_s']:.1f}s。下一步：evaluate -> benchmark -> save")

        return search

    def _train_with_traditional_search(self, X_train: np.ndarray, y_for_fit: np.ndarray):
        if self.config.params.get("use_halton_search", False) and self.config.params.get("use_random_search", True):
            return self._train_with_halton_search(X_train, y_for_fit)
        self._model_already_refit_full = False
        svr = SVR(kernel='rbf')
        param_grid = {
            'C': self.config.params['C_values'],
            'gamma': self.config.params['gamma_values'],
            'epsilon': self.config.params['epsilon_values']
        }

        if self.config.params['use_random_search']:
            param_distributions = param_grid
            if self.config.params.get('use_param_distributions', True):
                try:
                    from scipy.stats import loguniform
                    param_distributions = {
                        'C': loguniform(1e1, 1e4),
                        'gamma': loguniform(1e-4, 1.0),
                        'epsilon': loguniform(1e-4, 1e-1)
                    }
                except Exception:
                    param_distributions = param_grid

            search = RandomizedSearchCV(
                svr, param_distributions,
                n_iter=self.config.params['random_search_iter'],
                cv=self.config.params['cv_folds'],
                scoring='r2',
                n_jobs=self._safe_parallel_n_jobs(self.config.params.get('n_jobs', -1)),
                random_state=self.config.params.get('random_seed', 42),
                verbose=1
            )
        else:
            search = GridSearchCV(
                svr, param_grid,
                cv=self.config.params['cv_folds'],
                scoring='r2',
                n_jobs=self._safe_parallel_n_jobs(self.config.params.get('n_jobs', -1)),
                verbose=1
            )

        try:
            search.fit(X_train, y_for_fit)
        except Exception as e:
            print(f"⚠️  传统搜索失败: {e}")
            if self.config.params.get("use_halton_search", False):
                return self._train_with_halton_search(X_train, y_for_fit)
            raise
        self.model = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_

        return search

    def _train_with_optuna(self, X_train: np.ndarray, y_for_fit: np.ndarray, X_full: np.ndarray, y_full: np.ndarray):
        try:
            import optuna
            from sklearn.model_selection import cross_val_score, train_test_split
            from sklearn.svm import SVR
        except ImportError as e:
            print(f"⚠️  Optuna导入失败: {e}")
            print("请安装Optuna: pip install optuna")
            print("将回退到搜索策略")
            if self.config.params.get("use_halton_search", False):
                return self._train_with_halton_search(X_train, y_for_fit)
            return self._train_with_traditional_search(X_train, y_for_fit)

        print("✅ 使用Optuna进行贝叶斯优化...")

        X_train_sub = X_train
        y_train_sub = y_for_fit

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_sub, y_train_sub,
            test_size=0.2,
            random_state=self.config.params.get("random_seed", 42)
        )

        def objective(trial):
            t0 = time.time()
            if self.config.params.get("optuna_verbose", True):
                print(f"[Optuna] Trial {trial.number + 1} start...", flush=True)
            C = trial.suggest_categorical('C', self.config.params.get("C_values", [100, 500, 2000, 5000]))
            gamma = trial.suggest_categorical('gamma', self.config.params.get("gamma_values", [0.1, 0.5, 1.0, 5.0, 10.0]))
            epsilon = trial.suggest_categorical('epsilon', self.config.params.get("epsilon_values", [0.001, 0.01]))

            svr = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)

            try:
                scores = cross_val_score(
                    svr, X_tr, y_tr,
                    cv=self.config.params['cv_folds'],
                    scoring='r2',
                    n_jobs=self._safe_parallel_n_jobs(self.config.params.get('optuna_cv_n_jobs', 1))
                )
                val = float(scores.mean())
                elapsed = time.time() - t0
                print(f"[Optuna] Trial {trial.number + 1} finished in {elapsed:.1f}s | R2={val:.6f}")
                if elapsed > float(self.config.params.get("trial_timeout_s", 60)):
                    print(f"⚠️  Trial耗时超过{self.config.params.get('trial_timeout_s', 60)}s，记为无效")
                    return -float('inf')
                return val
            except Exception as e:
                print(f"⚠️  Optuna评估失败: {e}")
                return -float('inf')

        total_trials = int(self.config.params.get("n_trials", 50) or 1)

        def _optuna_callback(study, trial):
            try:
                current = trial.number + 1
                progress = current * 100.0 / max(1, total_trials)
                print(
                    f"[Optuna] Trial {current} done ({progress:.1f}%) | "
                    f"value={trial.value:.6f} | best={study.best_value:.6f}"
                )
            except Exception:
                pass

        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(
                objective,
                n_trials=self.config.params.get('n_trials', 50),
                n_jobs=self.config.params.get('optuna_n_jobs', 1),
                show_progress_bar=True,
                timeout=self.config.params.get("search_timeout_s", self.config.params.get("optuna_timeout_s", None)),
                callbacks=[_optuna_callback]
            )
        except Exception as e:
            print(f"⚠️  Optuna优化失败: {e}")
            if self.config.params.get("use_halton_search", False):
                return self._train_with_halton_search(X_train, y_for_fit)
            return self._train_with_traditional_search(X_train, y_for_fit)

        best_params = study.best_params
        print(f"✅ Optuna优化完成，最佳参数: {best_params}")
        print(f"✅ 最佳CV分数: {study.best_value:.6f}")

        try:
            from optuna.importance import get_param_importances
            self.param_importance_ = get_param_importances(study)
            print(f"✅ 参数重要性: {self.param_importance_}")
        except Exception as e:
            print(f"⚠️  参数重要性分析失败: {e}")

        self.model = SVR(
            kernel='rbf',
            C=best_params['C'],
            gamma=best_params['gamma'],
            epsilon=best_params['epsilon']
        )
        self.model.fit(X_full, y_full)
        self._model_already_refit_full = True

        self.best_params_ = best_params
        self.best_score_ = study.best_value

        class PseudoSearch:
            def __init__(self, model, best_params, best_score):
                self.best_estimator_ = model
                self.best_params_ = best_params
                self.best_score_ = best_score

        return PseudoSearch(self.model, self.best_params_, self.best_score_)

    def _train_with_halton_search(self, X_train: np.ndarray, y_for_fit: np.ndarray):
        from sklearn.model_selection import cross_val_score

        def _first_primes(n: int) -> List[int]:
            primes = []
            candidate = 2
            while len(primes) < n:
                is_prime = True
                for p in primes:
                    if candidate % p == 0:
                        is_prime = False
                        break
                if is_prime:
                    primes.append(candidate)
                candidate += 1
            return primes

        def _halton_sequence(size: int, dim: int, scramble: bool, seed: int) -> np.ndarray:
            rng = np.random.RandomState(seed)
            bases = _first_primes(dim)
            seq = np.zeros((size, dim), dtype=float)
            for d, base in enumerate(bases):
                perm = None
                if scramble:
                    perm = np.arange(base)
                    if base > 1:
                        p = np.arange(1, base)
                        rng.shuffle(p)
                        perm[1:] = p
                for i in range(size):
                    f = 1.0
                    r = 0.0
                    idx = i + 1
                    while idx > 0:
                        f /= base
                        digit = idx % base
                        if perm is not None:
                            digit = perm[digit]
                        r += f * digit
                        idx //= base
                    seq[i, d] = r
            return seq

        p = self.config.params
        C_vals = list(map(float, p.get("C_values", [100, 500, 2000, 5000])))
        gamma_vals = list(map(float, p.get("gamma_values", [0.1, 0.5, 1.0, 5.0, 10.0])))
        eps_vals = list(map(float, p.get("epsilon_values", [0.001, 0.01])))
        n_iter = int(p.get("random_search_iter", 30))
        batch_size = int(p.get("batch_size", 100))
        early_stop = int(p.get("early_stopping_rounds", 5))
        cv_folds = int(p.get("cv_folds", 3))
        seed = int(p.get("random_seed", 42))
        scramble = bool(p.get("halton_scramble", True))

        max_samples = p.get("search_max_samples", p.get("halton_max_samples", None))
        if isinstance(max_samples, (int, float)) and max_samples and len(X_train) > int(max_samples):
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(X_train), size=int(max_samples), replace=False)
            X_train_sub = X_train[idx]
            y_train_sub = y_for_fit[idx]
        else:
            X_train_sub = X_train
            y_train_sub = y_for_fit

        halton = _halton_sequence(n_iter, 3, scramble, seed)

        def _pick(vals: List[float], u: float) -> float:
            idx = min(len(vals) - 1, int(u * len(vals)))
            return vals[idx]

        best_score = -np.inf
        best_params = None
        no_improve = 0
        trial_counter = 0
        timeout_count = 0

        total_trials = len(halton)
        search_start = time.time()
        search_timeout = float(p.get("search_timeout_s", 600))
        for i in range(0, n_iter, batch_size):
            if (time.time() - search_start) > search_timeout:
                print(f"[Halton] 超时停止：>{search_timeout}s")
                break
            batch = halton[i:i + batch_size]
            for u_c, u_g, u_e in batch:
                trial_counter += 1
                progress = trial_counter * 100.0 / max(1, total_trials)
                params = {
                    "C": _pick(C_vals, u_c),
                    "gamma": _pick(gamma_vals, u_g),
                    "epsilon": _pick(eps_vals, u_e)
                }
                try:
                    if p.get("halton_verbose", True):
                        print(f"[Halton] Trial {trial_counter}/{total_trials} ({progress:.1f}%) start: {params}", flush=True)
                    t0 = time.time()
                    svr = SVR(kernel='rbf', **params)
                    scores = cross_val_score(
                        svr, X_train_sub, y_train_sub,
                        cv=cv_folds, scoring='r2', n_jobs=self._safe_parallel_n_jobs(p.get("n_jobs", 1))
                    )
                    score = float(np.mean(scores))
                    elapsed = time.time() - t0
                    if elapsed > float(p.get("trial_timeout_s", 60)):
                        print(f"⚠️  Trial耗时超过{p.get('trial_timeout_s', 60)}s，记为无效")
                        score = -np.inf
                        timeout_count += 1
                    if p.get("halton_verbose", True):
                        print(f"[Halton] Trial {trial_counter}/{total_trials} ({progress:.1f}%) done | R2={score:.6f}")
                except Exception as e:
                    print(f"⚠️  Halton评估失败: {e}")
                    score = -np.inf

                if score > best_score:
                    best_score = score
                    best_params = params
                    no_improve = 0
                elif np.isfinite(score):
                    no_improve += 1

                if no_improve >= early_stop:
                    break
            if no_improve >= early_stop:
                print(f"[Halton] 早停触发: {early_stop} 轮无提升 (timeout={timeout_count})")
                break

        if best_params is None:
            raise RuntimeError("Halton搜索未找到有效参数")

        self.model = SVR(kernel='rbf', **best_params)
        self.model.fit(X_train, y_for_fit)
        self._model_already_refit_full = True
        self.best_params_ = best_params
        self.best_score_ = best_score

        class PseudoSearch:
            def __init__(self, model, best_params, best_score):
                self.best_estimator_ = model
                self.best_params_ = best_params
                self.best_score_ = best_score

        return PseudoSearch(self.model, self.best_params_, self.best_score_)

    train_model = train

    @monitor_performance("svr.evaluate")
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, data: pd.DataFrame):
        return self.evaluator.evaluate(self, X_test, y_test, data)

    evaluate_model = evaluate

    def save_model(self) -> None:
        rc = self.config.params.get("_run_context") if isinstance(self.config.params.get("_run_context"), dict) else None

        model_meta = {
            "run_id": str((rc or {}).get("manifest_run_id") or self.config.params.get("_run_id", "")),
            "model_version": str(self.config.params.get("model_version", "svr-v1")),
            "best_params": self.best_params_ if isinstance(self.best_params_, dict) else {},
            "best_score": float(self.best_score_) if self.best_score_ is not None else None,
            "used_y_scaler": bool(self.y_scaler is not None),
        }
        wrapper = ModelWrapper(model=self.model, scaler=self.scaler, y_scaler=self.y_scaler, metadata=model_meta)
        wrapper.save(MODEL_BUNDLE_PKL, MODEL_BUNDLE_META_JSON, run_context=rc)
        self._io_write_count += 1

        # 兼容旧版工件（避免破坏既有脚本）
        joblib.dump(self.model, MODEL_PKL)
        self._io_write_count += 1
        joblib.dump(self.scaler, SCALER_PKL)
        self._io_write_count += 1
        print(f"Saved model to {MODEL_PKL} and scaler to {SCALER_PKL}")
        if self.y_scaler is not None:
            joblib.dump(self.y_scaler, Y_SCALER_PKL)
            self._io_write_count += 1
            print(f"Saved y_scaler to {Y_SCALER_PKL}")

        perf_summary = {
            "run_id": str(self.config.params.get("_run_id", "")),
            "stage": "stage1",
            "module": "training",
            "io_write_count": int(self._io_write_count),
        }
        perf_summary.update(self.search_timing_)
        write_performance_summary(OUTDIR, "performance_summary_stage1_training", perf_summary, run_context=rc)
        export_efficiency_decomposition(
            OUTDIR,
            config_params=self.config.params,
            search_timing=self.search_timing_,
            run_context=rc,
        )

    @monitor_performance("svr.benchmark_speed")
    def benchmark_speed(self, n_spectra: int = 100):
        return self.speed_benchmarker.benchmark(self, n_spectra=n_spectra)

    def run_thickness_extrapolation(self, df: pd.DataFrame,
                                    train_min_nm: int = None, train_max_nm: int = None,
                                    test_min_nm: int = None, test_max_nm: int = None):
        return self.extrapolation_evaluator.run(
            self, df,
            train_min_nm=train_min_nm,
            train_max_nm=train_max_nm,
            test_min_nm=test_min_nm,
            test_max_nm=test_max_nm,
        )
