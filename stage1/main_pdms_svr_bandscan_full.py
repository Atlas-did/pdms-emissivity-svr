#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDMS薄膜发射率 TMM -> SVR 系统（带分区功能与厚度带批量扫描，完整版）
"""
from __future__ import annotations

import os
import json
import time
import argparse
import sys
import traceback
import random
import shutil
import threading
from datetime import datetime
from dataclasses import dataclass, field
from math import inf
from typing import Dict, Tuple, List, Optional, Any, Union

import joblib
import numpy as np
import pandas as pd
import tmm
from joblib import Parallel, delayed
from scipy.interpolate import CubicSpline, PchipInterpolator, RegularGridInterpolator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

if __package__ is None or __package__ == "":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(_HERE)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from stage1.core.config_runtime import DEFAULTS as CORE_DEFAULTS
from stage1.core.io_utils import (
    convert_numpy_types as core_convert_numpy_types,
    load_json as core_load_json,
    save_json as core_save_json,
)
from stage1.core.model_wrapper import ModelWrapper
from stage1.core.observability import (
    ErrorContext,
    append_session_log,
    log_exception_with_context,
    repair_stale_started_manifest,
    write_run_manifest,
)
from stage1.core.run_context import build_run_context
from stage1.core.simulation import TMMSimulator, TimedCache
from stage1.core.training import (
    SVRExtrapolationEvaluator as CoreSVRExtrapolationEvaluator,
    SVRModelEvaluator as CoreSVRModelEvaluator,
    SVRSpeedBenchmarker as CoreSVRSpeedBenchmarker,
    SVRTrainer as CoreSVRTrainer,
)
from stage1.core.paths import (
    BANDSCAN_SUMMARY,
    MATERIAL_INFO_JSON,
    MODEL_BUNDLE_PKL,
    MODEL_INFO_JSON,
    MODEL_PKL,
    OUTDIR,
    REGION_CONFIG_JSON,
    REGION_MODELS_DIR,
    REGION_SUMMARY_JSON,
    SCALER_PKL,
    SPEED_RESULTS_JSON,
    TEST_RESULTS_JSON,
    TMM_CSV,
    Y_SCALER_PKL,
)
from stage1.config_model import normalize_runtime_config, validate_runtime_config
from stage1.experiment_tracker import ExperimentTracker
from stage1.tmm_cpu_threaded import batch_simulate_tmm

# ==================== 1. 常量与路径 ====================
from pathlib import Path


# ==================== 2. 帮助函数 ====================

def convert_numpy_types(obj: Any) -> Any:
    """兼容层：转发到 stage1.core.io_utils.convert_numpy_types。"""
    return core_convert_numpy_types(obj)


_DEFAULT_RUN_CONTEXT: Any = None


def save_json(path: Any, obj: Any, run_context: Any = None) -> None:
    """兼容层：转发到 stage1.core.io_utils.save_json。"""
    if run_context is None:
        run_context = _DEFAULT_RUN_CONTEXT
    core_save_json(path, obj, run_context=run_context)


def load_json(path: Any) -> Dict:
    """兼容层：转发到 stage1.core.io_utils.load_json。"""
    return core_load_json(path)


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)



# ==================== 3. 默认配置 ====================
DEFAULTS = CORE_DEFAULTS.copy()


# ==================== 4. Config类 ====================
@dataclass
class Config:
    params: Dict = field(default_factory=lambda: DEFAULTS.copy())
    config_file: str = os.path.join(OUTDIR, "config.json")

    def save(self) -> None:
        self.validate()
        save_json(self.config_file, self.params)

    def load(self) -> None:
        if os.path.exists(self.config_file):
            self.params.update(load_json(self.config_file))  # 让异常向上传播，不再静默忽略
        self.validate()

    def validate(self) -> None:
        """配置校验入口：先做兼容性归一化，再做Pydantic严格校验。"""
        normalized = normalize_runtime_config(self.params)
        self.params = validate_runtime_config(normalized)


# ==================== 5. 交互配置器 ====================
class InteractiveConfigurator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _ask_int(self, prompt: str, default: int, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
        for _ in range(3):
            s = input(prompt).strip()
            if s == "":
                return default
            try:
                v = int(s)
                if (min_val is not None and v < min_val) or (max_val is not None and v > max_val):
                    print("输入超出范围，请重试。")
                    continue
                return v
            except Exception:
                print("输入无效，请重试。")
        print("多次输入无效，使用默认值。")
        return default

    def _ask_float(self, prompt: str, default: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
        for _ in range(3):
            s = input(prompt).strip()
            if s == "":
                return default
            try:
                v = float(s)
                if (min_val is not None and v < min_val) or (max_val is not None and v > max_val):
                    print("输入超出范围，请重试。")
                    continue
                return v
            except Exception:
                print("输入无效，请重试。")
        print("多次输入无效，使用默认值。")
        return default

    def run(self) -> None:
        p = self.cfg.params
        print("\n=== Interactive configuration ===")
        s = input(
            f"Choose search method (0=grid,1=random) [default {'1' if p['use_random_search'] else '0'}]: ").strip()
        if s == "0":
            p["use_random_search"] = False
        elif s == "1":
            p["use_random_search"] = True

        use_default = input("Use default hyperparameters? (y/n) [y]: ").strip().lower() or "y"
        if use_default != "y":
            try:
                p["C_values"] = list(map(float, input(
                    f"C values (space separated) [default {' '.join(map(str, p['C_values']))}]: ").split() or p[
                                             "C_values"]))
                p["gamma_values"] = list(map(float, input(
                    f"gamma values [default {' '.join(map(str, p['gamma_values']))}]: ").split() or p["gamma_values"]))
                p["epsilon_values"] = list(map(float, input(
                    f"epsilon values [default {' '.join(map(str, p['epsilon_values']))}]: ").split() or p[
                                                   "epsilon_values"]))
                p["random_search_iter"] = self._ask_int(
                    f"random search iter [default {p['random_search_iter']}]: ", p["random_search_iter"], 1, 500)
            except Exception:
                print("Invalid input — keeping defaults.")

        p["cv_folds"] = self._ask_int(f"CV folds [default {p['cv_folds']}]: ", p["cv_folds"], 2, 10)
        p["test_size"] = self._ask_float(f"Test size (0-1) [default {p['test_size']}]: ", p["test_size"], 0.05, 0.5)

        try:
            current_n_jobs = int(p.get("n_jobs", -1))
        except Exception:
            current_n_jobs = -1
        nj = input(f"n_jobs for parallel search (-1=all cores, 1=no parallel) [default {current_n_jobs}]: ").strip()
        if nj:
            try:
                val = int(nj)
                if val == 0:
                    print("0 is not valid for n_jobs, using 1 instead.")
                    val = 1
                p["n_jobs"] = val
            except Exception:
                print("Invalid n_jobs input, keeping previous value.")

        use_ellip = input("Enable ellipsometry processing? (y/n) [n]: ").strip().lower() or "n"
        p["enable_ellipsometry"] = (use_ellip == "y")
        if p["enable_ellipsometry"]:
            ellip_file = input(f"Ellipsometry file path [default {p['ellipsometry_file']}]: ").strip()
            if ellip_file:
                p["ellipsometry_file"] = ellip_file
            p["ellipsometry_angle_deg"] = self._ask_float(
                f"Incidence angle (deg) [default {p['ellipsometry_angle_deg']}]: ",
                float(p["ellipsometry_angle_deg"]), 30.0, 89.0
            )
            p["ellipsometry_n0"] = self._ask_float(
                f"Ambient refractive index n0 [default {p['ellipsometry_n0']}]: ",
                float(p["ellipsometry_n0"]), 0.5, 2.0
            )
            apply_target = input("Apply ellipsometry n/k to (none/pdms/sio2) [none]: ").strip().lower() or "none"
            if apply_target in {"none", "pdms", "sio2"}:
                p["ellipsometry_apply_to_material"] = apply_target
            else:
                print("Invalid choice, keep 'none'.")
                p["ellipsometry_apply_to_material"] = "none"

        print("\nConfiguration updated and saved.")
        self.cfg.save()


# ==================== 6. 材料数据加载器 ====================
class MaterialDataLoader:
    def __init__(self, pdms_file: str = "pdms_nk.xlsx", sio2_file: str = "sio2_nk.xlsx", config: Optional[Config] = None):
        self.pdms_file = pdms_file
        self.sio2_file = sio2_file
        self.config = config

    @staticmethod
    def _is_uniform_grid(lam: np.ndarray, rtol: float = 1e-3) -> bool:
        if len(lam) < 3:
            return True
        diffs = np.diff(lam)
        return np.allclose(diffs, diffs[0], rtol=rtol, atol=0)

    @staticmethod
    def _wrap_interpolator(interpolator):
        def _f(x):
            arr = np.atleast_1d(x).astype(float)
            vals = interpolator(arr.reshape(-1, 1))
            return vals[0] if np.isscalar(x) else vals
        return _f

    @staticmethod
    def _build_interpolators(lam: np.ndarray, n: np.ndarray, k: np.ndarray, config: Optional[Config]):
        use_uniform = MaterialDataLoader._is_uniform_grid(lam)
        resample_uniform = bool(config.params.get("resample_uniform_grid", True)) if config else False
        grid_step = float(config.params.get("material_grid_step", 0.02)) if config else None

        if resample_uniform and grid_step is not None:
            lam_uniform = np.arange(lam.min(), lam.max() + grid_step, grid_step)
            if use_uniform:
                base_interp_n = CubicSpline(lam, n, extrapolate=True)
                base_interp_k = CubicSpline(lam, k, extrapolate=True)
            else:
                base_interp_n = PchipInterpolator(lam, n, extrapolate=True)
                base_interp_k = PchipInterpolator(lam, k, extrapolate=True)
            n_uniform = base_interp_n(lam_uniform)
            k_uniform = base_interp_k(lam_uniform)
            n_interp = RegularGridInterpolator((lam_uniform,), n_uniform, bounds_error=False, fill_value=None)
            k_interp = RegularGridInterpolator((lam_uniform,), k_uniform, bounds_error=False, fill_value=None)
            n_spline = MaterialDataLoader._wrap_interpolator(n_interp)
            k_spline = MaterialDataLoader._wrap_interpolator(k_interp)
            return n_spline, k_spline, lam.min(), lam.max()

        if use_uniform:
            n_spline = CubicSpline(lam, n, extrapolate=True)
            k_spline = CubicSpline(lam, k, extrapolate=True)
        else:
            n_spline = PchipInterpolator(lam, n, extrapolate=True)
            k_spline = PchipInterpolator(lam, k, extrapolate=True)
        return n_spline, k_spline, lam.min(), lam.max()

    @staticmethod
    def _read_and_check(file_path: str) -> pd.DataFrame:
        """读取并验证材料数据文件，保持与原始版本兼容"""
        # 1) 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Material file not found: {file_path}")
        
        # 2) 读取Excel
        df = pd.read_excel(file_path)
        
        # 3) 识别列名映射
        candidates = {
            "lambda": ["波长 (μm)", "wavelength (um)", "wavelength (μm)", "lambda_um", "lambda (um)"],
            "n": ["折射率 (n)", "n", "refractive_index_n"],
            "k": ["消光系数 (k)", "k", "extinction_k"]
        }
        col_map = {}
        for key, opts in candidates.items():
            for opt in opts:
                if opt in df.columns:
                    col_map[key] = opt
                    break
        if len(col_map) != 3:
            raise ValueError(f"File {file_path} missing expected columns. Found: {df.columns.tolist()}")
        
        # 4) 统一列名
        df = df.rename(columns={col_map["lambda"]: "lambda_um", col_map["n"]: "n", col_map["k"]: "k"})
        df = df[["lambda_um", "n", "k"]]
        
        # 5) 过滤空行
        df = df.dropna(how='all')
        
        if df.empty:
            raise ValueError(f"材料数据文件没有有效数据: {file_path}")
        
        # 6) 类型转换
        df['lambda_um'] = df['lambda_um'].astype(float)
        df['n'] = df['n'].astype(float)
        df['k'] = df['k'].astype(float)
        
        # 7) k值修正（兼容实验噪声）
        df.loc[df['k'] < -0.01, 'k'] = 0.0
        
        # 8) 返回清洗后的数据
        return df

    def load_data(self, file_path: str, sample_interval: int = 1) -> Tuple[Any, Any, int, float, float]:
        df = self._read_and_check(file_path)
        if sample_interval > 1:
            df = df.iloc[::sample_interval].reset_index(drop=True)
        lam = df["lambda_um"].astype(float).values
        n = df["n"].astype(float).values
        k = df["k"].astype(float).values
        n_spline, k_spline, lam_min, lam_max = self._build_interpolators(lam, n, k, self.config)
        return n_spline, k_spline, len(lam), lam_min, lam_max

    def load_all(self, pdms_sample: int = 3, sio2_sample: int = 2) -> Dict:
        pdms_n, pdms_k, pdms_count, pdms_lam_min, pdms_lam_max = self.load_data(self.pdms_file, pdms_sample)
        sio2_n, sio2_k, sio2_count, sio2_lam_min, sio2_lam_max = self.load_data(self.sio2_file, sio2_sample)
        data = {
            "pdms_n": pdms_n, "pdms_k": pdms_k,
            "sio2_n": sio2_n, "sio2_k": sio2_k,
            "pdms_count": pdms_count, "sio2_count": sio2_count,
            "pdms_lam_min": pdms_lam_min, "pdms_lam_max": pdms_lam_max,
            "sio2_lam_min": sio2_lam_min, "sio2_lam_max": sio2_lam_max
        }

        if self.config and self.config.params.get("enable_ellipsometry", False):
            try:
                ellip = EllipsometryProcessor(self.config).load_and_convert()
                if ellip:
                    data.update(ellip)
                    apply_target = self.config.params.get("ellipsometry_apply_to_material", "none")
                    if apply_target == "pdms":
                        data["pdms_n"] = ellip["ellipsometry_n"]
                        data["pdms_k"] = ellip["ellipsometry_k"]
                        data["pdms_count"] = ellip["ellipsometry_count"]
                        data["pdms_lam_min"] = ellip["ellipsometry_lam_min"]
                        data["pdms_lam_max"] = ellip["ellipsometry_lam_max"]
                        print("✅ 已使用椭偏测量结果替换PDMS n/k")
                    elif apply_target == "sio2":
                        data["sio2_n"] = ellip["ellipsometry_n"]
                        data["sio2_k"] = ellip["ellipsometry_k"]
                        data["sio2_count"] = ellip["ellipsometry_count"]
                        data["sio2_lam_min"] = ellip["ellipsometry_lam_min"]
                        data["sio2_lam_max"] = ellip["ellipsometry_lam_max"]
                        print("✅ 已使用椭偏测量结果替换SiO2 n/k")
            except FileNotFoundError as e:
                print(f"⚠️ 椭偏数据未找到: {e}")
            except Exception as e:
                print(f"⚠️ 椭偏数据处理失败: {e}")
        return data


class EllipsometryProcessor:
    """椭圆偏振测量数据处理模块（加载/标准化/转换为n,k并插值）。"""

    COLUMN_CANDIDATES = {
        "wavelength_um": ["波长", "wavelength", "lambda", "wl", "λ", "lambda_um", "wavelength_um", "波长_um", "波长_μm", "波长 (μm)", "wavelength (um)", "wavelength (μm)"],
        "psi_deg": ["psi", "ψ", "Psi", "Psi (deg)", "psi_deg", "Ψ(°)", "椭偏角psi", "椭偏psi"],
        "delta_deg": ["delta", "Δ", "Delta", "Delta (deg)", "delta_deg", "Δ(°)", "椭偏角delta", "椭偏delta"],
        "angle_deg": ["angle", "incidence", "theta", "angle_deg", "AOI", "入射角", "incidence_angle", "theta_deg"]
    }

    def __init__(self, config: Config, file_path: Optional[str] = None):
        self.config = config
        self.file_path = file_path or str(config.params.get("ellipsometry_file", ""))

    def _read_file(self) -> pd.DataFrame:
        if not self.file_path:
            raise FileNotFoundError("未提供椭偏数据文件路径")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(self.file_path)
        ext = Path(self.file_path).suffix.lower()
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(self.file_path)
        if ext in {".csv", ".txt"}:
            return pd.read_csv(self.file_path)
        raise ValueError(f"不支持的椭偏数据文件格式: {ext}")

    @staticmethod
    def _match_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        lower_map = {str(c).lower(): c for c in df.columns}
        for cand in candidates:
            cand_lower = str(cand).lower()
            for col_lower, col in lower_map.items():
                if cand_lower == col_lower or cand_lower in col_lower:
                    return col
        return None

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        col_map = {}
        for key, opts in self.COLUMN_CANDIDATES.items():
            match = self._match_column(df, opts)
            if match:
                col_map[key] = match

        missing = [k for k in ("wavelength_um", "psi_deg", "delta_deg") if k not in col_map]
        if missing:
            raise ValueError(f"椭偏数据缺少必要列: {missing}. 已发现列: {df.columns.tolist()}")

        df = df.rename(columns={v: k for k, v in col_map.items()})
        cols = ["wavelength_um", "psi_deg", "delta_deg"]
        if "angle_deg" in df.columns:
            cols.append("angle_deg")
        df = df[cols].copy()
        df = df.dropna(how="any")
        df["wavelength_um"] = df["wavelength_um"].astype(float)
        df["psi_deg"] = df["psi_deg"].astype(float)
        df["delta_deg"] = df["delta_deg"].astype(float)
        if "angle_deg" in df.columns:
            df["angle_deg"] = df["angle_deg"].astype(float)
        else:
            df["angle_deg"] = float(self.config.params.get("ellipsometry_angle_deg", 70.0))
        df = df.sort_values(by="wavelength_um").reset_index(drop=True)
        return df

    def load(self) -> pd.DataFrame:
        raw = self._read_file()
        return self._normalize_columns(raw)

    def convert_to_nk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将椭偏测量数据(Ψ, Δ)转换为 n,k。
        采用半无限均匀介质的解析公式：
        ε = sin^2θ * (1 + tan^2θ * ((1 - ρ)/(1 + ρ))^2)
        其中 ρ = tan(Ψ) * exp(iΔ)
        """
        wl = df["wavelength_um"].values.astype(float)
        psi = np.deg2rad(df["psi_deg"].values.astype(float))
        delta = np.deg2rad(df["delta_deg"].values.astype(float))
        angle = np.deg2rad(df["angle_deg"].values.astype(float))
        rho = np.tan(psi) * np.exp(1j * delta)

        sin_t = np.sin(angle)
        tan_t = np.tan(angle)
        
        # 避免除零风险：给分母加极小值
        denom = 1 + rho
        denom[np.abs(denom) < 1e-12] = 1e-12
        
        with np.errstate(divide='ignore', invalid='ignore'):
            eps = (sin_t ** 2) * (1.0 + (tan_t ** 2) * ((1 - rho) / denom) ** 2)

        eps_real = np.real(eps)
        eps_abs = np.abs(eps)
        n = np.sqrt((eps_abs + eps_real) / 2.0)
        k = np.sqrt(np.maximum((eps_abs - eps_real) / 2.0, 0.0))

        nk_df = pd.DataFrame({
            "wavelength_um": wl,
            "n": n.astype(float),
            "k": k.astype(float)
        })
        nk_df = nk_df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        return nk_df

    def load_and_convert(self) -> Optional[Dict[str, Any]]:
        df = self.load()
        nk_df = self.convert_to_nk(df)
        if nk_df.empty:
            raise ValueError("椭偏数据转换后为空")
        lam = nk_df["wavelength_um"].values.astype(float)
        n = nk_df["n"].values.astype(float)
        k = nk_df["k"].values.astype(float)
        n_spline, k_spline, lam_min, lam_max = MaterialDataLoader._build_interpolators(lam, n, k, self.config)
        return {
            "ellipsometry_n": n_spline,
            "ellipsometry_k": k_spline,
            "ellipsometry_count": len(lam),
            "ellipsometry_lam_min": lam_min,
            "ellipsometry_lam_max": lam_max
        }


# ==================== 7. TMM仿真器（已拆分至 stage1.core.simulation） ====================


# ==================== 8. SVR训练器（已拆分至 stage1.core.training） ====================


# 使用core.training中的实现覆盖同名类，保持主入口兼容并完成Phase 2训练层拆分接入。
SVRModelEvaluator = CoreSVRModelEvaluator
SVRSpeedBenchmarker = CoreSVRSpeedBenchmarker
SVRExtrapolationEvaluator = CoreSVRExtrapolationEvaluator
SVRTrainer = CoreSVRTrainer

# ==================== 9. RegionSplitter ====================
class RegionSplitter:
    """智能region分区管理器，支持波长和厚度的自动分区"""

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def _get_discrete_points(self) -> Dict[str, Any]:
        """获取所有离散采样点"""
        thickness_step = self.cfg["thickness_step"]
        lambda_step = self.cfg["lambda_step"]
        min_thickness = self.cfg.get("min_thickness_nm", 100.0)

        # 厚度离散值（包含端点）
        all_thicknesses = np.arange(
            min_thickness, self.cfg["max_thickness"] + 1e-9, thickness_step
        )

        # 波长离散值
        all_wavelengths = np.arange(
            self.cfg["lambda_min"], self.cfg["lambda_max"] + lambda_step, lambda_step
        )

        # SiO2基底厚度
        sio2_thicknesses = self.cfg["sio2_thicknesses_nm"]

        return {
            "thicknesses": all_thicknesses,
            "wavelengths": all_wavelengths,
            "sio2": sio2_thicknesses,
            "n_th": len(all_thicknesses),
            "n_wl": len(all_wavelengths),
            "n_sio2": len(sio2_thicknesses),
        }

    def calculate_optimal_regions(self, target_n_regions: int) -> Dict[str, Any]:
        """基于采样分布计算region划分方案。"""
        points = self._get_discrete_points()

        def _build_lambda_segments(n_segments: int) -> List[Tuple[float, float]]:
            lam_min = self.cfg["lambda_min"]
            lam_max = self.cfg["lambda_max"]
            prioritize_atm = bool(self.cfg.get("prioritize_atmospheric_window", True))
            atm_min, atm_max = 8.0, 13.0

            if prioritize_atm and lam_min < atm_min < atm_max < lam_max and n_segments >= 2:
                segments = []
                segments.append((atm_min, atm_max))
                remaining = n_segments - 1
                left_span = max(0.0, atm_min - lam_min)
                right_span = max(0.0, lam_max - atm_max)
                total_span = left_span + right_span

                left_n = int(round(remaining * (left_span / total_span))) if total_span > 0 else 0
                right_n = remaining - left_n

                if left_span > 0 and left_n == 0:
                    left_n = 1
                    right_n = max(0, remaining - 1)
                if right_span > 0 and right_n == 0:
                    right_n = 1
                    left_n = max(0, remaining - 1)

                if left_n > 0:
                    wl_per = left_span / left_n
                    for i in range(left_n):
                        segments.append((lam_min + i * wl_per, lam_min + (i + 1) * wl_per))
                if right_n > 0:
                    wl_per = right_span / right_n
                    for i in range(right_n):
                        segments.append((atm_max + i * wl_per, atm_max + (i + 1) * wl_per))

                return sorted(segments, key=lambda x: x[0])

            wl_span = lam_max - lam_min
            wl_per = wl_span / n_segments
            return [(lam_min + i * wl_per, lam_min + (i + 1) * wl_per) for i in range(n_segments)]

        lambda_segments = _build_lambda_segments(target_n_regions)

        regions = {}
        region_id = 0
        strategy = self.cfg.get("region_split_strategy", "lambda_only")

        if strategy == "lambda_only":
            for wl_min, wl_max in lambda_segments:
                n_wavelengths = int(round((wl_max - wl_min) / self.cfg["lambda_step"])) + 1
                estimated_samples = points["n_th"] * n_wavelengths * points["n_sio2"]
                overlap = float(self.cfg.get("overlap_ratio", 0.0))
                wl_span = wl_max - wl_min
                wl_min_ext = max(self.cfg["lambda_min"], wl_min - wl_span * overlap)
                wl_max_ext = min(self.cfg["lambda_max"], wl_max + wl_span * overlap)
                regions[f"auto_region_{region_id}"] = {
                    "lambda_min": round(wl_min, 2),
                    "lambda_max": round(wl_max, 2),
                    "lambda_min_ext": round(wl_min_ext, 2),
                    "lambda_max_ext": round(wl_max_ext, 2),
                    "th_min": int(points["thicknesses"][0]),
                    "th_max": int(points["thicknesses"][len(points["thicknesses"]) - 1]),
                    "n_thickness_values": points["n_th"],
                    "n_wavelengths": n_wavelengths,
                    "estimated_samples": estimated_samples,
                    "overlap_ratio": overlap,
                }
                region_id += 1
        else:
            thicknesses = points["thicknesses"]
            weights = 1.0 / (thicknesses - thicknesses.min() + 1.0)
            cum_w = np.cumsum(weights)
            total_w = cum_w[-1]
            n_segments = len(lambda_segments)
            bounds = [0.0] + [total_w * i / n_segments for i in range(1, n_segments)] + [total_w]

            def _idx_at_weight(w):
                return int(np.searchsorted(cum_w, w, side='left'))

            for i, (wl_min, wl_max) in enumerate(lambda_segments):
                start_idx = _idx_at_weight(bounds[i])
                end_idx = _idx_at_weight(bounds[i + 1])
                end_idx = max(end_idx, start_idx + 3)
                end_idx = min(end_idx, points["n_th"])

                if end_idx - start_idx < 3:
                    print(f"⚠️  region_{i} 厚度值不足3个，已合并到前一个region")
                    continue

                th_min = float(thicknesses[start_idx])
                th_max = float(thicknesses[end_idx - 1])
                n_wavelengths = int(round((wl_max - wl_min) / self.cfg["lambda_step"])) + 1
                estimated_samples = (
                        (end_idx - start_idx) * n_wavelengths * points["n_sio2"]
                )

                overlap = float(self.cfg.get("overlap_ratio", 0.0))
                wl_span = wl_max - wl_min
                wl_min_ext = max(self.cfg["lambda_min"], wl_min - wl_span * overlap)
                wl_max_ext = min(self.cfg["lambda_max"], wl_max + wl_span * overlap)

                regions[f"auto_region_{region_id}"] = {
                    "lambda_min": round(wl_min, 2),
                    "lambda_max": round(wl_max, 2),
                    "lambda_min_ext": round(wl_min_ext, 2),
                    "lambda_max_ext": round(wl_max_ext, 2),
                    "th_min": th_min,
                    "th_max": th_max,
                    "n_thickness_values": end_idx - start_idx,
                    "n_wavelengths": n_wavelengths,
                    "estimated_samples": estimated_samples,
                    "overlap_ratio": overlap,
                }

                region_id += 1

        # 训练时间预估
        avg_time_per_region = self.cfg.get("avg_time_per_region", 30)  # 每个region的平均训练时间（秒）
        total_estimated_time = len(regions) * avg_time_per_region

        return {
            "regions": regions,
            "total_estimated_samples": sum(r["estimated_samples"] for r in regions.values()),
            "total_estimated_time_seconds": total_estimated_time,
            "total_estimated_time_minutes": total_estimated_time / 60,
            "n_regions_actual": len(regions),
            "original_target": target_n_regions,
        }

    def interactive_split(self) -> Optional[Dict[str, Any]]:
        """交互式生成并保存region划分方案。"""
        print_section("智能Region分区推荐系统")
        print("\n当前采样参数:")
        print(f"  波长范围: {self.cfg['lambda_min']} - {self.cfg['lambda_max']} μm")
        print(f"  波长步长: {self.cfg['lambda_step']} μm")
        print(f"  厚度范围: {self.cfg.get('min_thickness_nm', 100.0)} - {self.cfg['max_thickness']} nm")
        print(f"  厚度步长: {self.cfg['thickness_step']} nm")
        print(f"  SiO2基底: {self.cfg['sio2_thicknesses_nm']} nm")
        points = self._get_discrete_points()
        total_samples = points["n_th"] * points["n_wl"] * points["n_sio2"]
        print(f"\n总离散采样点数: {total_samples:,}")
        print(f"  - 厚度值: {points['n_th']}")
        print(f"  - 波长值: {points['n_wl']}")
        print(f"  - SiO2基底: {points['n_sio2']}")
        try:
            raw = input("\n请输入期望的region数量 (回车=自动推荐, 建议3-8): ").strip()
            if raw == "":
                max_samples_per_region = int(self.cfg.get("max_samples_per_region", 250000))
                target_n = int(np.ceil(total_samples / max_samples_per_region))
                target_n = max(2, min(20, target_n))
                print(f"自动推荐region数量: {target_n} (max_samples_per_region={max_samples_per_region})")
            else:
                target_n = int(raw)
                if target_n < 2 or target_n > 20:
                    print("⚠️  数量超出合理范围 (2-20)，已重置为4")
                    target_n = 4
        except:
            print("⚠️  输入无效，使用默认值4")
            target_n = 4
        print(f"\n正在计算 {target_n} 个region的最优分区...")
        result = self.calculate_optimal_regions(target_n)
        print("\n" + "-" * 70)
        print(f"{'Region':<20} {'波长范围(μm)':<18} {'厚度范围(nm)':<18} {'预估样本数':<12}")
        print("-" * 70)
        for name, info in result["regions"].items():
            wl_range = f"{info['lambda_min']:.2f}-{info['lambda_max']:.2f}"
            th_range = f"{info['th_min']}-{info['th_max']}"
            print(f"{name:<20} {wl_range:<18} {th_range:<18} {info['estimated_samples']:<12}")
        print("-" * 70)
        print(
            f"\n总计: 实际region数量: {result['n_regions_actual']} | 预估总样本数: {result['total_estimated_samples']:,} | 预估训练时间: {result['total_estimated_time_minutes']:.1f} 分钟")
        confirm = input("\n是否接受此方案并保存? (y/n): ").strip().lower()
        if confirm == "y":
            save_json(REGION_CONFIG_JSON, result)
            print(f"✅ 分区配置已保存到 {REGION_CONFIG_JSON}")
            return result["regions"]
        else:
            print("\n已取消分区方案")
            return None

    def get_region_simulation_tasks(self, regions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据region配置生成仿真任务列表。"""
        tasks = []
        points = self._get_discrete_points()
        eps = float(self.cfg.get("region_edge_eps", 1e-6))
        for region_name, region_info in regions.items():
            wl_indices = np.where((points["wavelengths"] >= region_info["lambda_min"] - eps) &
                                  (points["wavelengths"] <= region_info["lambda_max"] + eps))[0]
            th_indices = np.where((points["thicknesses"] >= region_info["th_min"] - eps) &
                                  (points["thicknesses"] <= region_info["th_max"] + eps))[0]
            tasks.append({
                "region_name": region_name,
                "wavelengths": points["wavelengths"][wl_indices],
                "thicknesses": points["thicknesses"][th_indices],
                "sio2_thicknesses": points["sio2"],
                "region_info": region_info
            })
        return tasks


# ==================== 10. 批量扫描函数 ====================
def scan_thickness_bands(df: pd.DataFrame, outdir: Union[str, Path] = "simulation_data", r2_threshold: float = 0.92,
                         min_bandwidth: float = 200, min_counts: int = 500,
                         config: Optional[Config] = None) -> None:
    """
    批量扫描所有可能的厚度区间，训练SVR并评估
    """
    # 确保outdir是Path对象
    outdir = Path(outdir) if not isinstance(outdir, Path) else outdir
    print_section("批量扫描厚度带区间")

    # 获取所有唯一厚度值并排序
    pdms_all = [int(x) for x in sorted(df['PDMS厚度_nm'].unique())]
    print(f"找到 {len(pdms_all)} 个唯一厚度值: {pdms_all}")

    bands = []

    # 生成所有可能的区间（宽度≥min_bandwidth，至少包含3个厚度值）
    for i, th_min in enumerate(pdms_all):
        for j, th_max in enumerate(pdms_all[i + 2:], i + 2):  # 确保至少3个厚度值
            if th_max - th_min < min_bandwidth:
                continue
            bands.append((th_min, th_max))

    print(f"生成 {len(bands)} 个候选厚度带")

    soft_cap = int(config.params.get("scan_soft_cap", 400)) if config is not None else 400
    hard_cap = int(config.params.get("scan_hard_cap", 1500)) if config is not None else 1500
    if len(bands) > hard_cap:
        print(f"⚠️ 候选数={len(bands)} 超过硬上限 {hard_cap}，预计耗时极长。")
        print("建议：增大 thickness_step / lambda_step，或提高 min_bandwidth。")
        go_on = input("是否仍继续扫描? (y/N): ").strip().lower() == "y"
        if not go_on:
            print("已取消本次扫描。")
            return
    if len(bands) > soft_cap:
        print(f"⚠️ 候选数较大，自动降采样到 {soft_cap} 个以加速。")
        idxs = np.linspace(0, len(bands) - 1, num=soft_cap, dtype=int)
        bands = [bands[i] for i in idxs]

    max_train_samples = int(config.params.get("scan_max_train_samples", 12000)) if config is not None else 12000
    save_predictions = False
    if config is not None:
        save_predictions = bool(config.params.get("scan_save_predictions", False))
        if bool(config.params.get("scan_prompt_save_predictions", True)):
            ans = input("是否保存每个区间的y_test/y_pred? (y/N): ").strip().lower()
            if ans in {"y", "yes"}:
                save_predictions = True
            elif ans in {"n", "no", ""}:
                save_predictions = False

    seed = config.params.get("random_seed", 42) if config is not None else 42

    results = []

    # 处理每个厚度带
    for idx, (th_min, th_max) in enumerate(bands, 1):
        print(f"\r【进度 {idx}/{len(bands)}】扫描: {th_min}-{th_max} nm", end="")

        # 过滤数据
        mask = (df['PDMS厚度_nm'] >= th_min) & (df['PDMS厚度_nm'] <= th_max)
        sub = df.loc[mask]

        if len(sub) < min_counts:
            continue

        # 准备数据
        X = sub[['波长_μm', '基底厚度_nm', 'PDMS厚度_nm']].values.astype(float)
        y = sub['发射率ε'].values.astype(float)

        if len(sub) > max_train_samples:
            sub = sub.sample(n=max_train_samples, random_state=seed)
            X = sub[['波长_μm', '基底厚度_nm', 'PDMS厚度_nm']].values.astype(float)
            y = sub['发射率ε'].values.astype(float)

        # 标准化
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            Xs, y, test_size=0.2, random_state=seed,
        )

        # 训练SVR（快速模式，固定参数）
        svr = SVR(kernel='rbf', C=1000, gamma=0.01, epsilon=0.001)
        svr.fit(X_train, y_train)

        # 预测并评估
        y_pred = np.clip(svr.predict(X_test), 0, 1)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # 保存结果
        band_info = {
            'th_min': int(th_min),
            'th_max': int(th_max),
            'width': int(th_max - th_min),
            'count': int(len(sub)),
            'r2': float(r2),
            'rmse': float(rmse),
        }
        results.append(band_info)

        # 保存每个区间的详细结果（可选）
        if save_predictions:
            band_file = outdir / f"bandscan_band_{th_min}-{th_max}.json"
            save_json(band_file, {
                "band": band_info,
                "y_test": y_test.tolist(),
                "y_pred": y_pred.tolist()
            })

    print("\n" + "-" * 70)
    print(f"扫描完成！处理了 {len(results)} 个有效厚度带")

    # 导出总表
    if results:
        summary_df = pd.DataFrame(results)
        summary_file = outdir / "bandscan_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"汇总表已保存: {summary_file}")

        # 打印满足R²阈值的区间
        print(f"\n--- 满足R²≥{r2_threshold}的厚度带 ---")
        selected = [r for r in results if r['r2'] >= r2_threshold]
        selected = sorted(selected, key=lambda x: (-x['width'], -x['r2']))

        for r in selected:
            print(f"  厚度: {r['th_min']:>4}-{r['th_max']:<4} nm  | "
                  f"样本数: {r['count']:<5} | R²: {r['r2']:.4f} | RMSE: {r['rmse']:.4f}")

        print(f"\n共找到 {len(selected)} 个高质量的厚度带区间")
    else:
        print("没有找到满足条件的厚度带区间")


# ==================== 11. 批量训练分区模型函数 ====================
def train_and_save_region(region_name: str, sub_df: pd.DataFrame, config: Config, min_samples: int = 50) -> Optional[Dict[str, Any]]:
    """对一个region的数据区间训练SVR和Scaler并保存"""
    if len(sub_df) < min_samples:
        print(f"⚠️ {region_name}: 样本数 {len(sub_df)} 少于 {min_samples}，跳过")
        return None

    # 1) 与Stage1全局训练保持一致：先做特征工程，再划分数据，再仅在训练集上fit scaler
    tmp_trainer = SVRTrainer(config)
    X = tmp_trainer._build_features(sub_df)
    y = sub_df["发射率ε"].values.astype(float)

    split_kwargs = {
        "test_size": config.params.get("test_size", 0.2),
        "random_state": config.params.get("random_seed", 42),
    }
    # 优先按厚度分组划分，避免厚度泄露
    split_protocol = "group_split_by_thickness_with_random_fallback"
    if config.params.get("group_split_by_thickness", True):
        groups = sub_df["PDMS厚度_nm"].values
        try:
            gss = GroupShuffleSplit(n_splits=1, test_size=split_kwargs["test_size"], random_state=split_kwargs["random_state"])
            train_idx, test_idx = next(gss.split(X, y, groups))
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        except Exception:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, **split_kwargs)
            split_protocol = "random_split_fallback"
    else:
        split_protocol = "random_split"
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, **split_kwargs)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # 2) 使用与配置一致的搜索策略，避免固定取参数列表末项导致严重失配
    param_grid = {
        "C": list(config.params.get("C_values", [100, 500, 1000])),
        "gamma": list(config.params.get("gamma_values", [0.01, 0.1, 1.0])),
        "epsilon": list(config.params.get("epsilon_values", [0.001, 0.01])),
    }
    base = SVR(kernel='rbf')
    cv_folds = max(2, int(config.params.get("cv_folds", 3)))

    if config.params.get("use_random_search", True):
        n_iter = min(int(config.params.get("random_search_iter", 10)),
                     max(1, len(param_grid["C"]) * len(param_grid["gamma"]) * len(param_grid["epsilon"])))
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="r2",
            cv=cv_folds,
            random_state=config.params.get("random_seed", 42),
            n_jobs=1,
            verbose=0,
        )
    else:
        search = GridSearchCV(
            estimator=base,
            param_grid=param_grid,
            scoring="r2",
            cv=cv_folds,
            n_jobs=1,
            verbose=0,
        )

    search.fit(X_train, y_train)
    svr = search.best_estimator_
    y_pred = np.clip(svr.predict(X_test), 0.0, 1.0)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    n_support_vectors = int(getattr(svr, "support_", np.array([])).shape[0])

    # 保存模型和缩放器
    region_dir = REGION_MODELS_DIR / region_name
    region_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(svr, region_dir / "svr_model.pkl")
    joblib.dump(scaler, region_dir / "scaler.pkl")

    # 选保存一份训练用csv也方便调试
    sub_df.to_csv(region_dir / "region_training_data.csv", index=False, encoding="utf-8-sig")

    # Optional: save per-region evaluation predictions for traceability (paper tables).
    if bool(config.params.get("region_save_predictions", False)):
        fmt = str(config.params.get("region_predictions_format", "csv") or "csv").strip().lower()
        try:
            if fmt == "json":
                pred_obj = {
                    "meta": {
                        "run_id": str(config.params.get("_run_id", "")),
                        "region": region_name,
                        "split_protocol": split_protocol,
                        "test_size": float(split_kwargs["test_size"]),
                        "random_state": int(split_kwargs["random_state"]),
                    },
                    "y_test": [float(v) for v in y_test.tolist()],
                    "y_pred": [float(v) for v in y_pred.tolist()],
                }
                save_json(region_dir / "region_test_predictions.json", pred_obj)
            else:
                # Default CSV (more compact + Excel-friendly)
                pred_df = pd.DataFrame({"y_test": y_test.astype(float), "y_pred": y_pred.astype(float)})
                pred_df.to_csv(region_dir / "region_test_predictions.csv", index=False, encoding="utf-8-sig")
        except Exception as e:
            print(f"⚠️ {region_name}: 保存 region_test_predictions 失败: {e}")

    if r2 < 0:
        print(
            f"⚠️ {region_name}: R²={r2:.4f} (<0，表示比均值基线还差)。"
            f" best_params={search.best_params_}"
        )
    else:
        print(f"✅ {region_name}: 完成训练 R²={r2:.4f}，已保存至 {region_dir}")

    return {
        "run_id": str(config.params.get("_run_id", "")),
        "split_protocol": split_protocol,
        "test_size": float(split_kwargs["test_size"]),
        "random_state": int(split_kwargs["random_state"]),
        "region": region_name,
        "samples": len(sub_df),
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "support_vectors": n_support_vectors,
        "best_params": search.best_params_,
    }


def train_baseline_models(df: pd.DataFrame, config: Config) -> None:
    """训练多种基线模型（RF/MLP/Linear）并保存预测结果，用于论文对比图表(Figure 14)。"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    import time

    print_section("正在训练基线模型 (Baseline Comparison)")
    print(f"使用全部 {len(df)} 条样本进行基线训练...")

    X = df[["波长_μm", "基底厚度_nm", "PDMS厚度_nm"]].values
    y = df["发射率ε"].values
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    run_id = str(config.params.get("_run_id", ""))
    split_protocol = "random_split"
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest (n=50)": RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42),
        "MLP (100,50)": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=200, random_state=42),
        # SVR (Global) already exists but we can add small one for comparison
        # "SVR (RBF)": SVR(kernel='rbf', C=100, gamma=0.1) # Too slow for baselines usually
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"⏳ 正在训练 {name}...")
        t0 = time.time()
        model.fit(X_train_s, y_train)
        train_time = time.time() - t0
        
        t1 = time.time()
        y_pred = model.predict(X_test_s)
        pred_time = time.time() - t1
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"   -> {name}: R²={r2:.4f}, RMSE={rmse:.4f}, Time={train_time:.2f}s")
        
        results[name] = {
            "run_id": run_id,
            "split_protocol": split_protocol,
            "test_size": 0.2,
            "random_state": 42,
            "r2": r2,
            "rmse": rmse,
            "train_time_s": train_time,
            "pred_time_ms": (pred_time / len(X_test)) * 1000,
            "y_pred_sample": list(y_pred[:1000]), # Save subset for plotting
            "y_test_sample": list(y_test[:1000])
        }

    # 保存结果
    save_path = OUTDIR / "baseline_comparison_results.json"
    save_json(save_path, results)
    print(f"\n✅ 基线模型对比结果已保存至: {save_path}")


def train_all_regions(region_config_path, tmm_csv_path, config):
    """装载分区配置和全局数据，批量训练所有region各自SVR并保存。"""
    # 加载分区配置和数据
    region_conf_path = Path(region_config_path)
    tmm_path = Path(tmm_csv_path)
    region_conf = load_json(region_conf_path)
    all_regions = region_conf["regions"]
    full_df = pd.read_csv(tmm_path)

    # 支持重叠分区：同一数据点可复制到多个region，避免边界样本只落入首个region。
    print(f"正在为 {len(full_df)} 个数据点分配region（支持重叠）...")
    region_chunks: List[pd.DataFrame] = []
    cover_mask = np.zeros(len(full_df), dtype=bool)

    eps = float(config.params.get("region_edge_eps", 1e-6))
    for region_name, region_info in all_regions.items():
        lam_min = region_info.get("lambda_min_ext", region_info["lambda_min"])
        lam_max = region_info.get("lambda_max_ext", region_info["lambda_max"])
        cond = (
            (full_df["波长_μm"] >= lam_min - eps) &
            (full_df["波长_μm"] <= lam_max + eps) &
            (full_df["PDMS厚度_nm"] >= region_info["th_min"] - eps) &
            (full_df["PDMS厚度_nm"] <= region_info["th_max"] + eps)
        )
        matched = int(cond.sum())
        if matched == 0:
            continue
        cover_mask |= cond.values
        chunk = full_df.loc[cond].copy()
        chunk["region"] = region_name
        region_chunks.append(chunk)

    invalid_count = int((~cover_mask).sum())
    if invalid_count > 0:
        print(f"过滤了 {invalid_count} 个未分配到任何region的数据点")
    if not region_chunks:
        raise ValueError("没有任何数据点匹配到region，请检查region_config范围设置。")

    full_df = pd.concat(region_chunks, ignore_index=True)
    duplication_ratio = len(full_df) / max(1, int(cover_mask.sum()))
    print(f"region分配完成：覆盖点={int(cover_mask.sum())}，复制后样本={len(full_df)}，平均覆盖倍数={duplication_ratio:.3f}")
    
    # 按region分组并训练
    results = []
    grouped = full_df.groupby('region')
    
    print(f"共找到 {len(grouped)} 个有效的region分组")
    for region_name, sub_df in grouped:
        region_name = str(region_name)
        try:
            res = train_and_save_region(region_name, sub_df, config)
            if res: results.append(res)
        except Exception as e:
            print(f"❌ {region_name} 训练失败：{e}")
            traceback.print_exc()
    
    # 保存汇总信息供图表生成使用
    print(f"✅ 保存region训练汇总至: {REGION_SUMMARY_JSON}")
    rc = config.params.get("_run_context") if isinstance(config.params.get("_run_context"), dict) else None
    save_json(REGION_SUMMARY_JSON, {
        "run_id": str((rc or {}).get("manifest_run_id") or config.params.get("_run_id", "")),
        "split_protocol": str((rc or {}).get("split_protocol") or "region_train_group_split_by_thickness_with_fallback"),
        "regions_config": all_regions,
        "results": results,
    }, run_context=rc)

    print(f"==== Region批量训练全部完成，共生成{len(results)}个SVR分区模型 ====")
    return results


# ==================== 12. 主菜单 ====================
def print_menu() -> None:
    """显示主菜单。"""
    print("\n" + "=" * 60)
    print("PDMS薄膜发射率TMM-SVR系统（带分区功能）")
    print("=" * 60)
    print("1) 生成默认config.json（初始化配置）")
    print("2) Configure (interactive)")
    print("3) Run TMM simulation (full range)")
    print("4) Train SVR (and evaluate)")
    print("5) Evaluate only (use existing model)")
    print("6) Thickness extrapolation test")
    print("7) Run speed benchmark")
    print("8) Save / Load model")
    print("9) Full pipeline (simulate -> train -> eval -> speed -> save)")
    print("10) Region splitting (智能分区)")
    print("11) Run TMM simulation with regions (并行分区模拟)")
    print("12) 批量扫描厚度带区间并训练")
    print("13) 批量训练分区模型（基于已有分区配置和TMM数据）")
    print("14) 物理意义与结果解读说明（R²/速度/区间）")
    print("15) Exit")
    print("16) Train baseline models (RF/MLP/Linear for comparison) - for Figure 14")
    print("17) 一键清理/归档历史输出（防残留）")
    print("18) Region状态检查（配置数/样本数/模型文件）")
    print("=" * 60)


def set_random_seeds(seed: int) -> None:
    """设置所有随机种子以确保可重复性。"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def explain_model_physics() -> None:
    """打印模型物理意义与参数影响说明。"""
    print_section("物理意义与结果解读（简明手册）")
    print("\n【模型物理意义】")
    print("1) TMM 是物理真值：用传输矩阵法计算薄膜的反射R、透射T、发射率ε。")
    print("2) SVR 是替身模型：用机器学习拟合 TMM 结果，快速预测 ε。")
    print("3) 分边界偏差区模型：把波长/厚度空间分成多个region，各自训练，减少。")

    print("\n【R²与区间解释示例】")
    print("- 例如 region_1 的 R² ≥ 0.97：说明在该波长/厚度区间内，SVR 对 TMM 拟合非常好。")
    print("- 若 R² < 0.9：常见原因是样本不足、区间太宽、材料数据稀疏或物理变化剧烈。")
    print("- 建议：缩小region、增密采样、或开启特征工程以改善拟合。")

    print("\n【速度是否符合规律？】")
    print("- SVR 预测一次光谱的耗时应明显低于 TMM。通常加速比 >10× 属于合理范围。")
    print("- 若加速比过低：可能是模型维度过大、scaler开销高、CPU负载高或频繁IO。")
    print("- 若加速比异常高：通常是 TMM 很慢或样本点太多导致。")

    print("\n【关键参数的物理/数值影响】")
    print("- lambda_min/max：定义波长范围。范围越宽，物理覆盖越全，但样本量更大。")
    print("- lambda_step：波长步长越小，谱线更细，TMM更慢，SVR样本更多。")
    print("- min_thickness_nm/max_thickness：定义厚度空间覆盖范围。")
    print("- thickness_step：厚度步长越小，样本越密，训练更稳定，但耗时更高。")
    print("- energy_tol/strict_energy_check：能量守恒阈值与是否剔除违规点。严格会更物理，但数据变少。")
    print("- substrate_n_real/imag：基底光学常数直接影响R/T/ε的物理正确性。")
    print("- feature_engineering：相位特征能增强周期性表达，提升R²。")
    print("- group_split_by_thickness：避免厚度泄露，评估更真实但更难划分。")
    print("- allow_random_split：放宽划分约束，R²可能升高但泛化可信度下降。")
    print("- region_split_strategy/overlap_ratio：决定分区方式与重叠，重叠可降低边界误差。")
    print("- max_samples_per_region：控制单区样本量，防止训练过慢。")
    print("- use_random_search/cv_folds：搜索范围与交叉验证折数，影响超参稳定性与耗时。")

    print("\n【实践建议】")
    print("- 先用默认配置跑通流程，再逐步缩步长/加区间。")
    print("- 若R²偏低：优先提高样本密度或缩小region，再考虑复杂模型。")
    print("- 若速度不理想：减少波长点数或region数；或先用SVR评估再回到TMM验证。")


def cleanup_or_archive_outputs() -> None:
    """一键清理或归档历史输出，避免残留文件影响后续运行。"""
    print_section("清理/归档历史输出")
    print("1) 直接删除历史输出（保留config.json）")
    print("2) 归档历史输出到 archive_runs 时间戳目录（保留config.json）")
    print("3) 取消")

    mode = input("选择 [1/2/3]: ").strip() or "3"
    if mode == "3":
        print("已取消。")
        return
    if mode not in {"1", "2"}:
        print("无效选择，已取消。")
        return

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)
    archive_dir = None
    if mode == "2":
        ts = time.strftime("%Y%m%d_%H%M%S")
        archive_dir = outdir / "archive_runs" / ts
        archive_dir.mkdir(parents=True, exist_ok=True)

    files_to_handle = [
        Path(TMM_CSV),
        Path(MODEL_PKL),
        Path(SCALER_PKL),
        Path(Y_SCALER_PKL),
        Path(TEST_RESULTS_JSON),
        Path(SPEED_RESULTS_JSON),
        Path(MODEL_INFO_JSON),
        Path(MATERIAL_INFO_JSON),
        Path(REGION_SUMMARY_JSON),
        Path(BANDSCAN_SUMMARY),
        outdir / "performance_summary_stage1_tmm.json",
        outdir / "performance_summary_stage1_tmm.csv",
        outdir / "performance_summary_stage1_training.json",
        outdir / "performance_summary_stage1_training.csv",
        outdir / "performance_summary_index.json",
        outdir / "mlp_baseline.pkl",
    ]

    dynamic_patterns = [
        "bandscan_band_*.json",
        "extrap_thickness_*.json",
    ]

    handled = 0
    for fp in files_to_handle:
        if not fp.exists():
            continue
        try:
            if mode == "1":
                fp.unlink(missing_ok=True)
            else:
                if archive_dir is None:
                    raise RuntimeError("archive_dir is None in archive mode")
                dest = archive_dir / fp.name
                shutil.move(str(fp), str(dest))
            handled += 1
        except Exception as e:
            print(f"⚠️  处理失败: {fp} | {e}")

    for pattern in dynamic_patterns:
        for fp in outdir.glob(pattern):
            try:
                if mode == "1":
                    fp.unlink(missing_ok=True)
                else:
                    if archive_dir is None:
                        raise RuntimeError("archive_dir is None in archive mode")
                    dest = archive_dir / fp.name
                    shutil.move(str(fp), str(dest))
                handled += 1
            except Exception as e:
                print(f"⚠️  处理失败: {fp} | {e}")

    region_models_dir = Path(REGION_MODELS_DIR)
    if region_models_dir.exists():
        try:
            if mode == "1":
                shutil.rmtree(region_models_dir, ignore_errors=True)
            else:
                if archive_dir is None:
                    raise RuntimeError("archive_dir is None in archive mode")
                dest = archive_dir / "region_models"
                if dest.exists():
                    shutil.rmtree(dest, ignore_errors=True)
                shutil.move(str(region_models_dir), str(dest))
            handled += 1
        except Exception as e:
            print(f"⚠️  处理失败: {region_models_dir} | {e}")

    print(f"✅ 完成。共处理 {handled} 个输出项。")
    if archive_dir is not None:
        print(f"📦 归档目录: {archive_dir}")


def inspect_region_status(config: Config) -> None:
    """菜单18：一键检查region配置、样本数与模型文件状态。"""
    print_section("Region状态检查（训练前自检）")

    report: Dict[str, Any] = {
        "stage": "stage1",
        "module": "inspect_regions",
        "region_config_path": str(REGION_CONFIG_JSON),
        "tmm_csv_path": str(TMM_CSV),
        "regions": [],
        "issues": [],
    }

    rc = config.params.get("_run_context") if isinstance(config.params.get("_run_context"), dict) else None
    report["run_id"] = str((rc or {}).get("manifest_run_id") or config.params.get("_run_id", ""))
    report["split_protocol"] = str((rc or {}).get("split_protocol") or "")

    if not os.path.exists(REGION_CONFIG_JSON):
        print(f"❌ 未找到分区配置文件: {REGION_CONFIG_JSON}")
        print("请先运行菜单 10 生成分区配置。")
        report["issues"].append("missing_region_config")
        save_json(OUTDIR / "inspect_regions_report.json", report, run_context=rc)
        return

    try:
        region_conf = load_json(REGION_CONFIG_JSON)
    except Exception as e:
        print(f"❌ 读取分区配置失败: {e}")
        report["issues"].append(f"failed_to_load_region_config:{e}")
        save_json(OUTDIR / "inspect_regions_report.json", report, run_context=rc)
        return

    regions = region_conf.get("regions", {}) if isinstance(region_conf, dict) else {}
    print(f"📄 region_config: {REGION_CONFIG_JSON}")
    print(f"📌 当前配置region数量: {len(regions)}")
    report["region_count"] = int(len(regions))
    if not regions:
        print("⚠️ 配置中没有有效region。")
        report["issues"].append("no_regions_in_config")
        save_json(OUTDIR / "inspect_regions_report.json", report, run_context=rc)
        return

    df = None
    if os.path.exists(TMM_CSV):
        try:
            df = pd.read_csv(TMM_CSV)
            print(f"📊 TMM数据行数: {len(df)} | 文件: {TMM_CSV}")
            report["tmm_rows"] = int(len(df))
        except Exception as e:
            print(f"⚠️ 读取TMM数据失败: {e}")
            report["issues"].append(f"failed_to_read_tmm_csv:{e}")
    else:
        print(f"⚠️ 未找到TMM数据: {TMM_CSV}")
        report["issues"].append("missing_tmm_csv")

    eps = float(config.params.get("region_edge_eps", 1e-6))
    print("\n" + "-" * 100)
    print(f"{'Region':<16} {'λ范围(μm)':<18} {'厚度范围(nm)':<18} {'样本数(TMM)':<14} {'模型状态':<28}")
    print("-" * 100)

    for region_name, info in regions.items():
        lam_min = float(info.get("lambda_min", 0.0))
        lam_max = float(info.get("lambda_max", 0.0))
        th_min = float(info.get("th_min", 0.0))
        th_max = float(info.get("th_max", 0.0))

        sample_count = "N/A"
        if df is not None and {"波长_μm", "PDMS厚度_nm"}.issubset(df.columns):
            lam_min_ext = float(info.get("lambda_min_ext", lam_min))
            lam_max_ext = float(info.get("lambda_max_ext", lam_max))
            cond = (
                (df["波长_μm"] >= lam_min_ext - eps) &
                (df["波长_μm"] <= lam_max_ext + eps) &
                (df["PDMS厚度_nm"] >= th_min - eps) &
                (df["PDMS厚度_nm"] <= th_max + eps)
            )
            sample_count = str(int(cond.sum()))

        region_dir = REGION_MODELS_DIR / region_name
        model_ok = (region_dir / "svr_model.pkl").exists()
        scaler_ok = (region_dir / "scaler.pkl").exists()
        y_scaler_ok = (region_dir / "y_scaler.pkl").exists()
        if model_ok and scaler_ok:
            model_state = f"✅ model+scaler{' +y' if y_scaler_ok else ''}"
        elif region_dir.exists():
            model_state = "⚠️ 目录存在但文件不全"
        else:
            model_state = "❌ 未训练"

        print(
            f"{region_name:<16} "
            f"{f'{lam_min:.2f}-{lam_max:.2f}':<18} "
            f"{f'{th_min:.0f}-{th_max:.0f}':<18} "
            f"{sample_count:<14} "
            f"{model_state:<28}"
        )

        report["regions"].append(
            {
                "region": str(region_name),
                "lambda_min_um": float(lam_min),
                "lambda_max_um": float(lam_max),
                "th_min_nm": float(th_min),
                "th_max_nm": float(th_max),
                "tmm_samples": int(sample_count) if str(sample_count).isdigit() else None,
                "model_ok": bool(model_ok),
                "scaler_ok": bool(scaler_ok),
                "y_scaler_ok": bool(y_scaler_ok),
                "model_state": str(model_state),
                "region_dir": str(region_dir),
            }
        )

    print("-" * 100)
    print(f"📁 region_models目录: {REGION_MODELS_DIR}")
    print("提示：若“配置数量”和“模型数量”不一致，先运行 17 清理，再 10→11→13 重建。")

    save_json(OUTDIR / "inspect_regions_report.json", report, run_context=rc)
    print(f"✅ 已写入审计报告: {OUTDIR / 'inspect_regions_report.json'}")


# ==================== 13. 主函数 ====================
def main(
    scripted_choices: Optional[List[str]] = None,
    benchmark_n_spectra: int = 50,
    config_file: Optional[str] = None,
    action: Optional[str] = None,
) -> None:
    repair_stale_started_manifest(OUTDIR, stale_after_s=60 * 60)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_started_at = datetime.now().isoformat(timespec="seconds")
    run_t0 = time.time()
    run_status = "started"

    cfg = Config()
    if config_file:
        cfg.config_file = config_file
    cfg.load()
    cfg.params["_run_id"] = run_id

    action_name = action or "interactive"
    set_random_seeds(int(cfg.params.get("random_seed", 42)))
    ic = InteractiveConfigurator(cfg)
    loader = MaterialDataLoader(config=cfg)

    project_root = Path(__file__).resolve().parents[1]
    input_paths = [str(cfg.config_file), str(loader.pdms_file), str(loader.sio2_file)]
    if bool(cfg.params.get("enable_ellipsometry", False)):
        input_paths.append(str(cfg.params.get("ellipsometry_file", "")))

    split_protocol_override = None
    if action_name == "train_regions":
        split_protocol_override = "region_train_group_split_by_thickness_with_fallback"

    run_context = build_run_context(
        run_id=run_id,
        action=action_name,
        config_snapshot=dict(cfg.params),
        input_paths=input_paths,
        project_root=project_root,
        manifest_latest_path=str(Path(OUTDIR) / "run_manifest_latest.json"),
        split_protocol_override=split_protocol_override,
    )
    cfg.params["_run_context"] = {
        "run_id": run_context.run_id,
        "manifest_run_id": run_context.manifest_run_id,
        "action": run_context.action,
        "split_protocol": run_context.split_protocol,
        "data_fingerprint": run_context.data_fingerprint,
    }

    global _DEFAULT_RUN_CONTEXT
    _DEFAULT_RUN_CONTEXT = cfg.params["_run_context"]

    split_protocol = run_context.split_protocol
    trainer = SVRTrainer(cfg)
    region_splitter = RegionSplitter(cfg.params)
    append_session_log(
        OUTDIR,
        "session_start",
        {"config_file": cfg.config_file, "run_id": run_id, "action": action_name},
    )
    write_run_manifest(
        OUTDIR,
        run_id=run_id,
        status="started",
        started_at=run_started_at,
        config_file=str(cfg.config_file),
        config_snapshot=cfg.params,
        input_files=[str(cfg.config_file)],
        manifest_run_id=run_context.manifest_run_id,
        parent_run_id=None,
        action=action_name,
        action_run_id=run_id,
        split_protocol=split_protocol,
        data_fingerprint=run_context.data_fingerprint,
    )

    def _sample_range_from_cfg() -> str:
        p = cfg.params
        return (
            f"lambda={p.get('lambda_min', 'NA')}-{p.get('lambda_max', 'NA')},"
            f"pdms={p.get('min_thickness_nm', 'NA')}-{p.get('max_thickness', 'NA')},"
            f"sio2={p.get('sio2_thicknesses_nm', 'NA')}"
        )

    try:
        scripted_queue = list(scripted_choices or [])
        while True:
            print_menu()
            if scripted_queue:
                choice = scripted_queue.pop(0)
                print(f"[non-interactive] Select: {choice}")
            else:
                choice = input("Select: ").strip()
            append_session_log(OUTDIR, "menu_choice", {"choice": choice, "run_id": run_id})

            if choice == "1":
                # 生成默认config.json（初始化配置）
                if os.path.exists(cfg.config_file):
                    print(f"配置文件已存在: {cfg.config_file}")
                else:
                    cfg.params = DEFAULTS.copy()
                    cfg.save()
                    print(f"✅ 已生成默认配置文件: {cfg.config_file}")

            elif choice == "2":
                ic.run()

            elif choice == "3":
                try:
                    material_data = loader.load_all(pdms_sample=cfg.params.get("pdms_sample", 3),
                                                    sio2_sample=cfg.params.get("sio2_sample", 2))
                except Exception as e:
                    log_exception_with_context(
                        e,
                        ErrorContext(
                            stage="stage1.menu.option3.simulation",
                            region="global",
                            file=str(TMM_CSV),
                            sample_range="".join(_sample_range_from_cfg()),
                        ),
                        print_traceback=True,
                    )
                    continue
                simulator = TMMSimulator(material_data, cfg)
                df = simulator.run_simulation()
                save_json(MATERIAL_INFO_JSON, {
                    "pdms_file": loader.pdms_file,
                    "sio2_file": loader.sio2_file,
                    "pdms_sample": cfg.params.get("pdms_sample", 3),
                    "sio2_sample": cfg.params.get("sio2_sample", 2),
                    "config": cfg.params
                })

            elif choice == "4":
                if not os.path.exists(TMM_CSV):
                    print("No TMM data found, run simulation first (option 3).")
                    continue
                row_count = "NA"
                try:
                    df = pd.read_csv(TMM_CSV)
                    row_count = str(len(df))
                    X_train, X_test, y_train, y_test, processed_df = trainer.preprocess(df)
                    trainer.train(X_train, y_train)
                    trainer.evaluate(X_test, y_test, processed_df)
                    trainer.save_model()
                except Exception as e:
                    log_exception_with_context(
                        e,
                        ErrorContext(
                            stage="stage1.menu.option4.train",
                            region="global",
                            file=str(TMM_CSV),
                            sample_range=f"rows={row_count}",
                        ),
                        print_traceback=True,
                    )

            elif choice == "5":
                if not os.path.exists(MODEL_PKL) and not os.path.exists(MODEL_BUNDLE_PKL):
                    print("No saved model found. Train or load a model first.")
                    continue
                if trainer.model is None:
                    if os.path.exists(MODEL_BUNDLE_PKL):
                        trainer.model, trainer.scaler, trainer.y_scaler, _ = ModelWrapper.load(MODEL_BUNDLE_PKL)
                    else:
                        trainer.model = joblib.load(MODEL_PKL)
                        if trainer.scaler is None:
                            trainer.scaler = joblib.load(SCALER_PKL)
                        if os.path.exists(Y_SCALER_PKL):
                            trainer.y_scaler = joblib.load(Y_SCALER_PKL)
                if trainer.scaler is None and os.path.exists(SCALER_PKL):
                    trainer.scaler = joblib.load(SCALER_PKL)
                df = pd.read_csv(TMM_CSV)
                X_train, X_test, y_train, y_test, processed_df = trainer.preprocess(df)
                trainer.evaluate(X_test, y_test, processed_df)

            elif choice == "6":
                if not os.path.exists(TMM_CSV):
                    print("No TMM data found, run simulation first (option 3).")
                    continue
                df = pd.read_csv(TMM_CSV)
                required_cols = {"波长_μm", "基底厚度_nm", "PDMS厚度_nm", "发射率ε"}
                if not required_cols.issubset(df.columns):
                    print(f"Data missing required columns: {required_cols - set(df.columns)}")
                    continue
                try:
                    trainer.run_thickness_extrapolation(df)
                except Exception as e:
                    print("Extrapolation failed:", e)

            elif choice == "7":
                if trainer.model is None:
                    print("Model not trained/loaded yet.")
                    continue
                trainer.benchmark_speed(n_spectra=benchmark_n_spectra)

            elif choice == "8":
                s = input("1) Save model  2) Load model. Choose: ").strip()
                if s == "1":
                    if trainer.model is None:
                        print("No model to save.")
                    else:
                        trainer.save_model()
                        if trainer.mlp is not None:
                            joblib.dump(trainer.mlp, os.path.join(OUTDIR, "mlp_baseline.pkl"))
                        print("Saved model and scalers.")
                else:
                    if os.path.exists(MODEL_BUNDLE_PKL):
                        trainer.model, trainer.scaler, trainer.y_scaler, _ = ModelWrapper.load(MODEL_BUNDLE_PKL)
                        mlp_file = os.path.join(OUTDIR, "mlp_baseline.pkl")
                        if os.path.exists(mlp_file):
                            trainer.mlp = joblib.load(mlp_file)
                        print("Model loaded.")
                    elif os.path.exists(MODEL_PKL) and os.path.exists(SCALER_PKL):
                        trainer.model = joblib.load(MODEL_PKL)
                        trainer.scaler = joblib.load(SCALER_PKL)
                        if os.path.exists(Y_SCALER_PKL):
                            trainer.y_scaler = joblib.load(Y_SCALER_PKL)
                        mlp_file = os.path.join(OUTDIR, "mlp_baseline.pkl")
                        if os.path.exists(mlp_file):
                            trainer.mlp = joblib.load(mlp_file)
                        print("Model loaded.")
                    else:
                        print("Saved model files missing.")

            elif choice == "9":
                try:
                    material_data = loader.load_all(pdms_sample=cfg.params.get("pdms_sample", 3),
                                                    sio2_sample=cfg.params.get("sio2_sample", 2))
                except Exception as e:
                    log_exception_with_context(
                        e,
                        ErrorContext(
                            stage="stage1.menu.option9.full_pipeline.load",
                            region="global",
                            file=str(TMM_CSV),
                            sample_range="".join(_sample_range_from_cfg()),
                        ),
                        print_traceback=True,
                    )
                    continue
                row_count = "NA"
                try:
                    simulator = TMMSimulator(material_data, cfg)
                    df = simulator.run_simulation()
                    row_count = str(len(df))
                    trainer = SVRTrainer(cfg)
                    X_train, X_test, y_train, y_test, processed_df = trainer.preprocess(df)
                    trainer.train(X_train, y_train)
                    trainer.evaluate(X_test, y_test, processed_df)
                    if cfg.params.get("external_extrapolation", True):
                        try:
                            trainer.run_thickness_extrapolation(df)
                        except Exception as e:
                            print(f"⚠️ 外推评估失败（不影响主流程）: {e}")
                    trainer.benchmark_speed(n_spectra=benchmark_n_spectra)
                    trainer.save_model()
                    if trainer.mlp is not None:
                        joblib.dump(trainer.mlp, os.path.join(OUTDIR, "mlp_baseline.pkl"))
                    print("Full pipeline completed.")
                except Exception as e:
                    log_exception_with_context(
                        e,
                        ErrorContext(
                            stage="stage1.menu.option9.full_pipeline.run",
                            region="global",
                            file=str(OUTDIR),
                            sample_range=f"rows={row_count}",
                        ),
                        print_traceback=True,
                    )

            elif choice == "10":
                print("\n正在进入智能分区模式...")
                regions = region_splitter.interactive_split()
                if regions:
                    print("分区配置已保存，可以使用选项11进行并行分区模拟。")

            elif choice == "11":
                if not os.path.exists(REGION_CONFIG_JSON):
                    print("未找到分区配置文件，请先运行选项10进行智能分区。")
                    continue
                try:
                    material_data = loader.load_all(pdms_sample=cfg.params.get("pdms_sample", 3),
                                                    sio2_sample=cfg.params.get("sio2_sample", 2))
                    region_config = load_json(REGION_CONFIG_JSON)
                    regions = region_config.get("regions", {})
                    region_count = str(len(regions))
                    if not regions:
                        print("分区配置文件为空或格式错误。")
                        continue
                    print(f"加载了 {len(regions)} 个分区配置。")
                    simulator = TMMSimulator(material_data, cfg)
                    df = simulator.run_parallel_simulation_by_region(material_data, regions)
                    if not df.empty:
                        print(f"分区模拟完成，总数据点: {len(df)}")
                        save_json(MATERIAL_INFO_JSON, {
                            "pdms_file": loader.pdms_file,
                            "sio2_file": loader.sio2_file,
                            "pdms_sample": cfg.params.get("pdms_sample", 3),
                            "sio2_sample": cfg.params.get("sio2_sample", 2),
                            "config": cfg.params,
                            "regions_used": list(regions.keys())
                        })
                except Exception as e:
                    log_exception_with_context(
                        e,
                        ErrorContext(
                            stage="stage1.menu.option11.region_simulation",
                            region=f"n_regions={region_count if 'region_count' in locals() else 'NA'}",
                            file=str(REGION_CONFIG_JSON),
                            sample_range="".join(_sample_range_from_cfg()),
                        ),
                        print_traceback=True,
                    )

            elif choice == "12":
                # ===== 批量扫描厚度带功能 =====
                if not os.path.exists(TMM_CSV):
                    print("❌ 未找到TMM数据文件，请先运行选项3生成数据。")
                    continue

                print_section("批量扫描厚度带区间")

                # 加载数据
                df = pd.read_csv(TMM_CSV)
                print(f"✅ 已加载 {len(df)} 行TMM数据")

                # 获取用户参数
                try:
                    r2_thresh = float(input(f"R²阈值 [默认 0.92]: ").strip() or "0.92")
                    min_bw = float(input(f"最小带宽(nm) [默认 200]: ").strip() or "200")
                    min_cnt = int(input(f"最小样本数 [默认 500]: ").strip() or "500")
                except Exception:
                    print("⚠️ 输入无效，使用默认值")
                    r2_thresh, min_bw, min_cnt = 0.92, 200, 500

                # 执行扫描
                scan_thickness_bands(df, outdir=OUTDIR, r2_threshold=r2_thresh,
                                     min_bandwidth=min_bw, min_counts=min_cnt, config=cfg)

                print("\n✅ 批量扫描完成！")
                print(f"查看结果: {BANDSCAN_SUMMARY}")

            elif choice == "13":
                # 批量分区训练保存：自动生成 region_models/auto_region_x/ 的pkl
                if not os.path.exists(REGION_CONFIG_JSON):
                    print("未找到分区配置，请先进行region分区 (菜单10)")
                    continue
                if not os.path.exists(TMM_CSV):
                    print("未找到TMM数据，请先进行TMM仿真 (菜单3或11)")
                    continue
                try:
                    train_all_regions(REGION_CONFIG_JSON, TMM_CSV, cfg)
                except Exception as e:
                    log_exception_with_context(
                        e,
                        ErrorContext(
                            stage="stage1.menu.option13.train_all_regions",
                            region="multi-region",
                            file=str(REGION_CONFIG_JSON),
                            sample_range=str(TMM_CSV),
                        ),
                        print_traceback=True,
                    )

            elif choice == "14":
                explain_model_physics()

            elif choice == "15":
                append_session_log(OUTDIR, "session_exit", {"choice": choice, "run_id": run_id})
                print("Exiting.")
                run_status = "completed"
                break

            elif choice == "16":
                # 训练基线模型（Figure 14对比用）
                if not os.path.exists(TMM_CSV):
                    print("❌ 未找到TMM数据，请先进行TMM仿真 (菜单3或11) 或放置 tmm_emissivity_data.csv")
                    continue
                row_count = "NA"
                try:
                    df = pd.read_csv(TMM_CSV)
                    row_count = str(len(df))
                    train_baseline_models(df, cfg)
                except Exception as e:
                    log_exception_with_context(
                        e,
                        ErrorContext(
                            stage="stage1.menu.option16.baseline_training",
                            region="global",
                            file=str(TMM_CSV),
                            sample_range=f"rows={row_count}",
                        ),
                        print_traceback=True,
                    )

            elif choice == "17":
                cleanup_or_archive_outputs()

            elif choice == "18":
                inspect_region_status(cfg)

            else:
                print("Invalid selection. Try again.")
    except KeyboardInterrupt:
        run_status = "interrupted"
        append_session_log(OUTDIR, "session_interrupt", {"reason": "KeyboardInterrupt", "run_id": run_id})
        raise
    except Exception:
        run_status = "failed"
        raise
    finally:
        write_run_manifest(
            OUTDIR,
            run_id=run_id,
            status=run_status,
            started_at=run_started_at,
            duration_s=time.time() - run_t0,
            config_file=str(cfg.config_file),
            config_snapshot=cfg.params,
            input_files=[str(cfg.config_file), str(TMM_CSV), str(REGION_CONFIG_JSON)],
            manifest_run_id=run_context.manifest_run_id,
            parent_run_id=None,
            action=action_name,
            action_run_id=run_id,
            split_protocol=split_protocol,
            data_fingerprint=run_context.data_fingerprint,
        )


# ==================== 14. 入口点 ====================
def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage1 PDMS TMM->SVR pipeline")
    parser.add_argument(
        "--action",
        default="interactive",
        choices=[
            "interactive",
            "simulate",
            "split_regions",
            "train",
            "evaluate",
            "extrapolation",
            "benchmark",
            "full_pipeline",
            "region_simulation",
            "train_regions",
            "baselines",
            "inspect_regions",
        ],
        help="Run one reproducible non-interactive action, or keep interactive menu.",
    )
    parser.add_argument(
        "--benchmark-n-spectra",
        type=int,
        default=50,
        help="Benchmark spectra count for speed test actions.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="",
        help="Optional config.json path override for reproducible runs.",
    )
    return parser


if __name__ == "__main__":
    try:
        args = _build_cli_parser().parse_args()
        action_to_choices = {
            "simulate": ["3", "15"],
            "split_regions": ["10", "15"],
            "train": ["4", "15"],
            "evaluate": ["5", "15"],
            "extrapolation": ["6", "15"],
            "benchmark": ["5", "7", "15"],
            "full_pipeline": ["9", "15"],
            "region_simulation": ["11", "15"],
            "train_regions": ["13", "15"],
            "baselines": ["16", "15"],
            "inspect_regions": ["18", "15"],
        }
        scripted = None
        if args.action != "interactive":
            scripted = action_to_choices[args.action]
        main(
            scripted_choices=scripted,
            benchmark_n_spectra=max(1, int(args.benchmark_n_spectra)),
            config_file=(args.config_file.strip() or None),
            action=args.action,
        )
    except KeyboardInterrupt:
        append_session_log(OUTDIR, "session_interrupt", {"reason": "KeyboardInterrupt"})
        print("\nInterrupted by user, exiting.")
    except Exception as e:
        log_exception_with_context(
            e,
            ErrorContext(
                stage="stage1.main",
                region="global",
                file=str(OUTDIR),
                sample_range="main-loop",
            ),
            print_traceback=True,
        )