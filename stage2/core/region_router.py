from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np


def default_load_json(path: Any) -> Dict:
    import json
    path = Path(path) if not isinstance(path, Path) else path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class MultiRegionModelRouter:
    """管理多个region模型，自动路由到正确的模型。"""

    def __init__(self, region_models_dir: Any, datadir: Any, load_json_fn=default_load_json):
        self.region_models_dir = Path(region_models_dir) if not isinstance(region_models_dir, Path) else region_models_dir
        self.datadir = Path(datadir) if not isinstance(datadir, Path) else datadir
        self._load_json = load_json_fn

        cfg_path = self.datadir / "config.json"
        if cfg_path.exists():
            try:
                self.config = self._load_json(cfg_path)
            except Exception:
                self.config = {}
        else:
            self.config = {}

        self.region_configs = self._load_region_configs()
        self.region_models: Dict[str, Dict] = {}
        self._load_all_regions()

        if not self.region_models:
            self._load_global_model_as_fallback()

    def _load_all_regions(self):
        if not self.region_models_dir.exists():
            print(f"[WARN]  Region模型目录不存在: {self.region_models_dir}")
            return

        pattern = str(self.region_models_dir / "*")
        region_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
        print(f"\n[DIR] 扫描到 {len(region_dirs)} 个region模型目录")

        for region_dir in region_dirs:
            region_name = os.path.basename(region_dir)
            region_dir_path = Path(region_dir)
            model_file = region_dir_path / "svr_model.pkl"
            scaler_file = region_dir_path / "scaler.pkl"
            y_scaler_file = region_dir_path / "y_scaler.pkl"

            if model_file.exists() and scaler_file.exists():
                try:
                    lambda_min, lambda_max, th_min, th_max, lambda_min_ext, lambda_max_ext = self._extract_region_bounds(region_name, region_dir)
                    self.region_models[region_name] = {
                        "model": joblib.load(model_file),
                        "scaler": joblib.load(scaler_file),
                        "y_scaler": joblib.load(y_scaler_file) if y_scaler_file.exists() else None,
                        "lambda_min": lambda_min,
                        "lambda_max": lambda_max,
                        "lambda_min_ext": lambda_min_ext,
                        "lambda_max_ext": lambda_max_ext,
                        "th_min": th_min,
                        "th_max": th_max,
                        "region_dir": region_dir,
                    }
                    print(f"[OK] 加载region: {region_name:<25} 范围: {lambda_min:.1f}-{lambda_max:.1f}μm, {th_min:.0f}-{th_max:.0f}nm")
                except Exception as e:
                    print(f"[FAIL] 加载失败 {region_name}: {e}")
            else:
                print(f"[WARN]  模型文件缺失: {region_dir}")

        print(f"[DATA] 成功加载 {len(self.region_models)} 个region模型")

    def _load_region_configs(self) -> Dict[str, Dict]:
        configs = {}
        region_config_file = self.datadir / "region_config.json"
        if region_config_file.exists():
            try:
                full_config = self._load_json(region_config_file)
                configs = full_config.get("regions", {})
                print(f"[DIR] 从 {region_config_file} 加载了 {len(configs)} 个region配置")
            except Exception as e:
                print(f"[WARN]  加载region配置失败: {e}")
        return configs

    def _load_global_model_as_fallback(self):
        """尝试加载全局模型作为回退；若仍失败则保留空region_models并警告。"""
        global_model = self.datadir / "svr_emissivity_model.pkl"
        global_scaler = self.datadir / "scaler.pkl"
        global_y_scaler = self.datadir / "y_scaler.pkl"

        if global_model.exists() and global_scaler.exists():
            try:
                self.region_models["global"] = {
                    "model": joblib.load(global_model),
                    "scaler": joblib.load(global_scaler),
                    "y_scaler": joblib.load(global_y_scaler) if global_y_scaler.exists() else None,
                    "lambda_min": 0.3,
                    "lambda_max": 25.0,
                    "th_min": 100,
                    "th_max": 1000,
                    "region_dir": self.datadir,
                }
                print("[INFO]   使用全局模型作为回退")
            except Exception as e:
                print(f"[WARN]  全局模型加载失败: {e}（将在figure方法中动态构建）")
        else:
            print("[WARN]  未找到任何预训练模型（region或全局），将在figure方法中动态构建fallback模型")

    @property
    def is_empty(self) -> bool:
        """当没有任何可用region模型时返回True（包括global fallback）。"""
        return not bool(self.region_models)

    def _extract_region_bounds(self, region_name: str, region_dir: str) -> Tuple[float, float, float, float, float, float]:
        if region_name in self.region_configs:
            cfg = self.region_configs[region_name]
            lam_min = cfg.get("lambda_min", 0.3)
            lam_max = cfg.get("lambda_max", 25.0)
            lam_min_ext = cfg.get("lambda_min_ext", lam_min)
            lam_max_ext = cfg.get("lambda_max_ext", lam_max)
            return lam_min, lam_max, cfg.get("th_min", 100), cfg.get("th_max", 1000), lam_min_ext, lam_max_ext

        summary_file = self.datadir / "region_training_summary.json"
        if summary_file.exists():
            try:
                summary = self._load_json(summary_file)
                regions_cfg = summary.get("regions_config", {})
                if region_name in regions_cfg:
                    cfg = regions_cfg[region_name]
                    lam_min = cfg.get("lambda_min", 0.3)
                    lam_max = cfg.get("lambda_max", 25.0)
                    lam_min_ext = cfg.get("lambda_min_ext", lam_min)
                    lam_max_ext = cfg.get("lambda_max_ext", lam_max)
                    return lam_min, lam_max, cfg.get("th_min", 100), cfg.get("th_max", 1000), lam_min_ext, lam_max_ext
            except Exception:
                pass

        try:
            parts = region_name.split("_")
            th_min, th_max, lambda_min, lambda_max = 100, 1000, 0.3, 25.0
            for part in parts:
                if "-" in part and "nm" in part:
                    th_str = part.replace("nm", "")
                    th_min, th_max = map(float, th_str.split("-"))
                    break
            for part in parts:
                if "-" in part and "um" in part:
                    wl_str = part.replace("um", "")
                    lambda_min, lambda_max = map(float, wl_str.split("-"))
                    break
            return lambda_min, lambda_max, th_min, th_max, lambda_min, lambda_max
        except Exception:
            pass

        return 0.1, 30.0, 50.0, 1500.0, 0.1, 30.0

    def _prepare_model_input(self, scaler: Any, X_batch: np.ndarray) -> np.ndarray:
        """根据scaler期望维度自动构造模型输入（兼容3维基础特征与工程特征）。"""
        X_raw = np.asarray(X_batch, dtype=float)
        expected = int(getattr(scaler, "n_features_in_", X_raw.shape[1]))
        if expected == X_raw.shape[1]:
            return X_raw

        # 仅在原始3维输入时尝试按Stage1规则构造工程特征
        if X_raw.shape[1] == 3:
            wl = X_raw[:, 0]
            d_sio2 = X_raw[:, 1]
            d_pdms = X_raw[:, 2]

            n_pdms = float(self.config.get("n_pdms_eff", 1.41)) if isinstance(self.config, dict) else 1.41
            n_sio2 = float(self.config.get("n_sio2_eff", 1.46)) if isinstance(self.config, dict) else 1.46

            wl_m = wl * 1e-6
            pdms_m = d_pdms * 1e-9
            sio2_m = d_sio2 * 1e-9
            with np.errstate(divide="ignore", invalid="ignore"):
                delta_pdms = 2 * np.pi * n_pdms * pdms_m / wl_m
                delta_sio2 = 2 * np.pi * n_sio2 * sio2_m / wl_m
                inv_wl = 1.0 / wl
                d_over_wl = d_pdms / wl

            engineered = np.column_stack([
                X_raw,
                np.sin(delta_pdms), np.cos(delta_pdms),
                np.sin(delta_sio2), np.cos(delta_sio2),
                inv_wl, d_over_wl,
            ])

            use_inter = True
            if isinstance(self.config, dict):
                use_inter = bool(self.config.get("use_interaction_features", True))
            if use_inter:
                inter = np.column_stack([
                    np.sin(delta_pdms) * np.sin(delta_sio2),
                    np.cos(delta_pdms) * np.cos(delta_sio2),
                    np.sin(delta_pdms) * np.cos(delta_sio2),
                    np.cos(delta_pdms) * np.sin(delta_sio2),
                ])
                engineered = np.column_stack([engineered, inter])

            engineered = np.nan_to_num(engineered, nan=0.0, posinf=0.0, neginf=0.0)

            if engineered.shape[1] == expected:
                return engineered
            if engineered.shape[1] > expected:
                return engineered[:, :expected]

            pad = np.zeros((engineered.shape[0], expected - engineered.shape[1]), dtype=float)
            return np.column_stack([engineered, pad])

        if X_raw.shape[1] > expected:
            return X_raw[:, :expected]
        if X_raw.shape[1] < expected:
            pad = np.zeros((X_raw.shape[0], expected - X_raw.shape[1]), dtype=float)
            return np.column_stack([X_raw, pad])
        return X_raw

    def _predict_with_region(self, region: Dict, X_batch: np.ndarray) -> np.ndarray:
        X_model = self._prepare_model_input(region["scaler"], X_batch)
        X_scaled = region["scaler"].transform(X_model)
        pred = region["model"].predict(X_scaled)
        if region.get("y_scaler") is not None:
            try:
                pred = region["y_scaler"].inverse_transform(np.asarray(pred).reshape(-1, 1)).ravel()
            except Exception:
                pass
        return np.asarray(pred, dtype=float).ravel()

    def predict_emissivity(self, X: np.ndarray) -> np.ndarray:
        if not self.region_models:
            raise ValueError("没有可用的region模型")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("X必须是形状为(n, 3)的数组: [wavelength_um, sio2_thickness_nm, pdms_thickness_nm]")

        n_samples = X.shape[0]
        if n_samples == 0:
            return np.array([], dtype=float)

        wl_all = X[:, 0]
        pdms_all = X[:, 2]

        predictions = np.zeros(n_samples, dtype=float)
        used_regions = np.full(n_samples, "unknown", dtype=object)

        cfg = self.config if isinstance(self.config, dict) else {}
        min_wl_span = float(cfg.get("min_wl_span_um", 1e-3))
        min_th_span = float(cfg.get("min_th_span_nm", 1.0))
        min_weight = float(cfg.get("min_region_weight", 1e-6))
        wl_weight = float(cfg.get("wl_distance_weight", 1.0))
        th_weight = float(cfg.get("th_distance_weight", 1.0))
        weight_temperature = max(1e-6, float(cfg.get("region_weight_temperature", 1.0)))
        in_range_boost = max(1.0, float(cfg.get("in_range_weight_boost", 1.5)))

        region_items = list(self.region_models.items())

        candidate_counts = np.zeros(n_samples, dtype=int)
        primary_region_names = np.full(n_samples, "", dtype=object)
        region_match_masks: Dict[str, np.ndarray] = {}

        for region_name, region in region_items:
            wl_min_ext = region.get("lambda_min_ext", region["lambda_min"])
            wl_max_ext = region.get("lambda_max_ext", region["lambda_max"])
            mask = (
                (wl_all >= wl_min_ext) &
                (wl_all <= wl_max_ext) &
                (pdms_all >= region["th_min"]) &
                (pdms_all <= region["th_max"])
            )
            region_match_masks[region_name] = mask
            first_hit = mask & (candidate_counts == 0)
            primary_region_names[first_hit] = region_name
            candidate_counts += mask.astype(int)

        no_candidate_idx = np.where(candidate_counts == 0)[0]
        if no_candidate_idx.size > 0:
            wl_centers = np.array([(r["lambda_min"] + r["lambda_max"]) / 2 for _, r in region_items], dtype=float)
            th_centers = np.array([(r["th_min"] + r["th_max"]) / 2 for _, r in region_items], dtype=float)
            wl_spans = np.array([max(min_wl_span, r["lambda_max"] - r["lambda_min"]) for _, r in region_items], dtype=float)
            th_spans = np.array([max(min_th_span, r["th_max"] - r["th_min"]) for _, r in region_items], dtype=float)

            wl_dist = np.abs(wl_all[no_candidate_idx, None] - wl_centers[None, :]) / wl_spans[None, :]
            th_dist = np.abs(pdms_all[no_candidate_idx, None] - th_centers[None, :]) / th_spans[None, :]
            total_dist = wl_weight * wl_dist + th_weight * th_dist
            nearest = np.argmin(total_dist, axis=1)

            for k, sample_idx in enumerate(no_candidate_idx):
                region_name = region_items[int(nearest[k])][0]
                primary_region_names[sample_idx] = region_name
                candidate_counts[sample_idx] = 1

        single_idx = np.where(candidate_counts == 1)[0]
        if single_idx.size > 0:
            single_regions = primary_region_names[single_idx]
            for region_name in np.unique(single_regions):
                region = self.region_models[region_name]
                idx = single_idx[single_regions == region_name]
                predictions[idx] = self._predict_with_region(region, X[idx])
                used_regions[idx] = region_name

        multi_idx = np.where(candidate_counts > 1)[0]
        for i in multi_idx:
            wl, _, _ = X[i]
            candidates: List[Tuple[str, Dict]] = []
            for region_name, region in region_items:
                if region_match_masks[region_name][i]:
                    candidates.append((region_name, region))

            preds = []
            weights = []
            names = []
            for region_name, region in candidates:
                pred_val = float(self._predict_with_region(region, X[i:i + 1])[0])
                preds.append(pred_val)
                names.append(region_name)

                wl_center = (region["lambda_min"] + region["lambda_max"]) / 2
                wl_span = max(min_wl_span, region["lambda_max"] - region["lambda_min"])
                dist = abs(wl - wl_center) / wl_span
                weight = np.exp(-dist / weight_temperature)
                if region["lambda_min"] <= wl <= region["lambda_max"]:
                    weight *= in_range_boost
                weights.append(float(weight))

            weights = np.asarray(weights, dtype=float)
            weight_sum = float(weights.sum())
            if weight_sum <= 0.0 or not np.isfinite(weight_sum):
                weights = np.full(len(weights), 1.0 / len(weights), dtype=float)
            else:
                weights = (weights + min_weight) / (weight_sum + min_weight * len(weights))

            predictions[i] = float(np.dot(weights, np.asarray(preds, dtype=float)))
            used_regions[i] = "blend:" + "+".join(names)

        unique, counts = np.unique(used_regions, return_counts=True)
        if len(unique) > 1:
            print(f"\n[DATA] Region使用统计: {dict(zip(unique, counts))}")

        return np.clip(predictions, 0.0, 1.0)

    def predict_emissivity_hard(self, X: np.ndarray, *, use_extended_bounds: bool = False) -> np.ndarray:
        """Hard routing prediction without blending.

        This is primarily intended for ablation studies:
        - `use_extended_bounds=False` approximates a no-overlap strategy (use λ_min/λ_max only).
        - `use_extended_bounds=True` allows overlap ranges (λ_min_ext/λ_max_ext) but still selects a single region.
        """
        if not self.region_models:
            raise ValueError("没有可用的region模型")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("X必须是形状为(n, 3)的数组: [wavelength_um, sio2_thickness_nm, pdms_thickness_nm]")

        n_samples = X.shape[0]
        if n_samples == 0:
            return np.array([], dtype=float)

        wl_all = X[:, 0]
        pdms_all = X[:, 2]

        cfg = self.config if isinstance(self.config, dict) else {}
        min_wl_span = float(cfg.get("min_wl_span_um", 1e-3))
        min_th_span = float(cfg.get("min_th_span_nm", 1.0))
        wl_weight = float(cfg.get("wl_distance_weight", 1.0))
        th_weight = float(cfg.get("th_distance_weight", 1.0))

        region_items = list(self.region_models.items())

        # Candidate masks (vectorized)
        candidate_counts = np.zeros(n_samples, dtype=int)
        region_match_masks: Dict[str, np.ndarray] = {}
        for region_name, region in region_items:
            wl_min = region.get("lambda_min_ext", region["lambda_min"]) if use_extended_bounds else region["lambda_min"]
            wl_max = region.get("lambda_max_ext", region["lambda_max"]) if use_extended_bounds else region["lambda_max"]
            mask = (
                (wl_all >= wl_min) &
                (wl_all <= wl_max) &
                (pdms_all >= region["th_min"]) &
                (pdms_all <= region["th_max"])
            )
            region_match_masks[region_name] = mask
            candidate_counts += mask.astype(int)

        # Choose a single region for each sample.
        chosen_region = np.full(n_samples, "", dtype=object)

        # 1) Single candidate: pick it directly
        single_idx = np.where(candidate_counts == 1)[0]
        if single_idx.size > 0:
            for region_name, mask in region_match_masks.items():
                hit = mask[single_idx]
                if np.any(hit):
                    chosen_region[single_idx[hit]] = region_name

        # 2) Multiple candidates: pick nearest center (wl/th normalized)
        multi_idx = np.where(candidate_counts > 1)[0]
        if multi_idx.size > 0:
            wl_centers = np.array([(r["lambda_min"] + r["lambda_max"]) / 2 for _, r in region_items], dtype=float)
            th_centers = np.array([(r["th_min"] + r["th_max"]) / 2 for _, r in region_items], dtype=float)
            wl_spans = np.array([max(min_wl_span, r["lambda_max"] - r["lambda_min"]) for _, r in region_items], dtype=float)
            th_spans = np.array([max(min_th_span, r["th_max"] - r["th_min"]) for _, r in region_items], dtype=float)

            wl_dist = np.abs(wl_all[multi_idx, None] - wl_centers[None, :]) / wl_spans[None, :]
            th_dist = np.abs(pdms_all[multi_idx, None] - th_centers[None, :]) / th_spans[None, :]
            total_dist = wl_weight * wl_dist + th_weight * th_dist

            # Penalize regions that do not match bounds for this sample
            allowed = np.column_stack([
                region_match_masks[name][multi_idx] for name, _ in region_items
            ]).astype(bool)
            total_dist = np.where(allowed, total_dist, np.inf)
            nearest = np.argmin(total_dist, axis=1)
            for k, sample_idx in enumerate(multi_idx):
                chosen_region[sample_idx] = region_items[int(nearest[k])][0]

        # 3) No candidate: fallback to nearest region center
        no_candidate_idx = np.where(candidate_counts == 0)[0]
        if no_candidate_idx.size > 0:
            wl_centers = np.array([(r["lambda_min"] + r["lambda_max"]) / 2 for _, r in region_items], dtype=float)
            th_centers = np.array([(r["th_min"] + r["th_max"]) / 2 for _, r in region_items], dtype=float)
            wl_spans = np.array([max(min_wl_span, r["lambda_max"] - r["lambda_min"]) for _, r in region_items], dtype=float)
            th_spans = np.array([max(min_th_span, r["th_max"] - r["th_min"]) for _, r in region_items], dtype=float)

            wl_dist = np.abs(wl_all[no_candidate_idx, None] - wl_centers[None, :]) / wl_spans[None, :]
            th_dist = np.abs(pdms_all[no_candidate_idx, None] - th_centers[None, :]) / th_spans[None, :]
            total_dist = wl_weight * wl_dist + th_weight * th_dist
            nearest = np.argmin(total_dist, axis=1)
            for k, sample_idx in enumerate(no_candidate_idx):
                chosen_region[sample_idx] = region_items[int(nearest[k])][0]

        # Batch-predict per chosen region
        predictions = np.zeros(n_samples, dtype=float)
        for region_name in np.unique(chosen_region):
            if not region_name:
                continue
            idx = np.where(chosen_region == region_name)[0]
            if idx.size == 0:
                continue
            region = self.region_models[str(region_name)]
            predictions[idx] = self._predict_with_region(region, X[idx])

        return np.clip(predictions, 0.0, 1.0)

    def get_region_for_point(self, wavelength: float, thickness: float) -> Optional[str]:
        for region_name, region in self.region_models.items():
            if (region["lambda_min"] <= wavelength <= region["lambda_max"] and
                    region["th_min"] <= thickness <= region["th_max"]):
                return region_name
        return None
