from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from scipy.interpolate import CubicSpline


DEFAULT_MATERIAL_INFO = {
    "pdms_file": "pdms_nk.xlsx",
    "sio2_file": "sio2_nk.xlsx",
    "pdms_sample": 3,
    "sio2_sample": 2,
}


def load_material_data(material_info: Dict[str, Any], root: Path) -> Dict[str, Any]:
    """加载光学常数数据，并返回样条函数与采样点统计。"""
    pdms_file = material_info.get("pdms_file", str(root / "stage1" / "pdms_nk.xlsx"))
    sio2_file = material_info.get("sio2_file", str(root / "stage1" / "sio2_nk.xlsx"))
    pdms_sample = int(material_info.get("pdms_sample", 3))
    sio2_sample = int(material_info.get("sio2_sample", 2))

    def read_map(path: Any, sample: int):
        path = Path(path) if not isinstance(path, Path) else path

        if not path.exists():
            fallback_candidates = [
                Path(os.path.basename(path)),
                root / "stage1" / os.path.basename(path),
                root / "data" / "raw" / "material_nk" / os.path.basename(path),
            ]
            for alt_path in fallback_candidates:
                if alt_path.exists():
                    path = alt_path
                    break
            else:
                raise FileNotFoundError(f"找不到材料文件: {path}")

        df = pd.read_excel(path)
        candidates = {
            "lambda": ["波长 (μm)", "wavelength (um)", "lambda_um", "lambda (um)", "Wavelength(μm)"],
            "n": ["折射率 (n)", "n", "refractive_index_n", "Refractive index n"],
            "k": ["消光系数 (k)", "k", "extinction_k", "Extinction coefficient k"],
        }
        col_map = {}
        for key, opts in candidates.items():
            for opt in opts:
                if opt in df.columns:
                    col_map[key] = opt
                    break

        if len(col_map) != 3:
            raise ValueError(f"Material file {path} missing expected columns. Found: {df.columns.tolist()}")

        df = df.rename(columns={col_map["lambda"]: "lambda_um", col_map["n"]: "n", col_map["k"]: "k"})
        if sample > 1:
            df = df.iloc[::sample].reset_index(drop=True)
        lam = df["lambda_um"].astype(float).values
        n = df["n"].astype(float).values
        k = df["k"].astype(float).values
        return CubicSpline(lam, n, extrapolate=True), CubicSpline(lam, k, extrapolate=True), len(lam)

    pdms_n_s, pdms_k_s, pdms_count = read_map(pdms_file, pdms_sample)
    sio2_n_s, sio2_k_s, sio2_count = read_map(sio2_file, sio2_sample)

    return {
        "pdms_n": pdms_n_s,
        "pdms_k": pdms_k_s,
        "sio2_n": sio2_n_s,
        "sio2_k": sio2_k_s,
        "pdms_count": pdms_count,
        "sio2_count": sio2_count,
    }


def load_data_bundle(datadir: Any, region_models_dir: Any, load_json_fn, router_cls, root: Path) -> Dict[str, Any]:
    """集中加载DataLoader所需的IO资源，返回状态字典。"""
    datadir = Path(datadir) if not isinstance(datadir, Path) else datadir
    region_models_dir = Path(region_models_dir) if not isinstance(region_models_dir, Path) else region_models_dir

    result: Dict[str, Any] = {
        "config": {},
        "material_info": {},
        "material_data": {},
        "tmm_data": pd.DataFrame(),
        "processed_data": pd.DataFrame(),
        "model_router": None,
        "test_results": {},
        "speed_results": {},
        "model_info": {},
        "region_summary": {},
        "scaler": None,
    }

    cfg_path = datadir / "config.json"
    if cfg_path.exists():
        result["config"] = load_json_fn(cfg_path)

    mi_path = datadir / "material_info.json"
    if mi_path.exists():
        result["material_info"] = load_json_fn(mi_path)
    else:
        result["material_info"] = dict(DEFAULT_MATERIAL_INFO)

    result["material_data"] = load_material_data(result["material_info"], root)

    tmm_file = datadir / "tmm_emissivity_data.csv"
    if not tmm_file.exists():
        tmm_file_alt = datadir / "tmm_emissivity_data_regioned.csv"
        if tmm_file_alt.exists():
            tmm_file = tmm_file_alt
        else:
            # Fallback: look in stage1/simulation_data
            tmm_file_stage1 = root / "stage1" / "simulation_data" / "tmm_emissivity_data.csv"
            if tmm_file_stage1.exists():
                tmm_file = tmm_file_stage1
            else:
                print("[WARN]  未找到TMM数据文件，将创建空DataFrame（后续figure方法会生成合成数据）")
                result["tmm_data"] = pd.DataFrame(columns=["波长_μm", "基底厚度_nm", "PDMS厚度_nm", "发射率ε"])
                return result

    result["tmm_data"] = pd.read_csv(tmm_file).dropna(subset=["发射率ε"])

    proc_file = datadir / "processed_data.csv"
    if proc_file.exists():
        result["processed_data"] = pd.read_csv(proc_file).dropna(subset=["发射率ε"])
    else:
        result["processed_data"] = result["tmm_data"].copy()

    result["model_router"] = router_cls(region_models_dir, datadir)

    test_file = datadir / "test_results.json"
    if test_file.exists():
        result["test_results"] = load_json_fn(test_file)

    speed_file = datadir / "speed_results.json"
    if speed_file.exists():
        result["speed_results"] = load_json_fn(speed_file)

    model_info_file = datadir / "model_info.json"
    if model_info_file.exists():
        result["model_info"] = load_json_fn(model_info_file)

    region_summary_file = datadir / "region_training_summary.json"
    if region_summary_file.exists():
        result["region_summary"] = load_json_fn(region_summary_file)

    scaler_file = datadir / "scaler.pkl"
    if scaler_file.exists():
        result["scaler"] = joblib.load(scaler_file)

    return result
