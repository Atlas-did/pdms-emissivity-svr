from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator  # type: ignore[import-not-found]


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def normalize_runtime_config(params: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort coercion/clamping before strict model validation.

    This keeps compatibility with legacy config files while making
    `Config.validate()` concise and maintainable.
    """
    p = dict(params)

    p["test_size"] = min(0.5, max(0.05, _as_float(p.get("test_size", 0.3), 0.3)))
    p["cv_folds"] = max(2, _as_int(p.get("cv_folds", 3), 3))
    p["lambda_step"] = max(0.001, _as_float(p.get("lambda_step", 0.1), 0.1))
    p["thickness_step"] = max(1.0, _as_float(p.get("thickness_step", 50.0), 50.0))

    p["min_thickness_nm"] = max(1.0, _as_float(p.get("min_thickness_nm", 100.0), 100.0))
    p["max_thickness"] = max(
        p["min_thickness_nm"] + p["thickness_step"],
        _as_float(p.get("max_thickness", 1000.0), 1000.0),
    )

    p["lambda_min"] = max(0.01, _as_float(p.get("lambda_min", 0.3), 0.3))
    p["lambda_max"] = max(
        p["lambda_min"] + p["lambda_step"],
        _as_float(p.get("lambda_max", 25.0), 25.0),
    )
    if bool(p.get("limit_lambda_range", False)):
        p["lambda_min"] = max(2.0, p["lambda_min"])
        p["lambda_max"] = min(14.0, p["lambda_max"])
        if p["lambda_max"] <= p["lambda_min"]:
            p["lambda_max"] = p["lambda_min"] + p["lambda_step"]

    p["overlap_ratio"] = min(0.2, max(0.0, _as_float(p.get("overlap_ratio", 0.05), 0.05)))
    p["region_overlap_ratio"] = min(
        0.2,
        max(0.0, _as_float(p.get("region_overlap_ratio", p["overlap_ratio"]), p["overlap_ratio"])),
    )

    p["min_wl_span_um"] = max(1e-9, _as_float(p.get("min_wl_span_um", 1e-3), 1e-3))
    p["min_th_span_nm"] = max(1e-9, _as_float(p.get("min_th_span_nm", 1.0), 1.0))
    p["wl_distance_weight"] = max(1e-6, _as_float(p.get("wl_distance_weight", 1.0), 1.0))
    p["th_distance_weight"] = max(1e-6, _as_float(p.get("th_distance_weight", 1.0), 1.0))
    p["region_weight_temperature"] = max(1e-6, _as_float(p.get("region_weight_temperature", 1.0), 1.0))
    p["in_range_weight_boost"] = max(1.0, _as_float(p.get("in_range_weight_boost", 1.5), 1.5))
    p["min_region_weight"] = max(0.0, _as_float(p.get("min_region_weight", 1e-6), 1e-6))

    p["energy_check_n_points"] = max(50, _as_int(p.get("energy_check_n_points", 500), 500))
    p["extrapolation_n_samples"] = max(100, _as_int(p.get("extrapolation_n_samples", 1000), 1000))
    p["extrapolation_tmm_max_points"] = max(10, _as_int(p.get("extrapolation_tmm_max_points", 200), 200))

    p["max_samples_per_region"] = max(1000, _as_int(p.get("max_samples_per_region", 250000), 250000))
    p["batch_size"] = max(10, _as_int(p.get("batch_size", 100), 100))
    p["early_stopping_rounds"] = max(1, _as_int(p.get("early_stopping_rounds", 5), 5))
    p["cache_ttl_s"] = max(1.0, _as_float(p.get("cache_ttl_s", 180.0), 180.0))
    p["cache_maxsize"] = max(1000, _as_int(p.get("cache_maxsize", 60000), 60000))
    p["material_grid_step"] = max(0.001, _as_float(p.get("material_grid_step", p.get("lambda_step", 0.02)), 0.02))
    p["search_max_samples"] = max(100, _as_int(p.get("search_max_samples", 8000), 8000))
    p["trial_timeout_s"] = max(1.0, _as_float(p.get("trial_timeout_s", 60), 60.0))
    p["search_timeout_s"] = max(10.0, _as_float(p.get("search_timeout_s", 600), 600.0))

    p["ellipsometry_angle_deg"] = min(89.0, max(30.0, _as_float(p.get("ellipsometry_angle_deg", 70.0), 70.0)))
    p["ellipsometry_n0"] = max(0.5, _as_float(p.get("ellipsometry_n0", 1.0), 1.0))
    if p.get("ellipsometry_apply_to_material") not in {"none", "pdms", "sio2"}:
        p["ellipsometry_apply_to_material"] = "none"

    n_jobs = _as_int(p.get("n_jobs", -1), -1)
    if n_jobs == 0:
        n_jobs = 1
    p["n_jobs"] = n_jobs

    return p


class RuntimeConfigModel(BaseModel):
    """运行时配置模型（允许额外字段），用于替代裸字典校验。"""

    model_config = ConfigDict(extra="allow")

    # 关键参数（其余参数允许通过extra保留）
    test_size: float = Field(0.3, ge=0.05, le=0.5)
    cv_folds: int = Field(3, ge=2)

    lambda_step: float = Field(0.1, gt=0.0)
    thickness_step: float = Field(50.0, ge=1.0)
    lambda_min: float = Field(0.3, gt=0.0)
    lambda_max: float = Field(25.0, gt=0.0)

    min_thickness_nm: float = Field(100.0, ge=1.0)
    max_thickness: float = Field(1000.0, ge=1.0)

    overlap_ratio: float = Field(0.05, ge=0.0, le=0.2)
    region_overlap_ratio: float = Field(0.05, ge=0.0, le=0.2)

    min_wl_span_um: float = Field(1e-3, gt=0.0)
    min_th_span_nm: float = Field(1.0, gt=0.0)
    wl_distance_weight: float = Field(1.0, gt=0.0)
    th_distance_weight: float = Field(1.0, gt=0.0)
    region_weight_temperature: float = Field(1.0, gt=0.0)
    in_range_weight_boost: float = Field(1.5, ge=1.0)
    min_region_weight: float = Field(1e-6, ge=0.0)

    energy_check_n_points: int = Field(500, ge=50)
    extrapolation_n_samples: int = Field(1000, ge=100)
    extrapolation_tmm_max_points: int = Field(200, ge=10)

    cache_ttl_s: float = Field(180.0, ge=1.0)
    cache_maxsize: int = Field(60000, ge=1000)

    search_max_samples: int = Field(8000, ge=100)
    trial_timeout_s: float = Field(60.0, ge=1.0)
    search_timeout_s: float = Field(600.0, ge=10.0)

    ellipsometry_angle_deg: float = Field(70.0, ge=30.0, le=89.0)
    ellipsometry_n0: float = Field(1.0, ge=0.5)

    model_version: str = "svr-v1"
    enable_experiment_tracking: bool = False
    experiment_tracking_uri: str = ""
    experiment_name: str = "pdms_emissivity"

    use_gpu_tmm: bool = False
    gpu_backend: str = "auto"

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "RuntimeConfigModel":
        if self.lambda_max <= self.lambda_min:
            self.lambda_max = self.lambda_min + max(self.lambda_step, 1e-3)

        min_max_thickness = self.min_thickness_nm + self.thickness_step
        if self.max_thickness <= min_max_thickness:
            self.max_thickness = min_max_thickness

        # region overlap默认跟随overlap_ratio
        if self.region_overlap_ratio is None:
            self.region_overlap_ratio = self.overlap_ratio

        return self


def validate_runtime_config(params: Dict[str, Any]) -> Dict[str, Any]:
    """返回经过模型校验与归一化后的配置字典。"""
    try:
        model = RuntimeConfigModel(**params)
    except ValidationError as e:
        raise ValueError(f"配置校验失败: {e}") from e

    normalized = model.model_dump()
    # 保留额外字段
    for k, v in params.items():
        if k not in normalized:
            normalized[k] = v
    return normalized
