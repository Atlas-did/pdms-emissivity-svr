from __future__ import annotations

import threading
import time
from math import inf
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tmm
from joblib import Parallel, delayed

from stage1.core.io_utils import save_json
from stage1.core.observability import ErrorContext, log_exception_with_context, write_performance_summary
from stage1.core.performance import monitor_performance
from stage1.core.paths import OUTDIR, TMM_CSV
from stage1.tmm_cpu_threaded import batch_simulate_tmm


class TimedCache:
    """简单TTL缓存，用于TMM计算结果复用。"""

    def __init__(self, ttl_s: float = 120.0, maxsize: int = 50000):
        self.ttl_s = float(ttl_s)
        self.maxsize = int(maxsize)
        self._store: Dict[Any, Tuple[float, Any]] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        # Evidence for "hit_rate=0" explanation.
        self._total_requests = 0
        self._duplicate_key_count = 0
        self._seen_key_hashes: set[int] = set()
        self._seen_cap_reached = False
        self._seen_cap = 2_000_000

    def get(self, key: Any) -> Optional[Any]:
        with self._lock:
            self._total_requests += 1
            if not self._seen_cap_reached:
                try:
                    hk = hash(key)
                    if hk in self._seen_key_hashes:
                        self._duplicate_key_count += 1
                    else:
                        self._seen_key_hashes.add(hk)
                        if len(self._seen_key_hashes) >= self._seen_cap:
                            self._seen_cap_reached = True
                except Exception:
                    # Best-effort only; never break simulation due to stats.
                    pass
            now = time.time()
            item = self._store.get(key)
            if item is None:
                self._misses += 1
                return None
            ts, value = item
            if (now - ts) > self.ttl_s:
                self._store.pop(key, None)
                self._misses += 1
                return None
            self._hits += 1
            return value

    def set(self, key: Any, value: Any) -> None:
        with self._lock:
            if len(self._store) >= self.maxsize:
                for k in list(self._store.keys())[: max(1, self.maxsize // 10)]:
                    self._store.pop(k, None)
            self._store[key] = (time.time(), value)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": float(self._hits),
                "misses": float(self._misses),
                "hit_rate": (float(self._hits) / float(total)) if total > 0 else 0.0,
                "total_requests": int(self._total_requests),
                "unique_key_count": int(len(self._seen_key_hashes)),
                "duplicate_key_count": int(self._duplicate_key_count),
                "unique_key_count_capped": bool(self._seen_cap_reached),
            }


class TMMSimulator:
    MIN_REQUIRED_SAMPLES = 1974
    MIN_VALID_LAMBDA = 0.1
    MAX_VALID_LAMBDA = 50.0
    MIN_VALID_THICKNESS = 10.0
    MAX_VALID_THICKNESS = 10000.0
    SUBSTRATE_N = 1.0 + 100.0j

    @staticmethod
    def _quantize_to_bin(value: float, resolution: float) -> int:
        resolution = max(float(resolution), 1e-12)
        return int(np.floor(float(value) / resolution + 0.5))

    def __init__(self, material_data: Dict, cfg: Any):
        self.material_data = material_data
        self.cfg = cfg
        p = cfg.params

        self.lambda_min = max(self.MIN_VALID_LAMBDA, p.get("lambda_min", 0.3))
        self.lambda_max = min(self.MAX_VALID_LAMBDA, p.get("lambda_max", 25.0))
        if self.lambda_max <= self.lambda_min:
            self.lambda_max = self.lambda_min + 0.1

        self.lambda_step = max(0.01, p.get("lambda_step", 0.1))
        if self.lambda_step >= (self.lambda_max - self.lambda_min):
            self.lambda_step = (self.lambda_max - self.lambda_min) / 10

        max_thickness = max(self.MIN_VALID_THICKNESS, min(self.MAX_VALID_THICKNESS, p.get("max_thickness", 1000.0)))
        thickness_step = max(1.0, p.get("thickness_step", 50.0))
        if thickness_step >= max_thickness:
            thickness_step = max_thickness / 10

        min_thickness = max(self.MIN_VALID_THICKNESS, p.get("min_thickness_nm", 100.0))
        if min_thickness >= max_thickness:
            min_thickness = max(self.MIN_VALID_THICKNESS, max_thickness - thickness_step)
        self.pdms_thickness_range_nm = np.arange(min_thickness, max_thickness + 1e-9, thickness_step)

        self.sio2_thicknesses_nm = []
        for d in p.get("sio2_thicknesses_nm", [500, 1500, 2500]):
            if self.MIN_VALID_THICKNESS <= d <= self.MAX_VALID_THICKNESS:
                self.sio2_thicknesses_nm.append(d)
        if not self.sio2_thicknesses_nm:
            self.sio2_thicknesses_nm = [500]

        self.n_jobs = max(-1, p.get("n_jobs", -1))
        self.energy_tol = max(1e-6, p.get("energy_tol", 1e-3))
        # Keep default consistent with `stage1/core/config_runtime.py`.
        self.strict_energy_check = bool(p.get("strict_energy_check", True))
        self.energy_violations = []
        self._violations_lock = threading.RLock()
        self._cache = None
        self._io_write_count = 0
        self._batch_elapsed_s: list[float] = []
        if p.get("enable_tmm_cache", True):
            self._cache = TimedCache(p.get("cache_ttl_s", 180.0), p.get("cache_maxsize", 60000))

        substrate_n_real = p.get("substrate_n_real", None)
        substrate_n_imag = p.get("substrate_n_imag", None)
        allow_placeholder = bool(p.get("allow_placeholder_substrate", False))
        if substrate_n_real is None or substrate_n_imag is None:
            if allow_placeholder:
                print("⚠️  未提供基底光学常数，使用占位值 1.0+100.0j（仅用于调试，禁止用于论文）")
                self.substrate_n = self.SUBSTRATE_N
            else:
                raise ValueError("未配置基底光学常数（substrate_n_real/substrate_n_imag）。请在config.json中显式设置。")
        else:
            self.substrate_n = complex(float(substrate_n_real), float(substrate_n_imag))

        self.assume_opaque_substrate = bool(p.get("assume_opaque_substrate", False))

        print(f"初始化TMM仿真器: 波长范围 {self.lambda_min:.2f}-{self.lambda_max:.2f}μm, "
              f"PDMS厚度范围 {min(self.pdms_thickness_range_nm):.0f}-{max(self.pdms_thickness_range_nm):.0f}nm, "
              f"SiO2厚度 {self.sio2_thicknesses_nm}nm")
        print(f"基底光学常数: n={self.substrate_n.real:.4g}+{self.substrate_n.imag:.4g}j | assume_opaque_substrate={self.assume_opaque_substrate}")

    def simulate_point(self, lam_um: float, pdms_d_nm: float, sio2_d_nm: float) -> Dict[str, float]:
        try:
            cache_key = None
            if self._cache is not None:
                wl_q = self._quantize_to_bin(lam_um, self.cfg.params.get("cache_lambda_quant_um", 1e-4))
                pdms_q = self._quantize_to_bin(pdms_d_nm, self.cfg.params.get("cache_thickness_quant_nm", 0.1))
                sio2_q = self._quantize_to_bin(sio2_d_nm, self.cfg.params.get("cache_thickness_quant_nm", 0.1))
                cache_key = (
                    wl_q,
                    pdms_q,
                    sio2_q,
                    f"{self.substrate_n.real:.6g}",
                    f"{self.substrate_n.imag:.6g}",
                    int(self.assume_opaque_substrate),
                )
                cached = self._cache.get(cache_key)
                if cached is not None:
                    return cached

            lam = float(lam_um)
            pdms_d_nm = float(pdms_d_nm)
            sio2_d_nm = float(sio2_d_nm)

            if not (self.MIN_VALID_LAMBDA <= lam <= self.MAX_VALID_LAMBDA):
                raise ValueError(f"波长超出有效范围: {lam:.2f}μm")
            if not (self.MIN_VALID_THICKNESS <= pdms_d_nm <= self.MAX_VALID_THICKNESS):
                raise ValueError(f"PDMS厚度超出有效范围: {pdms_d_nm:.0f}nm")
            if not (self.MIN_VALID_THICKNESS <= sio2_d_nm <= self.MAX_VALID_THICKNESS):
                raise ValueError(f"SiO2厚度超出有效范围: {sio2_d_nm:.0f}nm")

            pdms_d_m = pdms_d_nm * 1e-9
            sio2_d_m = sio2_d_nm * 1e-9

            try:
                if "pdms_lam_min" in self.material_data and "pdms_lam_max" in self.material_data:
                    if not (self.material_data["pdms_lam_min"] <= lam <= self.material_data["pdms_lam_max"]):
                        if not self.cfg.params.get("external_extrapolation", True):
                            raise ValueError(
                                f"波长 {lam:.2f}μm 超出PDMS材料数据范围 ({self.material_data['pdms_lam_min']:.2f}-{self.material_data['pdms_lam_max']:.2f}μm)")
                if "sio2_lam_min" in self.material_data and "sio2_lam_max" in self.material_data:
                    if not (self.material_data["sio2_lam_min"] <= lam <= self.material_data["sio2_lam_max"]):
                        if not self.cfg.params.get("external_extrapolation", True):
                            raise ValueError(
                                f"波长 {lam:.2f}μm 超出SiO2材料数据范围 ({self.material_data['sio2_lam_min']:.2f}-{self.material_data['sio2_lam_max']:.2f}μm)")

                pdms_n_val = self.material_data["pdms_n"](lam)
                pdms_k_val = self.material_data["pdms_k"](lam)
                sio2_n_val = self.material_data["sio2_n"](lam)
                sio2_k_val = self.material_data["sio2_k"](lam)
            except Exception as e:
                raise RuntimeError(f"材料参数插值失败: {e}")

            if not np.isfinite(pdms_n_val) or pdms_n_val <= 0:
                raise ValueError(f"无效的PDMS折射率n: {pdms_n_val}")
            pdms_k_val = 0.0 if (not np.isfinite(pdms_k_val) or pdms_k_val < 0.0) else pdms_k_val

            if not np.isfinite(sio2_n_val) or sio2_n_val <= 0:
                raise ValueError(f"无效的SiO2折射率n: {sio2_n_val}")
            sio2_k_val = 0.0 if (not np.isfinite(sio2_k_val) or sio2_k_val < 0.0) else sio2_k_val

            pdms_n = complex(pdms_n_val, pdms_k_val)
            sio2_n = complex(sio2_n_val, sio2_k_val)
            n_list = [1.0, pdms_n, sio2_n, self.substrate_n]
            d_list = [inf, pdms_d_m, sio2_d_m, inf]

            try:
                coh_s = tmm.coh_tmm("s", n_list, d_list, 0.0, lam * 1e-6)
                coh_p = tmm.coh_tmm("p", n_list, d_list, 0.0, lam * 1e-6)
            except Exception as e:
                raise RuntimeError(f"TMM计算失败: {e}")

            R_s = coh_s.get("R", 0.0)
            R_p = coh_p.get("R", 0.0)
            T_s = coh_s.get("T", 0.0)
            T_p = coh_p.get("T", 0.0)

            if not (np.isfinite(R_s) and np.isfinite(R_p)):
                raise ValueError(f"TMM计算结果包含无效值: R_s={R_s}, R_p={R_p}")

            R = 0.5 * (R_s + R_p)
            T = 0.5 * (T_s + T_p)
            if self.assume_opaque_substrate:
                T = 0.0

            eps = 1.0 - R - T
            eps = float(np.clip(eps, 0.0, 1.0))
            R = float(np.clip(R, 0.0, 1.0))
            T = float(np.clip(T, 0.0, 1.0))

            energy = R + T + eps
            if abs(energy - 1.0) > self.energy_tol:
                entry = {"λ_μm": lam, "基底厚度_nm": float(sio2_d_nm), "PDMS厚度_nm": float(pdms_d_nm),
                         "R": R, "T": T, "ε": eps, "sum": float(energy)}
                with self._violations_lock:
                    self.energy_violations.append(entry)
                if self.strict_energy_check:
                    return {"波长_μm": round(lam, 6), "基底厚度_nm": int(round(sio2_d_nm)),
                            "PDMS厚度_nm": int(round(pdms_d_nm)),
                            "反射率R": np.nan, "透射率T": np.nan, "发射率ε": np.nan}

            result = {"波长_μm": round(lam, 6), "基底厚度_nm": int(round(sio2_d_nm)),
                      "PDMS厚度_nm": int(round(pdms_d_nm)),
                      "反射率R": R, "透射率T": T, "发射率ε": eps}
            if self._cache is not None and cache_key is not None:
                self._cache.set(cache_key, result)
            return result
        except Exception as e:
            log_exception_with_context(
                e,
                ErrorContext(
                    stage="stage1.tmm.simulate_point",
                    region="global",
                    file=str(TMM_CSV),
                    sample_range=f"λ={float(lam_um):.6g},pdms={float(pdms_d_nm):.6g},sio2={float(sio2_d_nm):.6g}",
                ),
                print_traceback=False,
            )
            return {"波长_μm": round(float(lam_um), 6) if np.isfinite(float(lam_um)) else np.nan,
                    "基底厚度_nm": int(round(sio2_d_nm)) if np.isfinite(sio2_d_nm) else np.nan,
                    "PDMS厚度_nm": int(round(pdms_d_nm)) if np.isfinite(pdms_d_nm) else np.nan,
                    "反射率R": np.nan, "透射率T": np.nan, "发射率ε": np.nan}

    @monitor_performance("tmm.run_simulation")
    def run_simulation(self) -> pd.DataFrame:
        print("\n[TMM] Starting simulation...")
        p = self.cfg.params
        wavelengths = np.arange(self.lambda_min, self.lambda_max + self.lambda_step, self.lambda_step)
        tasks = [(lam, pdms, sio2) for sio2 in self.sio2_thicknesses_nm
                 for pdms in self.pdms_thickness_range_nm for lam in wavelengths]
        total = len(tasks)
        min_required = int(p.get("min_required_samples", 0))
        if p.get("enforce_min_samples", True) and min_required > 0 and total < min_required:
            raise ValueError(f"采样点不足: {total} < {min_required}. 请减小步长或增大范围。")

        start = time.time()
        results = []
        batch_size = int(p.get("batch_size", 100))
        backend = p.get("joblib_backend", "threading")
        n_batches = max(1, (total + batch_size - 1) // batch_size)
        last_report = time.time()
        for b, i in enumerate(range(0, total, batch_size), start=1):
            batch = tasks[i:i + batch_size]
            batch_t0 = time.time()
            try:
                if p.get("use_gpu_tmm", False):
                    batch_results = batch_simulate_tmm(self, batch, backend=p.get("gpu_backend", "auto"))
                else:
                    batch_results = Parallel(n_jobs=self.n_jobs, backend=backend)(
                        delayed(self.simulate_point)(lam, pdms, sio2) for lam, pdms, sio2 in batch
                    )
                results.extend(batch_results)
            except Exception as e:
                log_exception_with_context(
                    e,
                    ErrorContext(
                        stage="stage1.tmm.run_simulation.batch",
                        region="global",
                        file=str(TMM_CSV),
                        sample_range=f"{i}-{i + len(batch)}",
                    ),
                    print_traceback=False,
                )
            finally:
                self._batch_elapsed_s.append(time.time() - batch_t0)

            now = time.time()
            if now - last_report > 3 or b == n_batches:
                done = min(i + len(batch), total)
                pct = 100.0 * done / max(1, total)
                print(f"[TMM] 进度 {b}/{n_batches} | {done}/{total} ({pct:.1f}%)")
                last_report = now

        df = pd.DataFrame(results).sort_values(by=["基底厚度_nm", "PDMS厚度_nm", "波长_μm"]).reset_index(drop=True)
        valid_df = df.dropna(subset=["反射率R", "透射率T", "发射率ε"])
        invalid_count = len(df) - len(valid_df)

        valid_df.to_csv(TMM_CSV, index=False, encoding="utf-8-sig")
        self._io_write_count += 1
        elapsed = time.time() - start
        print(f"[TMM] Done: {len(valid_df)} valid rows (filtered {invalid_count} invalid rows), elapsed {elapsed:.1f}s, saved to {TMM_CSV}")

        with self._violations_lock:
            violations_snapshot = list(self.energy_violations)
        if violations_snapshot:
            violations_snapshot.sort(key=lambda x: (x.get("λ_μm", 0.0), x.get("基底厚度_nm", 0), x.get("PDMS厚度_nm", 0)))
            ev_file = OUTDIR / "energy_violations.json"
            rc = self.cfg.params.get("_run_context") if isinstance(self.cfg.params.get("_run_context"), dict) else None
            save_json(ev_file, violations_snapshot, run_context=rc)
            self._io_write_count += 1
            print(f"[TMM] Found {len(violations_snapshot)} energy violations")
            if invalid_count > 0 and self.strict_energy_check:
                print(f"[TMM] 已过滤 {invalid_count} 个能量不守恒的数据点")
            print(f"[TMM] 详细违规信息已保存到: {ev_file}")

        cache_stats = self._cache.get_stats() if self._cache is not None else {
            "hits": 0.0,
            "misses": 0.0,
            "hit_rate": 0.0,
            "total_requests": 0,
            "unique_key_count": 0,
            "duplicate_key_count": 0,
            "unique_key_count_capped": False,
        }
        perf_summary = {
            "stage": "stage1",
            "module": "tmm_simulation",
            "tmm_rows_valid": int(len(valid_df)),
            "tmm_rows_invalid": int(invalid_count),
            "tmm_elapsed_s": float(elapsed),
            "tmm_batches": int(n_batches),
            "tmm_batch_elapsed_s_mean": float(np.mean(self._batch_elapsed_s)) if self._batch_elapsed_s else 0.0,
            "tmm_batch_elapsed_s_max": float(np.max(self._batch_elapsed_s)) if self._batch_elapsed_s else 0.0,
            "energy_tol": float(self.energy_tol),
            "strict_energy_check": bool(self.strict_energy_check),
            "tmm_cache_hits": float(cache_stats.get("hits", 0.0)),
            "tmm_cache_misses": float(cache_stats.get("misses", 0.0)),
            "tmm_cache_hit_rate": float(cache_stats.get("hit_rate", 0.0)),
            "tmm_cache_total_requests": int(cache_stats.get("total_requests", 0)),
            "tmm_cache_unique_key_count": int(cache_stats.get("unique_key_count", 0)),
            "tmm_cache_duplicate_key_count": int(cache_stats.get("duplicate_key_count", 0)),
            "tmm_cache_unique_key_count_capped": bool(cache_stats.get("unique_key_count_capped", False)),
            "io_write_count": int(self._io_write_count),
        }
        rc = self.cfg.params.get("_run_context") if isinstance(self.cfg.params.get("_run_context"), dict) else None
        write_performance_summary(OUTDIR, "performance_summary_stage1_tmm", perf_summary, run_context=rc)
        return valid_df

    def run_parallel_simulation_by_region(self, material_data: Dict, regions: Dict[str, Any]) -> pd.DataFrame:
        tasks = []
        p = self.cfg.params
        thickness_step = p["thickness_step"]
        lambda_step = p["lambda_step"]
        min_thickness = p.get("min_thickness_nm", 100.0)
        all_thicknesses = np.arange(min_thickness, p["max_thickness"] + 1e-9, thickness_step)
        all_wavelengths = np.arange(p["lambda_min"], p["lambda_max"] + lambda_step, lambda_step)
        eps = float(p.get("region_edge_eps", 1e-6))
        for region_name, region_info in regions.items():
            wl_indices = np.where(
                (all_wavelengths >= region_info["lambda_min"] - eps) &
                (all_wavelengths <= region_info["lambda_max"] + eps)
            )[0]
            th_indices = np.where(
                (all_thicknesses >= region_info["th_min"] - eps) &
                (all_thicknesses <= region_info["th_max"] + eps)
            )[0]
            for sio2 in p["sio2_thicknesses_nm"]:
                for pdms in all_thicknesses[th_indices]:
                    for lam in all_wavelengths[wl_indices]:
                        tasks.append((lam, pdms, sio2, region_name))

        min_required = int(p.get("min_required_samples", 0))
        if p.get("enforce_min_samples", True) and min_required > 0 and len(tasks) < min_required:
            raise ValueError(f"分区采样点不足: {len(tasks)} < {min_required}. 请调整分区或步长。")

        print(f"\n[RegionSplitter] 并行模拟 {len(tasks)} 个任务...")
        backend = p.get("joblib_backend", "loky")
        results = []
        batch_size = int(p.get("batch_size", 100))
        n_batches = max(1, (len(tasks) + batch_size - 1) // batch_size)
        last_report = time.time()
        for b, i in enumerate(range(0, len(tasks), batch_size), start=1):
            batch = tasks[i:i + batch_size]
            try:
                batch_results = Parallel(n_jobs=p.get("n_jobs", -1), backend=backend)(
                    delayed(self._simulate_point_with_region)(lam, pdms, sio2, region_name, material_data)
                    for lam, pdms, sio2, region_name in batch
                )
                results.extend(batch_results)
            except Exception as e:
                print(f"[RegionSplitter] 批处理失败: batch {i}-{i + len(batch)}，错误: {e}")

            now = time.time()
            if now - last_report > 3 or b == n_batches:
                done = min(i + len(batch), len(tasks))
                pct = 100.0 * done / max(1, len(tasks))
                print(f"[RegionSplitter] 进度 {b}/{n_batches} | {done}/{len(tasks)} ({pct:.1f}%)")
                last_report = now

        if results:
            full_df = pd.DataFrame(results)
            full_df = full_df.sort_values(by=["region", "基底厚度_nm", "PDMS厚度_nm", "波长_μm"]).reset_index(drop=True)

            valid_df = full_df.dropna(subset=["反射率R", "透射率T", "发射率ε"])
            invalid_count = len(full_df) - len(valid_df)

            output_file = OUTDIR / "tmm_emissivity_data_regioned.csv"
            valid_df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"\n[RegionSplitter] 所有region模拟完成! 有效行数: {len(valid_df)} (过滤了 {invalid_count} 个无效数据点)")
            print(f"结果已保存到: {output_file}")
            return valid_df

        print("[RegionSplitter] 没有生成任何结果!")
        return pd.DataFrame()

    def _simulate_point_with_region(self, lam_um, pdms_d_nm, sio2_d_nm, region_name, material_data):
        result = self.simulate_point(lam_um, pdms_d_nm, sio2_d_nm)
        result["region"] = region_name
        return result
