from __future__ import annotations

from pathlib import Path

# Resolve project root from this file location:
# .../stage1/core/paths.py -> project root is parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT_ROOT / "stage1" / "simulation_data"
OUTDIR.mkdir(parents=True, exist_ok=True)

TMM_CSV = OUTDIR / "tmm_emissivity_data.csv"
MODEL_PKL = OUTDIR / "svr_emissivity_model.pkl"
SCALER_PKL = OUTDIR / "scaler.pkl"
Y_SCALER_PKL = OUTDIR / "y_scaler.pkl"
MODEL_BUNDLE_PKL = OUTDIR / "model_bundle.joblib"
MODEL_BUNDLE_META_JSON = OUTDIR / "model_bundle_meta.json"
TEST_RESULTS_JSON = OUTDIR / "test_results.json"
SPEED_RESULTS_JSON = OUTDIR / "speed_results.json"
MODEL_INFO_JSON = OUTDIR / "model_info.json"
MATERIAL_INFO_JSON = OUTDIR / "material_info.json"
REGION_CONFIG_JSON = OUTDIR / "region_config.json"
REGION_SUMMARY_JSON = OUTDIR / "region_training_summary.json"
BANDSCAN_SUMMARY = OUTDIR / "bandscan_summary.csv"
REGION_MODELS_DIR = OUTDIR / "region_models"
