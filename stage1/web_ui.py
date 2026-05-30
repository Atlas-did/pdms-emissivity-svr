from __future__ import annotations

import io
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import streamlit as st  # type: ignore[import-not-found]

from stage1.main_pdms_svr_bandscan_full import (
    Config,
    MATERIAL_INFO_JSON,
    MaterialDataLoader,
    SVRTrainer,
    TMMSimulator,
    save_json,
)


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "simulation_data" / "ui_uploads"


def _save_uploads(pdms_file, sio2_file) -> tuple[Path, Path]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    pdms_path = UPLOAD_DIR / "pdms_nk_uploaded.xlsx"
    sio2_path = UPLOAD_DIR / "sio2_nk_uploaded.xlsx"
    pdms_path.write_bytes(pdms_file.getbuffer())
    sio2_path.write_bytes(sio2_file.getbuffer())
    return pdms_path, sio2_path


def _run_full_pipeline(cfg: Config, pdms_path: Path, sio2_path: Path, run_speed_test: bool) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        loader = MaterialDataLoader(pdms_file=str(pdms_path), sio2_file=str(sio2_path), config=cfg)
        material_data = loader.load_all(
            pdms_sample=int(cfg.params.get("pdms_sample", 3)),
            sio2_sample=int(cfg.params.get("sio2_sample", 2)),
        )

        simulator = TMMSimulator(material_data, cfg)
        df = simulator.run_simulation()

        trainer = SVRTrainer(cfg)
        X_train, X_test, y_train, y_test, processed_df = trainer.preprocess(df)
        trainer.train(X_train, y_train)
        trainer.evaluate(X_test, y_test, processed_df)
        trainer.save_model()

        if run_speed_test:
            trainer.benchmark_speed(n_spectra=100)

        save_json(
            MATERIAL_INFO_JSON,
            {
                "pdms_file": str(pdms_path),
                "sio2_file": str(sio2_path),
                "pdms_sample": cfg.params.get("pdms_sample", 3),
                "sio2_sample": cfg.params.get("sio2_sample", 2),
                "config": cfg.params,
            },
        )

    return buffer.getvalue()


def main() -> None:
    st.set_page_config(page_title="PDMS SVR 控制台", layout="wide")
    st.title("PDMS 发射率训练控制台（离线本地）")
    st.caption("该页面运行在本机 127.0.0.1，不依赖外网。")

    cfg = Config()
    cfg.load()

    st.subheader("材料表格导入")
    up1, up2 = st.columns(2)
    with up1:
        pdms_upload = st.file_uploader("PDMS 光学常数表格 (.xlsx)", type=["xlsx"], key="pdms")
    with up2:
        sio2_upload = st.file_uploader("SiO2 光学常数表格 (.xlsx)", type=["xlsx"], key="sio2")

    st.subheader("核心配置")
    c1, c2, c3 = st.columns(3)
    with c1:
        cfg.params["lambda_min"] = st.number_input("lambda_min", value=float(cfg.params.get("lambda_min", 2.0)))
        cfg.params["lambda_max"] = st.number_input("lambda_max", value=float(cfg.params.get("lambda_max", 14.0)))
    with c2:
        cfg.params["min_thickness_nm"] = st.number_input("min_thickness_nm", value=float(cfg.params.get("min_thickness_nm", 100.0)))
        cfg.params["max_thickness"] = st.number_input("max_thickness", value=float(cfg.params.get("max_thickness", 1000.0)))
    with c3:
        cfg.params["test_size"] = st.slider("test_size", min_value=0.05, max_value=0.5, value=float(cfg.params.get("test_size", 0.3)), step=0.01)
        cfg.params["cv_folds"] = st.number_input("cv_folds", value=int(cfg.params.get("cv_folds", 3)), min_value=2, max_value=10)

    if st.button("保存配置"):
        cfg.save()
        st.success("配置已保存")

    st.subheader("执行")
    run_speed_test = st.checkbox("附加速度测试", value=True)
    if st.button("一键运行（仿真 -> 训练 -> 评估 -> 保存）"):
        if pdms_upload is None or sio2_upload is None:
            st.error("请先上传 PDMS 与 SiO2 两个 .xlsx 表格。")
            return
        try:
            pdms_path, sio2_path = _save_uploads(pdms_upload, sio2_upload)
            with st.spinner("正在运行，请稍候..."):
                logs = _run_full_pipeline(cfg, pdms_path, sio2_path, run_speed_test)
            st.success("流程运行完成。")
            st.code((logs or "(无控制台输出)")[-12000:])
            st.info("结果文件在 simulation_data/ 目录。")
        except Exception as e:
            st.error(f"运行失败: {e}")
            st.code(traceback.format_exc()[-12000:])


if __name__ == "__main__":
    main()
