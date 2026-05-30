from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FigureTask:
    figure_id: str
    method_name: str
    output_name: str
    data_source: str = "stage2-loader"


def default_figure_tasks() -> List[FigureTask]:
    return [
        FigureTask("Figure 1", "figure1_optical_constants", "figure1_optical_constants.png"),
        FigureTask("Figure 2", "figure2_key_thickness_spectra", "figure2_key_thickness_spectra.png"),
        FigureTask("Figure 3", "figure3_3d_surface", "figure3_3d_surface.png"),
        FigureTask("Figure 4", "figure4_scatter_plot", "figure4_scatter_plot.png"),
        FigureTask("Figure 5", "figure5_residual_analysis", "figure5_residual_analysis.png"),
        FigureTask("Figure 6", "figure6_literature_comparison", "figure6_literature_comparison.png"),
        FigureTask("Figure 8", "figure8_learning_curves", "figure8_region_performance_overview.png"),
        FigureTask("Figure 9", "figure9_energy_conservation", "figure9_energy_conservation.png"),
        FigureTask("Figure 14", "figure14_model_comparison", "figure14_model_comparison.png"),
        FigureTask("Figure 5 Ablation", "figure5_ablation_study", "figure5_ablation_study.png"),
        FigureTask("Figure 15", "figure15_direct_validation", "figure15_direct_validation.png"),
        FigureTask("Figure 7", "figure7_region_metrics", "figure7_region_metrics.png"),
        FigureTask("Figure 6 Boundary", "figure6_boundary_error", "figure6_boundary_analysis.png"),
        FigureTask("Figure 3 Flow", "figure3_process_flow", "figure2_framework.png"),
        FigureTask("Figure 16", "figure16_extrapolation", "figure16_uncertainty.png"),
        FigureTask("Figure 4 Sampling", "figure4_sampling_strategy", "figure4_sampling_strategy.png"),
        FigureTask("Figure 8 Efficiency", "figure8_efficiency", "figure8_efficiency.png"),
        FigureTask("Figure 17", "figure17_physical_constraints_effect", "figure17_physical_constraints_effect.png"),
        FigureTask("Figure 18", "figure18_ellipsometry_validation", "figure18_ellipsometry_validation.png"),
        FigureTask("Figure 19", "figure19_gpr_uncertainty", "figure19_gpr_uncertainty.png"),
        FigureTask("Figure 20", "figure20_temperature_dependence", "figure20_temperature_dependence.png"),
    ]


def key_figure_ids() -> List[str]:
    return ["Figure 3", "Figure 7", "Figure 9", "Figure 14", "Figure 16",
            "Figure 18", "Figure 19", "Figure 20"]
