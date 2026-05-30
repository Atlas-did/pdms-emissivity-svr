"""Stage2 core modules for router and figure data preparation."""

from .region_router import MultiRegionModelRouter
from .figure_data_builder import (
	build_efficiency_metrics,
	build_region_overview_lines,
	load_energy_violations_dataframe,
)
from .data_loader_io import load_data_bundle, load_material_data
from .data_loader_reporting import (
	apply_data_bundle,
	validate_router_state,
	print_load_overview,
	print_optional_status,
)
from .data_loader_recovery import clear_loader_state, handle_load_failure
from .data_loader_pipeline import execute_data_loading_pipeline
from .figure_registry import FigureTask, default_figure_tasks, key_figure_ids
from .figure_renderer import (
	run_figure_task,
	run_figure_batch,
	summarize_figure_results,
	check_key_figure_outputs,
	persist_figure_batch_summary,
)
from .observability import ErrorContext, log_exception_with_context, write_lightweight_summary, build_figure_batch_payload

__all__ = [
	"MultiRegionModelRouter",
	"build_efficiency_metrics",
	"build_region_overview_lines",
	"load_energy_violations_dataframe",
	"load_data_bundle",
	"load_material_data",
	"apply_data_bundle",
	"validate_router_state",
	"print_load_overview",
	"print_optional_status",
	"clear_loader_state",
	"handle_load_failure",
	"execute_data_loading_pipeline",
	"FigureTask",
	"default_figure_tasks",
	"key_figure_ids",
	"run_figure_task",
	"run_figure_batch",
	"summarize_figure_results",
	"check_key_figure_outputs",
	"persist_figure_batch_summary",
	"ErrorContext",
	"log_exception_with_context",
	"write_lightweight_summary",
	"build_figure_batch_payload",
]
