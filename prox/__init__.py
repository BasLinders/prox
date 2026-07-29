"""
prox — Process Excavator engine.

Public API
----------
run_full_analysis       Full pipeline: filter → discover → conform → analyse → visualise
load_and_validate_csv   CSV ingestion with auto column mapping
create_analysis_config  Build a config dict for run_full_analysis
CONFIG                  Default configuration
"""

from .pipeline import run_full_analysis
from .data_manager import (
    load_and_validate_csv,
    refine_activity_labels,
    optimize_dataframe_memory,
    filter_event_log,
    sample_log_stratified,
    check_trace_length,
)
from .config import CONFIG, create_analysis_config, get_column_mappings
from .discovery import perform_process_discovery, DISCOVERY_ALGORITHMS
from .conformance import run_conformance_checking, CONFORMANCE_METHODS
from .analytics import (
    get_event_log_summary,
    analyze_process_performance,
    analyze_repeat_purchases,
    format_business_report,
)
from .visualizer import visualize_focused_insights, export_results

__all__ = [
    "run_full_analysis",
    "load_and_validate_csv",
    "refine_activity_labels",
    "optimize_dataframe_memory",
    "filter_event_log",
    "sample_log_stratified",
    "check_trace_length",
    "CONFIG",
    "create_analysis_config",
    "get_column_mappings",
    "perform_process_discovery",
    "DISCOVERY_ALGORITHMS",
    "run_conformance_checking",
    "CONFORMANCE_METHODS",
    "get_event_log_summary",
    "analyze_process_performance",
    "analyze_repeat_purchases",
    "format_business_report",
    "visualize_focused_insights",
    "export_results",
]
