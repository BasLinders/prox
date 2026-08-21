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
    check_data_quality,
    winsorize_series,
)
from .config import CONFIG, create_analysis_config, get_column_mappings
from .discovery import perform_process_discovery, DISCOVERY_ALGORITHMS
from .conformance import (
    run_conformance_checking,
    CONFORMANCE_METHODS,
    build_structured_reference_model,
    import_reference_model_bpmn,
    diff_reference_model_coverage,
)
from .analytics import (
    get_event_log_summary,
    analyze_process_performance,
    analyze_repeat_purchases,
    analyze_conversion_funnel,
    analyze_funnel_by_segment,
    format_business_report,
    classify_sessions,
    summarize_user_journeys,
)
from .visualizer import visualize_focused_insights, export_results, render_petri_net
from .report import (
    generate_html_report,
    generate_segment_comparison_report,
    generate_reference_conformance_report,
)
from .segments import compare_segments
from .mock_data import generate_mock_event_log, generate_mock_csv_bytes

__all__ = [
    "run_full_analysis",
    "load_and_validate_csv",
    "refine_activity_labels",
    "optimize_dataframe_memory",
    "filter_event_log",
    "sample_log_stratified",
    "check_trace_length",
    "check_data_quality",
    "winsorize_series",
    "CONFIG",
    "create_analysis_config",
    "get_column_mappings",
    "perform_process_discovery",
    "DISCOVERY_ALGORITHMS",
    "run_conformance_checking",
    "CONFORMANCE_METHODS",
    "build_structured_reference_model",
    "import_reference_model_bpmn",
    "diff_reference_model_coverage",
    "get_event_log_summary",
    "analyze_process_performance",
    "analyze_repeat_purchases",
    "analyze_conversion_funnel",
    "analyze_funnel_by_segment",
    "format_business_report",
    "classify_sessions",
    "summarize_user_journeys",
    "visualize_focused_insights",
    "export_results",
    "render_petri_net",
    "generate_html_report",
    "generate_segment_comparison_report",
    "generate_reference_conformance_report",
    "compare_segments",
    "generate_mock_event_log",
    "generate_mock_csv_bytes",
]
