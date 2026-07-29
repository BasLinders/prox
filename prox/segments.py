import logging
import re
from typing import Any, Dict

import pandas as pd

from .pipeline import run_full_analysis

logger = logging.getLogger(__name__)


def _safe_folder_name(value: Any) -> str:
    """Sanitizes a segment value into a filesystem-safe folder name fragment."""
    text = re.sub(r'[^A-Za-z0-9_-]+', '_', str(value)).strip('_')
    return text or 'segment'


def compare_segments(
    event_log_df: pd.DataFrame,
    segment_col: str,
    config: Dict[str, Any],
    top_n_segments: int = 5,
    output_folder: str = "output"
) -> Dict[str, Any]:
    """
    Runs the full analysis pipeline once per segment value (top N by case count)
    and returns per-segment results plus a comparison table.

    Segments are assigned per case using the first observed value of
    `segment_col` within each case, so mixed-value cases don't get split
    across segments mid-trace.

    Parameters
    ----------
    event_log_df : pd.DataFrame
        Pre-cleaned event log with XES-standard columns, plus `segment_col`.
    segment_col : str
        Column to split the log by (e.g. device type, traffic source).
    config : dict
        Pipeline configuration, shared across all segments. Use create_analysis_config().
    top_n_segments : int
        Maximum number of segment values to analyse, ranked by case count.
    output_folder : str
        Base folder; each segment's images/exports are written to a distinct
        subfolder here so segments don't overwrite each other's output.

    Returns
    -------
    dict with keys:
        'segments'          : {segment_value: run_full_analysis() results}
        'comparison_table'  : {segment_value: {cases, health_score, fitness_score,
                               precision_score, repeat_rate, top_variant}}
        'errors'            : list of str
    """
    errors = []

    if segment_col not in event_log_df.columns:
        errors.append(f"Critical Error: Segment column '{segment_col}' not found in event log.")
        return {'segments': {}, 'comparison_table': {}, 'errors': errors}

    case_segment = event_log_df.groupby('case:concept:name')[segment_col].first()
    segment_counts = case_segment.value_counts()

    if segment_counts.empty:
        errors.append(f"Critical Error: No segment values found in column '{segment_col}'.")
        return {'segments': {}, 'comparison_table': {}, 'errors': errors}

    top_segments = segment_counts.head(top_n_segments).index.tolist()
    logger.info(
        "Comparing %d segment(s) from '%s' (of %d total): %s",
        len(top_segments), segment_col, len(segment_counts), top_segments
    )

    segment_results = {}
    comparison_table = {}

    for value in top_segments:
        case_ids = case_segment[case_segment == value].index
        segment_df = event_log_df[event_log_df['case:concept:name'].isin(case_ids)].copy()
        segment_output_folder = f"{output_folder}/segment_{_safe_folder_name(value)}"

        logger.info("--- Segment '%s': %d case(s) ---", value, len(case_ids))
        results = run_full_analysis(segment_df, config=config, output_folder=segment_output_folder)

        if results is None:
            errors.append(f"Analysis failed for segment '{value}'. Skipped.")
            continue

        segment_results[value] = results

        overall = results.get('conformance', {}).get('overall_summary', {})
        stats = results.get('performance', {}).get('summary_statistics', {})
        top_variants = results.get('performance', {}).get('variant_performance', {}).get('top_variants', {})
        biz = results.get('repeat_purchase_analysis')

        comparison_table[value] = {
            'cases': len(case_ids),
            'health_score': stats.get('process_health_score', 0),
            'fitness_score': overall.get('fitness_score', 0),
            'precision_score': overall.get('precision_score', 0),
            'repeat_rate': biz.get('metrics', {}).get('repeat_rate', 0) if biz else 0,
            'top_variant': next(iter(top_variants), 'N/A'),
        }

    return {
        'segments': segment_results,
        'comparison_table': comparison_table,
        'errors': errors,
    }
