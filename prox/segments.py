import copy
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict

import pandas as pd

from .pipeline import run_full_analysis

logger = logging.getLogger(__name__)


def _safe_folder_name(value: Any) -> str:
    """Sanitizes a segment value into a filesystem-safe folder name fragment."""
    text = re.sub(r'[^A-Za-z0-9_-]+', '_', str(value)).strip('_')
    return text or 'segment'


def _run_segment_job(value: Any, segment_df: pd.DataFrame, config: Dict[str, Any], output_folder: str):
    """Runs one segment's full pipeline. Module-level so it's picklable for ProcessPoolExecutor."""
    results = run_full_analysis(segment_df, config=config, output_folder=output_folder)
    return value, results


def compare_segments(
    event_log_df: pd.DataFrame,
    segment_col: str,
    config: Dict[str, Any],
    top_n_segments: int = 5,
    output_folder: str = "output",
    parallel: bool = True,
    max_workers: int = None,
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
    parallel : bool
        If True (default), run one segment per worker process via
        ProcessPoolExecutor — each segment's run_full_analysis() call is fully
        independent (no shared state), so this is close to an N-fold wall-clock
        win on segment comparison specifically. Intended for local use (PRoX
        runs on a standard laptop, not a shared/hosted Streamlit deployment)
        where spawning worker processes is unrestricted. Each segment run is
        pinned to a single core in this mode (see max_workers) to avoid
        nesting PM4Py's own multi-core alignment pool inside multiple worker
        processes at once, which would oversubscribe the machine's cores.
    max_workers : int, optional
        Worker process cap when parallel=True. Defaults to
        min(number of segments, CPU count).

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

    jobs = []
    for value in top_segments:
        case_ids = case_segment[case_segment == value].index
        segment_df = event_log_df[event_log_df['case:concept:name'].isin(case_ids)].copy()
        segment_output_folder = f"{output_folder}/segment_{_safe_folder_name(value)}"
        jobs.append((value, case_ids, segment_df, segment_output_folder))

    raw_results: Dict[Any, tuple] = {}

    if parallel and len(jobs) > 1:
        worker_count = max_workers or max(1, min(len(jobs), os.cpu_count() or 1))

        # Each worker process runs a full pipeline, whose conformance stage can
        # itself request multiple cores (config['speed_params']['cores']). Left
        # as-is, N parallel segments x M cores/segment could request N*M cores
        # at once — pin each segment run to a single core here so the only
        # parallelism in play is across segments, not nested inside them too.
        segment_config = copy.deepcopy(config)
        segment_config.setdefault('speed_params', {})['cores'] = 1

        logger.info(
            "Running %d segment(s) in parallel across %d worker process(es).",
            len(jobs), worker_count
        )

        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_run_segment_job, value, segment_df, segment_config, seg_folder): value
                for value, _case_ids, segment_df, seg_folder in jobs
            }
            for future in as_completed(futures):
                value = futures[future]
                try:
                    _, results = future.result()
                except Exception as e:
                    errors.append(f"Analysis failed for segment '{value}': {e}")
                    continue
                if results is None:
                    errors.append(f"Analysis failed for segment '{value}'. Skipped.")
                    continue
                raw_results[value] = results
    else:
        for value, case_ids, segment_df, seg_folder in jobs:
            logger.info("--- Segment '%s': %d case(s) ---", value, len(case_ids))
            results = run_full_analysis(segment_df, config=config, output_folder=seg_folder)
            if results is None:
                errors.append(f"Analysis failed for segment '{value}'. Skipped.")
                continue
            raw_results[value] = results

    case_id_lookup = {value: case_ids for value, case_ids, _segment_df, _seg_folder in jobs}
    segment_results = {}
    comparison_table = {}

    # Iterate in original top_segments order for stable output regardless of
    # which worker process finished first.
    for value in top_segments:
        if value not in raw_results:
            continue
        results = raw_results[value]
        segment_results[value] = results

        overall = results.get('conformance', {}).get('overall_summary', {})
        stats = results.get('performance', {}).get('summary_statistics', {})
        top_variants = results.get('performance', {}).get('variant_performance', {}).get('top_variants', {})
        biz = results.get('repeat_purchase_analysis')

        comparison_table[value] = {
            'cases': len(case_id_lookup[value]),
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
