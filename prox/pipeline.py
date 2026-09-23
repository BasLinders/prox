import logging
import pandas as pd
from typing import Callable, Dict, Any

from .discovery import perform_process_discovery
from .conformance import run_conformance_checking
from .analytics import (
    analyze_process_performance, get_event_log_summary, analyze_repeat_purchases, analyze_conversion_funnel,
    classify_sessions, summarize_user_journeys
)
from .visualizer import visualize_focused_insights, export_results
from .data_manager import filter_event_log, FILTER_HANDLERS, sample_log_stratified

logger = logging.getLogger(__name__)

# User-facing stage count/labels for progress_callback - one tick per
# meaningfully time-consuming stage. Log summary and the variants CSV export
# are near-instant, so they're folded into the stage they sit next to rather
# than getting their own tick.
_PROGRESS_STAGES = [
    "Filtering & sampling events",
    "Discovering process model",
    "Checking conformance",
    "Analysing performance",
    "Generating visualisations",
    "Computing business insights",
]


def run_full_analysis(
    event_log_df: pd.DataFrame,
    config: Dict[str, Any],
    output_folder: str = "output",
    progress_callback: Callable[[int, int, str], None] | None = None
) -> Dict[str, Any] | None:
    """
    Executes the full process mining pipeline and returns a structured results dict.

    Stages (in order):
        1. Filter        — apply configured filter chain
        1b. Sampling     — if enabled, stratified case sampling applied once;
                           every later stage (including business insights)
                           shares this same subset, so results describe one
                           consistent dataset instead of mixing a sampled
                           conformance check with full-log discovery/insights
        2. Summary       — event log statistics
        3. Discovery     — Petri net model from event log
        4. Conformance   — fitness, precision, trace deviations
        5. Performance   — bottlenecks, lead times, variants
        6. Visualisation — BPMN process maps saved to disk
        7. Business      — repeat purchase loyalty metrics

    Parameters
    ----------
    event_log_df : pd.DataFrame
        Pre-cleaned event log with XES-standard columns.
    config : dict
        Pipeline configuration. Use config.create_analysis_config() to build one.
    output_folder : str
        Folder for generated images/CSVs. Give each concurrent or repeated run
        (e.g. one per segment in compare_segments()) a distinct folder, or later
        runs will silently overwrite earlier runs' images.
    progress_callback : callable, optional
        Called as progress_callback(stage_num, total_stages, stage_label) as
        each major stage completes (see _PROGRESS_STAGES), e.g. to drive a UI
        progress bar. Not called on early-failure returns. No-op if None.

    Returns
    -------
    dict or None
        Keyed results for each stage. Returns None on critical early failure.
    """
    logger.info("=" * 60)
    logger.info("START: Process Mining Pipeline")
    logger.info("=" * 60)

    total_stages = len(_PROGRESS_STAGES)

    def _report_progress(stage_num: int) -> None:
        if progress_callback:
            progress_callback(stage_num, total_stages, _PROGRESS_STAGES[stage_num - 1])

    pipeline_results: Dict[str, Any] = {}
    speed_params = config.get("speed_params", {})
    sampling_config = config.get("sampling_config", {})
    sampling_enabled = sampling_config.get("enabled", True)

    # -------------------------------------------------------------------------
    # Step 1: Filtering
    # -------------------------------------------------------------------------
    log_df = event_log_df.copy()
    filter_steps = config.get("filter_steps", [])

    if filter_steps:
        unknown_types = sorted({
            step.get('type') or '<missing>' for step in filter_steps
            if step.get('type') not in FILTER_HANDLERS
        })
        if unknown_types:
            logger.error(
                "Unknown or missing filter type(s) in config: %s. Valid options: %s.",
                unknown_types, ', '.join(FILTER_HANDLERS)
            )
            return None

        logger.info("Applying %d filter step(s).", len(filter_steps))
        for i, step_config in enumerate(filter_steps):
            params = step_config.copy()
            f_type = params.pop('type')

            log_df, messages = filter_event_log(log_df, filter_type=f_type, **params)
            for msg in messages:
                logger.info("  [filter %d] %s", i + 1, msg)

            if log_df is None or log_df.empty:
                logger.error("Filter step %d produced an empty dataset. Aborting.", i + 1)
                return None
    else:
        logger.info("No filters configured. Using full dataset.")

    n_events = len(log_df)
    n_cases = log_df['case:concept:name'].nunique()
    logger.info("Post-filter: %d events, %d cases.", n_events, n_cases)

    if n_events > 10_000:
        logger.warning("Dataset is large (>10k events). Consider enabling sampling.")

    # -------------------------------------------------------------------------
    # Step 1b: Sampling (applied once, upstream of every remaining stage)
    # -------------------------------------------------------------------------
    # Previously only conformance checking sampled - discovery, performance
    # analysis, visualisation, and business insights all ran on the full
    # filtered log regardless of this toggle, which both defeated the point
    # of enabling sampling (those unsampled stages dominate runtime on a
    # large/noisy log) and meant business insights could describe a
    # different dataset than the conformance results shown alongside them.
    if sampling_enabled:
        sample_size = sampling_config.get("total_sample_size", speed_params.get("max_align", 250))
        strata_col = sampling_config.get("strata_col", "purchase")
        max_priority_ratio = sampling_config.get("max_priority_ratio", 0.5)

        logger.info("Applying stratified sampling (target: %d cases, strata: '%s').", sample_size, strata_col)
        log_df, sample_messages = sample_log_stratified(
            log_df, strata_col,
            total_sample_size=sample_size,
            max_priority_ratio=max_priority_ratio
        )
        for msg in sample_messages:
            logger.info("  [sampling] %s", msg)

        n_events = len(log_df)
        n_cases = log_df['case:concept:name'].nunique()
        logger.info("Post-sampling: %d events, %d cases.", n_events, n_cases)

    # Exposed so callers outside this pipeline (e.g. the propensity-modeling
    # tab, which reruns its own feature extraction rather than reading a
    # precomputed stage result) can train/score on the same filtered+sampled
    # dataset the rest of these results describe, instead of silently
    # re-pulling the full unfiltered/unsampled log.
    pipeline_results['processed_log'] = log_df

    _report_progress(1)

    # -------------------------------------------------------------------------
    # Step 2: Log Summary
    # -------------------------------------------------------------------------
    logger.info("--- Step 2: Log Summary ---")
    summary, errors = get_event_log_summary(log_df)
    if errors:
        for e in errors:
            logger.warning(e)
    if summary is None:
        logger.error("Log summary failed. Aborting.")
        return None

    pipeline_results['log_summary'] = summary
    logger.info("Cases: %s | Events: %s | Activities: %s",
                summary.get('Number of Cases'),
                summary.get('Number of Events'),
                summary.get('Number of Unique Activities'))

    # -------------------------------------------------------------------------
    # Step 3: Process Discovery
    # -------------------------------------------------------------------------
    logger.info("--- Step 3: Process Discovery ---")
    disc_cfg = config.get("discovery_params", {})

    model_tuple, errors, messages = perform_process_discovery(
        log_df,
        discovery_algo=disc_cfg.get("algorithm", "inductive_miner"),
        noise_threshold=disc_cfg.get("noise_threshold", 0.2),
        dependency_threshold=disc_cfg.get("dependency_threshold", 0.5),
        activity_threshold=disc_cfg.get("activity_threshold", 0)
    )

    for msg in messages:
        logger.info(msg)
    if errors:
        for e in errors:
            logger.error(e)
        return None

    net, im, fm = model_tuple
    pipeline_results['model'] = {'net': net, 'im': im, 'fm': fm}

    _report_progress(2)

    # -------------------------------------------------------------------------
    # Step 4: Conformance Checking
    # -------------------------------------------------------------------------
    logger.info("--- Step 4: Conformance Checking ---")
    conf_cfg = config.get("conformance_params", {})

    conformance_results = run_conformance_checking(
        log_df, net, im, fm,
        max_align=speed_params.get("max_align", 250),
        max_prec_cases=speed_params.get("max_prec_traces", 250),
        cores=speed_params.get("cores", 1),
        alignment_variant=conf_cfg.get("algorithm", "state_equation_a_star"),
        enable_detailed_analysis=conf_cfg.get("calculate_precision", True),
        calculate_fitness=conf_cfg.get("calculate_fitness", False),
        optimize_variants=conf_cfg.get("optimize_variants", True),
        # log_df was already sampled once in Step 1b (when sampling_enabled) and
        # is reused as-is for every stage; sampling here too would re-stratify
        # an already-sampled subset for no benefit, on top of being redundant
        # with Step 1b's own sampling log messages.
        perform_sampling=False,
    )

    pipeline_results['conformance'] = conformance_results

    for err in conformance_results.get('errors', []):
        logger.warning(err)

    overall = conformance_results.get('overall_summary', {})
    logger.info(
        "Conformance — Fitness: %.2f | Precision: %.2f | Quality: %s",
        overall.get('fitness_score', 0),
        overall.get('precision_score', 0),
        overall.get('quality_assessment', 'N/A')
    )

    _report_progress(3)

    # -------------------------------------------------------------------------
    # Step 5: Performance Analysis
    # -------------------------------------------------------------------------
    perf_cfg = config.get("performance_params", {})
    time_unit = perf_cfg.get("time_unit", "hours")
    logger.info("--- Step 5: Performance Analysis (%s) ---", time_unit)

    performance_results = analyze_process_performance(
        log_df,
        time_unit=time_unit,
        bottleneck_threshold_percentile=perf_cfg.get("bottleneck_threshold_percentile", 75),
        include_variants=True
    )

    pipeline_results['performance'] = performance_results

    for err in performance_results.get('errors', []):
        logger.warning(err)

    case_perf = performance_results.get('case_performance', {}).get('duration_stats', {})
    top_bn = performance_results.get('bottlenecks', {}).get('summary', {}).get('top_activity_bottleneck')
    logger.info(
        "Performance — Avg lead time: %.2f %s | Top bottleneck: %s",
        case_perf.get('mean', 0), time_unit, top_bn or 'None'
    )

    _report_progress(4)

    # -------------------------------------------------------------------------
    # Step 5b: Visualisation
    # -------------------------------------------------------------------------
    logger.info("--- Step 5b: Visualisation ---")
    vis_cfg = config.get("visualisation_params", {})
    happy_img, main_img = visualize_focused_insights(
        log_df,
        output_folder=output_folder,
        bottleneck_top_k=vis_cfg.get("bottleneck_top_k", 15)
    )

    pipeline_results['visualizations'] = {
        'happy_path': happy_img,
        'bottlenecks': main_img
    }

    _report_progress(5)

    # -------------------------------------------------------------------------
    # Step 6: Export variants CSV
    # -------------------------------------------------------------------------
    variant_data = performance_results.get('variant_performance', {})
    top_variants_dict = variant_data.get('top_variants', {})
    if top_variants_dict:
        variants_df = pd.DataFrame.from_dict(top_variants_dict, orient='index')
        variants_df.index.name = 'Variant_Path'
        export_results(variants_df, "process_variant_analysis", "csv", output_folder=output_folder)

    # -------------------------------------------------------------------------
    # Step 7: Business Insights
    # -------------------------------------------------------------------------
    logger.info("--- Step 7: Business Insights ---")
    biz_cfg = config.get("business_params", {})

    repeat_stats = analyze_repeat_purchases(
        log_df,
        output_folder=output_folder,
        user_col=biz_cfg.get("user_col", "user_id"),
        purchase_values=biz_cfg.get("purchase_values", ['purchase', 'has_purchase']),
        cart_values=biz_cfg.get("cart_values"),
        revenue_col=biz_cfg.get("revenue_col", "event_value")
    )

    if repeat_stats:
        pipeline_results['repeat_purchase_analysis'] = repeat_stats

    funnel_stats = analyze_conversion_funnel(
        log_df,
        funnel_steps=biz_cfg.get("funnel_steps"),
    )
    if funnel_stats.get('stages'):
        pipeline_results['funnel_analysis'] = funnel_stats

    session_labels = classify_sessions(
        log_df,
        user_col=biz_cfg.get("user_col", "user_id"),
        purchase_values=biz_cfg.get("purchase_values", ['purchase', 'has_purchase']),
        cart_values=biz_cfg.get("cart_values"),
        research_keywords=biz_cfg.get("research_keywords"),
        research_min_events=biz_cfg.get("research_min_events", 3),
    )
    if not session_labels.empty:
        pipeline_results['session_insights'] = {
            'sessions': session_labels,
            'journeys': summarize_user_journeys(session_labels),
        }

    _report_progress(6)

    logger.info("=" * 60)
    logger.info("COMPLETE: Process Mining Pipeline")
    logger.info("=" * 60)

    return pipeline_results
