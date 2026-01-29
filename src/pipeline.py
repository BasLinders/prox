from typing import Dict, Any, List
import pandas as pd
import pm4py
import os
from IPython.display import Image, display

# Import custom modules
from discovery import perform_process_discovery
from conformance import run_conformance_checking
from analytics import (
    analyze_process_performance, 
    get_event_log_summary, 
    analyze_repeat_purchases
)
from visualizer import (
    visualize_focused_insights, 
    export_results
)
from data_manager import filter_event_log

def run_full_analysis(event_log_df: pd.DataFrame, config: Dict[str, Any]):
    """
    Executes the full process analysis based on a provided configuration dictionary.
    """
    print("="*80)
    print("START OF CONFIGURED PROCESS ANALYSIS")
    print("="*80)

    # Dictionary to store all results (for return)
    pipeline_results = {}

    # Speed params
    speed_params = config.get("speed_params", {})

    # Check if sampler is enabled
    sampling_config = config.get('sampling_config', {})
    sampling_status = sampling_config.get('enabled', True)
    
    # --- Step 1: Apply Filters ---
    log_df_for_analysis = event_log_df.copy()
    filter_steps = config.get("filter_steps", [])

    if filter_steps:
        print(f"\n--- Applying filters ({len(filter_steps)} step(s)) ---")

    for i, step_config in enumerate(filter_steps):
        # Make a copy to avoid modifying the original config
        params = step_config.copy()

        # Extract 'type' because filter_event_log expects 'filter_type' as a specific arg
        f_type = params.pop('type', None)

        if f_type:
            print(f"Step {i+1}: Filter type '{f_type}'...")
            # We pass f_type as the positional arg, and the rest as kwargs
            log_df_for_analysis, messages = filter_event_log(log_df_for_analysis, filter_type=f_type, **params)

        if messages:
            for msg in messages: print(f"   -> {msg}")

        # Stop if filter creates empty log
        if log_df_for_analysis is None or log_df_for_analysis.empty:
            print("Critical: Filter resulted in an empty dataset. Analysis stopped.")
            return None
        else:
            print(f"Warning: Filter step {i+1} missing 'type'. Skipped.")
    else:
        print("\n--- No filters applied, full dataset is being used ---")

    num_remaining_events = len(log_df_for_analysis)
    num_remaining_cases = log_df_for_analysis['case:concept:name'].nunique()
    
    print(f"\n--- DATA STATUS AFTER FILTERING ---")
    print(f"Events remaining: {num_remaining_events}")
    print(f"Cases remaining:  {num_remaining_cases}")
    
    # Safety check:
    if num_remaining_events > 10000:
        print("WARNING: Dataset is still very large (>10k events).")
        print("Consider activating sampling or use stricter filters.")

    # --- Step 2: Summary ---
    print("\n--- Summary of the analyzed Event Log ---")
    summary, errors = get_event_log_summary(log_df_for_analysis)

    if errors:
        print("Errors found during summary:", errors)
        return None

    for key, value in summary.items():
        print(f"- {key}: {value}")

    pipeline_results['log_summary'] = summary

    # --- Step 3: Process Discovery ---
    print("\n--- Discovering Process Model ---")
    discovery_cfg = config.get('discovery_params', {})

    # Defaults in case config is missing keys
    algo = discovery_cfg.get('algorithm', 'inductive_miner')
    noise = discovery_cfg.get('noise_threshold', 0.2)
    dep_thresh = discovery_cfg.get('dependency_threshold', 0.5)

    model_tuple, errors, messages = perform_process_discovery(
        log_df_for_analysis,
        discovery_algo=algo,
        noise_threshold=noise,
        dependency_threshold=dep_thresh
    )

    for msg in messages: print(f"✓ {msg}")

    if errors:
        print("ERROR during process discovery:", errors)
        return None

    net, im, fm = model_tuple
    pipeline_results['model'] = {'net': net, 'im': im, 'fm': fm}

    # --- Step 4: Conformance Checking ---
    print("\n--- Conformance Checking ---")

    conf_cfg = config.get('conformance_params', {})
    optimize_setting = conf_cfg.get('optimize_variants', True)

    # Map config keys to function arguments
    conformance_results = run_conformance_checking(
        log_df_for_analysis, net, im, fm,
        max_align = speed_params.get("max_align", 250),
        max_prec_cases = speed_params.get("max_prec_cases", 250),
        cores = 1,
        alignment_variant=conf_cfg.get('algorithm', 'state_equation_a_star'),
        enable_detailed_analysis=conf_cfg.get('calculate_precision', True),
        calculate_fitness=conf_cfg.get('calculate_fitness', False),
        optimize_variants=optimize_setting,
        perform_sampling=sampling_status
    )

    pipeline_results['conformance'] = conformance_results

    if conformance_results.get('errors'):
        print("Warnings / errors during conformance checking:")
    for error in conformance_results['errors']:
        print(f"  - {error}")

    summary = conformance_results.get('overall_summary', {})
    
    if summary:
        print("\n   --- General Quality Score ---")
        print(f"   - Assessment: {summary.get('quality_assessment', 'N/A')}")
        print(f"   - Fitness: {summary.get('fitness_score', 0):.2%}")
        # Only show precision if it was actually calculated
    if 'precision_score' in summary and summary['precision_score'] > 0:
        print(f"   - Precision: {summary.get('precision_score', 0):.2%}")

    print("\n   --- Recommendations (conformance) ---")
    recommendations = summary.get('recommendations', [])
    if recommendations:
        for rec in recommendations:
            print(f"   - {rec}")
    else:
        print("   - Conformance results look good. No specific recommendations.")

    # --- Step 5: Performance Analysis ---
    perf_cfg = config.get('performance_params', {})
    time_unit = perf_cfg.get('time_unit', 'hours')

    print(f"\n--- Performance Analysis (in {time_unit}) ---")

    performance_results = analyze_process_performance(
        log_df_for_analysis,
        time_unit=time_unit,
        bottleneck_threshold_percentile=perf_cfg.get('bottleneck_threshold_percentile', 75),
        include_variants=True
    )

    pipeline_results['performance'] = performance_results

    if performance_results.get('errors'):
        print("Warnings / errors during performance analysis:")
    for error in performance_results['errors']:
        print(f"   - {error}")

    # Extract insights
    perf_summary = performance_results.get('summary_statistics', {})
    variants = performance_results.get('variant_performance', {})
    case_perf = performance_results.get('case_performance', {}).get('duration_stats', {})

    # Safely get bottleneck
    bottleneck_info = performance_results.get('bottlenecks', {}).get('summary', {})
    top_bottleneck = bottleneck_info.get('top_activity_bottleneck', 'None')

    print(f"   --- Key performance insights ---")
    if perf_summary:
        print(f"    - Process health score: {perf_summary.get('process_health_score', 0):.1f} / 100")

    print(f"\n   - Average lead time: {case_perf.get('mean', 0):.2f} {time_unit}")
    print(f"   - Median lead time: {case_perf.get('median', 0): .2f} {time_unit}")
    print(f"   - Main bottleneck: {top_bottleneck}")

    if variants:
        print(f"   - Number of unique variants: {variants.get('total_variants', 0)}")
        coverage = variants.get('variant_coverage', {}).get('top_5_coverage', 0)
        print(f"   - Top 5 variants cover: {coverage:.1f}% of all cases")

    print(f"\n   --- Recommendations (Performance) ---")
    recommendations = perf_summary.get('recommendations', [])
    if recommendations:
        for rec in recommendations:
            print(f"   - {rec}")
    else:
        print("   - Performance results look good. No specific recommendations.")

     # --- Step 5b: Visualization ---

    print("\n--- Generating Visual Insights ---")
    
    # Input conversion
    if isinstance(log_df_for_analysis, pd.DataFrame):
        log_for_vis = pm4py.convert_to_event_log(log_df_for_analysis)
    else:
        log_for_vis = log_df_for_analysis

    vis_cfg = config.get('visualisation_params', {})
    top_k = vis_cfg.get('bottleneck_top_k', 15)

    # call fn
    happy_path_img, bottleneck_img = visualize_focused_insights(log_for_vis)
    
    pipeline_results['visualizations'] = {
        'happy_path': happy_path_img,
        'bottlenecks': bottleneck_img
    }

    # display logic
    from IPython.display import Image, display
    import os

    print("\n--- Process Visualizations ---")

    if happy_path_img and os.path.exists(happy_path_img):
        print("1. Happy Path BPMN (Most Frequent Variant):")
        display(Image(filename=happy_path_img))
    else:
        print("1. Happy Path BPMN could not be generated.")

    if bottleneck_img and os.path.exists(bottleneck_img):
        print("2. Bottleneck BPMN (worst paths):")
        display(Image(filename=bottleneck_img))
    else:
        print("2. Bottleneck BPMN could not be generated (Check Graphviz installation).")
    
    # --- Step 6: Export Results ---
    print("\n--- Exporting results ---")
    # Export variants to CSV if available
    if variants:
        top_variants_dict = variants.get('top_variants', {})
        if top_variants_dict:
            variants_df = pd.DataFrame.from_dict(top_variants_dict, orient='index')
            # Ensure index is named 'Variant'
            variants_df.index.name = 'Variant_Path'
            export_results(variants_df, "process_variant_analysis", "csv")

    conformance_data = pipeline_results.get('conformance', {})
    case_analysis = conformance_data.get('case_analysis', {})
    cases = case_analysis.get('cases', [])

    print("\n--- DEBUG: TRACE DEVIATION CHECK ---")
    if cases:
        # Find the first case that isn't perfect
        imperfect_cases = [c for c in cases if c['fitness'] < 1.0]

        if imperfect_cases:
            bad_case = imperfect_cases[0]
            print(f"Found a deviant case: {bad_case['case_id']}")
            print(f"Fitness: {bad_case['fitness']:.4f}")
            print(f"Deviations: {bad_case.get('deviations', 'No deviation data found')}")
        else:
            print("ALL sampled cases are perfect matches (100% Fitness).")
    else:
        # This will print if you are in Quick Scan mode (Token Replay)
        print("No detailed trace analysis available (Likely using Quick Scan / Token Replay).")

    # --- 7. Business Insights (repeat purchases)
    print("\n--- Generating Business Insights ---")

    # Find user id column
    potential_user_cols = ['user_id', 'customer', 'customer_id', 'actor', 'user']
    target_user_col = 'user_id'

    for col in potential_user_cols:
        if col in log_df_for_analysis.columns:
            target_user_col = col
            break

    repeat_stats = analyze_repeat_purchases(
        log_df_for_analysis,
        output_folder="output",
        user_col=target_user_col,
        purchase_values=['purchase']
    )

    if repeat_stats:
        pipeline_results['repeat_purchase_analysis'] = repeat_stats
        
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED")
    print("="*80)

    return pipeline_results
