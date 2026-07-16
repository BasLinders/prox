import logging
import gc
import os
import numpy as np
import pandas as pd
import traceback
from typing import Dict, Any

import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.petri_net.obj import Marking
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algorithm

from .data_manager import sample_log_stratified

logger = logging.getLogger(__name__)


def calculate_fitness_in_batches(log, net, im, fm, batch_size: int = 200) -> float:
    """
    Computes mean alignment fitness over an event log using batch processing.

    Batching limits peak memory usage during alignment, which is the most
    memory-intensive step in conformance checking.

    Returns
    -------
    float : Mean fitness across all traces (0.0–1.0).
    """
    total_fitness = 0.0
    total_traces = 0
    log_len = len(log)

    for i in range(0, log_len, batch_size):
        batch = log[i: i + batch_size]
        if not batch:
            continue
        results = alignments_algorithm.apply(batch, net, im, fm)
        for res in results:
            total_fitness += res['fitness']
            total_traces += 1
        del results, batch
        gc.collect()

    return total_fitness / total_traces if total_traces > 0 else 0.0


def parse_alignments(clean_log, alignments: list) -> list:
    """
    Parses raw alignment results into per-case fitness and deviation dicts.

    Returns
    -------
    list of dict, each containing:
        'case_id'   : str
        'fitness'   : float
        'deviations': {'skipped': list, 'unsolicited': list}
    """
    details = []

    for i, (trace, align) in enumerate(zip(clean_log, alignments)):
        if not isinstance(align, dict):
            continue

        if 'fitness' in align:
            t_fit = float(align['fitness'])
        else:
            cost = align.get('cost', 0)
            trace_len = len(trace)
            t_fit = max(0.0, 1.0 - (cost / (trace_len + 1)))

        skipped = []
        unsolicited = []

        for move in align.get('alignment', []):
            log_part = move[0]
            log_label = log_part[0] if isinstance(log_part, tuple) else log_part

            model_part = move[1]
            model_label = model_part[0] if isinstance(model_part, tuple) else model_part

            model_str = str(model_label) if model_label is not None else "None"
            log_str = str(log_label) if log_label is not None else "None"

            if (log_label == '>>' or log_label is None) and (model_label not in ('>>', None)):
                if not model_str.startswith(('tau', 'skip', 'init')):
                    skipped.append(model_str)
            elif (log_label not in ('>>', None)) and (model_label == '>>' or model_label is None):
                unsolicited.append(log_str)

        details.append({
            'case_id': str(trace.attributes.get('concept:name', f'Case_{i}')),
            'fitness': t_fit,
            'deviations': {'skipped': skipped, 'unsolicited': unsolicited}
        })

    return details


def run_conformance_checking(
    event_log_df: pd.DataFrame,
    process_model,
    initial_marking,
    final_marking,
    max_align: int = 250,
    max_prec_cases: int = 250,
    cores: int = 1,
    alignment_variant: str = 'state_equation_a_star',
    enable_detailed_analysis: bool = False,
    calculate_fitness: bool = False,
    optimize_variants: bool = True,
    perform_sampling: bool = True,
    strata_col: str = None,
    max_priority_ratio: float = 0.5
) -> Dict[str, Any]:
    """
    Orchestrates conformance checking: sampling, fitness, precision, and alignments.

    Parameters
    ----------
    event_log_df        : Input event log as DataFrame.
    process_model       : Petri net (net).
    initial_marking     : Petri net initial marking (im).
    final_marking       : Petri net final marking (fm).
    max_align           : Max traces for alignment computation.
    max_prec_cases      : Max traces for precision computation.
    cores               : CPU cores (0 = all available minus one).
    alignment_variant   : PM4Py alignment algorithm key.
    enable_detailed_analysis : If True, compute precision and per-trace deviations.
    calculate_fitness   : If True, compute standalone batched fitness score.
    optimize_variants   : If True, align once per unique variant (10-100x speedup).
    perform_sampling    : If True, use stratified sampling before alignment.
    strata_col          : Column for stratified sampling (default: 'purchase').
    max_priority_ratio  : Max share of the sample reserved for priority (strata) cases.

    Returns
    -------
    dict with keys: 'fitness', 'precision', 'alignments', 'case_analysis',
                    'overall_summary', 'errors'
    """
    results = {
        'fitness': {'log_fitness': 0, 'note': 'Not calculated'},
        'precision': {'precision_score': 0},
        'alignments': {},
        'case_analysis': {'cases': []},
        'errors': [],
        'overall_summary': {}
    }

    try:
        # --- 1. Prepare sampled log ---
        try:
            if perform_sampling:
                sc = strata_col if (strata_col and strata_col in event_log_df.columns) else (
                    'purchase' if 'purchase' in event_log_df.columns else 'case:concept:name'
                )
                s_df, _ = sample_log_stratified(
                    event_log_df, sc,
                    total_sample_size=max_align,
                    max_priority_ratio=max_priority_ratio
                )
                sampled_log = pm4py.convert_to_event_log(s_df)
            else:
                sampled_log = pm4py.convert_to_event_log(event_log_df.iloc[:max_align])
        except Exception:
            sampled_log = pm4py.convert_to_event_log(event_log_df)

        # --- 2. Standalone fitness (token replay path) ---
        if calculate_fitness:
            try:
                logger.info("Calculating fitness on %d traces.", len(sampled_log))
                avg_fitness = calculate_fitness_in_batches(
                    sampled_log, process_model, initial_marking, final_marking
                )
                results['fitness'] = {
                    'log_fitness': avg_fitness,
                    'note': "Calculated via batched alignment replay"
                }
            except Exception as e:
                results['errors'].append(f"Fitness calculation failed: {e}")

        # --- 3. Precision ---
        if enable_detailed_analysis:
            try:
                prec_input = sampled_log[:max_prec_cases] if len(sampled_log) > max_prec_cases else sampled_log
                logger.info("Calculating precision on %d traces.", len(prec_input))
                prec = precision_evaluator.apply(
                    prec_input, process_model, initial_marking, final_marking,
                    variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
                )
                results['precision'] = {
                    'precision_score': prec if isinstance(prec, float) else prec.get('precision', 0)
                }
            except Exception as e:
                results['errors'].append(f"Precision calculation failed: {e}")

        # --- 4. Alignments ---
        if alignment_variant != 'token_replay':
            input_log = sampled_log[:max_align] if len(sampled_log) > max_align else sampled_log

            # Rebuild a clean log with only string attributes to avoid PM4Py serialisation issues
            clean_log = EventLog()
            for trace in input_log:
                nt = Trace()
                nt.attributes['concept:name'] = str(trace.attributes.get('concept:name', 'Unknown'))
                for event in trace:
                    nt.append(Event({'concept:name': str(event['concept:name'])}))
                clean_log.append(nt)

            # Rebuild markings from net structure
            rim = Marking()
            rfm = Marking()
            for p in process_model.places:
                if not p.in_arcs:
                    rim[p] = 1
                if not p.out_arcs:
                    rfm[p] = 1
            if not rim:
                rim[list(process_model.places)[0]] = 1
            if not rfm:
                rfm[list(process_model.places)[-1]] = 1

            max_cores = max(1, os.cpu_count() - 1) if cores == 0 else cores
            params = {'cores': max_cores, 'ret_tuple_as_trans_desc': True}

            if optimize_variants:
                # Group identical traces and align once per unique variant
                variant_map: Dict[tuple, list] = {}
                unique_traces: list = []

                for i, trace in enumerate(clean_log):
                    sig = tuple(str(e['concept:name']) for e in trace)
                    if sig not in variant_map:
                        variant_map[sig] = []
                        unique_traces.append(trace)
                    variant_map[sig].append(i)

                logger.info(
                    "Aligning %d unique variants (from %d traces).",
                    len(unique_traces), len(clean_log)
                )
                variant_alignments = alignments_algorithm.apply(
                    unique_traces, process_model, rim, rfm, parameters=params
                )

                final_alignments = [None] * len(clean_log)
                for k, align_result in enumerate(variant_alignments):
                    sig = tuple(str(e['concept:name']) for e in unique_traces[k])
                    for idx in variant_map[sig]:
                        final_alignments[idx] = align_result
            else:
                logger.info("Aligning %d traces (no variant grouping).", len(clean_log))
                final_alignments = alignments_algorithm.apply(
                    clean_log, process_model, rim, rfm, parameters=params
                )

            valid = [a for a in final_alignments if isinstance(a, dict) and 'cost' in a]
            if valid:
                f_vals = []
                for i, align in enumerate(final_alignments):
                    if not isinstance(align, dict):
                        continue
                    if 'fitness' in align:
                        f_vals.append(float(align['fitness']))
                    else:
                        t_len = len(clean_log[i])
                        f_vals.append(max(0.0, 1.0 - (align['cost'] / (t_len + 1))))

                results['fitness'] = {
                    'log_fitness': float(np.mean(f_vals)) if f_vals else 0.0,
                    'note': "Derived from alignments"
                }
                results['alignments'] = {
                    'total': len(valid),
                    'average_cost': float(np.mean([a['cost'] for a in valid])),
                    'note': f"Calculated on {len(valid)} traces"
                }
                results['case_analysis'] = {'cases': parse_alignments(clean_log, final_alignments)}

    except Exception as e:
        results['errors'].append(f"Conformance error: {e}")
        results['errors'].append(traceback.format_exc())

    # --- Summary ---
    fit_score = results['fitness'].get('log_fitness', 0)
    prec_score = results['precision'].get('precision_score', 0)
    avg_score = (fit_score + prec_score) / 2 if prec_score > 0 else fit_score

    if avg_score > 0.8:
        quality = "Excellent"
    elif avg_score > 0.6:
        quality = "Good"
    elif avg_score > 0.4:
        quality = "Fair"
    else:
        quality = "Poor"

    results['overall_summary'] = {
        'fitness_score': fit_score,
        'precision_score': prec_score,
        'quality_assessment': quality,
        'recommendations': []
    }

    logger.info("Conformance complete. Fitness=%.2f, Precision=%.2f, Quality=%s",
                fit_score, prec_score, quality)
    return results
