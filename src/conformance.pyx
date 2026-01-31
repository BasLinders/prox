import pandas as pd
import numpy as np
import traceback
import gc
import os
from typing import Dict, Any

# Custom functions
from data_manager import sample_log_stratified

# PM4Py imports
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.petri_net.obj import Marking
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.objects.conversion.dfg import converter as dfg_converter
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algorithm

cpdef double calculate_fitness_in_batches(object log, object net, object im, object fm, int batch_size=200):
    """
    Calculates the average alignment fitness of an event log against a Petri net using batch processing.
    
    This Cython-optimized function computes the fitness by partitioning the event log 
    into smaller batches to manage memory consumption and performing alignment-based 
    conformance checking. Fitness is defined as the degree to which the observed 
    traces in the log can be replayed by the Petri net.
    
    Parameters
    ----------
    log : object
        The event log containing traces to be evaluated. Expected to be a 
        list-like object (e.g., PM4Py EventLog) that supports slicing.
    net : object
        The Petri net model (Accepting Petri Net) against which fitness is measured.
    im : object
        The initial marking of the Petri net.
    fm : object
        The final marking of the Petri net.
    batch_size : int, optional
        The number of traces to process in a single iteration (default is 200). 
        Adjusting this can optimize the trade-off between speed and memory usage.
    
    Returns
    -------
    double
        The mean fitness value across all processed traces. The value ranges 
        from 0.0 to 1.0, where 1.0 indicates perfect conformance.
    
    Notes
    -----
    The fitness calculation is based on the following formula:
    $$Fitness = \frac{\sum_{i=1}^{n} fitness(trace_i)}{n}$$
    where $n$ is the total number of traces. Manual garbage collection (`gc.collect()`) 
    is triggered after each batch to prevent memory fragmentation during large-scale 
    conformance checking.
    """
    cdef double total_fitness_sum = 0.0
    cdef Py_ssize_t total_traces = 0
    cdef Py_ssize_t log_len = len(log)
    cdef Py_ssize_t i
    cdef list batch, results
    cdef dict res

    for i in range(0, log_len, batch_size):
        batch = log[i : i + batch_size]
        if not batch: continue
        
        results = alignments_algorithm.apply(batch, net, im, fm)
        
        for res in results:
            total_fitness_sum += res['fitness']
            total_traces += 1
            
        del results
        del batch
        gc.collect()
        
    if total_traces > 0:
        return total_fitness_sum / total_traces
    return 0.0

cpdef list parse_alignments_cython(object clean_log, list alignments):
    """
    Parses alignment results into a structured format containing fitness and deviations.
    
    This Cython-optimized function iterates through the results of a process 
    alignment algorithm, calculating trace-level fitness and identifying 
    specific process deviations (skips and unsolicited moves). It maps log 
    traces to model moves to pinpoint exactly where a process went off-track.
    
    Parameters
    ----------
    clean_log : object
        The event log object (typically a PM4Py log) containing the original 
        traces. Used to extract trace attributes like 'concept:name'.
    alignments : list of dict
        A list of alignment results corresponding to the traces in `clean_log`. 
        Each dictionary is expected to contain 'alignment' (a list of moves) 
        and optionally 'fitness' or 'cost'.
    
    Returns
    -------
    list of dict
        A list of dictionaries, one for each trace, containing:
        * 'case_id' (str): The identifier for the case.
        * 'fitness' (float): The calculated fitness score for the trace.
        * 'deviations' (dict): A sub-dictionary containing:
            - 'skipped' (list): Model activities that did not occur in the log.
            - 'unsolicited' (list): Log activities that were not predicted 
              by the model.
    
    Notes
    -----
    The function distinguishes between log moves, model moves, and synchronous 
    moves. It specifically ignores "silent" transitions (e.g., those starting 
    with 'tau', 'skip', or 'init') when recording skipped steps.
    
    If a pre-calculated 'fitness' is not found in the alignment dictionary, 
    it is derived using the cost:
    $$Fitness = 1.0 - \frac{cost}{length(trace) + 1}$$
    """
    cdef list details = []
    cdef Py_ssize_t i
    cdef object trace
    cdef dict align
    cdef double t_fit
    cdef list skipped_steps
    cdef list unsolicited_steps
    cdef list alignment_sequence
    cdef tuple move
    cdef object log_part, model_part, log_label, model_label
    cdef str model_label_str, log_label_str
    
    # Variables for calculation
    cdef double cost
    cdef Py_ssize_t trace_len

    for i, (trace, align) in enumerate(zip(clean_log, alignments)):
        if not isinstance(align, dict): continue
        
        if 'fitness' in align:
            t_fit = float(align['fitness'])
        else:
            # Fallback calculation
            cost = align.get('cost', 0)
            trace_len = len(trace)
            t_fit = 1.0 - (cost / (trace_len + 1))
            if t_fit < 0.0: t_fit = 0.0

        # Parsing Logic
        skipped_steps = []
        unsolicited_steps = []
        alignment_sequence = align.get('alignment', [])

        for move in alignment_sequence:
            log_part = move[0]
            if isinstance(log_part, tuple):
                log_label = log_part[0]
            else:
                log_label = log_part

            model_part = move[1]
            if isinstance(model_part, tuple):
                model_label = model_part[0]
            else:
                model_label = model_part

            log_label_str = str(log_label) if log_label is not None else "None"
            model_label_str = str(model_label) if model_label is not None else "None"

            # Check logic (Skipped)
            if (log_label == '>>' or log_label is None) and (model_label != '>>' and model_label is not None):
                if not model_label_str.startswith(('tau', 'skip', 'init')):
                    skipped_steps.append(model_label_str)

            # Check logic (Unsolicited)
            elif (log_label != '>>' and log_label is not None) and (model_label == '>>' or model_label is None):
                unsolicited_steps.append(log_label_str)

        details.append({
            'case_id': str(trace.attributes.get('concept:name', f'Case_{i}')),
            'fitness': t_fit,
            'deviations': {
                'skipped': skipped_steps,
                'unsolicited': unsolicited_steps
            }
        })
    
    return details

cpdef dict run_conformance_checking(
    object event_log_df,
    object process_model,
    object initial_marking,
    object final_marking,
    int max_align = 250,
    int max_prec_cases = 250,
    int cores = 1,
    str alignment_variant = 'state_equation_a_star',
    bint enable_detailed_analysis = False, 
    bint calculate_fitness = False,
    bint optimize_variants = True,
    bint perform_sampling = True,
    str strata_col = None
):
    """
    Executes comprehensive conformance checking between an event log and a process model.
    
    This high-level function orchestrates data sampling, fitness calculation, precision 
    evaluation, and trace alignment. It utilizes Cython-optimized routines and 
    optional variant grouping to provide a structured analysis of how well the 
    observed behavior (event log) matches the theoretical behavior (Petri net).
    
    Parameters
    ----------
    event_log_df : pd.DataFrame
        The input event log in a Pandas DataFrame format.
    process_model : object
        The Petri net model (Accepting Petri Net) used for conformance checking.
    initial_marking : object
        The initial marking (start state) of the Petri net.
    final_marking : object
        The final marking (end state) of the Petri net.
    max_align : int, optional
        Maximum number of traces to align to manage computational load (default 250).
    max_prec_cases : int, optional
        Maximum number of traces to use for precision calculation (default 250).
    cores : int, optional
        Number of CPU cores for parallel processing. Set to 0 to use all 
        available cores minus one (default 1).
    alignment_variant : str, optional
        The PM4Py alignment algorithm variant to use (default 'state_equation_a_star').
    enable_detailed_analysis : bool, optional
        If True, calculates precision and detailed case-level deviations (default False).
    calculate_fitness : bool, optional
        If True, triggers a standalone batched fitness calculation (default False).
    optimize_variants : bool, optional
        If True, groups identical traces into variants to reduce the number of 
        alignment operations (default True).
    perform_sampling : bool, optional
        If True, uses stratified sampling to select traces for analysis 
        instead of simple head-slicing (default True).
    strata_col : str, optional
        The column name to use for stratified sampling. If None, defaults to 
        'case:concept:name' or 'purchase'.
    
    Returns
    -------
    results : dict
        A dictionary containing the analysis output:
        * 'fitness' (dict): Log-level fitness score and calculation method.
        * 'precision' (dict): Precision score (degree of model over-generalization).
        * 'alignments' (dict): Summary of alignment costs and trace counts.
        * 'case_analysis' (dict): Detailed list of deviations per case.
        * 'overall_summary' (dict): Aggregated quality assessment (Excellent to Poor).
        * 'errors' (list): Captured exceptions and stack traces.
    
    Notes
    -----
    Conformance checking evaluates two primary dimensions:
    1. **Fitness**: Can the model explain the traces in the log?
    2. **Precision**: Does the model allow for behavior not found in the log?
    """
    cdef dict results = {
        'fitness': {'log_fitness': 0, 'note': 'Pending calculation'},
        'precision': {'precision_score': 0},
        'alignments': {},
        'case_analysis': {'cases': []},
        'errors': [],
        'overall_summary': {}
    }

    cdef object sampled_log
    cdef double avg_fitness
    
    try:
        # --- 1. DATA PREPARATION ---
        try:
            if perform_sampling:
                sc = 'case:concept:name' # Default fallback
                if strata_col and strata_col in event_log_df.columns:
                    sc = strata_col
                elif 'purchase' in event_log_df.columns:
                    sc = 'purchase'
                    
                # Note: sample_log_stratified returns (df, messages)
                s_df = sample_log_stratified(event_log_df, sc, total_sample_size=100)[0]
                sampled_log = pm4py.convert_to_event_log(s_df)
            else:
                sampled_log = pm4py.convert_to_event_log(event_log_df.iloc[:max_align])
        except:
            sampled_log = pm4py.convert_to_event_log(event_log_df)

        # --- 2. FITNESS (CONDITIONAL) ---
        if calculate_fitness:
            try:
                print(f"--- Calculating Standalone Fitness on {len(sampled_log)} traces ---")
                avg_fitness = calculate_fitness_in_batches(sampled_log, process_model, initial_marking, final_marking, batch_size=200)
                results['fitness'] = {
                    'log_fitness': avg_fitness,
                    'note': f"Calculated via Batched Replay"
                }
            except Exception as e:
                print(f"Fitness calculation failed: {e}")

        # --- 3. PRECISION ---
        if enable_detailed_analysis:
            try:
                prec_input = sampled_log
                if len(sampled_log) > max_prec_cases:
                    prec_input = sampled_log[:max_prec_cases]
                    
                print(f"--- Calculating Precision on {len(prec_input)} traces ---")
                prec = precision_evaluator.apply(
                    prec_input, process_model, initial_marking, final_marking,
                    variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
                )
                results['precision'] = {'precision_score': prec if isinstance(prec, float) else prec.get('precision', 0)}
            except Exception as e:
                pass

        # --- 4. ALIGNMENTS ---
        if alignment_variant != 'token_replay':
            input_log = sampled_log
            if len(sampled_log) > max_align:
                input_log = sampled_log[:max_align]

            # Clean Clone
            clean_log = EventLog()
            for trace in input_log:
                nt = Trace()
                nt.attributes['concept:name'] = str(trace.attributes.get('concept:name', 'Unknown'))
                for event in trace:
                    nt.append(Event({'concept:name': str(event['concept:name'])}))
                clean_log.append(nt)

            # Rebuild Markings
            rim = Marking()
            rfm = Marking()
            for p in process_model.places:
                if not p.in_arcs: rim[p] = 1
                if not p.out_arcs: rfm[p] = 1
            if not rim: rim[list(process_model.places)[0]] = 1
            if not rfm: rfm[list(process_model.places)[-1]] = 1

            max_cores = max(1, os.cpu_count() - 1) if cores == 0 else cores
            
            print(f"--- Calculating Alignments with {max_cores} core(s) on {len(clean_log)} traces... ---")
            
            params = {'cores': max_cores, 'ret_tuple_as_trans_desc': True}
            # alignments = alignments_algorithm.apply(clean_log, process_model, rim, rfm, parameters=params)

            if optimize_variants:
                print(f"--- Optimizing: Grouping {len(clean_log)} traces into variants ---")
                
                variant_map = {}
                unique_traces = []
                
                for i, trace in enumerate(clean_log):
                    sig_list = []
                    for e in trace:
                        sig_list.append(str(e['concept:name']))
                    sig = tuple(sig_list)
                    
                    if sig not in variant_map:
                        variant_map[sig] = []
                        unique_traces.append(trace)
                    
                    variant_map[sig].append(i)
                
                print(f"--- Calculating Alignments on {len(unique_traces)} UNIQUE variants (was {len(clean_log)}) ---")
                
                variant_alignments = alignments_algorithm.apply(unique_traces, process_model, rim, rfm, parameters=params)
                
                final_alignments = [None] * len(clean_log)
                
                for k, align_result in enumerate(variant_alignments):
                    trace_obj = unique_traces[k]
                    sig_list = []
                    for e in trace_obj:
                        sig_list.append(str(e['concept:name']))
                    sig = tuple(sig_list)
                    
                    original_indices = variant_map[sig]
                    
                    for idx in original_indices:
                        final_alignments[idx] = align_result 
            else:
                # Fallback: slower method (event-level brute force)
                print(f"--- Calculating Alignments (No Optimization) on {len(clean_log)} traces... ---")
                final_alignments = alignments_algorithm.apply(clean_log, process_model, rim, rfm, parameters=params)

            alignments = final_alignments
            valid = [a for a in alignments if isinstance(a, dict) and 'cost' in a]
            if valid:
                costs = [a['cost'] for a in valid]
                
                # Fitness backfill
                f_vals = []
                for i, align in enumerate(alignments):
                    if not isinstance(align, dict): continue
                    
                    if 'fitness' in align:
                        f_vals.append(float(align['fitness']))
                    else:
                        t_len = len(clean_log[i])
                        f_vals.append(max(0.0, 1.0 - (align['cost'] / (t_len + 1))))
                
                results['fitness'] = {
                    'log_fitness': float(np.mean(f_vals)) if f_vals else 0.0,
                    'note': "Derived from Alignments (Gold Standard)"
                }

                results['alignments'] = {
                    'total': len(valid),
                    'average_cost': float(np.mean(costs)),
                    'note': f"Calculated on {len(valid)} traces"
                }

                # --- CALL CYTHON PARSER ---
                details = parse_alignments_cython(clean_log, alignments)
                results['case_analysis'] = {'cases': details}

    except Exception as e:
        results['errors'].append(f"Conformance Error: {e}")
        results['errors'].append(str(traceback.format_exc()))

    # --- FINAL SUMMARY ---
    fit_score = results['fitness'].get('log_fitness', 0)
    prec_score = results['precision'].get('precision_score', 0)
    
    quality = "Poor"
    if prec_score > 0:
        avg_score = (fit_score + prec_score) / 2
    else:
        avg_score = fit_score

    if avg_score > 0.8: quality = "Excellent"
    elif avg_score > 0.6: quality = "Good"
    elif avg_score > 0.4: quality = "Fair"

    results['overall_summary'] = {
        'fitness_score': fit_score,
        'precision_score': prec_score,
        'quality_assessment': quality,
        'recommendations': [] 
    }

    return results
