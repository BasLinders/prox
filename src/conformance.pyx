import pm4py
import pandas as pd
import numpy as np
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algorithm
from typing import List, Dict, Any

cpdef double calculate_fitness_in_batches(object log, object net, object im, object fm, int batch_size=200):
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
  
