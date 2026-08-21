import logging
import gc
import os
import tempfile
import numpy as np
import pandas as pd
import traceback
from typing import Dict, Any, Tuple

import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.petri_net.obj import Marking
from pm4py.objects.process_tree.obj import ProcessTree, Operator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algorithm

from .data_manager import sample_log_stratified

logger = logging.getLogger(__name__)

# Stage 'type' -> ProcessTree operator, for stage types that combine multiple
# activities into one node (choice = XOR between activities, parallel = AND).
# 'required'/'optional'/'repeatable' are single-activity stage types handled
# directly in _build_stage_node, since they need a synthetic tau leaf rather
# than combining two labelled activities.
_MULTI_ACTIVITY_STAGE_OPERATORS = {
    'choice': Operator.XOR,
    'parallel': Operator.PARALLEL,
}

_VALID_STAGE_TYPES = {'required', 'choice', 'optional', 'repeatable', 'parallel'}


def _leaf(label: str) -> ProcessTree:
    return ProcessTree(label=label)


def _tau() -> ProcessTree:
    """PM4Py's silent/unlabelled leaf - a step that consumes no event."""
    return ProcessTree()


def _attach(parent: ProcessTree, children: list) -> ProcessTree:
    parent.children = children
    for c in children:
        c.parent = parent
    return parent


def _build_stage_node(stage: dict) -> Tuple[ProcessTree | None, str | None]:
    """
    Builds a single ProcessTree node for one stage dict. Returns (node, None)
    on success or (None, error_message) on a malformed stage.
    """
    stage_type = stage.get('type')
    activities = stage.get('activities') or []

    if stage_type not in _VALID_STAGE_TYPES:
        return None, f"Unknown stage type '{stage_type}'. Valid types: {', '.join(sorted(_VALID_STAGE_TYPES))}."
    if not activities:
        return None, f"Stage of type '{stage_type}' has no activities."

    if stage_type == 'required':
        if len(activities) != 1:
            return None, "A 'required' stage must have exactly one activity."
        return _leaf(activities[0]), None

    if stage_type == 'optional':
        if len(activities) != 1:
            return None, "An 'optional' stage must have exactly one activity."
        # XOR between the activity and a silent leaf: the step may or may not happen.
        return _attach(ProcessTree(operator=Operator.XOR), [_leaf(activities[0]), _tau()]), None

    if stage_type == 'repeatable':
        if len(activities) != 1:
            return None, "A 'repeatable' stage must have exactly one activity."
        # LOOP(tau, activity): may happen 0+ times (tau as the "do" branch keeps
        # zero occurrences valid; activity as the "redo" branch allows more).
        return _attach(ProcessTree(operator=Operator.LOOP), [_tau(), _leaf(activities[0])]), None

    # 'choice' or 'parallel': combine 2+ activities under XOR/AND.
    if len(activities) < 2:
        return None, f"A '{stage_type}' stage needs at least two activities."
    operator = _MULTI_ACTIVITY_STAGE_OPERATORS[stage_type]
    return _attach(ProcessTree(operator=operator), [_leaf(a) for a in activities]), None


def build_structured_reference_model(stages: list) -> Tuple[tuple | None, list]:
    """
    Builds a reference Petri net from an ordered list of "stages" combined in
    SEQUENCE. Each stage describes what's allowed at that point:

      {'activities': ['a'],      'type': 'required'}   - exactly this step
      {'activities': ['a','b'],  'type': 'choice'}      - any ONE of these (XOR)
      {'activities': ['a'],      'type': 'optional'}    - may or may not happen
      {'activities': ['a'],      'type': 'repeatable'}  - may happen 0+ times (LOOP)
      {'activities': ['a','b'],  'type': 'parallel'}    - both, any order (AND)

    A stage list of all-'required' single-activity stages is exactly a plain
    ordered sequence.

    Returns
    -------
    (net, im, fm) or None, list of error strings.
    """
    errors = []

    if not isinstance(stages, list) or len(stages) < 2:
        errors.append("A reference model needs at least two stages.")
        return None, errors

    nodes = []
    for i, stage in enumerate(stages):
        node, err = _build_stage_node(stage)
        if err:
            errors.append(f"Stage {i + 1}: {err}")
        else:
            nodes.append(node)

    if errors:
        return None, errors

    root = _attach(ProcessTree(operator=Operator.SEQUENCE), nodes)

    try:
        net, im, fm = pm4py.convert_to_petri_net(root)
    except Exception as e:
        errors.append(f"Failed to convert reference model to a Petri net: {e}")
        return None, errors

    return (net, im, fm), errors


def import_reference_model_bpmn(file_bytes: bytes) -> Tuple[tuple | None, list]:
    """
    Imports a BPMN 2.0 XML file as a reference process model. Writes the
    uploaded bytes to a temp file (pm4py.read_bpmn requires a path, unlike
    load_and_validate_csv's BytesIO support), imports, and converts to a
    Petri net.

    Returns
    -------
    (net, im, fm) or None, list of error strings.
    """
    errors = []
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".bpmn", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        bpmn_graph = pm4py.read_bpmn(tmp_path)
        net, im, fm = pm4py.convert_to_petri_net(bpmn_graph)
        return (net, im, fm), errors
    except Exception as e:
        errors.append(f"Failed to import BPMN file: {e}")
        return None, errors
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def diff_reference_model_coverage(reference_net, log_df: pd.DataFrame) -> dict:
    """
    Two plain set differences between the reference model's transition labels
    and the raw log's actual activities (not the discovered model's transitions,
    which are themselves an approximation under noise_threshold - comparing
    against real observed activity is more direct and honest).

    Returns
    -------
    dict with keys:
        'unexpected_in_data' : sorted list - activities in the log but not in
                                the reference model (real behaviour the target
                                process doesn't account for).
        'never_observed'     : sorted list - activities the reference model
                                expects that never occur in the log at all
                                (dead/unused expected steps).
    """
    model_activities = {t.label for t in reference_net.transitions if t.label is not None}
    log_activities = set(log_df['concept:name'].dropna().astype(str).unique()) if 'concept:name' in log_df.columns else set()

    return {
        'unexpected_in_data': sorted(log_activities - model_activities),
        'never_observed': sorted(model_activities - log_activities),
    }


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


def _fitness_token_replay(sampled_log, process_model, initial_marking, final_marking, *, max_align, **_ignored):
    """Fast token-based replay: fitness only, no per-trace deviations."""
    input_log = sampled_log[:max_align] if len(sampled_log) > max_align else sampled_log
    logger.info("Calculating fitness via token-based replay on %d traces.", len(input_log))
    tbr = pm4py.fitness_token_based_replay(input_log, process_model, initial_marking, final_marking)
    return {
        'fitness': {
            'log_fitness': tbr.get('log_fitness', 0.0),
            'percentage_fit_traces': tbr.get('percentage_of_fitting_traces', 0.0),
            'note': "Calculated via token-based replay"
        }
    }


def _fitness_state_equation_alignments(
    sampled_log, process_model, initial_marking, final_marking, *,
    max_align, cores, optimize_variants, **_ignored
):
    """Exact alignment-based fitness plus per-trace skipped/unsolicited deviations."""
    input_log = sampled_log[:max_align] if len(sampled_log) > max_align else sampled_log

    # Rebuild a clean log with only string attributes to avoid PM4Py serialisation issues
    clean_log = EventLog()
    for trace in input_log:
        nt = Trace()
        nt.attributes['concept:name'] = str(trace.attributes.get('concept:name', 'Unknown'))
        for event in trace:
            nt.append(Event({'concept:name': str(event['concept:name'])}))
        clean_log.append(nt)

    # Use the real markings discovery/import already produced (same pattern
    # as _fitness_token_replay above) rather than guessing from net topology.
    # A "place with no in-arcs is the initial marking" heuristic happens to
    # match for most discovered nets, but silently diverges from the actual
    # im/fm for reference models built via build_structured_reference_model()
    # or import_reference_model_bpmn() - e.g. any net with loops or multiple
    # branches, where sourceless/sinkless places don't coincide with the real
    # start/end. Only fall back to the topology guess if no markings were
    # actually supplied.
    rim = initial_marking if initial_marking else Marking()
    rfm = final_marking if final_marking else Marking()
    if not rim:
        for p in process_model.places:
            if not p.in_arcs:
                rim[p] = 1
        if not rim:
            rim[list(process_model.places)[0]] = 1
    if not rfm:
        for p in process_model.places:
            if not p.out_arcs:
                rfm[p] = 1
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

    method_results = {}
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

        method_results['fitness'] = {
            'log_fitness': float(np.mean(f_vals)) if f_vals else 0.0,
            'note': "Derived from alignments"
        }
        method_results['alignments'] = {
            'total': len(valid),
            'average_cost': float(np.mean([a['cost'] for a in valid])),
            'note': f"Calculated on {len(valid)} traces"
        }
        method_results['case_analysis'] = {'cases': parse_alignments(clean_log, final_alignments)}

    return method_results


# Single source of truth for available conformance/fitness methods: the engine
# dispatches on this dict, and the Streamlit UI builds its selectbox options
# and help text from it, so a new method only needs an entry here.
CONFORMANCE_METHODS = {
    'token_replay': {
        'handler': _fitness_token_replay,
        'label': 'Token Replay',
        'help': 'Fast, gives fitness & precision (no per-trace deviations).',
    },
    'state_equation_a_star': {
        'handler': _fitness_state_equation_alignments,
        'label': 'State Equation A*',
        'help': 'Slower, gives exact per-trace deviations.',
    },
}


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
    alignment_variant   : Key into CONFORMANCE_METHODS - 'token_replay' for fast
                          token-based replay fitness (no per-trace deviations), or
                          'state_equation_a_star' for exact alignment-based fitness
                          with per-trace skipped/unsolicited deviations.
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

        # --- 2. Standalone fitness (optional batched-alignment override) ---
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

        # --- 4. Fitness via the selected conformance method ---
        entry = CONFORMANCE_METHODS.get(alignment_variant)
        if entry is None:
            valid = ', '.join(CONFORMANCE_METHODS)
            results['errors'].append(f"Unknown conformance method '{alignment_variant}'. Valid options: {valid}.")
        else:
            try:
                method_results = entry['handler'](
                    sampled_log, process_model, initial_marking, final_marking,
                    max_align=max_align, cores=cores, optimize_variants=optimize_variants
                )
                results.update(method_results)
            except Exception as e:
                results['errors'].append(f"{entry['label']} fitness calculation failed: {e}")

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
