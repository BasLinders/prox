import pandas as pd
import pm4py
from typing import Tuple, List, Union

# === IMPORT MINING ALGORITHMS ===
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner

# === IMPORT CONVERTERS ===
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.objects.conversion.dfg import converter as dfg_converter

def perform_process_discovery(
    event_log_df: pd.DataFrame,
    discovery_algo: str = 'inductive_miner',
    noise_threshold: float = 0.0,
    dependency_threshold: float = 0.5,
    activity_threshold: int = 0
) -> Tuple[tuple | dict | None, list, list]:
    """
    Applies a chosen process discovery algorithm to the event log to generate a process model.
    
    This function acts as a centralized interface for various process discovery 
    techniques. It transforms raw event data into a formal Petri net representation 
    (net, initial marking, final marking), handling the necessary conversions 
    between DataFrames and PM4Py EventLog objects while applying algorithm-specific 
    hyperparameters.
    
    Parameters
    ----------
    event_log_df : pd.DataFrame
        The input event log. Must contain the standard PM4Py columns: 
        'case:concept:name', 'concept:name', and 'time:timestamp'.
    discovery_algo : {'inductive_miner', 'dfg', 'alpha_miner', 'heuristics_miner'}, optional
        The algorithm to use for model discovery (default is 'inductive_miner').
        * 'inductive_miner': Guarantees sound process trees/Petri nets.
        * 'dfg': Directly-Follows Graph discovery, converted to a Petri net.
        * 'alpha_miner': The classic approach to process discovery.
        * 'heuristics_miner': Robust against noise by focusing on frequent paths.
    noise_threshold : float, optional
        Threshold used by the Inductive Miner to filter out infrequent behavior 
        (default is 0.0).
    dependency_threshold : float, optional
        Threshold used by the Heuristics Miner to determine the strength of 
        causal dependencies (default is 0.5).
    activity_threshold : int, optional
        Minimum occurrences required for an activity to be included in the 
        discovery (default is 0).
    
    Returns
    -------
    process_model : tuple (net, im, fm) or None
        A triple containing the Petri net, the initial marking, and the 
        final marking. Returns None if a critical error occurs.
    errors : list of str
        A list of error messages or stack traces if the discovery process fails.
    messages : list of str
        A list of informational notes regarding the discovery parameters used.
    
    Notes
    -----
    Discovery algorithms vary significantly in how they handle noise and 
    completeness. The Inductive Miner is generally preferred for producing 
    reproducible, sound models.
    
    See Also
    --------
    run_conformance_checking : Uses the discovered model to evaluate log fitness.
    """

    errors = []
    messages = []
    process_model = None

    # --- Validation ---
    required_pm4py_cols = ['case:concept:name', 'concept:name', 'time:timestamp']
    if not all(col in event_log_df.columns for col in required_pm4py_cols):
        errors.append("Critical Error: Event log doesn't contain essential PM4Py columns.")
        return None, errors, messages

    if event_log_df.empty:
        errors.append("Critical Error: Can't discover process from an empty event log.")
        return None, errors, messages

    # --- Conversion ---
    try:
        log = pm4py.convert_to_event_log(event_log_df)
    except Exception as e:
        errors.append(f"Error when converting from DataFrame to PM4Py EventLog: {e}")
        return None, errors, messages

    # --- Initiate algorithm ---
    try:
        if discovery_algo == 'inductive_miner':
            tree = inductive_miner.apply(log, parameters={'noise_threshold': noise_threshold})
            net, im, fm = pt_converter.apply(tree, variant=pt_converter.Variants.TO_PETRI_NET)
            process_model = (net, im, fm)
            messages.append(f"Info: Petri net discovered with Inductive Miner (noise threshold: {noise_threshold})")

        elif discovery_algo == 'dfg':
            dfg = dfg_discovery.apply(log)

            # Convert DFG to Petri Net so downstream functions don't crash
            net, im, fm = dfg_converter.apply(dfg, variant=dfg_converter.Variants.TO_PETRI_NET)
            process_model = (net, im, fm)
            messages.append(f"Info: Process discovered using DFG (activity threshold: {activity_threshold})")

        elif discovery_algo == 'alpha_miner':
            net, im, fm = alpha_miner.apply(log)
            process_model = (net, im, fm)
            messages.append("Info: Process discovered using Alpha Miner")

        elif discovery_algo == 'heuristics_miner':
            parameters = {
                "dependency_threshold": dependency_threshold,
                "min_act_count": activity_threshold
            }

            net, im, fm = heuristics_miner.apply(log, parameters=parameters)
            process_model = (net, im, fm)
            messages.append(f"Info: Process discovered using Heuristics Miner (dependency: {dependency_threshold}, activity: {activity_threshold})")

        else:
            errors.append(f"Critical Error: Unknown discovery algorithm '{discovery_algo}'.")
            return None, errors, messages

    except Exception as e:
        errors.append(f"Error applying discovery algorithm: {e}")
        import traceback
        errors.append(f"Traceback: {traceback.format_exc()}")
        return None, errors, messages

    if process_model is None:
        errors.append("Critical Error: Process model could not be generated.")
        return None, errors, messages

    return process_model, errors, messages
