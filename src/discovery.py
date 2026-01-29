import pandas as pd
import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner

def perform_process_discovery(
    event_log_df: pd.DataFrame,
    discovery_algo: str = 'inductive_miner',
    noise_threshold: float = 0.0,
    dependency_threshold: float = 0.5,
    activity_threshold: int = 0
) -> Tuple[tuple | dict | None, list, list]:
    """
    Purpose: Applies a chosen process discovery algorithm to the event log.
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
