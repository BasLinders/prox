import logging
import pandas as pd
import pm4py
from typing import Tuple

from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.objects.conversion.dfg import converter as dfg_converter

logger = logging.getLogger(__name__)


def _discover_inductive_miner(log, *, noise_threshold=0.2, **_ignored):
    tree = inductive_miner.apply(log, parameters={'noise_threshold': noise_threshold})
    net, im, fm = pt_converter.apply(tree, variant=pt_converter.Variants.TO_PETRI_NET)
    message = f"Petri net discovered via Inductive Miner (noise={noise_threshold})"
    return net, im, fm, message


def _discover_heuristics_miner(log, *, dependency_threshold=0.5, activity_threshold=0, **_ignored):
    parameters = {
        "dependency_threshold": dependency_threshold,
        "min_act_count": activity_threshold
    }
    net, im, fm = heuristics_miner.apply(log, parameters=parameters)
    message = (
        f"Process discovered via Heuristics Miner "
        f"(dependency={dependency_threshold}, activity={activity_threshold})"
    )
    return net, im, fm, message


def _discover_dfg(log, *, activity_threshold=0, **_ignored):
    dfg, start_activities, end_activities = pm4py.discover_dfg(log)
    variant = dfg_converter.Variants.VERSION_TO_PETRI_NET_ACTIVITY_DEFINES_PLACE
    net, im, fm = dfg_converter.apply(
        dfg,
        variant=variant,
        parameters={
            variant.value.Parameters.START_ACTIVITIES: start_activities,
            variant.value.Parameters.END_ACTIVITIES: end_activities,
        }
    )
    message = f"Process discovered via DFG (activity threshold={activity_threshold})"
    return net, im, fm, message


# Single source of truth for available discovery algorithms: the engine
# dispatches on this dict, and the Streamlit UI builds its selectbox options
# and help text from it, so a new algorithm only needs an entry here.
DISCOVERY_ALGORITHMS = {
    'inductive_miner': {
        'handler': _discover_inductive_miner,
        'label': 'Inductive Miner',
        'help': 'Guaranteed sound model, best for most datasets.',
    },
    'heuristics_miner': {
        'handler': _discover_heuristics_miner,
        'label': 'Heuristics Miner',
        'help': 'Better for noisy or very large logs.',
    },
    'dfg': {
        'handler': _discover_dfg,
        'label': 'DFG',
        'help': 'Directly-Follows Graph, fastest option, best for a quick first look.',
    },
}


def perform_process_discovery(
    event_log_df: pd.DataFrame,
    discovery_algo: str = 'inductive_miner',
    noise_threshold: float = 0.0,
    dependency_threshold: float = 0.5,
    activity_threshold: int = 0
) -> Tuple[tuple | None, list, list]:
    """
    Applies a chosen process discovery algorithm to an event log to produce a Petri net.

    Parameters
    ----------
    event_log_df : pd.DataFrame
        Must contain 'case:concept:name', 'concept:name', and 'time:timestamp'.
    discovery_algo : str
        Key into DISCOVERY_ALGORITHMS. Inductive Miner is recommended for production use.
    noise_threshold : float
        Inductive Miner noise filter (0.0–1.0). Higher = simpler model.
    dependency_threshold : float
        Heuristics Miner dependency strength threshold.
    activity_threshold : int
        Minimum activity occurrences for inclusion (Heuristics Miner only).

    Returns
    -------
    process_model : tuple (net, im, fm) or None
    errors : list of str
    messages : list of str
    """
    errors = []
    messages = []

    required = ['case:concept:name', 'concept:name', 'time:timestamp']
    if not all(col in event_log_df.columns for col in required):
        errors.append("Critical Error: Event log is missing required PM4Py columns.")
        return None, errors, messages

    if event_log_df.empty:
        errors.append("Critical Error: Cannot discover a process from an empty event log.")
        return None, errors, messages

    try:
        log = pm4py.convert_to_event_log(event_log_df)
    except Exception as e:
        errors.append(f"Error converting DataFrame to PM4Py EventLog: {e}")
        return None, errors, messages

    entry = DISCOVERY_ALGORITHMS.get(discovery_algo)
    if entry is None:
        valid = ', '.join(DISCOVERY_ALGORITHMS)
        errors.append(f"Critical Error: Unknown discovery algorithm '{discovery_algo}'. Valid options: {valid}.")
        return None, errors, messages

    try:
        net, im, fm, message = entry['handler'](
            log,
            noise_threshold=noise_threshold,
            dependency_threshold=dependency_threshold,
            activity_threshold=activity_threshold,
        )
        messages.append(message)
    except Exception as e:
        import traceback
        errors.append(f"Error applying discovery algorithm '{discovery_algo}': {e}")
        errors.append(traceback.format_exc())
        return None, errors, messages

    logger.info(messages[-1] if messages else "Discovery complete.")
    return (net, im, fm), errors, messages
