import logging
import os
import pandas as pd
import pm4py
from typing import Union, Dict, List, Tuple

from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer

logger = logging.getLogger(__name__)


def visualize_focused_insights(
    event_log,
    output_folder: str = "output",
    bottleneck_top_k: int = 15
) -> Tuple[str | None, str | None]:
    """
    Generates two BPMN process maps and saves them as PNGs.

    1. Happy Path   — single most frequent variant.
    2. Main Flow    — top K variants combined (noise_threshold=0.2 for readability).

    Falls back to a Petri Net visualisation if BPMN conversion fails.

    Returns
    -------
    happy_path_img : str or None   — absolute path to the happy path PNG.
    main_flow_img  : str or None   — absolute path to the main flow PNG.
    """
    abs_output = os.path.abspath(output_folder)
    os.makedirs(abs_output, exist_ok=True)

    # Normalise all activity labels to strings to prevent PM4Py serialisation issues
    for trace in event_log:
        trace.attributes['concept:name'] = str(trace.attributes.get('concept:name', 'Unknown'))
        for event in trace:
            event['concept:name'] = str(event['concept:name'])

    def _generate_bpmn(log_data, filename: str, title: str) -> str | None:
        output_path = os.path.join(abs_output, filename)
        threshold = 0.0 if "happy" in filename else 0.2
        try:
            logger.info("Generating %s BPMN.", title)
            tree = pm4py.discover_process_tree_inductive(log_data, noise_threshold=threshold)
            bpmn_graph = pm4py.convert_to_bpmn(tree)
            gviz = bpmn_visualizer.apply(bpmn_graph)
            gviz.format = 'png'
            with open(output_path, 'wb') as f:
                f.write(gviz.pipe())
            logger.info("%s saved: %s", title, output_path)
            return output_path
        except Exception as e:
            logger.warning("BPMN generation failed for %s: %s. Trying Petri Net fallback.", title, e)
            try:
                net, im, fm = pm4py.discover_petri_net_inductive(log_data, noise_threshold=threshold)
                gviz = pn_visualizer.apply(net, im, fm)
                gviz.format = 'png'
                with open(output_path, 'wb') as f:
                    f.write(gviz.pipe())
                logger.info("%s (Petri Net) saved: %s", title, output_path)
                return output_path
            except Exception as e2:
                logger.error("Petri Net fallback also failed for %s: %s", title, e2)
                return None

    # --- Happy Path ---
    happy_output = None
    try:
        variants = pm4py.get_variants_as_tuples(event_log)
        if variants:
            top_variant = max(variants, key=lambda x: len(variants[x]))
            happy_log = variants_filter.apply(event_log, [top_variant])
            happy_output = _generate_bpmn(happy_log, "happy_path_model.png", "Happy Path")
        else:
            logger.warning("No variants found in log for happy path generation.")
    except Exception as e:
        logger.error("Error preparing happy path data: %s", e)

    # --- Main Process Flow ---
    main_output = None
    try:
        filtered_log = pm4py.filter_variants_top_k(event_log, bottleneck_top_k)
        main_output = _generate_bpmn(filtered_log, "main_process_flow.png", "Main Process Flow")
    except Exception as e:
        logger.error("Error preparing main process flow: %s", e)

    return happy_output, main_output


def render_petri_net(net, im, fm, output_path: str) -> str | None:
    """
    Renders a raw (net, im, fm) Petri net to a PNG, for models that aren't
    discovered from an event log (e.g. a user-defined reference model) and
    so have no BPMN conversion path through visualize_focused_insights().

    Returns
    -------
    str or None - the output path on success, None on failure.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        gviz = pn_visualizer.apply(net, im, fm)
        gviz.format = 'png'
        with open(output_path, 'wb') as f:
            f.write(gviz.pipe())
        return output_path
    except Exception as e:
        logger.error("Failed to render reference model Petri net: %s", e)
        return None


def export_results(
    data: Union[pd.DataFrame, Dict, List[Dict]],
    filename: str,
    file_format: str = 'csv',
    output_folder: str = 'output'
) -> Tuple[bool, str]:
    """
    Saves analysis results to a file.

    Accepts a DataFrame, a dict (saved as Metric/Value pairs), or a list of dicts.
    Currently supports CSV only.

    Returns
    -------
    success : bool
    message : str
    """
    if data is None:
        return False, "Error: data is None."

    os.makedirs(output_folder, exist_ok=True)
    base, _ = os.path.splitext(filename)
    output_path = os.path.join(output_folder, f"{base}.{file_format}")

    try:
        if file_format == 'csv':
            if isinstance(data, pd.DataFrame):
                df_to_save = data
            elif isinstance(data, dict):
                df_to_save = pd.DataFrame(list(data.items()), columns=['Metric', 'Value'])
            elif isinstance(data, list) and all(isinstance(i, dict) for i in data):
                df_to_save = pd.DataFrame(data)
            else:
                return False, f"Error: Unsupported data type '{type(data).__name__}'."

            df_to_save.to_csv(output_path, index=False)
            msg = f"Exported to '{output_path}'."
            logger.info(msg)
            return True, msg
        else:
            return False, f"Unsupported format '{file_format}'. Only 'csv' is supported."

    except IOError as e:
        msg = f"Could not write '{output_path}': {e}"
        logger.error(msg)
        return False, msg
    except Exception as e:
        msg = f"Unexpected export error: {e}"
        logger.error(msg)
        return False, msg
