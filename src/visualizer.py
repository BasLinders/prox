import pandas as pd
import os
import pm4py
from typing import Union, Dict, List, Any
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer

def visualize_focused_insights(event_log, output_folder="output", bottleneck_top_k=15):
    """
    Generates specialized BPMN visualizations for process behavior and deviations.
    
    This function creates two distinct process maps to assist analysts in 
    understanding process execution:
    1. **Happy Path**: A visualization of the single most frequent variant, 
       representing the "standard" or intended process flow.
    2. **Main Process Flow**: A combined model of the top K variants, 
       highlighting common deviations and structural bottlenecks.
    
    Parameters
    ----------
    event_log : EventLog
        A PM4Py EventLog object or compatible collection of traces.
    output_folder : str, optional
        The directory where the generated PNG images will be stored 
        (default is "output").
    bottleneck_top_k : int, optional
        The number of most frequent variants to include in the "Main Process 
        Flow" model (default is 15).
    
    Returns
    -------
    happy_output : str or None
        The absolute file path to the saved Happy Path BPMN image. 
        Returns None if generation fails.
    process_output : str or None
        The absolute file path to the saved Main Process Flow image. 
        Returns None if generation fails.
    
    Notes
    -----
    The function uses the Inductive Miner algorithm to discover a process tree, 
    which is then converted to BPMN (Business Process Model and Notation). 
    If BPMN conversion fails, it attempts a fallback to a Petri Net 
    visualization.
    
    The "Main Process Flow" uses a noise threshold ($0.2$) to ensure 
    readability by filtering out infrequent transitions that would 
    otherwise create a "spaghetti" model.
    
    See Also
    --------
    pm4py.discover_process_tree_inductive : The underlying discovery algorithm.
    pm4py.filter_variants_top_k : Used to isolate the most frequent process paths.
    """
    print("--- Generating Focused Visualizations (BPMN Style) ---")
    
    abs_output_folder = os.path.abspath(output_folder)
    if not os.path.exists(abs_output_folder):
        os.makedirs(abs_output_folder)

    # 1. Clean Log Data (Force Strings)
    for trace in event_log:
        trace.attributes['concept:name'] = str(trace.attributes.get('concept:name', 'Unknown'))
        for event in trace:
            event['concept:name'] = str(event['concept:name'])

    # Helper function to generate BPMN safely
    def generate_bpmn(log_data, filename, title):
        output_path = os.path.join(abs_output_folder, filename)
        try:
            print(f"   -> Generating {title} (BPMN)...")
            
            # Discovery: Inductive Miner guarantees a sound model
            # For the main flow (bottlenecks), use a slight noise threshold (0.2)
            # to prevent the BPMN from becoming unreadable if variants are very different.
            threshold = 0.0 if "happy" in filename else 0.2
            
            tree = pm4py.discover_process_tree_inductive(log_data, noise_threshold=threshold)
            bpmn_graph = pm4py.convert_to_bpmn(tree)
            
            # Render and Save
            gviz = bpmn_visualizer.apply(bpmn_graph)
            gviz.format = 'png'
            with open(output_path, 'wb') as f:
                f.write(gviz.pipe())
            
            print(f"    -> Saved {title}: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"    -> BPMN generation failed for {title}: {e}")
            print("    -> Attempting Petri Net fallback...")
            try:
                # Fallback to Petri Net (very messy on spaghetti models)
                net, im, fm = pm4py.discover_petri_net_inductive(log_data, noise_threshold=threshold)
                gviz = pn_visualizer.apply(net, im, fm)
                gviz.format = 'png'
                with open(output_path, 'wb') as f:
                    f.write(gviz.pipe())
                print(f"    -> Saved {title} (Petri Net): {output_path}")
                return output_path
            except Exception as e2:
                print(f"    -> Fallback also failed: {e2}")
                return None

    # ---------------------------------------------------------
    # 1. Happy Path (Top 1 Variant)
    # ---------------------------------------------------------
    happy_output = None
    try:
        variants = pm4py.get_variants_as_tuples(event_log)
        if variants:
            most_frequent_variant = max(variants, key=lambda x: len(variants[x])) 
            happy_log = variants_filter.apply(event_log, [most_frequent_variant])
            
            happy_output = generate_bpmn(happy_log, "happy_path_model.png", "Happy Path")
        else:
            print("    -> Warning: No variants found in log.")
    except Exception as e:
        print(f"    -> Error preparing Happy Path data: {e}")

    # ---------------------------------------------------------
    # 2. Main Process Flow (Top K Bottlenecks)
    # ---------------------------------------------------------
    process_output = None
    try:
        print(f"   -> Filtering log for Main Process Flow (Top {bottleneck_top_k} bottlenecks)...")
        # Filter top K variants
        filtered_log = pm4py.filter_variants_top_k(event_log, bottleneck_top_k)
        
        process_output = generate_bpmn(filtered_log, "main_process_flow.png", "Main Process Flow")
        
    except Exception as e:
        print(f"   -> Error preparing Main Process data: {e}")

    return happy_output, process_output

def export_results(
    data: Union[pd.DataFrame, Dict, List[Dict]],
    filename: str,
    file_format: str = 'csv',
    output_folder: str = 'output'
) -> tuple[bool, str]:
    """
    Saves analytical results to a file, handling various input data structures.
    
    This utility function acts as a flexible data sink for the analysis pipeline. 
    It automatically transforms process mining artifacts (like summary 
    dictionaries or performance DataFrames) into a standardized format for 
    storage. It ensures the target directory exists and manages file extensions 
    to prevent path errors.
    
    Parameters
    ----------
    data : pd.DataFrame or dict or list of dict
        The data object to be exported. 
        * If `pd.DataFrame`: Saved directly.
        * If `dict`: Converted to a two-column 'Metric' and 'Value' format.
        * If `list` of `dict`: Flattened into a row-per-dictionary DataFrame.
    filename : str
        The base name of the file without the extension (e.g., "case_deviations").
    file_format : str, optional
        The target file extension. Currently, only 'csv' is supported 
        (default is 'csv').
    output_folder : str, optional
        The directory path where the file will be created. If the folder 
        does not exist, it will be generated automatically (default is 'output').
    
    Returns
    -------
    success : bool
        True if the file was written successfully, False otherwise.
    message : str
        A descriptive string confirming the output path or detailing the 
        specific failure (e.g., permission errors or unsupported types).
    
    Notes
    -----
    The function provides an abstraction layer over `pandas.to_csv`, 
    standardizing how process metrics are persisted for business reporting.
    
    Raises
    ------
    IOError
        If the file cannot be written due to disk space, naming conflicts, 
        or restricted folder permissions.
    """
    # --- 1. Input Validation ---
    if data is None:
        return False, "Error: The data to export is None."

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Ensure the filename has the correct extension
    base_filename, _ = os.path.splitext(filename)
    output_filename = os.path.join(output_folder, f"{base_filename}.{file_format}")

    try:
        if file_format == 'csv':
            # --- 2. Handle different data types ---
            if isinstance(data, pd.DataFrame):
                # Data is already a DataFrame, save it directly
                df_to_save = data
            elif isinstance(data, dict):
                # Convert a single dictionary to a DataFrame
                # We'll create a two-column DataFrame: 'Metric' and 'Value'
                df_to_save = pd.DataFrame(list(data.items()), columns=['Metric', 'Value'])
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                # Convert a list of dictionaries to a DataFrame
                df_to_save = pd.DataFrame(data)
            else:
                return False, f"Error: Unsupported data type for CSV export: {type(data).__name__}."

            # --- 3. Save the file ---
            df_to_save.to_csv(output_filename, index=False)
            message = f"Successfully exported data to '{output_filename}'."
            print(message)  # Provide feedback in non-UI environments like Colab
            return True, message

        else:
            return False, f"Error: Unsupported file format '{file_format}'. Only 'csv' is supported."

    except IOError as e:
        message = f"Error: Could not write to file '{output_filename}'. Check permissions. Details: {e}"
        print(message)
        return False, message
    except Exception as e:
        message = f"An unexpected error occurred during export: {e}"
        print(message)
        return False, message
