import pandas as pd
import os
from typing import Union, Dict, List, Any
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer

def visualize_focused_insights(event_log, output_folder="output", bottleneck_top_k=15):
    """ 
    Generates two specific visualizations for analysts using BPMN (Flowcharts).
    
    1. Happy Path: The single most frequent path.
    2. Main Process Flow: The top K variants combined (showing structure of deviations).
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
    # 2. Main Process Flow (Top K Variants)
    # ---------------------------------------------------------
    process_output = None
    try:
        print(f"   -> Filtering log for Main Process Flow (Top {bottleneck_top_k} variants)...")
        # Filter top K variants
        filtered_log = pm4py.filter_variants_top_k(event_log, bottleneck_top_k)
        
        process_output = generate_bpmn(filtered_log, "main_process_flow.png", "Main Process Flow")
        
    except Exception as e:
        print(f"   -> Error preparing Main Process data: {e}")

    return happy_output, process_output
