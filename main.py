import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

# === IMPORT YOUR MODULES ===
from src.config import CONFIG
from src.data_manager import load_and_validate_csv, refine_activity_labels
from src.conformance import run_conformance_checking
from src.pipeline import run_full_analysis
from src.visualizer import export_results
from src.data_manager import (
    load_and_validate_csv,
    refine_activity_labels,
    filter_by_trace_length
)

if __name__ == '__main__':
    uploaded_file = 'C:/Users/Bas Linders/Documents/pm_dummy_data.csv'
    df_clean, errors_or_notes, _ = load_and_validate_csv(uploaded_file)
    df_clean = df_clean[df_clean.groupby('case:concept:name')['case:concept:name'].transform('count') < 30]

    # 1. Load Data Check
    if 'df_clean' not in locals():
        print("Please load 'df_clean' first using load_and_validate_csv()")
    else:
        # Create working copy
        df_ready = df_clean.copy()

        # 2. PRE-PROCESSING: UNLOCK COLUMNS
        # Critical Fix: Convert categories back to objects so we can edit text/rename columns
        for col in df_ready.select_dtypes(include=['category']).columns:
            df_ready[col] = df_ready[col].astype('object')

        # 3. Activity Refinement
        if 'page_type' in df_ready.columns:
            print("Context Found: Refining 'page_view' using 'page_type'...")
            df_ready = refine_activity_labels(df_ready, target_activity='page_view', context_column='page_type')

        # 4. Determine & Apply Case ID
        # (Must happen after df_ready exists but before optimization)
        target_case_id = CONFIG.get("active_case_id", "auto")
        
        if target_case_id != "auto":
            if target_case_id in df_ready.columns:
                print(f"Switching Case ID to: {target_case_id}")
                df_ready.rename(columns={target_case_id: 'case:concept:name'}, inplace=True)
            else:
                print(f"Warning: Configured ID '{target_case_id}' not found. Using default mapping.")

        # 5. Optimize Memory (Re-lock for speed)
        print('\n--- Optimizing Memory ---')
        if 'optimize_dataframe_memory' in locals():
            optimize_dataframe_memory(df_ready)

        # 6. Select Config
        current_config = CONFIG

        # 7. Run Analysis
        print(f"Starting analysis on {len(df_ready)} events...")
        pipeline_results = run_full_analysis(df_ready, config=current_config)

        # =========================================================
        # 8. POST-ANALYSIS DASHBOARD
        # =========================================================
        if pipeline_results:
            print("\n" + "="*80)
            print("ANALYSIS DASHBOARD")
            print("="*80)

            # Show trace length
            check_trace_length(df_ready)

            # --- A. VISUALIZATION (Process Map) ---
            print("\n--- 1. Process Map (Petri Net) ---")
            model_data = pipeline_results.get('model')
            if model_data:
                net, im, fm = model_data['net'], model_data['im'], model_data['fm']

                # Save to file and display
                output_file = "process_model.png"
                pm4py.save_vis_petri_net(net, im, fm, output_file)
                display(Image(filename=output_file))
                print(f"Map saved to: {output_file}")
            else:
                print("No model generated.")

            # --- B. TOP VARIANTS (Happy Paths) ---
            print("\n--- 2. Top 10 Process Variants (The 'Happy Paths') ---")
            variants = pipeline_results.get('performance', {}).get('variant_performance', {}).get('top_variants', {})
            if variants:
                # Convert dictionary to clean DataFrame for display
                var_df = pd.DataFrame.from_dict(variants, orient='index')
                var_df = var_df[['frequency', 'percentage', 'num_activities']]
                display(var_df.head(10))
            else:
                print("No variant data available.")

            # --- C. BOTTLENECKS (Where is it slow?) ---
            print("\n--- 3. Top Bottlenecks (Slowest Activities) ---")
            bottlenecks = pipeline_results.get('performance', {}).get('bottlenecks', {}).get('activity_bottlenecks', {})
            if bottlenecks:
                # Create readable DataFrame
                bot_df = pd.DataFrame.from_dict(bottlenecks, orient='index')
                bot_df = bot_df[['mean_duration', 'frequency', 'severity']].sort_values('mean_duration', ascending=False)
                display(bot_df.head(10))
            else:
                print("No significant bottlenecks found.")

            # --- D. DEVIATIONS (Why is fitness low?) ---
            print("\n--- 4. Deviation Analysis (Conformance) ---")
            # Handle nested key safely
            fitness_data = pipeline_results.get('fitness', {})
            # Depending on function version, it might be nested or direct
            fitness = fitness_data.get('log_fitness', 0) if isinstance(fitness_data, dict) else 0
            
            print(f"Overall Fitness: {fitness:.2%}")

            cases = pipeline_results.get('conformance', {}).get('case_analysis', {}).get('cases', [])
            imperfect_cases = [c for c in cases if c['fitness'] < 1.0]
            if len(cases) == 0:
                print("No case details available.")
                if fitness == 0:
                    print("Verify Configuration: 'enable_fitness' and/or 'alignments' may have been set to 'False'.")
            elif imperfect_cases:
                print(f"Found {len(imperfect_cases)} deviant cases in the sample.")
                worst_case = min(imperfect_cases, key=lambda x: x['fitness'])
                print(f"\nWorst Case Example (ID: {worst_case['case_id']}):")
                print(f" - Fitness: {worst_case['fitness']:.2%}")
                # Use .get() with default string to avoid crashing if keys are missing (common in TBR)
                print(f" - Skipped: {worst_case.get('deviations', {}).get('skipped', 'N/A with Token Replay')}")
                print(f" - Unsolicited: {worst_case.get('deviations', {}).get('unsolicited', 'N/A with Token Replay')}")
            else:
                print("All sampled cases follow the model perfectly.")

        else:
            print("Analysis failed to return results.")
