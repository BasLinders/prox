import pandas as pd
import pm4py
import os
from typing import Dict, Any, Union, List
from config import COLUMN_MAPPINGS

def load_and_validate_csv(uploaded_file, max_file_size_mb=500, chunk_size=50000):
    """
    Main function: Loads, validates, and cleans the event log.
    Includes memory protection and strict logic ordering.
    """
    errors = []
    notes = []

    # --- File Size Check & Loading Strategy ---
    try:
        # 1. Check file size
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0, 2) # Move to end
            file_size_mb = uploaded_file.tell() / (1024 * 1024)
            uploaded_file.seek(0) # Reset to start
        else:
            file_size_mb = 0 # Cannot determine, assume small

        # 2. Determine Load Strategy
        CHUNK_THRESHOLD_MB = 50

        if file_size_mb > max_file_size_mb:
            errors.append(f"File too large ({file_size_mb:.2f} MB). Max allowed is {max_file_size_mb} MB.")
            return None, errors, False

        if file_size_mb > CHUNK_THRESHOLD_MB:
            notes.append(f"Large file detected ({file_size_mb:.1f} MB). Using optimized chunked loading.")
            try:
                df = _load_csv_chunked(uploaded_file, chunk_size=chunk_size)
            except Exception as e:
                errors.append(f"Chunk loading failed: {str(e)}")
                return None, errors, False
        else:
            # Standard load for smaller files (Faster overhead)
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip().str.lower()

        # Remove NaN values from category events
        if 'item_category' in df.columns:
            df['item_category'] = df['item_category'].fillna('General')
        if 'item_list_name' in df.columns:
            df['item_list_name'] = df['item_list_name'].fillna('General')
            
    except Exception as e:
        errors.append(f"Error loading file: {str(e)}. Please ensure it is a valid CSV.")
        return None, errors, False

    if df.empty:
        errors.append("The uploaded file is empty.")
        return None, errors, False

    # --- Deel 2: Column Mapping Handling ---
    df_columns = set(df.columns)
    rename_mapping = {}
    mapped_standards = set()

    # Priority 1: Exact Matches
    for standard_name, keywords in COLUMN_MAPPINGS.items():
        if standard_name in mapped_standards: continue
        exact_matches = keywords & df_columns
        if exact_matches:
            original_col = next(iter(exact_matches))
            if original_col not in rename_mapping:
                rename_mapping[original_col] = standard_name
                mapped_standards.add(standard_name)
                # If multiple matches found, warn user
                if len(exact_matches) > 1:
                    notes.append(f"Note: Multiple columns found for '{standard_name}'. Using '{original_col}'.")

    # Priority 2: Partial Matches
    for standard_name, keywords in COLUMN_MAPPINGS.items():
        if standard_name not in mapped_standards:
            for col_name in df_columns:
                if col_name not in rename_mapping:
                    if any(keyword in col_name for keyword in keywords):
                        rename_mapping[col_name] = standard_name
                        mapped_standards.add(standard_name)
                        notes.append(f"Note: Auto-mapped '{col_name}' -> '{standard_name}'")
                        break

    if rename_mapping:
        df.rename(columns=rename_mapping, inplace=True)

    # --- Deel 3: Critical Validation ---
    # We check for user_id because we need it for the composite key
    CRITICAL_COLS = {'case:concept:name', 'concept:name', 'time:timestamp', 'user_id'}
    missing_cols = CRITICAL_COLS - set(df.columns)

    if missing_cols:
        # Generate helpful error message
        details = []
        for col in missing_cols:
            keywords = COLUMN_MAPPINGS.get(col, {col})
            details.append(f"'{col}' (look for: {', '.join(list(keywords)[:3])})")

        errors.append(f"Missing required columns: {', '.join(details)}.")
        return None, errors + notes, False

    # --- Deel 4: Cleaning & Key Creation (ORDER IS CRITICAL) ---
    # 4.1 Drop NaNs in ID/Activity columns BEFORE creating keys
    initial_rows = len(df)

    # Create a subset list that actually exists in the dataframe
    subset_cols = [c for c in ['case:concept:name', 'user_id', 'concept:name'] if c in df.columns]
    df.dropna(subset=subset_cols, inplace=True)

    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        notes.append(f"Info: Removed {dropped_rows} rows due to missing ID or Activity.")

    if df.empty:
        errors.append("Critical Error: All rows contained missing IDs. Check data quality.")
        return None, errors + notes, False

    # 4.2 Create Composite Key
    try:
        df['user_id'] = df['user_id'].astype(str)
        df['case:concept:name'] = df['case:concept:name'].astype(str)

        # Overwrite case ID with composite key
        df['case:concept:name'] = df['user_id'] + '_' + df['case:concept:name']
        notes.append("Info: Composite Case ID created (User_ID + Session_ID).")
    except Exception as e:
        errors.append(f"Error creating composite key: {e}")
        return None, errors + notes, False

    # 4.3 Timestamp Parsing
    try:
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce')

        # Remove NaT rows
        invalid_mask = df['time:timestamp'].isna()
        invalid_count = invalid_mask.sum()

        if invalid_count > 0:
            df = df[~invalid_mask]
            notes.append(f"Warning: Removed {invalid_count} rows with invalid/unparseable timestamps.")

    except Exception as e:
        errors.append(f"Critical error parsing timestamps: {e}")
        return None, errors + notes, False

    # --- Deel 5: Optional Checks ---
    has_category = 'category' in df.columns
    if not has_category:
        notes.append("Note: No 'category' column. Category filters disabled.")

    if 'price' not in df.columns:
        notes.append("Note: No 'price' column. Revenue analysis disabled.")

    if 'purchase' not in df.columns and 'add_to_cart' not in df.columns:
        notes.append("Note: Ecommerce columns missing. Some specific metrics may be empty.")

    # Final Safety Check
    if df.empty:
        errors.append("Critical Error: No valid data remaining after cleaning.")
        return None, errors + notes, False

    return df, errors + notes, has_category

def _load_csv_chunked(uploaded_file, chunk_size=50000):
    """
    Loads large CSV files efficiently using chunking.
    """
    chunks = []

    try:
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)

        with pd.read_csv(uploaded_file, chunksize=chunk_size) as reader:
            for i, chunk in enumerate(reader):
                chunk.columns = chunk.columns.str.strip().str.lower()
                chunk.dropna(how='all', inplace=True)
                chunks.append(chunk)

        if not chunks:
            return pd.DataFrame()

        return pd.concat(chunks, ignore_index=True)

    except Exception as e:
        raise Exception(f"Error during chunked processing: {e}")

def optimize_dataframe_memory(df: pd.DataFrame):
    """
    Optimizes memory usage by converting object columns to categories.
    CRITICAL for 8GB RAM machines.
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            num_unique = len(df[col].unique())
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')
    return df
    
def check_trace_length(df_clean):
    case_lengths = df_clean.groupby('case:concept:name').size()
    
    print("--- Trace Length Statistics ---")
    print(case_lengths.describe())
    
    # See the 95th and 99th percentile
    print(f"\n95% of cases have fewer than {int(case_lengths.quantile(0.95))} events.")
    print(f"99% of cases have fewer than {int(case_lengths.quantile(0.99))} events.")
    
def get_trace_signature(trace):
     return tuple(str(e['concept:name']) for e in trace)

def refine_activity_labels(df: pd.DataFrame, target_activity='page_view', context_column=None):
    """
    Renames a generic activity (e.g., 'page_view') by appending context from another column
    (e.g., 'page_title' or 'page_path').

    Example: 'page_view' + 'checkout' -> 'page_view_CHECKOUT'
    """
    if context_column is not None and context_column not in df.columns:
        print(f"Warning: Context column '{context_column}' not found. Skipping refinement.")
        return df

    # Identify rows that match the target activity
    mask = df['concept:name'] == target_activity

    if not mask.any():
        return df

    print(f"Refining {mask.sum()} '{target_activity}' events using '{context_column}'...")

    # 1. Extract context values
    context_values = df.loc[mask, context_column].fillna('unknown').astype(str)

    # 2. CLEANING
    # Remove query parameters (everything after ?) to prevent "search?q=red"
    # and "search?q=blue" from becoming two different process steps.
    if not context_values.empty:
        # Check first element to see if it looks like a URL/Path
        first_val = str(context_values.iloc[0])
        if 'http' in first_val or '/' in first_val:
            context_values = context_values.str.split('?').str[0]  # Strip query params
            context_values = context_values.str.strip('/')          # Remove trailing slashes
            # Take last path segment
            context_values = context_values.apply(lambda x: x.split('/')[-1] if '/' in x else x)

    # 3. Apply the new name
    # Format: "page_view_CHECKOUT"
    df.loc[mask, 'concept:name'] = target_activity + "_" + context_values.str.upper()

    return df

def filter_event_log(
    event_log_df: pd.DataFrame,
    filter_type: str,
    **kwargs
) -> Tuple[pd.DataFrame | None, list]:
    """
    Purpose:
        Provides various filtering options for an event log DataFrame. This is crucial for
        drilling down into specific user segments, paths, or performance brackets.

    Args:
        event_log_df (pd.DataFrame): The input event log.
        filter_type (str): The type of filter to apply. Supported values:
            - 'case_duration': Filters cases based on their total duration.
            - 'activity': Filters cases that contain or do not contain specific activities.
            - 'attribute': Filters events based on the value of a specific attribute column.
            - 'top_variants': Keeps only the cases belonging to the most frequent variants.

    Keyword Args (**kwargs):
        For 'case_duration':
            - min_duration (float): The minimum duration to keep.
            - max_duration (float): The maximum duration to keep.
            - time_unit (str): 'seconds', 'minutes', 'hours', or 'days'.

        For 'activity':
            - activities (List[str]): A list of activity names to filter by.
            - mode (str): 'contains' (default) to keep cases with any of the activities,
                          or 'not_contains' to remove them.

        For 'attribute':
            - attribute_col (str): The name of the column to filter on.
            - attribute_values (List[Any]): A list of values to keep.

        For 'top_variants':
            - top_n (int): The number of most frequent variants to keep (e.g., 10).

    Returns:
        A tuple containing the filtered event log DataFrame and a list of info/error messages.
        Returns (None, errors) if a critical error occurs.
    """
    messages = []

    if event_log_df is None or event_log_df.empty:
        messages.append("Error: Input event log is empty.")
        return None, messages

    # Work on a copy to avoid modifying the original DataFrame
    filtered_df = event_log_df.copy()
    original_event_count = len(filtered_df)
    original_case_count = filtered_df['case:concept:name'].nunique()

    try:
        if filter_type == 'case_duration':
            # --- Filter by the total duration of a case ---
            min_d = kwargs.get('min_duration', 0)
            max_d = kwargs.get('max_duration', np.inf)
            time_unit = kwargs.get('time_unit', 'hours')

            time_conversions = {'seconds': 1, 'minutes': 60, 'hours': 3600, 'days': 86400}
            divisor = time_conversions.get(time_unit, 3600)

            case_durations = filtered_df.groupby('case:concept:name')['time:timestamp'].agg(['min', 'max'])
            case_durations['duration'] = (case_durations['max'] - case_durations['min']).dt.total_seconds() / divisor

            cases_to_keep = case_durations[
                (case_durations['duration'] >= min_d) & (case_durations['duration'] <= max_d)
            ].index

            filtered_df = filtered_df[filtered_df['case:concept:name'].isin(cases_to_keep)]
            messages.append(f"Filtered cases by duration (min: {min_d}, max: {max_d} {time_unit}).")

        elif filter_type == 'activity':
            # --- Filter cases based on the activities they contain ---
            activities = kwargs.get('activities', [])
            mode = kwargs.get('mode', 'contains')

            if not isinstance(activities, list) or not activities:
                messages.append("Error: 'activities' must be a non-empty list.")
                return None, messages

            cases_with_activities = filtered_df[filtered_df['concept:name'].isin(activities)]['case:concept:name'].unique()

            if mode == 'contains':
                filtered_df = filtered_df[filtered_df['case:concept:name'].isin(cases_with_activities)]
                messages.append(f"Kept cases containing activities: {activities}.")
            elif mode == 'not_contains':
                filtered_df = filtered_df[~filtered_df['case:concept:name'].isin(cases_with_activities)]
                messages.append(f"Removed cases containing activities: {activities}.")
            else:
                messages.append(f"Error: Invalid mode '{mode}' for activity filter.")
                return None, messages

        elif filter_type == 'attribute':
            # --- Filter events directly by an attribute's value ---
            attr_col = kwargs.get('attribute_col')
            attr_vals = kwargs.get('attribute_values', [])

            if not attr_col or attr_col not in filtered_df.columns:
                messages.append(f"Error: Attribute column '{attr_col}' not found.")
                return None, messages

            if not isinstance(attr_vals, list) or not attr_vals:
                messages.append("Error: 'attribute_values' must be a non-empty list.")
                return None, messages

            filtered_df = filtered_df[filtered_df[attr_col].isin(attr_vals)]
            messages.append(f"Filtered events where '{attr_col}' is in {attr_vals}.")

        elif filter_type == 'top_variants':
            # --- Keep only the cases belonging to the top N most frequent variants ---
            top_n = kwargs.get('top_n', 10)

            variants = filtered_df.groupby('case:concept:name')['concept:name'].apply(lambda x: ' -> '.join(x))
            top_variant_sequences = variants.value_counts().nlargest(top_n).index

            cases_to_keep = variants[variants.isin(top_variant_sequences)].index

            filtered_df = filtered_df[filtered_df['case:concept:name'].isin(cases_to_keep)]
            messages.append(f"Filtered log to keep the top {top_n} most frequent variants.")

        else:
            messages.append(f"Error: Unknown filter type '{filter_type}'.")
            return None, messages

        # --- Final Summary ---
        final_event_count = len(filtered_df)
        final_case_count = filtered_df['case:concept:name'].nunique()

        if final_event_count == 0:
            messages.append("Warning: The current filter combination resulted in an empty event log.")
        else:
            messages.append(f"Filtering complete. Log reduced from {original_case_count} cases ({original_event_count} events) "
                            f"to {final_case_count} cases ({final_event_count} events).")

        return filtered_df.reset_index(drop=True), messages

    except Exception as e:
        messages.append(f"An unexpected error occurred during filtering: {e}")
        import traceback
        messages.append(f"Traceback: {traceback.format_exc()}")
        return None, messages

def sample_log_stratified(
    event_log_df,
    strata_col,
    priority_value=1,
    total_sample_size=500,
    max_priority_ratio=0.5
):
    """
    Executes a stratified sample check on case-level.
    
    This is designed to *guarantee* rare, important cases (such as 'purchase == 1')
    in the sample, while the rest are randomly filled to the total sample size.
    
    Args:
    - event_log_df: The complete, prepared event log DataFrame.
    - strata_col: The column name to be stratified by (e.g., 'purchase').
    - priority_value: The value in the strata_col that has priority (e.g., 1).
    - total_sample_size: The *desired* total number of cases in the sample.
    - max_priority_ratio: The maximum proportion of 'priority' cases
      in the final sample (e.g., 0.5 = 50%).
    
    Returns:
    A DataFrame containing the sample and a list of messages.
    """
    messages = []
    
    # 1. Try Stratified Sampling if column exists
    if strata_col and strata_col in event_log_df.columns:
        try:
            # --- Main logic: Stratified sampling ---
            
            # Determine the strata for every case
            # If a case ever had the priority_value (ex. purchase=1),
            # Then the whole case gets that value
            case_strata = event_log_df.groupby('case:concept:name')[strata_col].max()
            
            priority_cases = case_strata[case_strata == priority_value].index
            other_cases = case_strata[case_strata != priority_value].index
            
            num_priority_cases = len(priority_cases)
            num_other_cases = len(other_cases)
            
            # Determine how many priority cases we take into account
            # We take 'em all, up until a maximum of (ex. 50% * 500) = 250
            max_priority_sample = int(total_sample_size * max_priority_ratio)
            priority_sample_size = min(num_priority_cases, max_priority_sample)
            
            # Determine how many other cases we take to fill the sample with
            other_sample_size = total_sample_size - priority_sample_size
            # Make sure that we don't ask for more than available
            other_sample_size = min(other_sample_size, num_other_cases)
            
            # Take the samples (with replace=False to prevent duplicates)
            # Handle edge case where sample size is 0
            if priority_sample_size > 0:
                priority_case_sample_ids = priority_cases.to_series().sample(priority_sample_size, replace=False)
            else:
                priority_case_sample_ids = pd.Series([], dtype=object)

            if other_sample_size > 0:
                other_case_sample_ids = other_cases.to_series().sample(other_sample_size, replace=False)
            else:
                other_case_sample_ids = pd.Series([], dtype=object)
            
            # Combine the IDs
            final_case_ids = pd.concat([priority_case_sample_ids, other_case_sample_ids]).tolist()
            
            # Filter the original DataFrame
            sampled_df = event_log_df[event_log_df['case:concept:name'].isin(final_case_ids)].copy()
            
            messages.append(f"Info: Stratified sample check executed (Total: {len(final_case_ids)} cases).")
            messages.append(f"   -> {len(priority_case_sample_ids)} '{strata_col}={priority_value}' cases (priority).")
            messages.append(f"   -> {len(other_case_sample_ids)} other cases.")
            
            return sampled_df, messages
            
        except Exception as e:
            messages.append(f"Error during stratified sampling: {e}. Executing normal random sample check.")
    else:
        if strata_col:
             messages.append(f"Warning: Stratification column '{strata_col}' not found. Executing random sample check.")
        else:
             messages.append("Info: No stratification column provided. Executing random sample check.")

    # 2. Fallback: Random Sampling
    all_case_ids = event_log_df['case:concept:name'].unique()
    sample_size = min(total_sample_size, len(all_case_ids))
    
    if len(all_case_ids) == 0:
        messages.append("Info: Log is empty, no sample taken.")
        return event_log_df.copy(), messages
    
    sampled_ids = pd.Series(all_case_ids).sample(sample_size, replace=False).tolist()
    messages.append(f"Info: Normal random sample check executed with ({len(sampled_ids)} cases).")
    
    return event_log_df[event_log_df['case:concept:name'].isin(sampled_ids)].copy(), messages
