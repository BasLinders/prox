import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from .config import COLUMN_MAPPINGS

logger = logging.getLogger(__name__)


def load_and_validate_csv(
    uploaded_file,
    max_file_size_mb: int = 500,
    chunk_threshold_mb: int = 50,
    chunk_size: int = 50000
) -> Tuple[pd.DataFrame | None, list, bool]:
    """
    Loads, validates, and cleans an event log CSV into a PM4Py-compatible DataFrame.

    Handles file size checks, auto-mapping of column names to XES standards,
    composite case key creation (user_id + session_id), and timestamp parsing.

    Returns
    -------
    df : pd.DataFrame or None
    messages : list of str
    has_category : bool
    """
    errors = []
    notes = []

    try:
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0, 2)
            file_size_mb = uploaded_file.tell() / (1024 * 1024)
            uploaded_file.seek(0)
        else:
            file_size_mb = 0

        if file_size_mb > max_file_size_mb:
            errors.append(f"File too large ({file_size_mb:.2f} MB). Max allowed: {max_file_size_mb} MB.")
            return None, errors, False

        if file_size_mb > chunk_threshold_mb:
            notes.append(f"Large file ({file_size_mb:.1f} MB). Using chunked loading.")
            try:
                df = _load_csv_chunked(uploaded_file, chunk_size=chunk_size)
            except Exception as e:
                errors.append(f"Chunk loading failed: {e}")
                return None, errors, False
        else:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip().str.lower()

        if 'item_category' in df.columns:
            df['item_category'] = df['item_category'].fillna('General')
        if 'item_list_name' in df.columns:
            df['item_list_name'] = df['item_list_name'].fillna('General')

    except Exception as e:
        errors.append(f"Error loading file: {e}")
        return None, errors, False

    if df.empty:
        errors.append("The uploaded file is empty.")
        return None, errors, False

    # --- Column Mapping ---
    df_columns = set(df.columns)
    rename_mapping = {}
    mapped_standards = set()

    for standard_name, keywords in COLUMN_MAPPINGS.items():
        if standard_name in mapped_standards:
            continue
        exact_matches = keywords & df_columns
        if exact_matches:
            original_col = next(iter(exact_matches))
            if original_col not in rename_mapping:
                rename_mapping[original_col] = standard_name
                mapped_standards.add(standard_name)
                if len(exact_matches) > 1:
                    notes.append(f"Multiple columns matched '{standard_name}'. Using '{original_col}'.")

    for standard_name, keywords in COLUMN_MAPPINGS.items():
        if standard_name not in mapped_standards:
            for col_name in df_columns:
                if col_name not in rename_mapping:
                    if any(keyword in col_name for keyword in keywords):
                        rename_mapping[col_name] = standard_name
                        mapped_standards.add(standard_name)
                        notes.append(f"Auto-mapped '{col_name}' -> '{standard_name}'")
                        break

    if rename_mapping:
        df.rename(columns=rename_mapping, inplace=True)

    # --- Critical Validation ---
    CRITICAL_COLS = {'case:concept:name', 'concept:name', 'time:timestamp', 'user_id'}
    missing_cols = CRITICAL_COLS - set(df.columns)

    if missing_cols:
        details = []
        for col in missing_cols:
            keywords = COLUMN_MAPPINGS.get(col, {col})
            details.append(f"'{col}' (look for: {', '.join(list(keywords)[:3])})")
        errors.append(f"Missing required columns: {', '.join(details)}.")
        return None, errors + notes, False

    # --- Cleaning & Key Creation ---
    initial_rows = len(df)
    subset_cols = [c for c in ['case:concept:name', 'user_id', 'concept:name'] if c in df.columns]
    df.dropna(subset=subset_cols, inplace=True)

    dropped = initial_rows - len(df)
    if dropped > 0:
        notes.append(f"Removed {dropped} rows with missing ID or Activity.")

    if df.empty:
        errors.append("Critical Error: All rows contained missing IDs. Check data quality.")
        return None, errors + notes, False

    try:
        df['user_id'] = df['user_id'].astype(str)
        df['case:concept:name'] = df['case:concept:name'].astype(str)
        df['case:concept:name'] = df['user_id'] + '_' + df['case:concept:name']
        notes.append("Composite Case ID created (user_id + session_id).")
    except Exception as e:
        errors.append(f"Error creating composite key: {e}")
        return None, errors + notes, False

    try:
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce')
        invalid_count = df['time:timestamp'].isna().sum()
        if invalid_count > 0:
            df = df[df['time:timestamp'].notna()]
            notes.append(f"Removed {invalid_count} rows with invalid timestamps.")
    except Exception as e:
        errors.append(f"Critical error parsing timestamps: {e}")
        return None, errors + notes, False

    has_category = 'category' in df.columns
    if not has_category:
        notes.append("No 'category' column found. Category filters disabled.")
    if 'price' not in df.columns:
        notes.append("No 'price' column found. Revenue analysis disabled.")
    if 'purchase' not in df.columns and 'add_to_cart' not in df.columns:
        notes.append("E-commerce columns missing. Some metrics may be unavailable.")

    if df.empty:
        errors.append("Critical Error: No valid data remaining after cleaning.")
        return None, errors + notes, False

    logger.info("CSV loaded: %d events, %d cases.", len(df), df['case:concept:name'].nunique())
    return df, errors + notes, has_category


def _load_csv_chunked(uploaded_file, chunk_size: int = 50000) -> pd.DataFrame:
    chunks = []
    if hasattr(uploaded_file, 'seek'):
        uploaded_file.seek(0)
    with pd.read_csv(uploaded_file, chunksize=chunk_size) as reader:
        for chunk in reader:
            chunk.columns = chunk.columns.str.strip().str.lower()
            chunk.dropna(how='all', inplace=True)
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)



# pm4py.convert_to_event_log() requires case:concept:name and concept:name to be
# plain string dtype, not category — both are structurally low-cardinality
# (every case has multiple events, activities repeat by nature), so they'd
# otherwise get swept up by the category-downcasting below and break discovery.
_PM4PY_REQUIRED_STRING_COLUMNS = frozenset(['case:concept:name', 'concept:name'])


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Converts low-cardinality string columns to category dtype to reduce RAM usage.

    Uses is_string_dtype rather than a bare `== 'object'` check, since pandas 2.x+
    infers plain string columns as a dedicated 'str' dtype rather than 'object'.
    """
    for col in df.columns:
        if col in _PM4PY_REQUIRED_STRING_COLUMNS:
            continue
        if pd.api.types.is_string_dtype(df[col].dtype):
            if len(df[col].unique()) / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')
    return df


def check_trace_length(df: pd.DataFrame) -> Dict[str, Any]:
    """Returns descriptive statistics for trace lengths (events per case)."""
    case_lengths = df.groupby('case:concept:name').size()
    stats = case_lengths.describe().to_dict()
    stats['p95'] = int(case_lengths.quantile(0.95))
    stats['p99'] = int(case_lengths.quantile(0.99))
    return stats


def get_trace_signature(trace) -> tuple:
    """Returns a hashable tuple of activity names representing a trace variant."""
    return tuple(str(e['concept:name']) for e in trace)


def refine_activity_labels(
    df: pd.DataFrame,
    target_activity: str = 'page_view',
    context_column: str = None
) -> pd.DataFrame:
    """
    Refines generic activity labels by appending context from another column.

    For example, 'page_view' + page_type='CHECKOUT' becomes 'page_view_CHECKOUT'.
    URL paths are cleaned: query params stripped, only the last path segment kept.
    """
    if context_column is None or context_column not in df.columns:
        if context_column:
            logger.warning("Context column '%s' not found. Skipping label refinement.", context_column)
        return df

    mask = df['concept:name'] == target_activity
    if not mask.any():
        return df

    logger.info("Refining %d '%s' events using '%s'.", mask.sum(), target_activity, context_column)

    context_values = df.loc[mask, context_column].fillna('unknown').astype(str)

    if not context_values.empty:
        first_val = str(context_values.iloc[0])
        if 'http' in first_val or '/' in first_val:
            context_values = context_values.str.split('?').str[0]
            context_values = context_values.str.strip('/')
            context_values = context_values.apply(lambda x: x.split('/')[-1] if '/' in x else x)

    df.loc[mask, 'concept:name'] = target_activity + '_' + context_values.str.upper()
    return df


def _filter_case_duration(filtered_df, min_duration=0, max_duration=np.inf, time_unit='hours', **_ignored):
    messages = []
    divisor = {'seconds': 1, 'minutes': 60, 'hours': 3600, 'days': 86400}.get(time_unit, 3600)

    case_durations = filtered_df.groupby('case:concept:name')['time:timestamp'].agg(['min', 'max'])
    case_durations['duration'] = (
        case_durations['max'] - case_durations['min']
    ).dt.total_seconds() / divisor
    keep = case_durations[(case_durations['duration'] >= min_duration) & (case_durations['duration'] <= max_duration)].index
    filtered_df = filtered_df[filtered_df['case:concept:name'].isin(keep)]
    messages.append(f"Filtered by duration ({min_duration}–{max_duration} {time_unit}).")
    return filtered_df, messages


def _filter_activity(filtered_df, activities=None, mode='contains', **_ignored):
    messages = []
    activities = activities or []

    if not isinstance(activities, list) or not activities:
        messages.append("Error: 'activities' must be a non-empty list.")
        return None, messages

    if mode == 'contains':
        cases = filtered_df[filtered_df['concept:name'].isin(activities)]['case:concept:name'].unique()
        filtered_df = filtered_df[filtered_df['case:concept:name'].isin(cases)]
        messages.append(f"Kept cases containing activities: {activities}.")
    elif mode == 'not_contains':
        cases = filtered_df[filtered_df['concept:name'].isin(activities)]['case:concept:name'].unique()
        filtered_df = filtered_df[~filtered_df['case:concept:name'].isin(cases)]
        messages.append(f"Removed cases containing activities: {activities}.")
    elif mode == 'remove_events':
        filtered_df = filtered_df[~filtered_df['concept:name'].isin(activities)]
        messages.append(f"Removed events matching: {activities}.")
    elif mode == 'keep_events':
        filtered_df = filtered_df[filtered_df['concept:name'].isin(activities)]
        messages.append(f"Kept only events matching: {activities}.")
    else:
        messages.append(f"Error: Invalid mode '{mode}' for activity filter.")
        return None, messages

    return filtered_df, messages


def _filter_crop(filtered_df, activity=None, top_n=None, **_ignored):
    messages = []
    target_input = activity
    if not target_input:
        messages.append("Error: 'activity' must be specified for crop filter.")
        return None, messages

    targets = [target_input] if isinstance(target_input, str) else target_input
    target_hits = pd.DataFrame()
    used_target = ""

    for target in targets:
        if target in filtered_df['concept:name'].values:
            target_hits = filtered_df[filtered_df['concept:name'] == target]
            used_target = target
            messages.append(f"Crop target found as activity name '{target}'.")
            break
        elif target in filtered_df.columns:
            try:
                vals = pd.to_numeric(filtered_df[target], errors='coerce').fillna(0)
                temp_hits = filtered_df[vals > 0]
                if not temp_hits.empty:
                    target_hits = temp_hits
                    used_target = target
                    filtered_df.loc[target_hits.index, 'concept:name'] = target
                    messages.append(f"Injected activity '{target}' from column values > 0.")
                    break
            except Exception:
                pass

    if target_hits.empty:
        messages.append(f"Warning: None of {targets} found as activity or active column. Skipping crop.")
        return filtered_df, messages

    final_hits = filtered_df[filtered_df['concept:name'] == used_target]
    cutoff_times = (
        final_hits.groupby('case:concept:name')['time:timestamp']
        .min()
        .reset_index()
        .rename(columns={'time:timestamp': 'cutoff_time'})
    )
    merged = filtered_df.merge(cutoff_times, on='case:concept:name', how='inner')
    filtered_df = merged[merged['time:timestamp'] <= merged['cutoff_time']].drop(columns=['cutoff_time'])
    messages.append(f"Cropped traces at '{used_target}'. Cases without it were removed.")

    if top_n:
        variants = filtered_df.groupby('case:concept:name')['concept:name'].apply(lambda x: ' -> '.join(x))
        top_seqs = variants.value_counts().nlargest(top_n).index
        keep_cases = variants[variants.isin(top_seqs)].index
        filtered_df = filtered_df[filtered_df['case:concept:name'].isin(keep_cases)]
        messages.append(f"Kept top {top_n} most frequent variants after crop.")

    return filtered_df, messages


def _filter_endpoints(filtered_df, start_activities=None, end_activities=None, **_ignored):
    messages = []
    start_acts = start_activities or []
    end_acts = end_activities or []

    if start_acts:
        starts = filtered_df.sort_values('time:timestamp').groupby('case:concept:name').first()['concept:name']
        keep = starts[starts.isin(start_acts)].index
        filtered_df = filtered_df[filtered_df['case:concept:name'].isin(keep)]
        messages.append(f"Kept cases starting with: {start_acts}.")
    if end_acts:
        ends = filtered_df.sort_values('time:timestamp').groupby('case:concept:name').last()['concept:name']
        keep = ends[ends.isin(end_acts)].index
        filtered_df = filtered_df[filtered_df['case:concept:name'].isin(keep)]
        messages.append(f"Kept cases ending with: {end_acts}.")

    return filtered_df, messages


def _filter_attribute(filtered_df, attribute_col=None, attribute_values=None, **_ignored):
    messages = []
    attr_vals = attribute_values or []

    if not attribute_col or attribute_col not in filtered_df.columns:
        messages.append(f"Error: Attribute column '{attribute_col}' not found.")
        return None, messages
    if not isinstance(attr_vals, list) or not attr_vals:
        messages.append("Error: 'attribute_values' must be a non-empty list.")
        return None, messages

    filtered_df = filtered_df[filtered_df[attribute_col].isin(attr_vals)]
    messages.append(f"Kept events where '{attribute_col}' in {attr_vals}.")
    return filtered_df, messages


def _filter_top_variants(filtered_df, top_n=10, **_ignored):
    messages = []
    variants = filtered_df.groupby('case:concept:name')['concept:name'].apply(lambda x: ' -> '.join(x))
    top_seqs = variants.value_counts().nlargest(top_n).index
    keep_cases = variants[variants.isin(top_seqs)].index
    filtered_df = filtered_df[filtered_df['case:concept:name'].isin(keep_cases)]
    messages.append(f"Kept top {top_n} most frequent variants.")
    return filtered_df, messages


# Single source of truth for available filter types: filter_event_log dispatches
# on this dict, and pipeline.py validates filter_steps config against it upfront,
# so a new filter type only needs an entry here.
FILTER_HANDLERS = {
    'activity': _filter_activity,
    'case_duration': _filter_case_duration,
    'crop': _filter_crop,
    'endpoints': _filter_endpoints,
    'attribute': _filter_attribute,
    'top_variants': _filter_top_variants,
}


def filter_event_log(
    event_log_df: pd.DataFrame,
    filter_type: str,
    **kwargs
) -> Tuple[pd.DataFrame | None, list]:
    """
    Applies a named filter to an event log DataFrame. See FILTER_HANDLERS for
    available filter_type keys.

    filter_type options
    -------------------
    'activity'      : keep/remove cases or events by activity name
                      kwargs: activities (list), mode ('contains'|'not_contains'|'remove_events'|'keep_events')
    'case_duration' : filter cases by throughput time
                      kwargs: min_duration, max_duration, time_unit
    'crop'          : trim traces to stop at a target activity or flag column
                      kwargs: activity (str|list), top_n (int, optional)
    'endpoints'     : filter by first/last activity
                      kwargs: start_activities (list), end_activities (list)
    'attribute'     : keep events where a column matches given values
                      kwargs: attribute_col (str), attribute_values (list)
    'top_variants'  : retain only the top N most frequent variants
                      kwargs: top_n (int)
    """
    messages = []

    if event_log_df is None or event_log_df.empty:
        messages.append("Error: Input event log is empty.")
        return None, messages

    filtered_df = event_log_df.copy()
    original_events = len(filtered_df)
    original_cases = filtered_df['case:concept:name'].nunique()

    handler = FILTER_HANDLERS.get(filter_type)
    if handler is None:
        valid = ', '.join(FILTER_HANDLERS)
        messages.append(f"Error: Unknown filter type '{filter_type}'. Valid options: {valid}.")
        return None, messages

    try:
        filtered_df, handler_messages = handler(filtered_df, **kwargs)
        messages.extend(handler_messages)

        if filtered_df is None:
            return None, messages

        final_events = len(filtered_df)
        final_cases = filtered_df['case:concept:name'].nunique()

        if final_events == 0:
            messages.append("Warning: Filter produced an empty event log.")
        else:
            messages.append(
                f"Filter complete: {original_cases} cases → {final_cases} cases "
                f"({original_events} → {final_events} events)."
            )

        return filtered_df.reset_index(drop=True), messages

    except Exception as e:
        import traceback
        messages.append(f"Unexpected error during filtering: {e}")
        messages.append(traceback.format_exc())
        return None, messages


def sample_log_stratified(
    event_log_df: pd.DataFrame,
    strata_col: str,
    priority_value=1,
    total_sample_size: int = 500,
    max_priority_ratio: float = 0.5
) -> Tuple[pd.DataFrame, list]:
    """
    Stratified case-level sampling that preserves rare high-priority events (e.g. purchases).

    Selects up to total_sample_size * max_priority_ratio priority cases, then fills
    remaining slots with random standard cases. Falls back to random sampling if
    strata_col is missing or raises an error.
    """
    messages = []

    if strata_col and strata_col in event_log_df.columns:
        try:
            case_strata = event_log_df.groupby('case:concept:name')[strata_col].max()
            priority_cases = case_strata[case_strata == priority_value].index
            other_cases = case_strata[case_strata != priority_value].index

            max_priority = int(total_sample_size * max_priority_ratio)
            n_priority = min(len(priority_cases), max_priority)
            n_other = min(total_sample_size - n_priority, len(other_cases))

            priority_ids = (
                priority_cases.to_series().sample(n_priority, replace=False)
                if n_priority > 0 else pd.Series([], dtype=object)
            )
            other_ids = (
                other_cases.to_series().sample(n_other, replace=False)
                if n_other > 0 else pd.Series([], dtype=object)
            )

            final_ids = pd.concat([priority_ids, other_ids]).tolist()
            sampled_df = event_log_df[event_log_df['case:concept:name'].isin(final_ids)].copy()

            messages.append(f"Stratified sample: {len(final_ids)} cases total.")
            messages.append(f"  {len(priority_ids)} priority ('{strata_col}'={priority_value}), {len(other_ids)} other.")
            return sampled_df, messages

        except Exception as e:
            messages.append(f"Stratified sampling failed: {e}. Falling back to random sample.")
    else:
        if strata_col:
            messages.append(f"Column '{strata_col}' not found. Using random sample.")

    all_ids = event_log_df['case:concept:name'].unique()
    n = min(total_sample_size, len(all_ids))
    if len(all_ids) == 0:
        return event_log_df.copy(), messages

    sampled_ids = pd.Series(all_ids).sample(n, replace=False).tolist()
    messages.append(f"Random sample: {len(sampled_ids)} cases.")
    return event_log_df[event_log_df['case:concept:name'].isin(sampled_ids)].copy(), messages
