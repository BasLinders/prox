import io

import pandas as pd
import pytest

from prox.data_manager import (
    load_and_validate_csv,
    filter_event_log,
    sample_log_stratified,
    optimize_dataframe_memory,
    refine_activity_labels,
)

from conftest import make_event_log


def make_csv_bytes(df):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def make_raw_log_df():
    """Column names that only match via COLUMN_MAPPINGS aliases, not XES standard names."""
    return pd.DataFrame({
        'session_id': ['s1', 's1', 's2'],
        'user_id': ['u1', 'u1', 'u2'],
        'event_name': ['a', 'b', 'a'],
        'timestamp': ['2024-01-01 00:00:00', '2024-01-01 00:01:00', '2024-01-01 00:00:00'],
    })


# --- load_and_validate_csv ---

def test_load_and_validate_csv_success_maps_columns_and_builds_composite_key():
    df, messages, has_category = load_and_validate_csv(make_csv_bytes(make_raw_log_df()))

    assert df is not None
    assert has_category is False
    assert {'case:concept:name', 'concept:name', 'time:timestamp', 'user_id'}.issubset(df.columns)
    assert df['case:concept:name'].iloc[0] == 'u1_s1'


def test_load_and_validate_csv_missing_required_column():
    raw = pd.DataFrame({'session_id': ['s1'], 'event_name': ['a'], 'timestamp': ['2024-01-01']})
    df, messages, has_category = load_and_validate_csv(make_csv_bytes(raw))

    assert df is None
    assert any('user_id' in m for m in messages)


def test_load_and_validate_csv_rejects_oversized_file():
    df, messages, has_category = load_and_validate_csv(
        make_csv_bytes(make_raw_log_df()), max_file_size_mb=0
    )
    assert df is None
    assert any('too large' in m.lower() for m in messages)


# --- filter_event_log ---

def test_filter_event_log_remove_events(simple_event_log):
    filtered, messages = filter_event_log(
        simple_event_log, filter_type='activity', activities=['b'], mode='remove_events'
    )
    assert filtered is not None
    assert 'b' not in filtered['concept:name'].values
    assert set(filtered['concept:name'].unique()) == {'a', 'c'}


def test_filter_event_log_unknown_type(simple_event_log):
    filtered, messages = filter_event_log(simple_event_log, filter_type='not_a_real_filter')
    assert filtered is None
    assert any('unknown filter type' in m.lower() for m in messages)


def test_filter_event_log_top_variants_keeps_only_most_frequent():
    rows = []
    base = pd.Timestamp('2024-01-01')
    # 3 cases of a->b->c, 1 case of a->c (minority variant)
    for i in range(3):
        rows += [(f'm{i}', 'a', base), (f'm{i}', 'b', base), (f'm{i}', 'c', base)]
    rows += [('odd', 'a', base), ('odd', 'c', base)]
    df = make_event_log(rows)

    filtered, messages = filter_event_log(df, filter_type='top_variants', top_n=1)
    assert filtered is not None
    assert 'odd' not in filtered['case:concept:name'].unique()


def test_filter_event_log_case_duration():
    rows = [
        ('short', 'a', pd.Timestamp('2024-01-01 00:00')),
        ('short', 'b', pd.Timestamp('2024-01-01 00:05')),
        ('long', 'a', pd.Timestamp('2024-01-01 00:00')),
        ('long', 'b', pd.Timestamp('2024-01-01 05:00')),
    ]
    df = make_event_log(rows)
    filtered, messages = filter_event_log(
        df, filter_type='case_duration', min_duration=1, time_unit='hours'
    )
    assert filtered is not None
    assert set(filtered['case:concept:name'].unique()) == {'long'}


def test_filter_event_log_endpoints_keeps_matching_start():
    rows = [
        ('1', 'start_a', pd.Timestamp('2024-01-01 00:00')),
        ('1', 'end', pd.Timestamp('2024-01-01 00:01')),
        ('2', 'start_b', pd.Timestamp('2024-01-01 00:00')),
        ('2', 'end', pd.Timestamp('2024-01-01 00:01')),
    ]
    df = make_event_log(rows)
    filtered, messages = filter_event_log(df, filter_type='endpoints', start_activities=['start_a'])
    assert filtered is not None
    assert set(filtered['case:concept:name'].unique()) == {'1'}


def test_filter_event_log_attribute_missing_column_returns_error(simple_event_log):
    filtered, messages = filter_event_log(
        simple_event_log, filter_type='attribute', attribute_col='does_not_exist', attribute_values=['x']
    )
    assert filtered is None
    assert any('not found' in m.lower() for m in messages)


def test_filter_event_log_unknown_type_lists_valid_options(simple_event_log):
    filtered, messages = filter_event_log(simple_event_log, filter_type='bogus')
    assert filtered is None
    joined = ' '.join(messages).lower()
    assert 'activity' in joined and 'top_variants' in joined


def test_filter_event_log_empty_input_returns_error():
    filtered, messages = filter_event_log(pd.DataFrame(), filter_type='top_variants', top_n=1)
    assert filtered is None
    assert any('empty' in m.lower() for m in messages)


# --- sample_log_stratified ---

def test_sample_log_stratified_preserves_all_priority_cases():
    rows = []
    for i in range(10):
        rows.append((f'n{i}', 'a', pd.Timestamp('2024-01-01'), 0))
    for i in range(2):
        rows.append((f'p{i}', 'a', pd.Timestamp('2024-01-01'), 1))
    df = pd.DataFrame(rows, columns=['case:concept:name', 'concept:name', 'time:timestamp', 'flag'])

    sampled, messages = sample_log_stratified(
        df, strata_col='flag', priority_value=1, total_sample_size=3, max_priority_ratio=1.0
    )
    sampled_cases = set(sampled['case:concept:name'].unique())
    assert {'p0', 'p1'}.issubset(sampled_cases)
    assert len(sampled_cases) == 3


def test_sample_log_stratified_falls_back_to_random_when_column_missing():
    df = make_event_log([('1', 'a', pd.Timestamp('2024-01-01')), ('2', 'a', pd.Timestamp('2024-01-01'))])
    sampled, messages = sample_log_stratified(df, strata_col='missing_col', total_sample_size=1)
    assert len(sampled['case:concept:name'].unique()) == 1
    assert any('not found' in m.lower() for m in messages)


# --- optimize_dataframe_memory ---

def test_optimize_dataframe_memory_downcasts_low_cardinality_only():
    df = pd.DataFrame({
        'low_cardinality': ['x'] * 80 + ['y'] * 20,
        'high_cardinality': [str(i) for i in range(100)],
    })
    optimize_dataframe_memory(df)
    assert str(df['low_cardinality'].dtype) == 'category'
    assert str(df['high_cardinality'].dtype) != 'category'


# --- refine_activity_labels ---

def test_refine_activity_labels_appends_context():
    df = pd.DataFrame({
        'concept:name': ['page_view', 'page_view', 'click'],
        'page_type': ['checkout', 'home', None],
    })
    result = refine_activity_labels(df.copy(), target_activity='page_view', context_column='page_type')
    assert result.loc[0, 'concept:name'] == 'page_view_CHECKOUT'
    assert result.loc[1, 'concept:name'] == 'page_view_HOME'
    assert result.loc[2, 'concept:name'] == 'click'


def test_refine_activity_labels_missing_context_column_is_noop():
    df = pd.DataFrame({'concept:name': ['page_view']})
    result = refine_activity_labels(df.copy(), target_activity='page_view', context_column='does_not_exist')
    assert result.loc[0, 'concept:name'] == 'page_view'
