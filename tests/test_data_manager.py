import io

import pandas as pd
import pytest

from prox.data_manager import (
    load_and_validate_csv,
    filter_event_log,
    sample_log_stratified,
    optimize_dataframe_memory,
    refine_activity_labels,
    check_data_quality,
    winsorize_series,
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

def test_load_and_validate_csv_success_maps_columns_and_defaults_case_to_user():
    df, messages, has_category = load_and_validate_csv(make_csv_bytes(make_raw_log_df()))

    assert df is not None
    assert has_category is False
    assert {'case:concept:name', 'concept:name', 'time:timestamp', 'user_id', 'session_id'}.issubset(df.columns)
    # Default case_grouping="user": case:concept:name is the bare user_id, and
    # both of u1's rows (from session s1) collapse into one case.
    assert df['case:concept:name'].iloc[0] == 'u1'
    assert set(df['case:concept:name']) == {'u1', 'u2'}
    # session_id is always kept, composite-safe, regardless of case_grouping.
    assert df['session_id'].iloc[0] == 'u1_s1'


def test_load_and_validate_csv_case_grouping_session_builds_composite_key():
    df, messages, has_category = load_and_validate_csv(
        make_csv_bytes(make_raw_log_df()), case_grouping='session'
    )

    assert df is not None
    assert df['case:concept:name'].iloc[0] == 'u1_s1'
    assert df['case:concept:name'].iloc[0] == df['session_id'].iloc[0]


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


def test_load_and_validate_csv_does_not_rename_resource_column_to_user_id():
    """Regression test: COLUMN_MAPPINGS used to alias 'resource'/'org:resource'
    (the XES standard name for who performed a step) into 'user_id' (whose
    case this is) - two different concepts. That made analyze_process_
    performance()'s own resource-column detection unreachable, since the
    column it looks for had already been renamed away by the time it ran."""
    raw = make_raw_log_df()
    raw['resource'] = ['agent_A', 'agent_A', 'agent_B']
    df, messages, has_category = load_and_validate_csv(make_csv_bytes(raw))

    assert df is not None
    assert 'resource' in df.columns
    assert df['user_id'].tolist() == ['u1', 'u1', 'u2']  # unaffected by the resource column


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


def test_filter_event_log_top_variants_ignores_row_order_uses_timestamp():
    """Regression: variant strings must be built from time:timestamp order,
    not incoming DataFrame row order. A BigQuery/CSV source that comes back
    with rows out of chronological order per case must not corrupt which
    variant is 'most frequent'.

    3 cases share the true chronological variant a -> b -> c (t0, t1, t2)
    but are inserted into the DataFrame in different row orders; 1 case is
    a genuinely different, minority variant. Without re-sorting by
    time:timestamp before joining, the 3 majority cases would produce 3
    different-looking strings (one per row-order permutation) instead of
    one shared string, and top_n=1 would keep an arbitrary single case
    instead of all 3 true majority cases.
    """
    t0 = pd.Timestamp('2024-01-01 00:00')
    t1 = pd.Timestamp('2024-01-01 00:01')
    t2 = pd.Timestamp('2024-01-01 00:02')
    rows = [
        ('m0', 'a', t0), ('m0', 'b', t1), ('m0', 'c', t2),  # in order
        ('m1', 'b', t1), ('m1', 'a', t0), ('m1', 'c', t2),  # shuffled
        ('m2', 'c', t2), ('m2', 'b', t1), ('m2', 'a', t0),  # reversed
        ('odd', 'x', t0), ('odd', 'c', t1),                 # true minority variant
    ]
    df = make_event_log(rows)

    filtered, messages = filter_event_log(df, filter_type='top_variants', top_n=1)

    assert filtered is not None
    assert set(filtered['case:concept:name'].unique()) == {'m0', 'm1', 'm2'}


def test_filter_event_log_crop_top_n_ignores_row_order_uses_timestamp():
    """Same regression as top_variants, for the crop filter's top_n step."""
    t0 = pd.Timestamp('2024-01-01 00:00')
    t1 = pd.Timestamp('2024-01-01 00:01')
    t2 = pd.Timestamp('2024-01-01 00:02')
    rows = [
        ('m0', 'a', t0), ('m0', 'b', t1), ('m0', 'checkout', t2),
        ('m1', 'b', t1), ('m1', 'a', t0), ('m1', 'checkout', t2),
        ('m2', 'checkout', t2), ('m2', 'b', t1), ('m2', 'a', t0),
        ('odd', 'x', t0), ('odd', 'checkout', t1),
    ]
    df = make_event_log(rows)

    filtered, messages = filter_event_log(df, filter_type='crop', activity='checkout', top_n=1)

    assert filtered is not None
    assert set(filtered['case:concept:name'].unique()) == {'m0', 'm1', 'm2'}


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


def test_optimize_dataframe_memory_never_categorizes_pm4py_required_columns():
    """case:concept:name and concept:name must stay string dtype - pm4py.convert_to_event_log()
    rejects category dtype, and both columns are structurally low-cardinality (every case has
    multiple events, activities repeat), so they'd otherwise always get swept into category."""
    df = pd.DataFrame({
        'case:concept:name': ['1', '1', '2', '2'],
        'concept:name': ['a', 'b', 'a', 'b'],
    })
    optimize_dataframe_memory(df)
    assert str(df['case:concept:name'].dtype) != 'category'
    assert str(df['concept:name'].dtype) != 'category'


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


# --- check_data_quality ---

def test_check_data_quality_clean_log_has_no_issues(simple_event_log):
    result = check_data_quality(simple_event_log)
    assert result['issues'] == []
    assert result['duplicate_events'] == 0
    assert result['single_event_cases'] == 0
    assert result['out_of_order_events'] == 0


def test_check_data_quality_detects_duplicate_events():
    df = make_event_log([
        ('c1', 'a', pd.Timestamp('2024-01-01 00:00:00')),
        ('c1', 'a', pd.Timestamp('2024-01-01 00:00:00')),  # exact duplicate of the row above
        ('c1', 'b', pd.Timestamp('2024-01-01 00:01:00')),
    ])
    result = check_data_quality(df)
    assert result['duplicate_events'] == 2  # both rows in the duplicate pair are counted
    assert any('duplicate event' in issue.lower() for issue in result['issues'])


def test_check_data_quality_detects_single_event_cases():
    df = make_event_log([
        ('c1', 'a', pd.Timestamp('2024-01-01 00:00:00')),  # only event in its case
        ('c2', 'a', pd.Timestamp('2024-01-01 00:00:00')),
        ('c2', 'b', pd.Timestamp('2024-01-01 00:01:00')),
    ])
    result = check_data_quality(df)
    assert result['single_event_cases'] == 1
    assert any('one event' in issue.lower() for issue in result['issues'])


def test_check_data_quality_detects_out_of_order_events():
    df = make_event_log([
        ('c1', 'a', pd.Timestamp('2024-01-01 00:05:00')),
        ('c1', 'b', pd.Timestamp('2024-01-01 00:00:00')),  # logged after 'a' but timestamped earlier
    ])
    result = check_data_quality(df)
    assert result['out_of_order_events'] == 1
    assert any('earlier than the previous event' in issue.lower() for issue in result['issues'])


def test_check_data_quality_empty_df_returns_no_issues():
    result = check_data_quality(pd.DataFrame())
    assert result['issues'] == []
    assert result['duplicate_events'] == 0


# --- winsorize_series ---

def test_winsorize_series_percentile_caps_a_single_extreme_outlier():
    series = pd.Series([10.0, 12.0, 11.0, 9.0, 13.0, 10.0, 11.0, 12.0, 9.0, 999999.0])
    clipped, lower, upper = winsorize_series(series, method='percentile', param=10.0)

    assert clipped.max() == pytest.approx(upper)
    assert clipped.max() < 999999.0
    # Every non-outlier value is well inside the band, so only the injected
    # outlier should actually get capped.
    assert (series[:-1] == clipped[:-1]).all()


def test_winsorize_series_std_caps_at_mean_plus_n_std():
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    clipped, lower, upper = winsorize_series(series, method='std', param=1.0)

    mean, std = series.mean(), series.std()
    assert lower == pytest.approx(mean - std)
    assert upper == pytest.approx(mean + std)
    assert clipped.min() >= lower
    assert clipped.max() <= upper


def test_winsorize_series_preserves_nan_positions():
    series = pd.Series([10.0, None, 30.0, None, 9999.0])
    clipped, lower, upper = winsorize_series(series, method='percentile', param=10.0)

    assert clipped.isna().tolist() == [False, True, False, True, False]


def test_winsorize_series_no_outliers_leaves_values_unchanged():
    series = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    clipped, lower, upper = winsorize_series(series, method='std', param=3.0)

    assert (clipped == series).all()


def test_winsorize_series_empty_series_returns_zero_bounds():
    clipped, lower, upper = winsorize_series(pd.Series([], dtype=float))
    assert clipped.empty
    assert lower == 0.0
    assert upper == 0.0


def test_winsorize_series_all_nan_returns_zero_bounds():
    clipped, lower, upper = winsorize_series(pd.Series([None, None], dtype=float))
    assert clipped.isna().all()
    assert lower == 0.0
    assert upper == 0.0


# --- refine_activity_labels ---

def test_refine_activity_labels_cleans_urls_per_row_not_by_first_row_only():
    """Regression test: the URL-cleaning step (strip query string, keep last
    path segment) used to decide whether to apply at all based only on the
    first matched row's value, then applied that single decision to the
    whole column. A column mixing plain and URL-like values had every
    non-first-style row leak raw slashes/query strings into the activity
    name."""
    df = pd.DataFrame({
        'concept:name': ['page_view', 'page_view'],
        'page_type': ['product', '/category/product?ref=x'],
        'time:timestamp': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-01 00:01')],
        'case:concept:name': ['c1', 'c1'],
    })
    out = refine_activity_labels(df, target_activity='page_view', context_column='page_type')
    assert out['concept:name'].tolist() == ['page_view_PRODUCT', 'page_view_PRODUCT']
