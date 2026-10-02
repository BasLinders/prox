import io

import pandas as pd
import pytest

from prox.data_manager import (
    load_and_validate_csv,
    filter_event_log,
    sample_log_stratified,
    optimize_dataframe_memory,
    refine_activity_labels,
    merge_page_views_into_page_events,
    check_data_quality,
    drop_duplicate_events,
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


def test_load_and_validate_csv_derives_purchase_flag_from_activity_names():
    """Regression test: a live BigQuery extract or a plain event-name CSV only
    has 'purchase' as a *value* inside concept:name, not as its own 0/1
    column - unlike the mock dataset, which bakes a literal 'purchase' column
    into every row. Without deriving one here, steps that require a literal
    binary column (e.g. main.py's sampling stratify-column picker) silently
    found nothing to offer on real data even when purchase events existed."""
    raw = make_raw_log_df()
    raw['event_name'] = ['view_item', 'purchase', 'view_item']
    df, messages, has_category = load_and_validate_csv(make_csv_bytes(raw))

    assert df is not None
    assert 'purchase' in df.columns
    assert df['purchase'].tolist() == [False, True, False]
    assert df['purchase'].nunique() == 2
    assert any("Derived 'purchase' column" in m for m in messages)


def test_load_and_validate_csv_does_not_overwrite_existing_purchase_column():
    raw = make_raw_log_df()
    raw['purchase'] = [1, 0, 0]
    df, messages, has_category = load_and_validate_csv(make_csv_bytes(raw))

    assert df is not None
    assert df['purchase'].tolist() == [1, 0, 0]
    assert not any("Derived 'purchase' column" in m for m in messages)


def test_load_and_validate_csv_no_purchase_activity_leaves_column_absent():
    raw = make_raw_log_df()  # event_name values are just 'a' and 'b'
    df, messages, has_category = load_and_validate_csv(make_csv_bytes(raw))

    assert df is not None
    assert 'purchase' not in df.columns
    assert 'add_to_cart' not in df.columns


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


# --- drop_duplicate_events ---

def test_drop_duplicate_events_keeps_first_occurrence():
    df = make_event_log([
        ('c1', 'a', pd.Timestamp('2024-01-01 00:00:00')),
        ('c1', 'a', pd.Timestamp('2024-01-01 00:00:00')),  # exact duplicate of the row above
        ('c1', 'a', pd.Timestamp('2024-01-01 00:00:00')),  # and another
        ('c1', 'b', pd.Timestamp('2024-01-01 00:01:00')),
    ])
    df['price'] = [1.0, 2.0, 3.0, 4.0]
    result, n_removed = drop_duplicate_events(df)
    assert n_removed == 2
    assert len(result) == 2
    assert result['price'].tolist() == [1.0, 4.0]


def test_drop_duplicate_events_same_activity_different_timestamp_is_kept():
    df = make_event_log([
        ('c1', 'a', pd.Timestamp('2024-01-01 00:00:00')),
        ('c1', 'a', pd.Timestamp('2024-01-01 00:00:01')),
        ('c2', 'a', pd.Timestamp('2024-01-01 00:00:00')),  # same activity/time, other case
    ])
    result, n_removed = drop_duplicate_events(df)
    assert n_removed == 0
    assert len(result) == 3


def test_drop_duplicate_events_matches_check_data_quality():
    """Every flagged duplicate group collapses to one event, and a
    deduplicated log comes back clean."""
    df = make_event_log([
        ('c1', 'a', pd.Timestamp('2024-01-01 00:00:00')),
        ('c1', 'a', pd.Timestamp('2024-01-01 00:00:00')),
        ('c1', 'b', pd.Timestamp('2024-01-01 00:01:00')),
    ])
    assert check_data_quality(df)['duplicate_events'] == 2
    result, n_removed = drop_duplicate_events(df)
    assert n_removed == 1
    assert check_data_quality(result)['duplicate_events'] == 0


def test_drop_duplicate_events_same_rows_removed_from_copies():
    """raw_df and df_ready are deduplicated separately - both must lose the same rows."""
    df = make_event_log([
        ('c1', 'a', pd.Timestamp('2024-01-01 00:00:00')),
        ('c1', 'b', pd.Timestamp('2024-01-01 00:01:00')),
        ('c1', 'a', pd.Timestamp('2024-01-01 00:00:00')),
    ])
    first, _ = drop_duplicate_events(df)
    second, _ = drop_duplicate_events(df.copy())
    assert first.index.tolist() == second.index.tolist() == [0, 1]


def test_drop_duplicate_events_empty_df():
    result, n_removed = drop_duplicate_events(pd.DataFrame())
    assert n_removed == 0
    assert result.empty


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


def test_to_pm4py_frame_is_slim_and_time_sorted():
    from prox.data_manager import to_pm4py_frame
    df = pd.DataFrame({
        'case:concept:name': ['b', 'a', 'a', 'b'],
        'concept:name': pd.Categorical(['y', 'x2', 'x1', 'z']),
        'time:timestamp': pd.to_datetime(['2026-01-01 10:00', '2026-01-01 09:05', '2026-01-01 09:00', '2026-01-01 09:30']),
        'user_id': ['u', 'u', 'u', 'u'],
    })
    out = to_pm4py_frame(df)
    assert list(out.columns) == ['case:concept:name', 'concept:name', 'time:timestamp']
    assert out['concept:name'].tolist() == ['x1', 'x2', 'z', 'y']
    assert not isinstance(out['concept:name'].dtype, pd.CategoricalDtype)
    assert 'user_id' in df.columns  # input untouched


# --- merge_page_views_into_page_events ---

def _pv_log(rows, **extra):
    t0 = pd.Timestamp('2026-01-01 10:00:00')
    df = pd.DataFrame(
        [('c1', name, t0 + pd.Timedelta(seconds=sec)) for name, sec in rows],
        columns=['case:concept:name', 'concept:name', 'time:timestamp'],
    )
    for col, values in extra.items():
        df[col] = values
    return df


def test_merge_page_views_drops_page_view_next_to_page_specific_event():
    df = _pv_log([('page_view', 0), ('view_item', 1), ('add_to_cart', 8)])
    result, removed = merge_page_views_into_page_events(df)
    assert removed == 1
    assert result['concept:name'].tolist() == ['view_item', 'add_to_cart']


def test_merge_page_views_keeps_cms_page_views():
    df = _pv_log([('page_view', 0), ('page_view', 20), ('view_item_list', 21)])
    result, removed = merge_page_views_into_page_events(df)
    assert removed == 1
    assert result['concept:name'].tolist() == ['page_view', 'view_item_list']
    assert result['time:timestamp'].iloc[0] == pd.Timestamp('2026-01-01 10:00:00')


def test_merge_page_views_is_independent_of_tied_row_order():
    a = _pv_log([('page_view', 0), ('view_item', 0)])
    b = _pv_log([('view_item', 0), ('page_view', 0)])
    for df in (a, b):
        result, removed = merge_page_views_into_page_events(df)
        assert removed == 1
        assert result['concept:name'].tolist() == ['view_item']


def test_merge_page_views_event_before_page_view_belongs_to_previous_page():
    # view_item at 0, then 3s later a CMS page_view: view_item must not absorb it.
    df = _pv_log([('view_item', 0), ('page_view', 3)])
    result, removed = merge_page_views_into_page_events(df)
    assert removed == 0
    assert result['concept:name'].tolist() == ['view_item', 'page_view']


def test_merge_page_views_fast_click_through_keeps_cms_page_view():
    # CMS page_view at 0, listing page_view at 3 with its view_item_list at 3.2:
    # the list event merges only the nearest preceding page_view.
    df = _pv_log([('page_view', 0), ('page_view', 3), ('view_item_list', 3.2)])
    result, removed = merge_page_views_into_page_events(df)
    assert removed == 1
    assert result['time:timestamp'].tolist()[0] == pd.Timestamp('2026-01-01 10:00:00')
    assert result['concept:name'].tolist() == ['page_view', 'view_item_list']


def test_merge_page_views_respects_window_and_case():
    df = _pv_log([('page_view', 0), ('view_item', 30)])
    assert merge_page_views_into_page_events(df)[1] == 0
    other_case = _pv_log([('page_view', 0), ('view_item', 1)])
    other_case.loc[1, 'case:concept:name'] = 'c2'
    assert merge_page_views_into_page_events(other_case)[1] == 0


def test_merge_page_views_requires_same_url_when_column_present():
    same = _pv_log([('page_view', 0), ('view_item', 1)], page_location=['/p/1', '/p/1'])
    assert merge_page_views_into_page_events(same)[1] == 1
    different = _pv_log([('page_view', 0), ('view_item', 1)], page_location=['/home', '/p/1'])
    assert merge_page_views_into_page_events(different)[1] == 0


def test_merge_page_views_noop_without_both_kinds():
    only_pv = _pv_log([('page_view', 0), ('click', 1)])
    result, removed = merge_page_views_into_page_events(only_pv)
    assert removed == 0
    assert len(result) == 2


def test_merge_page_views_handles_non_unique_index():
    second = _pv_log([('page_view', 0), ('view_item', 1)])
    second['case:concept:name'] = 'c2'
    df = pd.concat([_pv_log([('page_view', 0), ('view_item', 1)]), second])  # index labels repeat
    result, removed = merge_page_views_into_page_events(df)
    assert removed == 2
    assert result['concept:name'].tolist() == ['view_item', 'view_item']
