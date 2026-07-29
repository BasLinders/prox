import pandas as pd
import pytest

from prox.analytics import (
    get_event_log_summary,
    analyze_process_performance,
    analyze_repeat_purchases,
    format_business_report,
)

from conftest import make_event_log


def make_bottleneck_log(n_cases=5):
    """Every case: a -> (60 min gap) -> b -> (1 min gap) -> c.

    'b' is a deliberate, deterministic bottleneck relative to 'c'; 'a' has no
    prior activity so its time_since_prev is always NaN (treated as 0).
    """
    rows = []
    base = pd.Timestamp('2024-01-01 00:00:00')
    for i in range(n_cases):
        case_id = str(i + 1)
        rows.append((case_id, 'a', base))
        rows.append((case_id, 'b', base + pd.Timedelta(minutes=60)))
        rows.append((case_id, 'c', base + pd.Timedelta(minutes=61)))
    return make_event_log(rows)


def make_purchase_log():
    """u1 buys twice (repeat buyer), u2 buys once (one-time buyer)."""
    rows = [
        dict(user_id='u1', **{'case:concept:name': 'c1', 'concept:name': 'purchase',
                               'time:timestamp': pd.Timestamp('2024-01-01'), 'event_value': 100}),
        dict(user_id='u1', **{'case:concept:name': 'c2', 'concept:name': 'purchase',
                               'time:timestamp': pd.Timestamp('2024-01-06'), 'event_value': 100}),
        dict(user_id='u2', **{'case:concept:name': 'c3', 'concept:name': 'purchase',
                               'time:timestamp': pd.Timestamp('2024-01-02'), 'event_value': 50}),
    ]
    return pd.DataFrame(rows)


# --- get_event_log_summary ---

def test_get_event_log_summary_basic(simple_event_log):
    summary, errors = get_event_log_summary(simple_event_log)
    assert errors == []
    assert summary['Number of Cases'] == 3
    assert summary['Number of Events'] == 9
    assert summary['Number of Unique Activities'] == 3


def test_get_event_log_summary_empty_df():
    summary, errors = get_event_log_summary(pd.DataFrame())
    assert summary is None
    assert any('empty' in e.lower() for e in errors)


def test_get_event_log_summary_missing_columns():
    summary, errors = get_event_log_summary(pd.DataFrame({'foo': [1, 2]}))
    assert summary is None
    assert any('missing' in e.lower() for e in errors)


# --- analyze_process_performance: bottlenecks & health score ---

def test_activity_bottleneck_detection_and_impact_score():
    df = make_bottleneck_log()
    results = analyze_process_performance(df, time_unit='minutes', bottleneck_threshold_percentile=75)

    bottlenecks = results['bottlenecks']['activity_bottlenecks']
    assert 'b' in bottlenecks
    assert 'a' not in bottlenecks
    assert 'c' not in bottlenecks

    assert bottlenecks['b']['mean_duration'] == pytest.approx(60.0)
    assert bottlenecks['b']['frequency'] == 5
    assert bottlenecks['b']['impact_score'] == pytest.approx(60.0 * 5)
    assert bottlenecks['b']['severity'] == 'medium'


def test_health_score_within_bounds():
    df = make_bottleneck_log()
    results = analyze_process_performance(df, time_unit='minutes')
    score = results['summary_statistics']['process_health_score']
    assert 0 <= score <= 100


def test_analyze_process_performance_missing_columns():
    results = analyze_process_performance(pd.DataFrame({'foo': [1]}))
    assert any('missing' in e.lower() for e in results['errors'])
    assert results['bottlenecks'] == {}


# --- analyze_repeat_purchases ---

def test_analyze_repeat_purchases_detects_repeat_buyer(tmp_path):
    df = make_purchase_log()
    result = analyze_repeat_purchases(df, output_folder=str(tmp_path))
    metrics = result['metrics']

    assert metrics['total_buyers'] == 2
    assert metrics['repeat_rate'] == pytest.approx(50.0)
    assert metrics['median_days_between'] == pytest.approx(5.0, abs=0.1)
    assert metrics['revenue_stats']['multiplier'] == pytest.approx(4.0)


def test_analyze_repeat_purchases_no_purchase_activity(tmp_path):
    df = pd.DataFrame({
        'user_id': ['u1', 'u1'],
        'case:concept:name': ['c1', 'c1'],
        'concept:name': ['view', 'view'],
        'time:timestamp': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-01 00:05')],
    })
    result = analyze_repeat_purchases(df, output_folder=str(tmp_path))
    assert result['metrics']['total_buyers'] == 0


# --- format_business_report ---

def test_format_business_report_handles_none():
    assert "No business insights" in format_business_report(None)


def test_format_business_report_includes_repeat_rate(tmp_path):
    df = make_purchase_log()
    result = analyze_repeat_purchases(df, output_folder=str(tmp_path))
    report = format_business_report(result)
    assert "50.00%" in report
