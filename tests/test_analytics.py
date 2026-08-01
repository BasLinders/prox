import pandas as pd
import pytest

from prox.analytics import (
    get_event_log_summary,
    analyze_process_performance,
    analyze_repeat_purchases,
    analyze_conversion_funnel,
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


def test_analyze_repeat_purchases_ignores_revenue_on_non_purchase_events(tmp_path):
    """Regression test: event_value/price on browsing events (view_item, add_to_cart)
    must not be treated as purchase evidence - GA4-style logs commonly attach a value
    to those events too, and this previously counted cart-abandoners as buyers."""
    df = pd.DataFrame([
        dict(user_id='u1', **{'case:concept:name': 'c1', 'concept:name': 'view_item',
                               'time:timestamp': pd.Timestamp('2024-01-01 10:00'), 'event_value': 80}),
        dict(user_id='u1', **{'case:concept:name': 'c1', 'concept:name': 'add_to_cart',
                               'time:timestamp': pd.Timestamp('2024-01-01 10:01'), 'event_value': 80}),
    ])
    result = analyze_repeat_purchases(df, output_folder=str(tmp_path))
    assert result['metrics']['total_buyers'] == 0


def test_analyze_repeat_purchases_uses_purchase_row_value_not_case_max(tmp_path):
    """Regression test: order value must come from the actual purchase event, not
    the highest-priced item merely viewed earlier in the same case."""
    df = pd.DataFrame([
        dict(user_id='u1', **{'case:concept:name': 'c1', 'concept:name': 'view_item',
                               'time:timestamp': pd.Timestamp('2024-01-01 10:00'), 'event_value': 120}),
        dict(user_id='u1', **{'case:concept:name': 'c1', 'concept:name': 'purchase',
                               'time:timestamp': pd.Timestamp('2024-01-01 10:05'), 'event_value': 30}),
    ])
    result = analyze_repeat_purchases(df, output_folder=str(tmp_path))
    assert result['metrics']['revenue_stats']['avg_value_one_time'] == pytest.approx(30.0)


def test_analyze_repeat_purchases_sums_multiple_purchase_rows_in_same_case(tmp_path):
    """A case with two purchase-tagged line items should sum to the order total,
    not take the max of the two."""
    df = pd.DataFrame([
        dict(user_id='u1', **{'case:concept:name': 'c1', 'concept:name': 'purchase',
                               'time:timestamp': pd.Timestamp('2024-01-01 10:00'), 'event_value': 20}),
        dict(user_id='u1', **{'case:concept:name': 'c1', 'concept:name': 'purchase',
                               'time:timestamp': pd.Timestamp('2024-01-01 10:00'), 'event_value': 15}),
    ])
    result = analyze_repeat_purchases(df, output_folder=str(tmp_path))
    assert result['metrics']['revenue_stats']['avg_value_one_time'] == pytest.approx(35.0)


def test_analyze_repeat_purchases_default_purchase_values_excludes_checkout_steps(tmp_path):
    """Regression test: the default purchase_values list previously included 'payment'
    and 'order', which false-matched GA4 checkout-funnel steps like add_payment_info
    that don't mean a purchase completed."""
    df = pd.DataFrame([
        dict(user_id='u1', **{'case:concept:name': 'c1', 'concept:name': 'add_payment_info',
                               'time:timestamp': pd.Timestamp('2024-01-01 10:00'), 'event_value': 0}),
    ])
    result = analyze_repeat_purchases(df, output_folder=str(tmp_path))
    assert result['metrics']['total_buyers'] == 0


def test_analyze_repeat_purchases_no_purchase_activity(tmp_path):
    df = pd.DataFrame({
        'user_id': ['u1', 'u1'],
        'case:concept:name': ['c1', 'c1'],
        'concept:name': ['view', 'view'],
        'time:timestamp': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-01 00:05')],
    })
    result = analyze_repeat_purchases(df, output_folder=str(tmp_path))
    assert result['metrics']['total_buyers'] == 0


def make_ecommerce_log():
    """u1: adds to cart & buys Electronics ($100), later buys Books ($20).
    u2: adds to cart, abandons (never buys)."""
    base = pd.Timestamp('2024-01-01')
    rows = [
        dict(user_id='u1', **{'case:concept:name': 'c1', 'concept:name': 'add_to_cart',
                               'time:timestamp': base, 'event_value': 100, 'category': 'Electronics'}),
        dict(user_id='u1', **{'case:concept:name': 'c1', 'concept:name': 'purchase',
                               'time:timestamp': base + pd.Timedelta(minutes=1),
                               'event_value': 100, 'category': 'Electronics'}),
        dict(user_id='u1', **{'case:concept:name': 'c2', 'concept:name': 'purchase',
                               'time:timestamp': base + pd.Timedelta(days=5),
                               'event_value': 20, 'category': 'Books'}),
        dict(user_id='u2', **{'case:concept:name': 'c3', 'concept:name': 'add_to_cart',
                               'time:timestamp': base, 'event_value': 40, 'category': 'Books'}),
    ]
    return pd.DataFrame(rows)


def test_analyze_repeat_purchases_cart_abandonment_rate(tmp_path):
    df = make_ecommerce_log()
    result = analyze_repeat_purchases(df, output_folder=str(tmp_path))
    cart = result['metrics']['cart_abandonment']

    assert cart['cases_added_to_cart'] == 2  # c1, c3
    assert cart['cases_purchased_after_cart'] == 1  # c1 only
    assert cart['abandonment_rate'] == pytest.approx(50.0)


def test_analyze_repeat_purchases_cart_abandonment_without_any_purchases(tmp_path):
    """100% abandonment must still be reported even when there are zero purchases."""
    df = pd.DataFrame([
        dict(user_id='u1', **{'case:concept:name': 'c1', 'concept:name': 'add_to_cart',
                               'time:timestamp': pd.Timestamp('2024-01-01'), 'event_value': 40}),
    ])
    result = analyze_repeat_purchases(df, output_folder=str(tmp_path))
    assert result['metrics']['total_buyers'] == 0
    assert result['metrics']['cart_abandonment'] == {
        'cases_added_to_cart': 1, 'cases_purchased_after_cart': 0, 'abandonment_rate': 100.0
    }


def test_analyze_repeat_purchases_average_order_value(tmp_path):
    df = make_ecommerce_log()
    result = analyze_repeat_purchases(df, output_folder=str(tmp_path))
    assert result['metrics']['average_order_value'] == pytest.approx(60.0)  # (100 + 20) / 2


def test_analyze_repeat_purchases_category_breakdown(tmp_path):
    df = make_ecommerce_log()
    result = analyze_repeat_purchases(df, output_folder=str(tmp_path))
    breakdown = result['metrics']['category_breakdown']

    assert breakdown['Electronics'] == {'revenue': 100.0, 'orders': 1}
    assert breakdown['Books'] == {'revenue': 20.0, 'orders': 1}
    assert result['charts']['category'] is not None


def test_analyze_repeat_purchases_category_breakdown_absent_without_category_column(tmp_path):
    df = make_purchase_log()
    result = analyze_repeat_purchases(df, output_folder=str(tmp_path))
    assert result['metrics']['category_breakdown'] == {}
    assert result['charts']['category'] is None


def test_analyze_repeat_purchases_revenue_trend(tmp_path):
    df = make_ecommerce_log()
    result = analyze_repeat_purchases(df, output_folder=str(tmp_path))
    trend = result['metrics']['revenue_trend']

    assert trend['2024-01-01'] == {'orders': 1, 'revenue': 100.0}
    assert trend['2024-01-06'] == {'orders': 1, 'revenue': 20.0}


# --- analyze_conversion_funnel ---

def make_funnel_log():
    """10 sessions: all view, 7 add_to_cart, 4 begin_checkout, 2 purchase."""
    rows = []
    base = pd.Timestamp('2024-01-01')

    def add(case, acts):
        t = base
        for a in acts:
            rows.append({'case:concept:name': case, 'concept:name': a, 'time:timestamp': t})
            t += pd.Timedelta(minutes=1)

    for i in range(10):
        acts = ['view_item']
        if i < 7:
            acts.append('add_to_cart')
        if i < 4:
            acts.append('begin_checkout')
        if i < 2:
            acts.append('purchase')
        add(f'c{i}', acts)

    return pd.DataFrame(rows)


def test_analyze_conversion_funnel_explicit_steps():
    df = make_funnel_log()
    result = analyze_conversion_funnel(
        df, funnel_steps=['view_item', 'add_to_cart', 'begin_checkout', 'purchase']
    )

    assert result['total_cases'] == 10
    assert result['stages']['view_item']['cases_reached'] == 10
    assert result['stages']['add_to_cart']['cases_reached'] == 7
    assert result['stages']['begin_checkout']['cases_reached'] == 4
    assert result['stages']['purchase']['cases_reached'] == 2
    assert result['stages']['purchase']['pct_of_total'] == pytest.approx(20.0)
    assert result['biggest_drop_off'] == 'purchase'
    assert result['errors'] == []


def test_analyze_conversion_funnel_auto_derives_order_matching_explicit():
    """Auto-derivation should recover the same order as the explicit funnel_steps
    on a clean, linearly-ordered funnel."""
    df = make_funnel_log()
    result = analyze_conversion_funnel(df)
    assert result['funnel_steps'] == ['view_item', 'add_to_cart', 'begin_checkout', 'purchase']


def test_analyze_conversion_funnel_truncates_at_first_zero_stage():
    """A stage with zero cases must stop the funnel there, not list further
    meaningless zero-stages after it."""
    df = make_funnel_log()
    result = analyze_conversion_funnel(
        df, funnel_steps=['view_item', 'nonexistent_step', 'add_to_cart']
    )
    assert list(result['stages'].keys()) == ['view_item', 'nonexistent_step']
    assert result['stages']['nonexistent_step']['cases_reached'] == 0
    assert result['funnel_steps'] == ['view_item', 'nonexistent_step']


def test_analyze_conversion_funnel_missing_columns():
    result = analyze_conversion_funnel(pd.DataFrame({'foo': [1]}))
    assert result['stages'] == {}
    assert any('missing' in e.lower() for e in result['errors'])


def test_analyze_conversion_funnel_empty_log():
    result = analyze_conversion_funnel(pd.DataFrame(columns=['case:concept:name', 'concept:name']))
    assert result['stages'] == {}
    assert any('empty' in e.lower() for e in result['errors'])


# --- format_business_report ---

def test_format_business_report_handles_none():
    assert "No business insights" in format_business_report(None)


def test_format_business_report_includes_repeat_rate(tmp_path):
    df = make_purchase_log()
    result = analyze_repeat_purchases(df, output_folder=str(tmp_path))
    report = format_business_report(result)
    assert "50.00%" in report
