import pandas as pd
import pytest

from pdf_builder import (
    _available_sections,
    _section_business,
    _section_sessions,
    _section_variants,
    _styles,
    build_pdf_report,
)


def make_results_with_business_and_sessions(tmp_path):
    """Minimal results dict covering just the sections these tests exercise -
    build_pdf_report/section builders only read the keys they need, so an
    otherwise-empty dict is fine for the untested sections."""
    sessions_df = pd.DataFrame({
        'session_id': ['u1_sA', 'u1_sB', 'u2_sA'],
        'user_id': ['u1', 'u1', 'u2'],
        'label': ['Browsing', 'Buying', 'Buying'],
        'event_count': [2, 3, 1],
        'first_activity': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-01']),
        'last_activity': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-01']),
    })
    journeys_df = pd.DataFrame({
        'user_id': ['u1', 'u2'],
        'session_count': [2, 1],
        'journey': ['Browsing -> Buying', 'Buying'],
    })
    return {
        'log_summary': {'Number of Cases': 2, 'Number of Events': 6, 'Number of Unique Activities': 3},
        'repeat_purchase_analysis': {
            'metrics': {
                'total_buyers': 2, 'repeat_rate': 50.0, 'average_order_value': 75.0,
                'cart_abandonment': {'abandonment_rate': 10.0},
                'category_breakdown': {'shoes': {'revenue': 100.0, 'orders': 2}},
                'revenue_stats': {}, 'avg_days_between': 0, 'median_days_between': 0,
            },
            'charts': {},
        },
        'session_insights': {'sessions': sessions_df, 'journeys': journeys_df},
    }


def test_available_sections_empty_results_is_empty():
    assert _available_sections({}, None) == {}


def test_available_sections_only_lists_sections_with_data(tmp_path):
    results = make_results_with_business_and_sessions(tmp_path)
    available = _available_sections(results, None)
    assert set(available.keys()) == {'business', 'sessions'}
    assert 'variants' not in available
    assert 'segments' not in available


def test_available_sections_includes_segments_when_comparison_present():
    segment_result = {'comparison_table': {'mobile': {'cases': 10}}}
    available = _available_sections({}, segment_result)
    assert set(available.keys()) == {'segments'}


def test_section_business_renders_metrics_when_present(tmp_path):
    results = make_results_with_business_and_sessions(tmp_path)
    story = _section_business(results, None, _styles())
    assert len(story) > 1  # heading + metrics table + more, not just a "no data" line


def test_section_business_reports_no_data_when_absent():
    story = _section_business({}, None, _styles())
    # heading + a single "no data" paragraph
    assert len(story) == 2


def test_section_sessions_includes_journey_table(tmp_path):
    results = make_results_with_business_and_sessions(tmp_path)
    story = _section_sessions(results, None, _styles())
    assert len(story) > 2


def test_section_variants_reports_no_data_when_absent():
    story = _section_variants({}, None, _styles())
    assert len(story) == 2


def test_build_pdf_report_produces_valid_pdf_bytes(tmp_path):
    results = make_results_with_business_and_sessions(tmp_path)
    pdf_bytes = build_pdf_report(results, ['business', 'sessions'])
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b'%PDF')
    assert len(pdf_bytes) > 500


def test_build_pdf_report_only_includes_requested_sections(tmp_path):
    """Selecting a subset of available sections is the whole point of the
    opt-in checkboxes - a request for just 'sessions' shouldn't silently
    pull in 'business' too."""
    results = make_results_with_business_and_sessions(tmp_path)
    both = build_pdf_report(results, ['business', 'sessions'])
    sessions_only = build_pdf_report(results, ['sessions'])
    # Not a strict guarantee for arbitrary PDFs (compression), but with one
    # fewer section (metrics table, category table, chart captions, report
    # text) the output is reliably and substantially smaller here.
    assert len(sessions_only) < len(both)


def test_build_pdf_report_ignores_unknown_section_keys(tmp_path):
    results = make_results_with_business_and_sessions(tmp_path)
    pdf_bytes = build_pdf_report(results, ['sessions', 'does_not_exist'])
    assert pdf_bytes.startswith(b'%PDF')
