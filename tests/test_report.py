import pandas as pd

from prox.config import create_analysis_config
from prox.pipeline import run_full_analysis
from prox.report import generate_html_report, generate_segment_comparison_report
from prox.segments import compare_segments

from conftest import make_event_log, make_simple_variant_log


def test_generate_html_report_contains_key_sections():
    df = make_simple_variant_log(n_cases=3)
    config = create_analysis_config(filter_steps=[], sample_size=10)
    results = run_full_analysis(df, config=config)
    assert results is not None

    report = generate_html_report(results)

    assert report.startswith("<!doctype html>")
    assert "PRoX Process Mining Report" in report
    assert "Activity Bottlenecks" in report
    assert "Slowest Transitions" in report
    assert "Top Variants" in report
    assert "Conformance" in report


def test_generate_html_report_contains_plain_language_executive_summary():
    df = make_simple_variant_log(n_cases=3)
    config = create_analysis_config(filter_steps=[], sample_size=10)
    results = run_full_analysis(df, config=config)
    assert results is not None

    report = generate_html_report(results)

    assert "Executive Summary" in report
    assert "health score" in report.lower()
    # One of the plain-language health verdicts should be present as a badge label.
    assert any(label in report for label in ("Healthy", "Needs attention", "Critical"))
    # The most common journey should be rendered as an arrow-separated sentence, not raw ' -> '.
    assert "→" in report


def test_generate_html_report_images_are_zoomable():
    """Process map images (the flow charts) must be clickable to view full-size,
    since they're unreadable at the embedded thumbnail size."""
    df = make_simple_variant_log(n_cases=3)
    config = create_analysis_config(filter_steps=[], sample_size=10)
    results = run_full_analysis(df, config=config)
    assert results is not None

    report = generate_html_report(results)
    assert 'class="zoomable"' in report
    assert 'id="lightbox-overlay"' in report
    assert 'id="lightbox-img"' in report


def test_generate_html_report_handles_missing_sections_gracefully():
    report = generate_html_report({})
    assert report.startswith("<!doctype html>")
    assert "No data available." in report
    assert "Executive Summary" not in report


def test_generate_html_report_escapes_activity_names():
    df = make_simple_variant_log(n_cases=3)
    df.loc[df.index[0], 'concept:name'] = '<script>alert(1)</script>'
    config = create_analysis_config(filter_steps=[], sample_size=10)
    results = run_full_analysis(df, config=config)
    assert results is not None

    report = generate_html_report(results)
    assert '<script>alert(1)</script>' not in report
    assert '&lt;script&gt;' in report


def _make_segmented_log():
    rows = []
    base = pd.Timestamp('2024-01-01')
    for i in range(3):
        case_id = f'm{i}'
        for j, act in enumerate(['a', 'b', 'c']):
            rows.append((case_id, act, base + pd.Timedelta(minutes=j), 'mobile'))
    for i in range(2):
        case_id = f'd{i}'
        for j, act in enumerate(['a', 'c']):
            rows.append((case_id, act, base + pd.Timedelta(minutes=j), 'desktop'))

    df = make_event_log([(r[0], r[1], r[2]) for r in rows])
    df['device'] = [r[3] for r in rows]
    return df


def test_generate_segment_comparison_report_contains_key_sections(tmp_path):
    df = _make_segmented_log()
    config = create_analysis_config(filter_steps=[], sample_size=10)
    segment_result = compare_segments(df, segment_col='device', config=config, output_folder=str(tmp_path))

    report = generate_segment_comparison_report(segment_result)

    assert report.startswith("<!doctype html>")
    assert "PRoX Segment Comparison Report" in report
    assert "Executive Summary" in report
    assert "Segment Comparison" in report
    assert "mobile" in report
    assert "desktop" in report
    assert 'class="zoomable"' in report


def test_generate_segment_comparison_report_handles_no_data():
    report = generate_segment_comparison_report({'segments': {}, 'comparison_table': {}, 'errors': []})
    assert report.startswith("<!doctype html>")
    assert "No segment data available." in report


def test_generate_segment_comparison_report_escapes_segment_values(tmp_path):
    df = _make_segmented_log()
    df['device'] = df['device'].replace('mobile', '<script>alert(1)</script>')
    config = create_analysis_config(filter_steps=[], sample_size=10)
    segment_result = compare_segments(df, segment_col='device', config=config, output_folder=str(tmp_path))

    report = generate_segment_comparison_report(segment_result)
    assert '<script>alert(1)</script>' not in report
    assert '&lt;script&gt;' in report
