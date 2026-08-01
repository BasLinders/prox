import base64

import pandas as pd

from prox.config import create_analysis_config
from prox.pipeline import run_full_analysis
from prox.report import generate_html_report, generate_segment_comparison_report
from prox.segments import compare_segments

from conftest import make_event_log, make_simple_variant_log

# Minimal valid 1x1 PNG, used to test image-embedding/zoom behavior without
# depending on Graphviz actually being installed (CI doesn't have it - see
# visualizer.py's documented "hard to assert on meaningfully" status).
_DUMMY_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _write_dummy_png(path):
    with open(path, "wb") as f:
        f.write(_DUMMY_PNG_BYTES)


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


def test_generate_html_report_images_are_zoomable(tmp_path):
    """Process map images (the flow charts) must be clickable to view full-size,
    since they're unreadable at the embedded thumbnail size. Uses a synthetic
    image path rather than a real pipeline run, since actual image generation
    needs Graphviz installed, which CI doesn't have."""
    img_path = tmp_path / "happy_path.png"
    _write_dummy_png(img_path)

    results = {'visualizations': {'happy_path': str(img_path), 'bottlenecks': None}}
    report = generate_html_report(results)

    assert 'class="zoomable"' in report
    assert 'id="lightbox-overlay"' in report
    assert 'id="lightbox-img"' in report


def test_generate_html_report_handles_missing_sections_gracefully():
    report = generate_html_report({})
    assert report.startswith("<!doctype html>")
    assert "No data available." in report
    assert "Executive Summary" not in report


def _make_ecommerce_funnel_log():
    """A small e-commerce-shaped log with cart, purchase, and category data,
    used to exercise the new business-insight sections end to end."""
    base = pd.Timestamp('2024-01-01')
    rows = []
    for i in range(5):
        case_id = f'c{i}'
        rows.append({'case:concept:name': case_id, 'concept:name': 'view_item',
                      'time:timestamp': base + pd.Timedelta(hours=i), 'user_id': f'u{i}',
                      'event_value': 0, 'category': 'Electronics'})
        rows.append({'case:concept:name': case_id, 'concept:name': 'add_to_cart',
                      'time:timestamp': base + pd.Timedelta(hours=i, minutes=1), 'user_id': f'u{i}',
                      'event_value': 0, 'category': 'Electronics'})
        if i < 3:
            rows.append({'case:concept:name': case_id, 'concept:name': 'purchase',
                          'time:timestamp': base + pd.Timedelta(hours=i, minutes=2), 'user_id': f'u{i}',
                          'event_value': 50 + i, 'category': 'Electronics'})
    return pd.DataFrame(rows)


def test_generate_html_report_includes_funnel_and_extended_business_sections():
    df = _make_ecommerce_funnel_log()
    config = create_analysis_config(filter_steps=[], sample_size=10)
    results = run_full_analysis(df, config=config)
    assert results is not None

    report = generate_html_report(results)

    assert "Conversion Funnel" in report
    assert "Average Order Value" in report
    assert "Cart Abandonment" in report
    assert "Revenue by Category" in report
    assert "Electronics" in report


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


def test_generate_segment_comparison_report_images_are_zoomable(tmp_path):
    """Per-segment happy path images must be clickable, same as the main report.
    Uses a synthetic image path since real generation needs Graphviz, which CI
    doesn't have."""
    img_path = tmp_path / "happy_path.png"
    _write_dummy_png(img_path)

    segment_result = {
        'segments': {
            'mobile': {'visualizations': {'happy_path': str(img_path)}},
            'desktop': {'visualizations': {'happy_path': str(img_path)}},
        },
        'comparison_table': {
            'mobile': {'cases': 3, 'health_score': 100, 'fitness_score': 1.0,
                       'precision_score': 1.0, 'repeat_rate': 0.0, 'top_variant': 'a -> b -> c'},
            'desktop': {'cases': 2, 'health_score': 90, 'fitness_score': 1.0,
                        'precision_score': 1.0, 'repeat_rate': 0.0, 'top_variant': 'a -> c'},
        },
        'errors': [],
    }

    report = generate_segment_comparison_report(segment_result)
    assert 'class="zoomable"' in report
    assert 'id="lightbox-overlay"' in report


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
