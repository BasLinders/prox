from prox.config import create_analysis_config
from prox.pipeline import run_full_analysis
from prox.report import generate_html_report

from conftest import make_simple_variant_log


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


def test_generate_html_report_handles_missing_sections_gracefully():
    report = generate_html_report({})
    assert report.startswith("<!doctype html>")
    assert "No data available." in report


def test_generate_html_report_escapes_activity_names():
    df = make_simple_variant_log(n_cases=3)
    df.loc[df.index[0], 'concept:name'] = '<script>alert(1)</script>'
    config = create_analysis_config(filter_steps=[], sample_size=10)
    results = run_full_analysis(df, config=config)
    assert results is not None

    report = generate_html_report(results)
    assert '<script>alert(1)</script>' not in report
    assert '&lt;script&gt;' in report
