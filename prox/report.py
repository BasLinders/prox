import base64
import html
import os
from typing import Any, Dict

from .analytics import format_business_report


def _embed_image(path: str) -> str:
    """Returns a base64 data URI for a local PNG, or '' if unavailable."""
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _table_rows(data: Dict[str, Dict[str, Any]], columns: list) -> str:
    """Renders a dict of {name: stats} into <tr> rows, HTML-escaping all values."""
    rows = []
    for name, stats in data.items():
        cells = "".join(f"<td>{html.escape(str(stats.get(c, '')))}</td>" for c in columns)
        rows.append(f"<tr><td>{html.escape(str(name))}</td>{cells}</tr>")
    return "\n".join(rows) if rows else '<tr><td colspan="99">No data available.</td></tr>'


def generate_html_report(results: Dict[str, Any]) -> str:
    """
    Builds a single, self-contained HTML report from a run_full_analysis()
    results dict, for sharing outside a live Streamlit session. Images are
    embedded as base64 data URIs so the file has no external dependencies.
    """
    summary = results.get('log_summary', {}) or {}
    performance = results.get('performance', {}) or {}
    conformance = results.get('conformance', {}) or {}
    viz = results.get('visualizations', {}) or {}
    biz = results.get('repeat_purchase_analysis')

    overall = conformance.get('overall_summary', {})
    stats = performance.get('summary_statistics', {})
    case_stats = performance.get('case_performance', {}).get('duration_stats', {})
    time_unit = html.escape(str(case_stats.get('unit', '')))
    variant_perf = performance.get('variant_performance', {})
    activity_bn = performance.get('bottlenecks', {}).get('activity_bottlenecks', {})
    transition_bn = performance.get('bottlenecks', {}).get('transition_bottlenecks', {})

    happy_path_img = _embed_image(viz.get('happy_path'))
    main_flow_img = _embed_image(viz.get('bottlenecks'))

    business_section = ""
    if biz:
        business_section = f"""
<h2>Business Insights</h2>
<pre>{html.escape(format_business_report(biz))}</pre>
"""

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>PRoX Process Mining Report</title>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; margin: 2rem; color: #1a1a1a; }}
  h1, h2 {{ border-bottom: 2px solid #ddd; padding-bottom: 0.3rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.9rem; }}
  th {{ background: #f5f5f5; }}
  .metrics {{ display: flex; gap: 2rem; margin-bottom: 1.5rem; flex-wrap: wrap; }}
  .metric {{ background: #f5f5f5; padding: 0.75rem 1.25rem; border-radius: 6px; }}
  .metric .label {{ font-size: 0.8rem; color: #666; }}
  .metric .value {{ font-size: 1.4rem; font-weight: 600; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  .images {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
  .images > div {{ flex: 1; min-width: 300px; }}
  pre {{ background: #f5f5f5; padding: 1rem; border-radius: 6px; white-space: pre-wrap; }}
</style>
</head>
<body>
<h1>PRoX Process Mining Report</h1>

<div class="metrics">
  <div class="metric"><div class="label">Cases</div><div class="value">{summary.get('Number of Cases', 0):,}</div></div>
  <div class="metric"><div class="label">Events</div><div class="value">{summary.get('Number of Events', 0):,}</div></div>
  <div class="metric"><div class="label">Activities</div><div class="value">{summary.get('Number of Unique Activities', 0)}</div></div>
  <div class="metric"><div class="label">Duration (days)</div><div class="value">{summary.get('Total Duration (Days)', 0)}</div></div>
  <div class="metric"><div class="label">Health Score</div><div class="value">{stats.get('process_health_score', 0):.0f}/100</div></div>
</div>

<h2>Process Maps</h2>
<div class="images">
  <div><h3>Happy Path</h3>{f'<img src="{happy_path_img}">' if happy_path_img else '<p>Not available.</p>'}</div>
  <div><h3>Main Process Flow</h3>{f'<img src="{main_flow_img}">' if main_flow_img else '<p>Not available.</p>'}</div>
</div>

<h2>Conformance</h2>
<div class="metrics">
  <div class="metric"><div class="label">Fitness</div><div class="value">{overall.get('fitness_score', 0):.1%}</div></div>
  <div class="metric"><div class="label">Precision</div><div class="value">{overall.get('precision_score', 0):.1%}</div></div>
  <div class="metric"><div class="label">Quality</div><div class="value">{html.escape(str(overall.get('quality_assessment', 'N/A')))}</div></div>
</div>

<h2>Activity Bottlenecks</h2>
<table>
<tr><th>Activity</th><th>Mean Duration ({time_unit})</th><th>Frequency</th><th>Impact Score</th><th>Severity</th></tr>
{_table_rows(activity_bn, ['mean_duration', 'frequency', 'impact_score', 'severity'])}
</table>

<h2>Slowest Transitions</h2>
<table>
<tr><th>Transition</th><th>Mean Duration ({time_unit})</th><th>Frequency</th><th>Impact Score</th><th>Severity</th></tr>
{_table_rows(transition_bn, ['mean_duration', 'frequency', 'impact_score', 'severity'])}
</table>

<h2>Top Variants</h2>
<table>
<tr><th>Variant</th><th>Frequency</th><th>Percentage</th></tr>
{_table_rows(variant_perf.get('top_variants', {}), ['frequency', 'percentage'])}
</table>
{business_section}
</body>
</html>"""
