import base64
import html
import os
from typing import Any, Dict

from .analytics import format_business_report

_STYLE = """
  body { font-family: -apple-system, Arial, sans-serif; margin: 2rem; color: #1a1a1a; max-width: 1100px; }
  h1, h2 { border-bottom: 2px solid #ddd; padding-bottom: 0.3rem; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }
  th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.9rem; }
  th { background: #f5f5f5; }
  .metrics { display: flex; gap: 2rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
  .metric { background: #f5f5f5; padding: 0.75rem 1.25rem; border-radius: 6px; }
  .metric .label { font-size: 0.8rem; color: #666; }
  .metric .value { font-size: 1.4rem; font-weight: 600; }
  img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
  img.zoomable { cursor: zoom-in; }
  .images { display: flex; gap: 1rem; flex-wrap: wrap; }
  .images > div { flex: 1; min-width: 300px; }
  pre { background: #f5f5f5; padding: 1rem; border-radius: 6px; white-space: pre-wrap; }
  .summary-box { background: #f8f9fb; border: 1px solid #e0e0e0; border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem; }
  .summary-box p { line-height: 1.5; }
  .summary-box ul { margin: 0.5rem 0 0; padding-left: 1.25rem; }
  .summary-box li { margin-bottom: 0.4rem; }
  .badge { display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px; font-weight: 600; font-size: 0.85rem; margin-bottom: 0.75rem; }
  .badge.good { background: #e6f4ea; color: #1e7e34; }
  .badge.warn { background: #fff8e1; color: #8a6100; }
  .badge.bad { background: #fdecea; color: #b3261e; }
  #lightbox-overlay {
    display: none; position: fixed; inset: 0; z-index: 1000;
    background: rgba(0, 0, 0, 0.85); align-items: center; justify-content: center;
    padding: 3rem; cursor: zoom-out;
  }
  #lightbox-overlay.active { display: flex; }
  #lightbox-overlay img { max-width: 95vw; max-height: 95vh; border: none; border-radius: 4px; }
"""

# Vanilla JS lightbox: click any .zoomable image to view it full-size, click
# again (or press Escape) to close. No external dependencies, since this
# report is a standalone file with no guaranteed internet access.
_LIGHTBOX_HTML = """
<div id="lightbox-overlay">
  <img id="lightbox-img" src="" alt="Enlarged view">
</div>
<script>
document.addEventListener('click', function (e) {
  var overlay = document.getElementById('lightbox-overlay');
  if (e.target.classList.contains('zoomable')) {
    document.getElementById('lightbox-img').src = e.target.src;
    overlay.classList.add('active');
  } else if (e.target === overlay || e.target.id === 'lightbox-img') {
    overlay.classList.remove('active');
  }
});
document.addEventListener('keydown', function (e) {
  if (e.key === 'Escape') {
    document.getElementById('lightbox-overlay').classList.remove('active');
  }
});
</script>
"""


def _html_shell(title: str, body: str) -> str:
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{_STYLE}</style>
</head>
<body>
{body}
{_LIGHTBOX_HTML}
</body>
</html>"""


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


def _health_verdict(score: float) -> tuple:
    """Translates a 0-100 process health score into a plain-language verdict + badge class."""
    if score >= 80:
        return "Healthy", "good"
    if score >= 50:
        return "Needs attention", "warn"
    return "Critical", "bad"


def _format_variant(variant: str) -> str:
    """Renders an ' -> '-joined variant path as an arrow-separated, HTML-escaped string."""
    return html.escape(str(variant).replace(' -> ', ' → '))


def _build_narrative_summary(results: Dict[str, Any]) -> str:
    """Builds a plain-language 'Executive Summary' section for non-technical readers,
    translating scores and stats already computed elsewhere in the pipeline into sentences."""
    summary = results.get('log_summary', {}) or {}
    performance = results.get('performance', {}) or {}
    conformance = results.get('conformance', {}) or {}
    biz = results.get('repeat_purchase_analysis')

    overall = conformance.get('overall_summary', {})
    stats = performance.get('summary_statistics', {})
    bn_summary = performance.get('bottlenecks', {}).get('summary', {})
    top_variants = performance.get('variant_performance', {}).get('top_variants', {})

    health = stats.get('process_health_score', 0)
    verdict_label, verdict_cls = _health_verdict(health)
    cases = summary.get('Number of Cases', 0)

    paragraphs = [
        f"<p>Based on <strong>{cases:,}</strong> customer journeys analysed, this process is "
        f"currently <strong>{html.escape(verdict_label.lower())}</strong>, with a health score of "
        f"<strong>{health:.0f} out of 100</strong>.</p>"
    ]

    fitness = overall.get('fitness_score', 0)
    quality = overall.get('quality_assessment', 'N/A')
    paragraphs.append(
        f"<p>PRoX checked how closely a sample of customer journeys match the process's expected "
        f"flow: <strong>{fitness:.0%}</strong> followed a recognised, valid path — rated "
        f"<strong>{html.escape(str(quality))}</strong> overall.</p>"
    )

    if top_variants:
        top_variant, top_stats = next(iter(top_variants.items()))
        pct = top_stats.get('percentage', 0)
        paragraphs.append(
            f"<p>The most common customer journey (shown as the Happy Path diagram below) was:<br>"
            f"<strong>{_format_variant(top_variant)}</strong><br>"
            f"— taken by <strong>{pct:.1f}%</strong> of customers.</p>"
        )

    top_bn = bn_summary.get('top_activity_bottleneck')
    top_t_bn = bn_summary.get('top_transition_bottleneck')
    if top_bn:
        sentence = (
            f"<p>The biggest slow-down in the journey happens at the "
            f"<strong>'{html.escape(str(top_bn))}'</strong> step"
        )
        if top_t_bn:
            sentence += f", specifically the move from <strong>'{_format_variant(top_t_bn)}'</strong>"
        sentence += ".</p>"
        paragraphs.append(sentence)

    if biz:
        m = biz.get('metrics', {})
        rate = m.get('repeat_rate', 0)
        buyers = m.get('total_buyers', 0)
        mult = m.get('revenue_stats', {}).get('multiplier', 0)
        aov = m.get('average_order_value', 0)
        sentence = (
            f"<p><strong>{rate:.1f}%</strong> of {buyers:,} identified buyers purchased more than "
            f"once"
        )
        if mult > 0:
            sentence += f", and repeat buyers are worth <strong>{mult:.1f}x</strong> more, on average, than one-time buyers"
        if aov > 0:
            sentence += f". The average order is worth <strong>{aov:,.2f}</strong>"
        sentence += ".</p>"
        paragraphs.append(sentence)

        cart = m.get('cart_abandonment')
        if cart:
            paragraphs.append(
                f"<p><strong>{cart['abandonment_rate']:.1f}%</strong> of sessions that added something "
                f"to their cart didn't go on to buy it — "
                f"<strong>{cart['cases_added_to_cart'] - cart['cases_purchased_after_cart']:,}</strong> "
                f"out of <strong>{cart['cases_added_to_cart']:,}</strong> carts were abandoned.</p>"
            )

    funnel = results.get('funnel_analysis')
    if funnel and funnel.get('stages'):
        drop_step = funnel.get('biggest_drop_off')
        if drop_step:
            drop_pct = funnel['stages'][drop_step]['drop_off_pct']
            paragraphs.append(
                f"<p>Across the customer journey's funnel steps, the biggest single drop-off is at "
                f"<strong>'{html.escape(str(drop_step))}'</strong>, where <strong>{drop_pct:.0f}%</strong> "
                f"of customers who reached that point didn't continue.</p>"
            )

    recommendations = stats.get('recommendations', [])
    rec_html = ""
    if recommendations:
        items = "".join(f"<li>{html.escape(r)}</li>" for r in recommendations)
        rec_html = f"<p><strong>What this means:</strong></p><ul>{items}</ul>"

    return f"""
<h2>Executive Summary</h2>
<div class="summary-box">
  <span class="badge {verdict_cls}">{html.escape(verdict_label)}</span>
  {''.join(paragraphs)}
  {rec_html}
</div>
"""


def generate_html_report(results: Dict[str, Any]) -> str:
    """
    Builds a single, self-contained HTML report from a run_full_analysis()
    results dict, for sharing outside a live Streamlit session. Images are
    embedded as base64 data URIs so the file has no external dependencies.

    Opens with a plain-language Executive Summary aimed at non-technical
    stakeholders, ahead of the detailed technical tables below it.
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

    narrative_summary = _build_narrative_summary(results) if results else ""

    business_section = ""
    if biz:
        biz_metrics = biz.get('metrics', {}) or {}
        cart = biz_metrics.get('cart_abandonment')
        category_breakdown = biz_metrics.get('category_breakdown', {}) or {}

        charts = biz.get('charts', {}) or {}
        chart_captions = [
            ('distribution', 'Orders per Customer'),
            ('timing', 'Time Between Purchases'),
            ('revenue', 'Avg. Lifetime Value'),
            ('category', 'Revenue by Category'),
            ('trend', 'Revenue Over Time'),
        ]
        chart_divs = [
            f'<div><h3>{caption}</h3><img class="zoomable" title="Click to enlarge" src="{img}"></div>'
            for key, caption in chart_captions
            if (img := _embed_image(charts.get(key)))
        ]
        charts_html = f'<div class="images">{"".join(chart_divs)}</div>' if chart_divs else ""

        extra_metrics_html = f"""
<div class="metrics">
  <div class="metric"><div class="label">Average Order Value</div><div class="value">{biz_metrics.get('average_order_value', 0):,.2f}</div></div>
  <div class="metric"><div class="label">Cart Abandonment</div><div class="value">{f"{cart['abandonment_rate']:.1f}%" if cart else 'N/A'}</div></div>
</div>
"""

        category_table_html = ""
        if category_breakdown:
            category_table_html = f"""
<h3>Revenue by Category</h3>
<table>
<tr><th>Category</th><th>Revenue</th><th>Orders</th></tr>
{_table_rows(category_breakdown, ['revenue', 'orders'])}
</table>
"""

        business_section = f"""
<h2>Business Insights</h2>
{extra_metrics_html}
{charts_html}
{category_table_html}
<pre>{html.escape(format_business_report(biz))}</pre>
"""

    funnel = results.get('funnel_analysis')
    funnel_section = ""
    if funnel and funnel.get('stages'):
        funnel_rows = "".join(
            f"<tr><td>{html.escape(str(step))}</td><td>{s['cases_reached']:,}</td>"
            f"<td>{s['pct_of_total']:.1f}%</td><td>{s['pct_of_previous_stage']:.1f}%</td>"
            f"<td>{s['drop_off_pct']:.1f}%</td></tr>"
            for step, s in funnel['stages'].items()
        )
        funnel_section = f"""
<h2>Conversion Funnel</h2>
<table>
<tr><th>Stage</th><th>Cases Reached</th><th>% of Total</th><th>% of Previous Stage</th><th>Drop-off</th></tr>
{funnel_rows}
</table>
"""

    body = f"""<h1>PRoX Process Mining Report</h1>

{narrative_summary}

<div class="metrics">
  <div class="metric"><div class="label">Cases</div><div class="value">{summary.get('Number of Cases', 0):,}</div></div>
  <div class="metric"><div class="label">Events</div><div class="value">{summary.get('Number of Events', 0):,}</div></div>
  <div class="metric"><div class="label">Activities</div><div class="value">{summary.get('Number of Unique Activities', 0)}</div></div>
  <div class="metric"><div class="label">Duration (days)</div><div class="value">{summary.get('Total Duration (Days)', 0)}</div></div>
  <div class="metric"><div class="label">Health Score</div><div class="value">{stats.get('process_health_score', 0):.0f}/100</div></div>
</div>

<h2>Process Maps</h2>
<div class="images">
  <div><h3>Happy Path</h3>{f'<img class="zoomable" title="Click to enlarge" src="{happy_path_img}">' if happy_path_img else '<p>Not available.</p>'}</div>
  <div><h3>Main Process Flow</h3>{f'<img class="zoomable" title="Click to enlarge" src="{main_flow_img}">' if main_flow_img else '<p>Not available.</p>'}</div>
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
{funnel_section}
{business_section}"""

    return _html_shell("PRoX Process Mining Report", body)


def _format_comparison_cell(column: str, value: Any) -> str:
    """Formats a comparison_table cell for display, matching each metric's natural unit."""
    if column in ('fitness_score', 'precision_score'):
        try:
            return f"{float(value):.1%}"
        except (TypeError, ValueError):
            return str(value)
    if column == 'health_score':
        try:
            return f"{float(value):.0f}"
        except (TypeError, ValueError):
            return str(value)
    if column == 'repeat_rate':
        try:
            return f"{float(value):.1f}%"
        except (TypeError, ValueError):
            return str(value)
    if column == 'top_variant':
        return str(value).replace(' -> ', ' → ')
    return str(value)


def generate_segment_comparison_report(segment_result: Dict[str, Any]) -> str:
    """
    Builds a self-contained HTML report from a compare_segments() results dict,
    for sharing outside a live Streamlit session. Opens with a plain-language
    summary of which segment performs best/worst, followed by the comparison
    table and each segment's Happy Path diagram.
    """
    comparison_table = segment_result.get('comparison_table', {}) or {}
    segments = segment_result.get('segments', {}) or {}

    if not comparison_table:
        return _html_shell(
            "PRoX Segment Comparison Report",
            "<h1>PRoX Segment Comparison Report</h1><p>No segment data available.</p>",
        )

    best_health = max(comparison_table.items(), key=lambda kv: kv[1].get('health_score', 0))
    worst_health = min(comparison_table.items(), key=lambda kv: kv[1].get('health_score', 0))
    best_repeat = max(comparison_table.items(), key=lambda kv: kv[1].get('repeat_rate', 0))

    narrative = f"""
<h2>Executive Summary</h2>
<div class="summary-box">
<p>Comparing <strong>{len(comparison_table)}</strong> segments: <strong>{html.escape(str(best_health[0]))}</strong>
has the healthiest process (health score <strong>{best_health[1].get('health_score', 0):.0f}/100</strong>), while
<strong>{html.escape(str(worst_health[0]))}</strong> shows the most friction
(health score <strong>{worst_health[1].get('health_score', 0):.0f}/100</strong>).</p>
<p><strong>{html.escape(str(best_repeat[0]))}</strong> has the highest repeat purchase rate, at
<strong>{best_repeat[1].get('repeat_rate', 0):.1f}%</strong>.</p>
</div>
"""

    columns = ['cases', 'health_score', 'fitness_score', 'precision_score', 'repeat_rate', 'top_variant']
    headers = ['Cases', 'Health Score', 'Fitness', 'Precision', 'Repeat Rate', 'Most Common Path']
    rows = []
    for value, row in comparison_table.items():
        cells = "".join(
            f"<td>{html.escape(_format_comparison_cell(col, row.get(col, '')))}</td>" for col in columns
        )
        rows.append(f"<tr><td>{html.escape(str(value))}</td>{cells}</tr>")

    table_html = f"""
<h2>Segment Comparison</h2>
<table>
<tr><th>Segment</th>{"".join(f"<th>{html.escape(h)}</th>" for h in headers)}</tr>
{"".join(rows) if rows else '<tr><td colspan="99">No data available.</td></tr>'}
</table>
"""

    image_divs = [
        f'<div><h3>{html.escape(str(value))}</h3>'
        + (f'<img class="zoomable" title="Click to enlarge" src="{img}">' if (img := _embed_image(seg_results.get('visualizations', {}).get('happy_path'))) else '<p>Not available.</p>')
        + '</div>'
        for value, seg_results in segments.items()
    ]
    images_html = (
        f'<h2>Happy Path per Segment</h2><div class="images">{"".join(image_divs)}</div>'
        if image_divs else ""
    )

    body = f"<h1>PRoX Segment Comparison Report</h1>{narrative}{table_html}{images_html}"
    return _html_shell("PRoX Segment Comparison Report", body)
