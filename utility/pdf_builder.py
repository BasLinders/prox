"""
pdf_builder.py

Modular PDF report builder for PRoX. Unlike `generate_html_report()` (in
prox/report.py), which always bundles every section into one fixed HTML
file, this lets the user opt into which result tabs to include via
checkboxes - so a shareable PDF can be scoped to just what's relevant
(e.g. only Business Insights + Session Insights) instead of everything.

Built with reportlab: pure-Python, no system-level rendering dependency
(no wkhtmltopdf binary, no Cairo/Pango), consistent with PRoX's "runs on a
standard laptop" design goal. Kept out of prox/report.py and out of the
prox/ package entirely, since prox/ is a Streamlit-free analysis engine and
this module both renders its own Streamlit UI and depends on reportlab, a
dependency none of the rest of prox/ needs.

Public entry point: render_pdf_builder(results, segment_result=None,
ai_conclusion=None, predictive=None) - draws the section checkboxes + Generate/Download flow
in the current Streamlit container.
"""

from __future__ import annotations

import html
import io
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, Preformatted, KeepTogether,
)

from prox.analytics import format_business_report
from prox.loops import LOOP_INFERENCE_NOTE

_PAGE_SIZE = letter
_MARGIN = 0.75 * inch
_CONTENT_WIDTH = _PAGE_SIZE[0] - 2 * _MARGIN
_MAX_TABLE_ROWS = 20
_MAX_JOURNEY_ROWS = 25


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="PRoXTitle", parent=styles["Title"], fontSize=24, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="PRoXSubtitle", parent=styles["Normal"], fontSize=11,
        textColor=colors.HexColor("#666666"), spaceAfter=24,
    ))
    styles.add(ParagraphStyle(
        name="PRoXSectionHeading", parent=styles["Heading1"], fontSize=16,
        spaceBefore=6, spaceAfter=10, textColor=colors.HexColor("#1a1a1a"),
    ))
    return styles


def _heading(text: str, styles) -> Paragraph:
    return Paragraph(text, styles["PRoXSectionHeading"])


def _subheading(text: str, styles) -> Paragraph:
    return Paragraph(text, styles["Heading3"])


def _scaled_image(path: str, max_width: float = _CONTENT_WIDTH, max_height: float = 4.5 * inch):
    """Loads a local image and scales it to fit within max_width/max_height,
    preserving aspect ratio. Returns None if the file can't be read."""
    if not path or not os.path.exists(path):
        return None
    try:
        reader = ImageReader(path)
        iw, ih = reader.getSize()
    except Exception:
        return None
    if iw <= 0 or ih <= 0:
        return None
    scale = min(max_width / iw, max_height / ih, 1.0)
    return Image(path, width=iw * scale, height=ih * scale)


def _metrics_table(pairs: List[tuple], styles) -> Table:
    """A compact label/value table for headline metrics, laid out in pairs
    per row (two metrics side by side) to save vertical space."""
    rows = []
    for i in range(0, len(pairs), 2):
        row = []
        for label, value in pairs[i:i + 2]:
            safe_label, safe_value = html.escape(str(label)), html.escape(str(value))
            row.append(Paragraph(f"<b>{safe_value}</b><br/><font size=8 color='#666666'>{safe_label}</font>", styles["Normal"]))
        if len(row) == 1:
            row.append("")
        rows.append(row)
    table = Table(rows, colWidths=[_CONTENT_WIDTH / 2] * 2)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f5f5")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return table


def _dict_table(data: Dict[str, Dict[str, Any]], columns: List[str], headers: List[str], max_rows: int = _MAX_TABLE_ROWS) -> Optional[Table]:
    """Renders a {name: {col: value}} dict (as produced throughout
    prox/analytics.py) into a reportlab Table, truncated to max_rows with a
    trailing note - mirrors _table_rows() in prox/report.py but for PDF."""
    if not data:
        return None
    header_row = ["Name"] + headers
    body_rows = []
    items = list(data.items())
    for name, stats in items[:max_rows]:
        row = [str(name)] + [_fmt(stats.get(c, "")) for c in columns]
        body_rows.append(row)
    table = Table([header_row] + body_rows, colWidths=[_CONTENT_WIDTH * 0.4] + [_CONTENT_WIDTH * 0.6 / len(columns)] * len(columns))
    table.setStyle(_table_style())
    return table


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def _table_style() -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f5f5")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])


def _df_table(df: pd.DataFrame, max_rows: int = _MAX_JOURNEY_ROWS) -> Optional[Table]:
    if df is None or df.empty:
        return None
    display_df = df.head(max_rows)
    header_row = [str(c) for c in display_df.columns]
    body_rows = [[_fmt(v) for v in row] for row in display_df.itertuples(index=False)]
    n_cols = len(header_row)
    table = Table([header_row] + body_rows, colWidths=[_CONTENT_WIDTH / n_cols] * n_cols)
    table.setStyle(_table_style())
    return table


def _truncation_note(total: int, shown: int, styles) -> Optional[Paragraph]:
    if total <= shown:
        return None
    return Paragraph(
        f"<i>Showing {shown} of {total:,} rows.</i>",
        ParagraphStyle(name="Note", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#888888")),
    )


# ---------------------------------------------------------------------------
# Section builders - each takes (results, segment_result, styles) and
# returns a list of flowables. Called only when the section has data AND
# the user checked it, so an unchecked or empty tab contributes nothing.
# ---------------------------------------------------------------------------

def _section_ai_conclusion(conclusion: Dict[str, Any], styles) -> List:
    """The AI-generated summary, key findings and next steps (see
    utility/ai_client.py). Kept outside _SECTION_DEFS since it doesn't come
    from `results` - it's passed to build_pdf_report() separately, and
    always goes first, as the report's executive summary."""
    bullet = ParagraphStyle(name="PRoXBullet", parent=styles["Normal"], leftIndent=16, bulletIndent=4, spaceAfter=4)
    story = [_heading("AI Summary & Next Steps", styles)]
    for paragraph in (conclusion.get("summary") or "").split("\n\n"):
        if paragraph.strip():
            story.append(Paragraph(html.escape(paragraph.strip()).replace("\n", "<br/>"), styles["Normal"]))
            story.append(Spacer(1, 6))

    findings = conclusion.get("key_findings") or []
    if findings:
        story.append(_subheading("Key Findings", styles))
        for finding in findings:
            story.append(Paragraph(html.escape(finding), bullet, bulletText="•"))

    next_steps = conclusion.get("next_steps") or []
    if next_steps:
        story.append(_subheading("Next Steps", styles))
        for i, step in enumerate(next_steps, start=1):
            story.append(Paragraph(html.escape(step), bullet, bulletText=f"{i}."))

    model = conclusion.get("model")
    generated_at = conclusion.get("generated_at")
    provenance = "Generated by AI" + (f" ({html.escape(model)})" if model else "")
    if generated_at:
        provenance += f" on {html.escape(generated_at)}"
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"<i>{provenance}, from the aggregate results in this report. Check it against the figures in the "
        "sections that follow before acting on it.</i>",
        ParagraphStyle(name="PRoXAINote", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#888888")),
    ))
    return story


def _section_predictive(predictive: Dict[str, Any], styles) -> List:
    """The Predictive Insights tab's trained propensity model: its
    cross-validated metrics, top drivers and in-progress score summary. Like
    the AI conclusion, kept outside _SECTION_DEFS since the model lives in
    session state rather than `results` - passed to build_pdf_report()
    separately as {"model_bundle", "drivers", "summary"}."""
    bundle = predictive.get("model_bundle") or {}
    metrics = bundle.get("metrics") or {}
    training = bundle.get("training_config") or {}
    bullet = ParagraphStyle(name="PRoXPredictiveBullet", parent=styles["Normal"], leftIndent=16, bulletIndent=4, spaceAfter=4)
    note = ParagraphStyle(name="PRoXPredictiveNote", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#888888"))

    outcome = training.get("outcome_activity")
    outcomes = outcome if isinstance(outcome, (list, tuple)) else [outcome]
    attributes = training.get("case_attribute_cols") or []
    story = [_heading("Predictive Insights", styles)]
    story.append(Paragraph(
        "A logistic regression model predicting whether a case reaches "
        f"<b>{html.escape(' or '.join(str(o) for o in outcomes))}</b>, trained on completed cases"
        + (f" using case attributes {html.escape(', '.join(attributes))}" if attributes else "")
        + ". A prediction, not a measurement.",
        styles["Normal"],
    ))
    story.append(Spacer(1, 6))

    story.append(_subheading("Validation", styles))
    story.append(_metrics_table([
        ("Completed cases", f"{metrics.get('n_completed', 0):,}"),
        ("Reached / didn't reach the outcome", f"{metrics.get('n_positive', 0):,} / {metrics.get('n_negative', 0):,}"),
        ("Accuracy", f"{metrics['accuracy_mean']:.1%} ± {metrics['accuracy_std']:.1%}"),
        ("Precision", f"{metrics['precision_mean']:.1%} ± {metrics['precision_std']:.1%}"),
        ("Recall", f"{metrics['recall_mean']:.1%} ± {metrics['recall_std']:.1%}"),
        ("ROC AUC", f"{metrics['roc_auc_mean']:.2f} ± {metrics['roc_auc_std']:.2f}"),
    ], styles))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Stratified k-fold cross-validation. The model scoring in-progress cases is refit on all "
        "completed cases afterward, so these metrics describe the cross-validation process's typical "
        "performance, not literally that exact refit model.",
        note,
    ))

    story.append(Spacer(1, 16))
    story.append(_subheading("Top Drivers", styles))
    drivers = predictive.get("drivers") or []
    if drivers:
        story.append(Paragraph(
            "<i>Associative, not causal - what tends to go together with reaching the outcome.</i>",
            styles["Normal"],
        ))
        story.append(Spacer(1, 4))
        for sentence in drivers:
            story.append(Paragraph(html.escape(sentence), bullet, bulletText="•"))
    else:
        story.append(Paragraph("No drivers to report.", styles["Normal"]))

    story.append(Spacer(1, 16))
    story.append(_subheading("In-Progress Cases", styles))
    summary = predictive.get("summary") or {}
    if summary.get("n_scored"):
        hist_rate = summary.get("historical_positive_rate")
        story.append(_metrics_table([
            ("In-progress cases scored", f"{summary['n_scored']:,}"),
            ("Average propensity", f"{summary['mean_score']:.1%}"),
            ("Historical conversion rate", f"{hist_rate:.1%}" if hist_rate is not None else "N/A"),
            ("Median propensity (IQR)", f"{summary['median_score']:.1%} ({summary['q1_score']:.1%} - {summary['q3_score']:.1%})"),
        ], styles))
        story.append(Spacer(1, 6))
        tiers = summary.get("risk_tiers") or {}
        tier_table = Table(
            [["Propensity tier", "Cases"]] + [
                [label, f"{tiers.get(key, 0):,}"]
                for key, label in [
                    ("high_risk", "High risk (lowest-scoring third)"),
                    ("medium_risk", "Medium risk"),
                    ("low_risk", "Low risk (highest-scoring third)"),
                ]
            ],
            colWidths=[_CONTENT_WIDTH * 0.7, _CONTENT_WIDTH * 0.3],
        )
        tier_table.setStyle(_table_style())
        story.append(tier_table)
        story.append(Spacer(1, 6))
        for sentence in summary.get("narrative") or []:
            story.append(Paragraph(html.escape(sentence), bullet, bulletText="•"))
    else:
        story.append(Paragraph(
            "No in-progress cases to score - every case in the log has either reached the outcome or aged out.",
            styles["Normal"],
        ))
    return story


def _section_process_maps(results, segment_result, styles) -> List:
    viz = results.get("visualizations", {}) or {}
    happy, main_flow = viz.get("happy_path"), viz.get("bottlenecks")
    story = [_heading("Process Maps", styles)]
    added = False
    for caption, path in [("Happy Path (most frequent variant)", happy), ("Main Process Flow", main_flow)]:
        img = _scaled_image(path)
        if img:
            story.append(_subheading(caption, styles))
            story.append(img)
            story.append(Spacer(1, 12))
            added = True
    if not added:
        story.append(Paragraph("Not available.", styles["Normal"]))
    return story


def _section_variants(results, segment_result, styles) -> List:
    top_variants = (results.get("performance", {}).get("variant_performance", {}) or {}).get("top_variants", {})
    story = [_heading("Variants", styles)]
    table = _dict_table(top_variants, ["frequency", "percentage"], ["Frequency", "Percentage"])
    if table is None:
        story.append(Paragraph("No variant data available.", styles["Normal"]))
    else:
        story.append(table)
        note = _truncation_note(len(top_variants), min(len(top_variants), _MAX_TABLE_ROWS), styles)
        if note:
            story.append(Spacer(1, 6))
            story.append(note)

    outcomes = results.get("variant_outcomes") or {}
    routes = outcomes.get("routes") or {}
    if routes:
        has_revenue = "total_revenue" in (outcomes.get("summary") or {})
        cols = ["cases", "conversion_rate"] + (["revenue", "revenue_share", "revenue_per_case"] if has_revenue else [])
        headers = ["Cases", "Conversion (%)"] + (["Revenue", "Revenue Share (%)", "Revenue per Case"] if has_revenue else [])
        story.append(Spacer(1, 16))
        story.append(_subheading("Conversion and revenue per route", styles))
        story.append(Paragraph(
            "A route is a variant cut off at the first purchase, so buyers and non-buyers who took "
            "the same steps count as one route.",
            styles["Normal"],
        ))
        story.append(Spacer(1, 6))
        story.append(_dict_table(routes, cols, headers))
        note = _truncation_note(len(routes), min(len(routes), _MAX_TABLE_ROWS), styles)
        if note:
            story.append(Spacer(1, 6))
            story.append(note)
    return story


def _section_bottlenecks(results, segment_result, styles) -> List:
    bottlenecks = results.get("performance", {}).get("bottlenecks", {}) or {}
    activity_bn = bottlenecks.get("activity_bottlenecks", {})
    transition_bn = bottlenecks.get("transition_bottlenecks", {})
    cols, headers = ["mean_duration", "frequency", "impact_score", "severity"], ["Mean Duration", "Frequency", "Impact", "Severity"]

    story = [_heading("Bottlenecks", styles)]
    story.append(_subheading("Activity Bottlenecks", styles))
    table = _dict_table(activity_bn, cols, headers)
    story.append(table if table else Paragraph("No activity bottlenecks found.", styles["Normal"]))
    story.append(Spacer(1, 16))
    story.append(_subheading("Slowest Transitions", styles))
    table = _dict_table(transition_bn, cols, headers)
    story.append(table if table else Paragraph("No transition bottlenecks found.", styles["Normal"]))
    return story


def _section_conformance(results, segment_result, styles) -> List:
    overall = results.get("conformance", {}).get("overall_summary", {}) or {}
    story = [_heading("Conformance", styles)]
    if not overall:
        story.append(Paragraph("No conformance data available.", styles["Normal"]))
        return story
    pairs = [
        ("Fitness", f"{overall.get('fitness_score', 0):.1%}"),
        ("Precision", f"{overall.get('precision_score', 0):.1%}"),
        ("Quality", str(overall.get("quality_assessment", "N/A"))),
    ]
    story.append(_metrics_table(pairs, styles))
    return story


def _section_funnel(results, segment_result, styles) -> List:
    funnel = results.get("funnel_analysis") or {}
    stages = funnel.get("stages", {})
    story = [_heading("Conversion Funnel", styles)]
    if not stages:
        story.append(Paragraph("No funnel data available.", styles["Normal"]))
        return story
    header_row = ["Stage", "Cases Reached", "% of Total", "% of Previous", "Drop-off"]
    body_rows = [
        [str(step), f"{s['cases_reached']:,}", f"{s['pct_of_total']:.1f}%",
         f"{s['pct_of_previous_stage']:.1f}%", f"{s['drop_off_pct']:.1f}%"]
        for step, s in stages.items()
    ]
    table = Table([header_row] + body_rows, colWidths=[_CONTENT_WIDTH * w for w in (0.3, 0.2, 0.175, 0.175, 0.15)])
    table.setStyle(_table_style())
    story.append(table)
    return story


def _section_business(results, segment_result, styles) -> List:
    biz = results.get("repeat_purchase_analysis")
    story = [_heading("Business Insights", styles)]
    if not biz:
        story.append(Paragraph("No business insight data available.", styles["Normal"]))
        return story

    metrics = biz.get("metrics", {}) or {}
    cart = metrics.get("cart_abandonment")
    pairs = [
        ("Total Buyers", f"{metrics.get('total_buyers', 0):,}"),
        ("Repeat Rate", f"{metrics.get('repeat_rate', 0):.1f}%"),
        ("Average Order Value", f"{metrics.get('average_order_value', 0):,.2f}"),
        ("Cart Abandonment", f"{cart['abandonment_rate']:.1f}%" if cart else "N/A"),
    ]
    story.append(_metrics_table(pairs, styles))
    story.append(Spacer(1, 12))

    charts = {k: v for k, v in (biz.get("charts", {}) or {}).items() if v and os.path.exists(v)}
    chart_captions = {
        "distribution": "Orders per Customer", "timing": "Time Between Purchases",
        "revenue": "Avg. Lifetime Value", "category": "Revenue by Category",
        "trend": "Revenue Over Time",
    }
    for key, path in charts.items():
        img = _scaled_image(path, max_height=3 * inch)
        if img:
            story.append(_subheading(chart_captions.get(key, key.title()), styles))
            story.append(img)
            story.append(Spacer(1, 10))

    category_breakdown = metrics.get("category_breakdown", {}) or {}
    if category_breakdown:
        story.append(_subheading("Revenue by Category", styles))
        table = _dict_table(category_breakdown, ["revenue", "orders"], ["Revenue", "Orders"])
        if table:
            story.append(table)
            story.append(Spacer(1, 12))

    story.append(_subheading("Summary", styles))
    story.append(Preformatted(format_business_report(biz), styles["Code"]))
    return story


def _section_sessions(results, segment_result, styles) -> List:
    session_insights = results.get("session_insights")
    story = [_heading("Session Insights", styles)]
    if not session_insights:
        story.append(Paragraph("No session insight data available.", styles["Normal"]))
        return story

    sessions_df = session_insights.get("sessions")
    journeys_df = session_insights.get("journeys")

    story.append(Paragraph(
        "Each session is labelled Browsing, Researching, Cart Abandonment, or "
        "Buying from a priority-ordered rule over its activities.",
        styles["Normal"],
    ))
    story.append(Spacer(1, 8))

    if sessions_df is not None and not sessions_df.empty:
        label_counts = sessions_df["label"].value_counts()
        pairs = [(label, f"{count:,}") for label, count in label_counts.items()]
        story.append(_metrics_table(pairs, styles))
        story.append(Spacer(1, 12))

    if journeys_df is not None and not journeys_df.empty:
        story.append(_subheading("Per-User Session Journeys", styles))
        table = _df_table(journeys_df[["user_id", "session_count", "journey"]])
        if table:
            story.append(table)
            note = _truncation_note(len(journeys_df), min(len(journeys_df), _MAX_JOURNEY_ROWS), styles)
            if note:
                story.append(Spacer(1, 6))
                story.append(note)
    return story


def _section_loops(results, segment_result, styles) -> List:
    loops = results.get("loop_analysis") or {}
    summary = loops.get("summary") or {}
    story = [_heading("Loops & Switches", styles)]
    if not summary:
        story.append(Paragraph("No loop data available.", styles["Normal"]))
        return story
    story.append(Paragraph(html.escape(LOOP_INFERENCE_NOTE), styles["Normal"]))
    story.append(Spacer(1, 6))
    story.append(_metrics_table([
        ("Cases with back-navigation", f"{summary['back_nav_case_share']:.1f}%"),
        ("Cases with repeat visits", f"{summary['repeat_case_share']:.1f}%"),
        ("Back-navigation events", f"{summary['back_nav_events']:,}"),
        ("Repeat-visit events", f"{summary['repeat_events']:,}"),
    ], styles))

    patterns = loops.get("back_navigation") or {}
    if patterns:
        story.append(Spacer(1, 16))
        story.append(_subheading("Most common back-navigation", styles))
        story.append(_dict_table(
            patterns,
            ["occurrences", "cases", "case_share"], ["Occurrences", "Cases", "% of Cases"],
        ))

    repeats = loops.get("repeats") or {}
    if repeats:
        story.append(Spacer(1, 16))
        story.append(_subheading("Repeated activities", styles))
        story.append(_dict_table(
            repeats, ["repeat_case_share", "repeat_events", "self_loop_events", "visits_per_case"],
            ["% of Its Cases Repeating", "Repeat Events", "Immediate (A -> A)", "Visits per Case"],
        ))

    after = (loops.get("after_loop") or {}).get("back_navigation") or {}
    if after.get("loops"):
        story.append(Spacer(1, 16))
        story.append(_subheading("After a back-navigation", styles))
        position = after.get("position") or {}
        story.append(Paragraph(
            f"{after['exit_share']:.1f}% of back-navigations end the case. Position in the journey: "
            f"early {position.get('early', 0):.0f}%, middle {position.get('middle', 0):.0f}%, "
            f"late {position.get('late', 0):.0f}%.",
            styles["Normal"],
        ))
        story.append(Spacer(1, 6))
        story.append(_dict_table(after.get("next_activity") or {}, ["count", "share"], ["Count", "Share (%)"]))

    switches = loops.get("group_switches")
    if switches and switches.get("transitions"):
        story.append(Spacer(1, 16))
        story.append(_subheading("Switches between groups", styles))
        story.append(Paragraph(
            f"{switches['summary']['switch_case_share']:.1f}% of cases switch group at least once.",
            styles["Normal"],
        ))
        story.append(Spacer(1, 6))
        story.append(_dict_table(
            switches["transitions"],
            ["occurrences", "cases", "case_share"], ["Occurrences", "Cases", "% of Cases"],
        ))
    return story


def _section_segments(results, segment_result, styles) -> List:
    story = [_heading("Segment Comparison", styles)]
    comparison_table = (segment_result or {}).get("comparison_table", {})
    if not comparison_table:
        story.append(Paragraph("No segment comparison data available.", styles["Normal"]))
        return story

    header_row = ["Segment", "Cases", "Health Score", "Fitness", "Precision", "Repeat Rate"]
    body_rows = []
    for value, row in comparison_table.items():
        body_rows.append([
            str(value), f"{row.get('cases', 0):,}", f"{row.get('health_score', 0):.0f}",
            f"{row.get('fitness_score', 0):.1%}", f"{row.get('precision_score', 0):.1%}",
            f"{row.get('repeat_rate', 0):.1f}%",
        ])
    table = Table([header_row] + body_rows, colWidths=[_CONTENT_WIDTH / len(header_row)] * len(header_row))
    table.setStyle(_table_style())
    story.append(table)
    return story


# key, label, has_data predicate, builder
_SECTION_DEFS: List[tuple] = [
    ("process_maps", "Process Maps",
     lambda r, sr: bool((r.get("visualizations") or {}).get("happy_path") or (r.get("visualizations") or {}).get("bottlenecks")),
     _section_process_maps),
    ("variants", "Variants",
     lambda r, sr: bool((r.get("performance", {}).get("variant_performance", {}) or {}).get("top_variants")),
     _section_variants),
    ("bottlenecks", "Bottlenecks",
     lambda r, sr: bool((r.get("performance", {}).get("bottlenecks", {}) or {}).get("activity_bottlenecks")
                         or (r.get("performance", {}).get("bottlenecks", {}) or {}).get("transition_bottlenecks")),
     _section_bottlenecks),
    ("conformance", "Conformance",
     lambda r, sr: bool(r.get("conformance", {}).get("overall_summary")),
     _section_conformance),
    ("funnel", "Funnel",
     lambda r, sr: bool((r.get("funnel_analysis") or {}).get("stages")),
     _section_funnel),
    ("loops", "Loops & Switches",
     lambda r, sr: bool((r.get("loop_analysis") or {}).get("summary")),
     _section_loops),
    ("business", "Business Insights",
     lambda r, sr: bool(r.get("repeat_purchase_analysis")),
     _section_business),
    ("sessions", "Session Insights",
     lambda r, sr: bool(r.get("session_insights")),
     _section_sessions),
    ("segments", "Segment Comparison",
     lambda r, sr: bool((sr or {}).get("comparison_table")),
     _section_segments),
]


def _available_sections(results: Dict[str, Any], segment_result: Optional[Dict[str, Any]]) -> Dict[str, str]:
    results = results or {}
    return {
        key: label for key, label, has_data, _ in _SECTION_DEFS
        if has_data(results, segment_result)
    }


def _cover_page(section_labels: List[str], results: Dict[str, Any], styles) -> List:
    summary = results.get("log_summary", {}) or {}
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    story = [
        Spacer(1, 1.5 * inch),
        Paragraph("PRoX Process Mining Report", styles["PRoXTitle"]),
        Paragraph(f"Generated {generated}", styles["PRoXSubtitle"]),
        Paragraph(
            f"{summary.get('Number of Cases', 0):,} cases · "
            f"{summary.get('Number of Events', 0):,} events · "
            f"{summary.get('Number of Unique Activities', 0)} activities",
            styles["Normal"],
        ),
        Spacer(1, 24),
        Paragraph("<b>Sections included in this report:</b>", styles["Normal"]),
    ]
    for label in section_labels:
        story.append(Paragraph(f"• {html.escape(str(label))}", styles["Normal"]))
    story.append(PageBreak())
    return story


AI_CONCLUSION_KEY = "ai_conclusion"
AI_CONCLUSION_LABEL = "AI Summary & Next Steps"
PREDICTIVE_KEY = "predictive"
PREDICTIVE_LABEL = "Predictive Insights"


def _has_trained_model(predictive: Optional[Dict[str, Any]]) -> bool:
    return bool(predictive) and (predictive.get("model_bundle") or {}).get("model") is not None


def build_pdf_report(
    results: Dict[str, Any],
    section_keys: List[str],
    segment_result: Optional[Dict[str, Any]] = None,
    ai_conclusion: Optional[Dict[str, Any]] = None,
    predictive: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Builds a PDF containing only the requested sections (each a key from
    _SECTION_DEFS, or AI_CONCLUSION_KEY). Unknown keys, or keys whose section
    has no data, are silently skipped - the caller decides what to offer via
    `list_available_sections`.

    `ai_conclusion` is a parsed ai_client conclusion ({"summary",
    "key_findings", "next_steps"}, plus optional "model"/"generated_at");
    when given and AI_CONCLUSION_KEY is selected, it opens the report.

    `predictive` is {"model_bundle": train_propensity_model() output,
    "drivers": analyze_propensity_drivers() sentences, "summary":
    summarize_propensity_scores() output}; when its model is trained and
    PREDICTIVE_KEY is selected, it closes the report, matching the tab order.

    Returns the PDF as bytes, ready for a Streamlit download_button.
    """
    styles = _styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=_PAGE_SIZE,
        leftMargin=_MARGIN, rightMargin=_MARGIN, topMargin=_MARGIN, bottomMargin=_MARGIN,
        title="PRoX Process Mining Report",
    )

    builders_by_key = {
        key: (label, lambda b=builder: b(results, segment_result, styles))
        for key, label, _, builder in _SECTION_DEFS
    }
    if ai_conclusion:
        builders_by_key[AI_CONCLUSION_KEY] = (AI_CONCLUSION_LABEL, lambda: _section_ai_conclusion(ai_conclusion, styles))
    if _has_trained_model(predictive):
        builders_by_key[PREDICTIVE_KEY] = (PREDICTIVE_LABEL, lambda: _section_predictive(predictive, styles))
    section_order = [AI_CONCLUSION_KEY] + [key for key, *_ in _SECTION_DEFS] + [PREDICTIVE_KEY]
    ordered_keys = [key for key in section_order if key in section_keys and key in builders_by_key]
    section_labels = [builders_by_key[k][0] for k in ordered_keys]

    story: List = _cover_page(section_labels, results, styles)
    for i, key in enumerate(ordered_keys):
        _, build_section = builders_by_key[key]
        section_story = build_section()
        # Keep the section heading glued to its first content flowable so a
        # heading never ends up alone at the bottom of a page.
        glue_count = min(2, len(section_story))
        story.append(KeepTogether(section_story[:glue_count]))
        story.extend(section_story[glue_count:])
        if i < len(ordered_keys) - 1:
            story.append(PageBreak())

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#999999"))
        canvas.drawRightString(_PAGE_SIZE[0] - _MARGIN, 0.5 * inch, f"Page {doc_.page}")
        canvas.drawString(_MARGIN, 0.5 * inch, "PRoX - Process Excavator")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()


def render_pdf_builder(
    results: Optional[Dict[str, Any]],
    segment_result: Optional[Dict[str, Any]] = None,
    ai_conclusion: Optional[Dict[str, Any]] = None,
    predictive: Optional[Dict[str, Any]] = None,
) -> None:
    """Renders the section checkboxes + Generate/Download flow. Call this
    from main.py after `results` (and optionally `segment_result` /
    `ai_conclusion` / `predictive`) exist."""
    st.caption(
        "Pick which sections to include - the PDF only contains what's "
        "checked below, so you can share a focused report (e.g. just "
        "Business + Session Insights) instead of exporting everything."
    )

    available = _available_sections(results or {}, segment_result)
    if available and ai_conclusion:
        available = {AI_CONCLUSION_KEY: AI_CONCLUSION_LABEL, **available}
    if available and _has_trained_model(predictive):
        available[PREDICTIVE_KEY] = PREDICTIVE_LABEL
    if not available:
        st.info("Run an analysis first to build a PDF report.")
        return

    select_all = st.checkbox("Select all", value=True, key="pdf_select_all")
    cols = st.columns(2)
    selected_keys = []
    for i, (key, label) in enumerate(available.items()):
        with cols[i % 2]:
            if st.checkbox(label, value=select_all, key=f"pdf_section_{key}"):
                selected_keys.append(key)

    if not selected_keys:
        st.warning("Select at least one section to include.")
        return

    if st.button("Generate PDF Report", type="primary", width='stretch'):
        with st.spinner("Building PDF..."):
            st.session_state["pdf_report_bytes"] = build_pdf_report(
                results, selected_keys, segment_result, ai_conclusion, predictive,
            )

    pdf_bytes = st.session_state.get("pdf_report_bytes")
    if pdf_bytes:
        st.download_button(
            "Download PDF Report", data=pdf_bytes, file_name="prox_report.pdf",
            mime="application/pdf", width='stretch',
        )
