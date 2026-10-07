"""
Directly-follows process graph for the interactive process explorer.

The static BPMN maps (visualizer.py) come from a process tree, so their nodes
include gateways and their edges don't correspond to observed transitions.
Click-to-filter needs the opposite: nodes that are activities and edges that
are transitions actually seen in the log. This module builds that graph as
plain JSON-serialisable dicts - no Graphviz, no pm4py - so it can be
computed in the pipeline, pickled with saved runs and handed to a front-end
component as-is.

Graph shape
-----------
{
  "time_unit":  "minutes",
  "n_cases":    int,
  "nodes": [{"id", "label", "kind", "cases", "events"}, ...],
  "edges": [{"source", "target", "frequency", "cases", "mean_time",
             "median_time", "happy"}, ...],
  "happy_path": [activity, ...],     # most frequent variant, in order
}

`kind` is "activity", "start" or "end". Start/end are synthetic nodes joined
to each case's first/last activity; their edges carry a frequency but no
times (there's no preceding or following event to measure against).
"""
import logging
import math
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

START_ID = "<start>"
END_ID = "<end>"

_TIME_DIVISORS = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}


def _empty_graph(time_unit: str) -> Dict[str, Any]:
    return {"time_unit": time_unit, "n_cases": 0, "nodes": [], "edges": [], "happy_path": []}


def build_process_graph(event_log_df: pd.DataFrame, time_unit: str = "minutes") -> Dict[str, Any]:
    """
    Builds the full (unpruned) directly-follows graph of an event log.

    Edge times are the gap between an event and the next one in the same case,
    in `time_unit`. An empty or malformed log gives an empty graph rather than
    raising, so a failure here can never abort the pipeline.
    """
    if time_unit not in _TIME_DIVISORS:
        logger.warning("Unknown time unit '%s' for process graph. Defaulting to minutes.", time_unit)
        time_unit = "minutes"

    required = ["case:concept:name", "concept:name", "time:timestamp"]
    if (
        event_log_df is None
        or event_log_df.empty
        or any(c not in event_log_df.columns for c in required)
    ):
        return _empty_graph(time_unit)

    # Only the three columns needed; a stable sort keeps the original order of
    # events that share a timestamp.
    df = event_log_df[required].copy()
    df["concept:name"] = df["concept:name"].astype(str)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")
    df = df.dropna(subset=["time:timestamp"])
    if df.empty:
        return _empty_graph(time_unit)
    df = df.sort_values(["case:concept:name", "time:timestamp"], kind="stable")

    case_col = df["case:concept:name"]
    n_cases = int(case_col.nunique())

    # --- Nodes ---
    events_per_activity = df["concept:name"].value_counts()
    cases_per_activity = df.groupby("concept:name")["case:concept:name"].nunique()
    nodes: List[Dict[str, Any]] = [
        {"id": START_ID, "label": "Start", "kind": "start", "cases": n_cases, "events": n_cases},
        {"id": END_ID, "label": "End", "kind": "end", "cases": n_cases, "events": n_cases},
    ]
    for activity, n_events in events_per_activity.items():
        nodes.append({
            "id": activity,
            "label": activity,
            "kind": "activity",
            "cases": int(cases_per_activity[activity]),
            "events": int(n_events),
        })

    # --- Happy path (most frequent variant) ---
    variants = df.groupby("case:concept:name", sort=False)["concept:name"].agg(tuple)
    happy_path = list(variants.value_counts().index[0])
    happy_edges = set(zip([START_ID, *happy_path], [*happy_path, END_ID]))

    # --- Activity -> activity edges ---
    next_activity = df.groupby("case:concept:name", sort=False)["concept:name"].shift(-1)
    next_timestamp = df.groupby("case:concept:name", sort=False)["time:timestamp"].shift(-1)
    has_next = next_activity.notna()
    transitions = pd.DataFrame({
        "case": case_col[has_next],
        "source": df["concept:name"][has_next],
        "target": next_activity[has_next],
        "seconds": (next_timestamp[has_next] - df["time:timestamp"][has_next]).dt.total_seconds(),
    })
    divisor = _TIME_DIVISORS[time_unit]
    edges: List[Dict[str, Any]] = []
    if not transitions.empty:
        grouped = transitions.groupby(["source", "target"]).agg(
            frequency=("seconds", "size"),
            cases=("case", "nunique"),
            mean_time=("seconds", "mean"),
            median_time=("seconds", "median"),
        )
        for (source, target), row in grouped.iterrows():
            edges.append({
                "source": source,
                "target": target,
                "frequency": int(row["frequency"]),
                "cases": int(row["cases"]),
                "mean_time": float(row["mean_time"]) / divisor,
                "median_time": float(row["median_time"]) / divisor,
                "happy": (source, target) in happy_edges,
            })

    # --- Start / end edges: one per case, so frequency == cases ---
    first_activity = df.groupby("case:concept:name", sort=False)["concept:name"].first()
    last_activity = df.groupby("case:concept:name", sort=False)["concept:name"].last()
    for activity, count in first_activity.value_counts().items():
        edges.append({
            "source": START_ID, "target": activity, "frequency": int(count), "cases": int(count),
            "mean_time": None, "median_time": None, "happy": (START_ID, activity) in happy_edges,
        })
    for activity, count in last_activity.value_counts().items():
        edges.append({
            "source": activity, "target": END_ID, "frequency": int(count), "cases": int(count),
            "mean_time": None, "median_time": None, "happy": (activity, END_ID) in happy_edges,
        })

    return {
        "time_unit": time_unit,
        "n_cases": n_cases,
        "nodes": nodes,
        "edges": edges,
        "happy_path": happy_path,
    }


def prune_process_graph(
    graph: Dict[str, Any],
    activity_pct: float = 100.0,
    edge_pct: float = 100.0,
) -> Dict[str, Any]:
    """
    Reduces a graph to its most frequent parts, so a high-variety clickstream
    log doesn't render as an unreadable hairball.

    activity_pct: percentage of activities kept, ranked by event count.
    edge_pct:     percentage of the remaining edges kept, ranked by frequency.

    The happy path's activities and edges are always kept, so the dominant
    route stays connected end to end whatever the sliders are set to. Other
    paths are *not* bridged: if an activity is dropped, the edges into and out
    of it go with it rather than being rerouted around it. An activity left
    with no edges is dropped too (the synthetic start/end are always kept).
    Returns a new graph; the input is not modified.
    """
    activity_pct = min(max(float(activity_pct), 0.0), 100.0)
    edge_pct = min(max(float(edge_pct), 0.0), 100.0)

    activities = sorted(
        (n for n in graph["nodes"] if n["kind"] == "activity"),
        key=lambda n: (-n["events"], n["id"]),
    )
    keep_count = max(1, math.ceil(len(activities) * activity_pct / 100)) if activities else 0
    keep_ids = {n["id"] for n in activities[:keep_count]}
    keep_ids.update(graph["happy_path"])
    keep_ids.update({START_ID, END_ID})

    candidate_edges = [e for e in graph["edges"] if e["source"] in keep_ids and e["target"] in keep_ids]
    optional = sorted((e for e in candidate_edges if not e["happy"]), key=lambda e: -e["frequency"])
    edge_keep_count = math.ceil(len(optional) * edge_pct / 100)
    kept_optional = {id(e) for e in optional[:edge_keep_count]}
    kept_edges = [e for e in candidate_edges if e["happy"] or id(e) in kept_optional]

    connected = {e["source"] for e in kept_edges} | {e["target"] for e in kept_edges}
    kept_nodes = [
        n for n in graph["nodes"]
        if n["kind"] != "activity" or n["id"] in connected
    ]
    return {**graph, "nodes": kept_nodes, "edges": kept_edges}
