import pandas as pd

from prox.process_graph import END_ID, START_ID, build_process_graph, prune_process_graph

from conftest import make_event_log, make_simple_variant_log


def _edge(graph, source, target):
    return next((e for e in graph["edges"] if e["source"] == source and e["target"] == target), None)


def _node_ids(graph):
    return {n["id"] for n in graph["nodes"]}


def _mixed_log():
    """3 cases a->b->c, 1 case a->c, 1 case a->d->c: happy path is a,b,c."""
    base = pd.Timestamp("2024-01-01")
    rows = []
    for case, acts in [("1", "abc"), ("2", "abc"), ("3", "abc"), ("4", "ac"), ("5", "adc")]:
        for i, act in enumerate(acts):
            rows.append((case, act, base + pd.Timedelta(minutes=i)))
    return make_event_log(rows)


def test_build_graph_counts_nodes_and_edges():
    graph = build_process_graph(make_simple_variant_log(n_cases=3))

    assert graph["n_cases"] == 3
    assert graph["happy_path"] == ["a", "b", "c"]
    assert _node_ids(graph) == {START_ID, END_ID, "a", "b", "c"}
    node_a = next(n for n in graph["nodes"] if n["id"] == "a")
    assert node_a["cases"] == 3 and node_a["events"] == 3

    ab = _edge(graph, "a", "b")
    assert ab["frequency"] == 3 and ab["cases"] == 3
    assert ab["mean_time"] == 1.0 and ab["median_time"] == 1.0  # one minute apart, in minutes
    assert _edge(graph, START_ID, "a")["frequency"] == 3
    assert _edge(graph, "c", END_ID)["frequency"] == 3
    assert _edge(graph, START_ID, "a")["mean_time"] is None


def test_build_graph_converts_edge_times_to_requested_unit():
    graph = build_process_graph(make_simple_variant_log(n_cases=2), time_unit="seconds")
    assert _edge(graph, "a", "b")["mean_time"] == 60.0


def test_build_graph_unknown_time_unit_falls_back_to_minutes():
    graph = build_process_graph(make_simple_variant_log(n_cases=2), time_unit="fortnights")
    assert graph["time_unit"] == "minutes"


def test_build_graph_flags_happy_path_edges_only():
    graph = build_process_graph(_mixed_log())

    assert graph["happy_path"] == ["a", "b", "c"]
    assert _edge(graph, "a", "b")["happy"]
    assert _edge(graph, START_ID, "a")["happy"]
    assert _edge(graph, "c", END_ID)["happy"]
    assert not _edge(graph, "a", "c")["happy"]
    assert not _edge(graph, "a", "d")["happy"]


def test_build_graph_orders_events_by_timestamp_not_row_order():
    base = pd.Timestamp("2024-01-01")
    log = make_event_log([
        ("1", "c", base + pd.Timedelta(minutes=2)),
        ("1", "a", base),
        ("1", "b", base + pd.Timedelta(minutes=1)),
    ])
    graph = build_process_graph(log)
    assert graph["happy_path"] == ["a", "b", "c"]


def test_build_graph_keeps_self_loops():
    base = pd.Timestamp("2024-01-01")
    log = make_event_log([("1", "a", base), ("1", "a", base + pd.Timedelta(minutes=1))])
    loop = _edge(build_process_graph(log), "a", "a")
    assert loop["frequency"] == 1


def test_build_graph_single_event_case_connects_start_to_end():
    log = make_event_log([("1", "a", pd.Timestamp("2024-01-01"))])
    graph = build_process_graph(log)
    assert _edge(graph, START_ID, "a")["frequency"] == 1
    assert _edge(graph, "a", END_ID)["frequency"] == 1


def test_build_graph_stringifies_non_string_activities():
    base = pd.Timestamp("2024-01-01")
    log = make_event_log([("1", 1, base), ("1", 2, base + pd.Timedelta(minutes=1))])
    assert _edge(build_process_graph(log), "1", "2") is not None


def test_build_graph_handles_empty_and_malformed_input():
    for bad in (None, pd.DataFrame(), pd.DataFrame({"x": [1]})):
        graph = build_process_graph(bad)
        assert graph["nodes"] == [] and graph["edges"] == []


def test_build_graph_is_json_serialisable():
    import json
    json.dumps(build_process_graph(_mixed_log()))


def test_prune_at_full_percentages_keeps_everything():
    graph = build_process_graph(_mixed_log())
    pruned = prune_process_graph(graph, 100, 100)
    assert _node_ids(pruned) == _node_ids(graph)
    assert len(pruned["edges"]) == len(graph["edges"])


def test_prune_drops_rare_activity_but_keeps_happy_path():
    graph = build_process_graph(_mixed_log())
    pruned = prune_process_graph(graph, activity_pct=10, edge_pct=100)

    ids = _node_ids(pruned)
    assert "d" not in ids
    assert {"a", "b", "c", START_ID, END_ID} <= ids
    assert _edge(pruned, "a", "d") is None and _edge(pruned, "d", "c") is None
    assert _edge(pruned, "a", "c") is not None  # both ends survive, so the shortcut does too


def test_prune_edges_keeps_happy_edges_at_zero_percent():
    graph = build_process_graph(_mixed_log())
    pruned = prune_process_graph(graph, activity_pct=100, edge_pct=0)

    assert pruned["edges"], "happy-path edges must survive"
    assert all(e["happy"] for e in pruned["edges"])
    assert _node_ids(pruned) == {START_ID, END_ID, "a", "b", "c"}


def test_prune_does_not_modify_input():
    graph = build_process_graph(_mixed_log())
    n_nodes, n_edges = len(graph["nodes"]), len(graph["edges"])
    prune_process_graph(graph, 10, 0)
    assert len(graph["nodes"]) == n_nodes and len(graph["edges"]) == n_edges


def test_prune_clamps_out_of_range_percentages():
    graph = build_process_graph(_mixed_log())
    assert len(prune_process_graph(graph, 500, 500)["edges"]) == len(graph["edges"])
    assert prune_process_graph(graph, -5, -5)["edges"]


def test_prune_empty_graph():
    pruned = prune_process_graph(build_process_graph(None), 50, 50)
    assert pruned["nodes"] == [] and pruned["edges"] == []
