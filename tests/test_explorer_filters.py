import pandas as pd
import pytest

from prox.data_manager import filter_event_log
from prox.explorer_filters import (
    add_step, available_actions, blocked_actions, build_step, describe_step, split_filter_steps,
)
from prox.process_graph import END_ID, START_ID


def _node(name="view", kind="activity"):
    return {"type": "node", "id": name, "label": name, "kind": kind, "cases": 3, "events": 3}


def _edge(source, target):
    return {"type": "edge", "source": source, "target": target, "frequency": 3, "cases": 3}


def test_no_selection_offers_nothing():
    assert available_actions(None) == []


def test_activity_node_offers_event_and_case_actions():
    assert [a for a, _ in available_actions(_node())] == [
        "remove_events", "keep_cases_with", "drop_cases_with"
    ]


def test_start_and_end_nodes_offer_nothing():
    assert available_actions(_node(START_ID, kind="start")) == []
    assert available_actions(_node(END_ID, kind="end")) == []


def test_activity_edge_offers_directly_follows_actions():
    assert [a for a, _ in available_actions(_edge("a", "b"))] == [
        "keep_cases_following", "drop_cases_following"
    ]


def test_start_and_end_edges_map_onto_endpoint_filters():
    assert [a for a, _ in available_actions(_edge(START_ID, "a"))] == ["keep_cases_starting"]
    assert [a for a, _ in available_actions(_edge("a", END_ID))] == ["keep_cases_ending"]


@pytest.mark.parametrize("action,selection,expected", [
    ("remove_events", _node("x"), {"type": "activity", "activities": ["x"], "mode": "remove_events"}),
    ("keep_cases_with", _node("x"), {"type": "activity", "activities": ["x"], "mode": "contains"}),
    ("drop_cases_with", _node("x"), {"type": "activity", "activities": ["x"], "mode": "not_contains"}),
    ("keep_cases_following", _edge("a", "b"),
     {"type": "directly_follows", "source": "a", "target": "b", "mode": "contains"}),
    ("drop_cases_following", _edge("a", "b"),
     {"type": "directly_follows", "source": "a", "target": "b", "mode": "not_contains"}),
    ("keep_cases_starting", _edge(START_ID, "a"), {"type": "endpoints", "start_activities": ["a"]}),
    ("keep_cases_ending", _edge("a", END_ID), {"type": "endpoints", "end_activities": ["a"]}),
])
def test_build_step(action, selection, expected):
    assert build_step(action, selection) == expected


def test_build_step_rejects_action_not_offered_for_the_selection():
    with pytest.raises(ValueError):
        build_step("keep_cases_following", _node())


def test_every_offered_action_builds_a_step_the_pipeline_accepts():
    df = pd.DataFrame({
        "case:concept:name": ["1", "1", "2", "2"],
        "concept:name": ["a", "b", "a", "c"],
        "time:timestamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:01"] * 2),
    })
    selections = [_node("a"), _edge("a", "b"), _edge(START_ID, "a"), _edge("b", END_ID)]
    for selection in selections:
        for action_id, _ in available_actions(selection):
            step = dict(build_step(action_id, selection))
            filtered, _ = filter_event_log(df, filter_type=step.pop("type"), **step)
            assert filtered is not None, (action_id, selection)


def test_describe_step_reads_naturally():
    assert describe_step(build_step("remove_events", _node("x"))) == "Remove events 'x'"
    assert describe_step(build_step("keep_cases_with", _node("x"))) == "Keep only cases containing 'x'"
    assert describe_step(build_step("drop_cases_with", _node("x"))) == "Remove cases containing 'x'"
    assert describe_step(build_step("drop_cases_following", _edge("a", "b"))) == \
        "Remove cases where 'a' is directly followed by 'b'"
    assert describe_step(build_step("keep_cases_ending", _edge("a", END_ID))) == \
        "Keep only cases ending with 'a'"


def test_add_step_skips_duplicates_and_does_not_mutate():
    step = build_step("remove_events", _node("x"))
    original = []
    once = add_step(original, step)
    assert once == [step] and original == []
    assert add_step(once, dict(step)) == [step]


def test_split_filter_steps_separates_form_steps_from_explorer_steps():
    form_activity = {"type": "activity", "activities": ["scroll"], "mode": "remove_events"}
    form_crop = {"type": "crop", "activity": ["purchase"]}
    staged_remove = {"type": "activity", "activities": ["x"], "mode": "remove_events"}
    staged_df = {"type": "directly_follows", "source": "a", "target": "b", "mode": "contains"}

    form, explorer = split_filter_steps([form_activity, form_crop, staged_remove, staged_df])
    assert form == [form_activity, form_crop]
    assert explorer == [staged_remove, staged_df]


def test_split_filter_steps_without_form_steps():
    staged = {"type": "activity", "activities": ["x"], "mode": "contains"}
    assert split_filter_steps([staged]) == ([], [staged])
    assert split_filter_steps(None) == ([], [])


def test_split_filter_steps_staged_event_removal_alone_is_form_owned_by_shape():
    # Indistinguishable from a form step by shape; the form claims it. Callers
    # that stage explorer steps always append them after the form's, so a
    # round trip through the form's own output stays lossless.
    step = {"type": "activity", "activities": ["x"], "mode": "remove_events"}
    assert split_filter_steps([step]) == ([step], [])


def test_describe_step_covers_the_form_filters_too():
    assert describe_step({"type": "crop", "activity": ["purchase"]}) == \
        "Crop traces at 'purchase' (cases that never reach it are removed)"
    assert describe_step({"type": "activity", "activities": ["x"], "mode": "keep_events"}) == \
        "Keep only events 'x'"
    assert describe_step({"type": "top_variants", "top_n": 5}) == "Keep only the top 5 variants"


def _cropped_log():
    """Every case runs a -> b -> c and ends at c, like a log cropped at its end point."""
    base = pd.Timestamp("2024-01-01")
    rows = [(case, act, base + pd.Timedelta(minutes=i))
            for case in ("1", "2") for i, act in enumerate("abc")]
    return pd.DataFrame(rows, columns=["case:concept:name", "concept:name", "time:timestamp"])


def test_blocked_actions_flags_actions_that_would_leave_no_cases():
    blocked = blocked_actions(_cropped_log(), _node("b"))
    assert blocked == {"remove_events": False, "keep_cases_with": False, "drop_cases_with": True}

    blocked = blocked_actions(_cropped_log(), _edge("b", END_ID))
    assert blocked == {"keep_cases_ending": True}  # every case ends at c, not b

    blocked = blocked_actions(_cropped_log(), _edge("a", "b"))
    assert blocked == {"keep_cases_following": False, "drop_cases_following": True}


def test_blocked_actions_without_a_selection_is_empty():
    assert blocked_actions(_cropped_log(), None) == {}
