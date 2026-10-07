"""
Turns clicks in the interactive process explorer into filter steps.

A click on a node or edge offers a few actions; picking one stages a regular
`filter_steps` entry (the same dicts `create_analysis_config` takes), which is
listed beside the Filter Events form and merged into the config after the
form's own steps. No new pipeline machinery: every action maps onto an
existing filter type (`activity`, `endpoints`) or `directly_follows`.

The Filter Events form owns at most one event-removal/keep step and one crop
step. `split_filter_steps` tells those apart from explorer-staged steps, so a
saved run or preset restores each to the right place.
"""
from typing import Any, Dict, List, Optional, Tuple

from .process_graph import END_ID, START_ID

# Selection dicts are what the component reports (see render_process_explorer).
Selection = Dict[str, Any]
Step = Dict[str, Any]

_FORM_ACTIVITY_MODES = {"remove_events", "keep_events"}


def available_actions(selection: Optional[Selection]) -> List[Tuple[str, str]]:
    """(action_id, label) pairs offered for a clicked node or edge."""
    if not selection:
        return []

    if selection.get("type") == "node":
        if selection.get("kind") != "activity":
            return []
        name = selection["label"]
        return [
            ("remove_events", f"Remove all '{name}' events"),
            ("keep_cases_with", f"Keep only cases containing '{name}'"),
            ("drop_cases_with", f"Remove cases containing '{name}'"),
        ]

    if selection.get("type") == "edge":
        source, target = selection["source"], selection["target"]
        if source == START_ID:
            return [("keep_cases_starting", f"Keep only cases starting with '{target}'")]
        if target == END_ID:
            return [("keep_cases_ending", f"Keep only cases ending with '{source}'")]
        return [
            ("keep_cases_following", f"Keep only cases where '{source}' is directly followed by '{target}'"),
            ("drop_cases_following", f"Remove cases where '{source}' is directly followed by '{target}'"),
        ]

    return []


def build_step(action_id: str, selection: Selection) -> Step:
    """The filter step for an action offered by available_actions()."""
    if selection.get("type") == "node":
        name = selection["id"]
        if action_id == "remove_events":
            return {"type": "activity", "activities": [name], "mode": "remove_events"}
        if action_id == "keep_cases_with":
            return {"type": "activity", "activities": [name], "mode": "contains"}
        if action_id == "drop_cases_with":
            return {"type": "activity", "activities": [name], "mode": "not_contains"}
    elif selection.get("type") == "edge":
        source, target = selection["source"], selection["target"]
        if action_id == "keep_cases_starting":
            return {"type": "endpoints", "start_activities": [target]}
        if action_id == "keep_cases_ending":
            return {"type": "endpoints", "end_activities": [source]}
        if action_id == "keep_cases_following":
            return {"type": "directly_follows", "source": source, "target": target, "mode": "contains"}
        if action_id == "drop_cases_following":
            return {"type": "directly_follows", "source": source, "target": target, "mode": "not_contains"}
    raise ValueError(f"Unknown action '{action_id}' for a {selection.get('type')!r} selection.")


def describe_step(step: Step) -> str:
    """One readable line for a staged step, e.g. for a chip or list entry."""
    kind = step.get("type")
    if kind == "activity":
        names = ", ".join(f"'{a}'" for a in step.get("activities") or [])
        return {
            "remove_events": f"Remove events {names}",
            "keep_events": f"Keep only events {names}",
            "not_contains": f"Remove cases containing {names}",
        }.get(step.get("mode", "contains"), f"Keep only cases containing {names}")
    if kind == "directly_follows":
        verb = "Remove" if step.get("mode") == "not_contains" else "Keep only"
        return f"{verb} cases where '{step.get('source')}' is directly followed by '{step.get('target')}'"
    if kind == "endpoints":
        parts = []
        if step.get("start_activities"):
            parts.append("starting with " + ", ".join(f"'{a}'" for a in step["start_activities"]))
        if step.get("end_activities"):
            parts.append("ending with " + ", ".join(f"'{a}'" for a in step["end_activities"]))
        return "Keep only cases " + " and ".join(parts)
    return str(kind)


def add_step(steps: List[Step], step: Step) -> List[Step]:
    """steps plus step; unchanged if the same step is already staged."""
    return steps if step in steps else [*steps, step]


def split_filter_steps(steps: Optional[List[Step]]) -> Tuple[List[Step], List[Step]]:
    """
    Splits a saved filter_steps list into (form_steps, explorer_steps).

    The form writes its steps first, in a fixed shape: one event remove/keep
    `activity` step, then one `crop`. Steps are taken from the front while
    they fit that shape; everything after the first step that doesn't (or that
    repeats a form step) was staged from the explorer.
    """
    steps = list(steps or [])
    form_steps: List[Step] = []
    seen_activity = seen_crop = False
    for step in steps:
        is_activity = step.get("type") == "activity" and step.get("mode") in _FORM_ACTIVITY_MODES
        is_crop = step.get("type") == "crop"
        if is_activity and not seen_activity and not seen_crop:
            seen_activity = True
        elif is_crop and not seen_crop:
            seen_crop = True
        else:
            break
        form_steps.append(step)
    return form_steps, steps[len(form_steps):]
