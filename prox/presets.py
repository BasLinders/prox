"""
Config presets: a small JSON file holding the analysis settings a repeat
analyst re-enters every session (sidebar, filter, sampling) plus the funnel
definition, that can be applied to any new upload.

Distinct from saved runs (prox/saved_runs.py): a saved run restores its
settings only together with its own event log and results. A preset carries
no data at all, so it applies to whatever log is loaded next - which is why
`fit_config_to_log` / `fit_funnel_to_log` exist: the new log may not contain
an activity, column or funnel step the preset names, and those have to be
dropped (and reported) rather than fed to a widget that can't show them.

Only reusable, machine-independent settings are kept. CPU cores, chunk sizes
and other tuning that depends on the machine or the file are left out.

The funnel lives beside the config, not inside it (the pipeline's config has
no funnel), in the shape produced by `build_funnel_settings`. That shape and
the fit helper are deliberately not preset-specific: when saved runs learn to
store a funnel, they can put the same dict in their manifest under "funnel"
and reuse everything here.

Layout:

    <presets_dir>/<preset_id>.preset.json

A preset is reused by name - saving "Acme checkout" again updates it, unlike
saved runs where every save is its own entry.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_PRESETS_DIR = ".prox_presets"
PRESET_VERSION = 1

FUNNEL_MODES = ("manual", "auto")

# Sections of the analysis config a preset keeps, and within each the fields.
# None keeps the whole section.
_CONFIG_FIELDS = {
    "data_loading": ("remove_duplicates",),
    "discovery_params": ("algorithm", "noise_threshold"),
    "conformance_params": ("algorithm", "calculate_precision"),
    "sampling_config": None,
    "filter_steps": None,
}

_SAMPLING_NONE_SENTINEL = "case:concept:name"


class PresetError(ValueError):
    """A preset file is unreadable or not a valid preset."""


def _safe_preset_id(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip()) or "preset"


def _preset_path(preset_id: str, presets_dir: str) -> str:
    return os.path.join(presets_dir, f"{preset_id}.preset.json")


def extract_preset_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """The reusable subset of a run_full_analysis config."""
    out: Dict[str, Any] = {}
    for section, fields in _CONFIG_FIELDS.items():
        if section not in config:
            continue
        value = config[section]
        if fields is None:
            out[section] = value
        else:
            out[section] = {f: value[f] for f in fields if f in value}
    return out


def build_funnel_settings(
    mode: str, steps: Optional[List[str]] = None, segment_col: Optional[str] = None
) -> Dict[str, Any]:
    """A funnel definition: "manual" (explicit ordered steps) or "auto"
    (detected from the data, so no steps are stored)."""
    if mode not in FUNNEL_MODES:
        raise ValueError(f"funnel mode must be one of {FUNNEL_MODES}, got {mode!r}")
    return {
        "mode": mode,
        "steps": [str(s) for s in steps or []] if mode == "manual" else [],
        "segment_col": segment_col or None,
    }


def build_preset(name: str, config: Dict[str, Any], funnel: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    name = name.strip()
    if not name:
        raise PresetError("A preset needs a name.")
    return {
        "version": PRESET_VERSION,
        "name": name,
        "config": extract_preset_config(config),
        "funnel": funnel,
    }


def validate_preset(obj: Any) -> Dict[str, Any]:
    """Returns obj if it's a usable preset, else raises PresetError."""
    if not isinstance(obj, dict):
        raise PresetError("Not a PRoX preset (expected a JSON object).")
    version = obj.get("version")
    if version != PRESET_VERSION:
        raise PresetError(
            f"Unsupported preset version {version!r} (this PRoX reads version {PRESET_VERSION})."
        )
    if not isinstance(obj.get("name"), str) or not obj["name"].strip():
        raise PresetError("The preset has no name.")
    config = obj.get("config")
    if not isinstance(config, dict):
        raise PresetError("The preset has no settings ('config' must be an object).")
    for section, fields in _CONFIG_FIELDS.items():
        if section not in config:
            continue
        expected = list if section == "filter_steps" else dict
        if not isinstance(config[section], expected):
            raise PresetError(f"'{section}' must be a {expected.__name__}.")
    for step in config.get("filter_steps") or []:
        if not isinstance(step, dict) or not isinstance(step.get("type"), str):
            raise PresetError("Each filter step must be an object with a 'type'.")
    funnel = obj.get("funnel")
    if funnel is not None:
        if not isinstance(funnel, dict) or funnel.get("mode") not in FUNNEL_MODES:
            raise PresetError(f"'funnel' must be null or an object with a mode of {FUNNEL_MODES}.")
        if not isinstance(funnel.get("steps", []), list):
            raise PresetError("'funnel.steps' must be a list.")
    return obj


def preset_to_json(preset: Dict[str, Any]) -> str:
    return json.dumps(preset, indent=2)


def preset_from_json(text) -> Dict[str, Any]:
    """Parses and validates preset JSON (str or bytes)."""
    try:
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        obj = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise PresetError(f"Not valid JSON: {e}") from e
    return validate_preset(obj)


def save_preset(preset: Dict[str, Any], presets_dir: str = DEFAULT_PRESETS_DIR) -> str:
    """Writes the preset, replacing any earlier one with the same name.
    Returns its preset_id."""
    validate_preset(preset)
    os.makedirs(presets_dir, exist_ok=True)
    preset_id = _safe_preset_id(preset["name"])
    path = _preset_path(preset_id, presets_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(preset_to_json(preset))
    os.replace(tmp_path, path)  # atomic on the same filesystem
    return preset_id


def load_preset(preset_id: str, presets_dir: str = DEFAULT_PRESETS_DIR) -> Dict[str, Any]:
    try:
        with open(_preset_path(preset_id, presets_dir), "r", encoding="utf-8") as f:
            return preset_from_json(f.read())
    except OSError as e:
        raise PresetError(f"Preset '{preset_id}' can't be read: {e}") from e


def list_presets(presets_dir: str = DEFAULT_PRESETS_DIR) -> List[Tuple[str, Dict[str, Any]]]:
    """(preset_id, preset) for every readable preset, sorted by name."""
    if not os.path.isdir(presets_dir):
        return []
    found = []
    for fname in os.listdir(presets_dir):
        if not fname.endswith(".preset.json"):
            continue
        preset_id = fname[: -len(".preset.json")]
        try:
            found.append((preset_id, load_preset(preset_id, presets_dir)))
        except PresetError as e:
            logger.warning("Skipping unusable preset %s: %s", fname, e)
    found.sort(key=lambda p: p[1]["name"].lower())
    return found


def delete_preset(preset_id: str, presets_dir: str = DEFAULT_PRESETS_DIR) -> bool:
    path = _preset_path(preset_id, presets_dir)
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False


def fit_config_to_log(
    config: Dict[str, Any], activities: List[str], columns: List[str]
) -> Tuple[Dict[str, Any], List[str]]:
    """Returns (config, notes): config without the filter activities, crop
    end point and sampling column the log doesn't have. notes is one
    readable line per thing dropped."""
    known_acts = set(activities)
    known_cols = set(columns)
    notes: List[str] = []
    fitted = dict(config)

    steps = []
    for step in config.get("filter_steps") or []:
        step = dict(step)
        if step.get("type") == "activity":
            kept = [a for a in step.get("activities") or [] if a in known_acts]
            missing = [a for a in step.get("activities") or [] if a not in known_acts]
            if missing:
                notes.append(f"Filter: activities not in this log were dropped: {', '.join(missing)}.")
            if not kept:
                continue
            step["activities"] = kept
        elif step.get("type") == "crop":
            missing = [a for a in step.get("activity") or [] if a not in known_acts]
            if missing:
                notes.append(
                    f"Process end point '{missing[0]}' is not in this log - traces are not cropped."
                )
                continue
        steps.append(step)
    if "filter_steps" in config:
        fitted["filter_steps"] = steps

    sampling = config.get("sampling_config") or {}
    strata = sampling.get("strata_col")
    if strata and strata != _SAMPLING_NONE_SENTINEL and strata not in known_cols:
        notes.append(f"Sampling: priority column '{strata}' is not in this log - plain random sample used.")
        fitted["sampling_config"] = {**sampling, "strata_col": _SAMPLING_NONE_SENTINEL}

    return fitted, notes


def fit_funnel_to_log(
    funnel: Optional[Dict[str, Any]], activities: List[str], segment_candidates: List[str]
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Returns (funnel, notes): funnel without the steps / segment column the
    log doesn't offer. activities should be the activities the funnel can
    actually be built from (i.e. after the run's filters)."""
    if funnel is None:
        return None, []
    notes: List[str] = []
    steps = list(funnel.get("steps") or [])
    kept = [s for s in steps if s in set(activities)]
    if len(kept) < len(steps):
        missing = [s for s in steps if s not in set(activities)]
        notes.append(f"Funnel: steps not available in this log were dropped: {', '.join(missing)}.")
    segment_col = funnel.get("segment_col")
    if segment_col and segment_col not in segment_candidates:
        notes.append(f"Funnel: segment column '{segment_col}' is not available in this log - no split applied.")
        segment_col = None
    return {**funnel, "steps": kept, "segment_col": segment_col}, notes
