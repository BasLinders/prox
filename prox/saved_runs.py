"""
Saved runs: a library of event logs the analyst has explicitly chosen to keep
around, labelled by hand (e.g. a client name) so a later session can come
back and pick "which run" to load instead of re-uploading or re-querying.

Distinct from prox/incremental.py's cache: that module keys one recurring
dataset by a *reused* label (the same "GA4 checkout funnel" id always points
at the same accumulating cache). Saved runs intentionally do the opposite -
every save gets its own entry even if the label repeats (e.g. "Acme Corp"
saved once a month, each a separate snapshot to come back to).

Coupling with the incremental cache: if the event log being saved is exactly
what's already sitting in an incremental-cache dataset_id (the run was loaded
via "Load cached dataset", or merged into one via Incremental Analysis), the
saved-run entry records that `source_dataset_id` and does NOT write its own
second copy of the data - it would just be a duplicate of bytes already
durably on disk. Loading such an entry reads through to that cache instead.
A run with no linked cache (a plain CSV/BigQuery pull never merged into an
incremental dataset) gets its own on-disk copy, same as the cache module.

Layout (one manifest, plus a data file only when not cache-linked):

    <save_dir>/<run_id>.manifest.json
    <save_dir>/<run_id>.data.csv.gz   - only present when source_dataset_id is None

`run_id` embeds a timestamp + random suffix (not just the label) precisely
because labels are expected to repeat across saves - unlike incremental.py's
dataset_id, reusing the same label here must not overwrite a prior entry.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .incremental import DEFAULT_CACHE_DIR, load_cached_dataset

logger = logging.getLogger(__name__)

DEFAULT_SAVED_RUNS_DIR = ".prox_saved_runs"

# Mirrors incremental.py's _READ_CSV_NA_KWARGS: only an empty field round-trips
# to NaN, so a case/activity value that happens to read "NA"/"null" etc. isn't
# silently corrupted by pandas' default na_values list on reload.
_READ_CSV_NA_KWARGS = {"keep_default_na": False, "na_values": [""]}


def _safe_run_id(label: str) -> str:
    """Builds a unique, filesystem-safe key for one save. Unlike
    incremental._safe_dataset_id, this must never collide on a repeated label -
    saving "Acme Corp" twice in one month has to produce two entries, not
    overwrite the first - so the key includes a timestamp + random suffix
    rather than being derived from the label alone."""
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in label.strip()) or "run"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    digest = hashlib.sha1(f"{label}-{ts}-{os.urandom(4).hex()}".encode("utf-8")).hexdigest()[:8]
    return f"{cleaned}-{ts}-{digest}"


def _manifest_path(run_id: str, save_dir: str) -> str:
    return os.path.join(save_dir, f"{run_id}.manifest.json")


def _data_path(run_id: str, save_dir: str) -> str:
    return os.path.join(save_dir, f"{run_id}.data.csv.gz")


def list_saved_runs(save_dir: str = DEFAULT_SAVED_RUNS_DIR) -> list:
    """Returns manifests for every saved run, newest first."""
    if not os.path.isdir(save_dir):
        return []
    manifests = []
    for name in os.listdir(save_dir):
        if name.endswith(".manifest.json"):
            try:
                with open(os.path.join(save_dir, name), "r", encoding="utf-8") as f:
                    manifests.append(json.load(f))
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Skipping unreadable saved-run manifest %s: %s", name, e)
    manifests.sort(key=lambda m: m.get("saved_at", ""), reverse=True)
    return manifests


def save_run(
    df: pd.DataFrame,
    label: str,
    case_grouping: str,
    source_dataset_id: Optional[str] = None,
    save_dir: str = DEFAULT_SAVED_RUNS_DIR,
) -> Dict[str, Any]:
    """Registers df in the saved-runs library under a fresh run_id.

    If source_dataset_id is given, this event log is already durably stored
    in the incremental cache under that id - the manifest links to it instead
    of writing a second copy. Otherwise df is written to its own gzip CSV,
    same layout as incremental.save_cached_dataset.
    """
    os.makedirs(save_dir, exist_ok=True)
    run_id = _safe_run_id(label)

    if source_dataset_id is None:
        data_path = _data_path(run_id, save_dir)
        tmp_path = data_path + ".tmp"
        df.to_csv(tmp_path, index=False, compression="gzip")
        os.replace(tmp_path, data_path)  # atomic on the same filesystem

    manifest = {
        "run_id": run_id,
        "label": label.strip(),
        "source_dataset_id": source_dataset_id,
        "case_grouping": case_grouping,
        "columns": sorted(str(c) for c in df.columns),
        "n_events": int(len(df)),
        "n_cases": int(df['case:concept:name'].nunique()) if 'case:concept:name' in df.columns else None,
        "min_timestamp": df['time:timestamp'].min().isoformat() if 'time:timestamp' in df.columns and not df.empty else None,
        "max_timestamp": df['time:timestamp'].max().isoformat() if 'time:timestamp' in df.columns and not df.empty else None,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = _manifest_path(run_id, save_dir)
    tmp_manifest_path = manifest_path + ".tmp"
    with open(tmp_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_manifest_path, manifest_path)
    return manifest


def load_saved_run(
    run_id: str, save_dir: str = DEFAULT_SAVED_RUNS_DIR, cache_dir: str = DEFAULT_CACHE_DIR
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], Optional[str]]:
    """Returns (df, manifest, error_message). df/manifest are None and
    error_message is set if the entry (or, for a cache-linked entry, the
    cache it points to) is missing or unreadable."""
    manifest_path = _manifest_path(run_id, save_dir)
    if not os.path.isfile(manifest_path):
        return None, None, f"Saved run '{run_id}' no longer exists."

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return None, None, f"Saved run '{run_id}' is unreadable: {e}"

    source_dataset_id = manifest.get("source_dataset_id")
    if source_dataset_id is not None:
        df, cache_manifest = load_cached_dataset(source_dataset_id, cache_dir=cache_dir)
        if df is None:
            return None, None, (
                f"'{manifest.get('label', run_id)}' was saved as a link to cached dataset "
                f"'{source_dataset_id}', but that cache has since been cleared. "
                "The saved event log is no longer available."
            )
        return df, manifest, None

    data_path = _data_path(run_id, save_dir)
    if not os.path.isfile(data_path):
        return None, None, f"Saved run '{run_id}' is missing its data file."

    try:
        df = pd.read_csv(data_path, compression="gzip", **_READ_CSV_NA_KWARGS)
        if 'time:timestamp' in df.columns:
            df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
        if 'case:concept:name' in df.columns:
            df['case:concept:name'] = df['case:concept:name'].astype(str)
        return df, manifest, None
    except (OSError, pd.errors.ParserError) as e:
        return None, None, f"Saved run '{run_id}' is unreadable: {e}"


def delete_saved_run(run_id: str, save_dir: str = DEFAULT_SAVED_RUNS_DIR) -> bool:
    """Deletes a saved-run entry (manifest + its own data file, if any).
    Never touches a linked incremental-cache dataset - that's a shared
    resource managed separately via clear_cached_dataset(). Returns True if
    anything was removed."""
    removed = False
    for path in (_manifest_path(run_id, save_dir), _data_path(run_id, save_dir)):
        if os.path.isfile(path):
            os.remove(path)
            removed = True
    return removed
