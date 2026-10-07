"""
Interactive process explorer: a Cytoscape.js directly-follows graph, mounted
as a Streamlit v2 component (st.components.v2).

Everything is bundled in the repo - Cytoscape.js and cytoscape-dagre are
vendored under vendor/ (MIT, licences alongside) - so the app works offline.
The libraries are UMD builds, but a v2 component's `js` has to be one ES
module string, so they're wrapped in CommonJS-style scopes and concatenated
in front of explorer.js rather than loaded as separate scripts.
"""
import functools
import os
import re
from typing import Any, Dict, Optional, Tuple

import streamlit as st

_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_DIR, "vendor")


def _read(*parts: str) -> str:
    with open(os.path.join(_DIR, *parts), encoding="utf-8") as f:
        return f.read()


def _umd_module(filename: str, requires: Optional[Dict[str, str]] = None) -> str:
    """Source of a UMD library as an expression that evaluates to its exports.
    `requires` maps a require()'d name to a variable already in scope."""
    # A leftover sourceMappingURL points at a .map file that isn't shipped.
    source = re.sub(r"^//# sourceMappingURL=.*$", "", _read("vendor", filename), flags=re.MULTILINE)
    shim = ", ".join(f"{name!r}: {var}" for name, var in (requires or {}).items())
    return (
        "(function () {\n"
        "  const module = { exports: {} };\n"
        "  const exports = module.exports;\n"
        f"  const require = (name) => ({{{shim}}})[name];\n"
        f"{source}\n"
        "  return module.exports;\n"
        "}).call(globalThis)"
    )


def _build_js() -> str:
    return (
        f"const cytoscape = {_umd_module('cytoscape.min.js')};\n"
        f"cytoscape.use({_umd_module('cytoscape-dagre.min.js', {'cytoscape': 'cytoscape'})});\n"
        f"{_read('explorer.js')}"
    )


@functools.lru_cache(maxsize=1)
def _assets() -> Tuple[str, str]:
    # Built once per process: the JS is ~500KB read off disk and concatenated.
    return _read("explorer.css"), _build_js()


def _get_component():
    # Registered on every render rather than held in a module global: the
    # registry belongs to the running Streamlit runtime, so a handle cached
    # from an earlier runtime (a restart, or another test's AppTest) would
    # point at a component that is no longer registered. Re-registering an
    # identical definition is a silent overwrite.
    css, js = _assets()
    return st.components.v2.component("prox_process_explorer", css=css, js=js)


def render_process_explorer(graph: Dict[str, Any], key: str, height: int = 620, metric: str = "frequency"):
    """
    Renders `graph` (see prox.process_graph) and returns the component result.
    `result.selection` is None, or a dict describing the last clicked node
    ({"type": "node", "id", "label", "kind", "cases", "events"}) or edge
    ({"type": "edge", "source", "target", "frequency", "cases", "mean_time",
    "median_time"}).

    metric: "frequency" labels edges with their count; "time" labels and
    colours them by mean transition time. Edge width is always frequency.
    """
    return _get_component()(
        key=key,
        data={"graph": graph, "height": height, "metric": metric},
        default={"selection": None},
        on_selection_change=lambda: None,
    )
