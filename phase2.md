# Phase 2 — Establish a Safety Net

Context: PRoX has no automated tests, no CI, no linter enforcement, no `.gitignore`, and dependencies are pinned with bare `>=` floors and no ceilings. Phase 1 (fixing the token-replay gap and finishing config parameterization) closes correctness issues; Phase 2 makes sure future changes to this codebase can be verified without re-deriving correctness by hand every time, especially given the multi-month gaps between development bursts on this project.

## 1. Test suite (`tests/`)

Add `pytest` and a `tests/` directory. Priority order, highest-value first:

1. **`prox/analytics.py`** — the least visually-checkable code in the repo. Cover:
   - `analyze_process_performance`: activity/transition bottleneck detection and `impact_score` calculation (`mean_duration * frequency`), severity thresholds (percentile-based), process health score.
   - `analyze_repeat_purchases`: multi-strategy purchase detection (activity-name pattern match, flag columns, revenue > 0), repeat-rate calculation, inter-purchase timing, revenue multiplier.
   - Use small synthetic event logs (a handful of cases/activities built directly as DataFrames) rather than real exported data — these functions are pure enough over a DataFrame that they don't need PM4Py's heavier internals mocked.

2. **`prox/data_manager.py`**:
   - Column auto-mapping against `COLUMN_MAPPINGS` (including the Dutch aliases).
   - `filter_event_log` dispatch for each `filter_type` (`activity`, `case_duration`, `crop`, `endpoints`, `attribute`, `top_variants`).
   - `sample_log_stratified` — priority-preserving behavior and the random-fallback path when `strata_col` is missing.
   - Composite case-ID creation (`user_id` + `session_id`).

3. **`prox/conformance.py`**:
   - Once the Phase 1 token-replay fix lands, verify it actually produces a non-zero fitness score independent of the alignment path.
   - Batched fitness (`calculate_fitness_in_batches`) against a known small log with a known expected fitness value.

Not a priority for Phase 2: `visualizer.py` (PNG output is hard to assert on meaningfully) and `pipeline.py` (better covered by one or two end-to-end smoke tests once the unit layer exists).

## 2. CI

Add a minimal GitHub Actions workflow (`.github/workflows/ci.yml`) that runs on every PR:
- `pip install -r requirements.txt`
- `pytest`
- `pyflakes` (or `ruff check`) across `prox/` and `main.py`

Keep it minimal at first — the goal is a fast, always-green gate, not exhaustive static analysis. Expand later once the baseline is trusted.

## 3. Housekeeping

- Add a `.gitignore` covering `__pycache__/`, `output/`, `.venv/` — stops the recurring untracked `prox/__pycache__/` noise seen in `git status`.
- Pin dependencies more strictly. Options, in order of effort:
  - Add upper bounds to `requirements.txt`/`setup.py` (e.g. `pm4py>=2.7.0,<3.0.0`) as a low-effort first step.
  - Generate a lockfile (`pip freeze` into `requirements-lock.txt`, or migrate to `poetry`/`pip-tools`) for full reproducibility.
  - `pm4py` and `streamlit` are the two fast-moving, breaking-change-prone dependencies here and should get the most conservative bounds.

## Why this order

Tests before CI before expansion: without a test suite, a CI workflow only checks that the code imports and lints, not that it's correct. Without CI, a test suite only protects the person who remembers to run it locally. Both need to exist before Phase 3 (extensibility refactors) and Phase 4 (new features) — refactoring the discovery/conformance dispatch or adding new analysis capability is much lower-risk once there's something to catch regressions automatically, especially given how much time can pass between sessions on this project.
