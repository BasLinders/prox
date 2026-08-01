# Phase 2 — Establish a Safety Net

**Status: Complete.**

- Added `tests/` with 32 pytest tests covering `analytics.py`, `data_manager.py`, `conformance.py`, and `discovery.py` — all passing.
- While writing the `optimize_dataframe_memory` test, found and fixed a **real, silent bug**: it checked `dtype == 'object'`, but pandas 2.x+/3.x infers plain string columns as a dedicated `str` dtype rather than `object` — so on the currently-installed pandas (3.0.3), memory optimization was a silent no-op for every run, never converting any column to `category`. Fixed with `pd.api.types.is_string_dtype()`, which correctly matches both `object` and `str` while still excluding already-categorical columns. This is a direct, concrete instance of the dependency-drift risk described below — not hypothetical.
- Added `.github/workflows/ci.yml`: runs `pyflakes` then `pytest` on every PR and push to `main`.
- Fixed the one pre-existing unused import (`prox/visualizer.py`'s unused `Any`) so the new lint gate starts green instead of red on day one.
- Added `.gitignore` (`__pycache__/`, `output/`, `.venv/`, `.pytest_cache/`, etc.) and `requirements-dev.txt` (`pytest`, `pyflakes`) for local development.
- Pinned dependencies with upper bounds one major above the currently-installed, test-verified versions (`pandas<4.0.0`, `numpy<3.0.0`, `matplotlib<4.0.0`, `seaborn<1.0.0`, `pm4py<3.0.0`, `streamlit<2.0.0`) in both `requirements.txt` and `setup.py`. Did not generate a lockfile — the upper-bound step alone was judged sufficient for now; revisit if reproducibility issues actually surface.

Original context: PRoX has no automated tests, no CI, no linter enforcement, no `.gitignore`, and dependencies are pinned with bare `>=` floors and no ceilings. Phase 1 is now complete: token-based replay is genuinely implemented (`pm4py.fitness_token_based_replay`, `prox/conformance.py`), `create_analysis_config()` is fully parameterized (no more values hardcoded inside the function body), and `dfg` is exposed in the UI's discovery selectbox alongside Inductive Miner and Heuristics Miner. That pass also caught and fixed a real pm4py API-drift bug in the DFG-to-Petri-net conversion (`dfg_converter.Variants.TO_PETRI_NET` no longer exists in pm4py 2.7.23 — replaced with `VERSION_TO_PETRI_NET_ACTIVITY_DEFINES_PLACE`), which is exactly the kind of breakage the loose `>=`-only pinning below was flagged as risking. Phase 2 makes sure future changes to this codebase can be verified without re-deriving correctness by hand every time, especially given the multi-month gaps between development bursts on this project.

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
   - Token-based replay (`alignment_variant='token_replay'`) produces a non-zero, plausible fitness score independent of the alignment path — regression-test this now that it's real, since a future pm4py upgrade is exactly the kind of change that could silently break it again (as it already did once for the DFG converter).
   - State-equation A* (`alignment_variant='state_equation_a_star'`) still produces per-trace deviations as expected.
   - Batched fitness (`calculate_fitness_in_batches`) against a known small log with a known expected fitness value.

4. **`prox/discovery.py`**:
   - All three discovery algorithms (`inductive_miner`, `heuristics_miner`, `dfg`) produce a usable `(net, im, fm)` tuple on a small synthetic log — this would have caught the `dfg` converter API break immediately instead of only surfacing when someone happened to select it in the UI.

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

## Phase 3 — Harden the extensibility seams

**Status: Complete.**

- **Discovery dispatch**: `prox/discovery.py`'s `if/elif` chain over `discovery_algo` is replaced with a `DISCOVERY_ALGORITHMS` registry (dict of `{'handler', 'label', 'help'}` per algorithm). `perform_process_discovery` now dispatches via lookup instead of branching, and an unknown key produces a clear error listing valid options instead of a bare `'Unknown discovery algorithm'` message.
- **Conformance dispatch**: same pattern in `prox/conformance.py` — the token-replay-vs-alignments branch is now a `CONFORMANCE_METHODS` registry, with each method's logic extracted into its own function (`_fitness_token_replay`, `_fitness_state_equation_alignments`).
- **UI now derives from the registries, not a second hardcoded list**: `main.py`'s discovery and conformance `st.selectbox` options and help text are built from `DISCOVERY_ALGORITHMS`/`CONFORMANCE_METHODS` (exported via `prox/__init__.py`) instead of separately maintained lists. Adding a new algorithm or conformance method now means adding one registry entry in `prox/` — the UI picks it up automatically, closing the "edit two files" seam described in the original status report.
- **`filter_steps` registry**: `prox/data_manager.py`'s big `if/elif` in `filter_event_log` is split into six named handler functions (`_filter_activity`, `_filter_case_duration`, `_filter_crop`, `_filter_endpoints`, `_filter_attribute`, `_filter_top_variants`) registered in `FILTER_HANDLERS`. Unknown filter types now report the valid option list, not just the bad one.
- **Fail-fast config validation**: `pipeline.py` now validates every `filter_steps` entry's `type` against `FILTER_HANDLERS` *before* running any filter step, instead of discovering a typo mid-run (previously step 3 of 5 failing looked like "produced an empty dataset" rather than "unknown filter type").
- Verified with 8 new/updated tests (`tests/test_pipeline.py`, plus additions to `test_conformance.py`/`test_data_manager.py` covering the new error-listing behavior) and a live headless Streamlit boot of `main.py` to confirm the registry-driven selectboxes render and behave correctly.
- All behavior is otherwise unchanged — this was a pure structural refactor verified against the existing Phase 2 test suite before any new tests were added, which is exactly the lower-risk-refactor payoff Phase 2 was meant to unlock.

## Phase 4 — Expand capability

**Status: In progress.**

- **4a. Full-analysis HTML report export — Complete.** `prox/report.py`'s `generate_html_report()` builds a single, self-contained HTML report (metrics, embedded base64 process-map images, bottleneck/variant tables, conformance summary, business insights) from a `run_full_analysis()` results dict. Wired into `main.py` as a "Download Full Report" button. All user-derived strings are HTML-escaped (verified with an XSS test). Shipped directly to `main` (e84026f).
- **Alpha Miner — considered and rejected.** It has no soundness guarantee, poor short-loop handling, and no noise tolerance — strictly weaker than the existing Inductive Miner (sound/robust), Heuristics Miner (noisy logs), and DFG (fast overview) for this domain (noisy website event logs). Not worth adding just because the Phase 3 registry makes it a one-entry change.
- **Segment comparison v1 — Complete.**
  - `prox/segments.py`'s `compare_segments(df, segment_col, config, top_n_segments=5)`: picks the top-N segment values by case count, runs `run_full_analysis()` once per segment, returns `{segment_value: results}` plus a `comparison_table` (cases, health score, fitness, precision, repeat rate, top variant — one row per segment).
  - UI: an optional "Segment by" selectbox (columns with sane cardinality, ~2-20 unique values) and a "Segment Comparison" tab showing the table plus per-segment happy-path images side by side. Shipped `dbf0a5f`.
  - Reuses existing building blocks (`filter_event_log(filter_type='attribute', ...)`, `run_full_analysis()`) — mostly orchestration, not new analysis logic.
  - Runs the full pipeline N times (once per segment) — since Phase 4b, this now runs in parallel across worker processes by default (see below), which was the direct motivation for parallelizing `compare_segments()` rather than just alignment.
  - **Deferred to v2**: automatically diffing golden paths between segments (e.g. "segment A visits checkout, segment B doesn't"). That's genuinely new algorithmic work, not orchestration — worth revisiting once the side-by-side v1 view proves useful in practice.
- **4c. Report overhaul for non-technical stakeholders — Complete.** `generate_html_report()` now opens with a plain-language "Executive Summary" (health verdict badge, fitness/precision translated to a sentence, most common journey, biggest bottleneck, business/funnel highlights, and the existing recommendations list), embeds the business-insight charts that were previously computed but never shown in the report, and adds a click-to-zoom lightbox (vanilla JS/CSS, no dependencies) since the process-map diagrams are unreadable at thumbnail size. `generate_segment_comparison_report()` is a new function bringing the same treatment (executive summary, comparison table, per-segment happy paths, lightbox) to segment comparison, with its own "Download Segment Comparison Report" button — closing the gap where segment comparison had no export at all.
- **4d. Business insights: bug fixes and new metrics — Complete.** Found and fixed three correctness bugs in `analyze_repeat_purchases()`, each with a reproduced-before-and-after regression test: revenue/price values alone were being treated as purchase evidence (GA4-style logs attach `event_value`/`price` to browsing events too, so cart-abandoners were counted as buyers); order value was `max(price)` across every row in a case instead of the actual purchase event (a merely-viewed pricier item could be reported as the order total); and three different, inconsistent default `purchase_values` lists existed across `analytics.py`/`config.py`/`pipeline.py`, one of which false-matched GA4's `add_payment_info` checkout step. Added cart abandonment rate, average order value, category-level revenue breakdown, and a revenue-over-time trend to the same function, plus a new `analyze_conversion_funnel()` (cases reaching each stage, drop-off between stages, explicit or auto-derived stage order). A dedicated **Funnel** tab lets the user define their own funnel from any activities in the log, in any order — explicitly industry-agnostic, since auto-derivation alone skews toward e-commerce-shaped funnels and isn't a substitute for the user's own process knowledge.

## Phase 4b — Optimization

**Status: Complete.** `cores` exposed in the UI (below), `compare_segments()`
parallelization, pipeline profiling, and Streamlit-layer caching are all
shipped — see `dev_optimization.md` for measured results. The one remaining
item (pre-discovery downsampling for very large logs) is explicitly
deprioritized pending a concrete pain report.

Went through vectorization, clustering, batching, multiprocessing, and CUDA against the actual codebase rather than in the abstract:

- **Vectorization**: already done where it matters — `analytics.py` is groupby/agg-based throughout. No meaningful remaining opportunity.
- **Clustering**: already implemented as a speed technique, not just an idea — `optimize_variants` in `prox/conformance.py` groups identical traces and aligns once per unique variant instead of once per case (a documented 10-100x speedup on the most expensive stage). If "clustering" instead means auto-discovering segments as a feature (rather than using a known column), that's closer to segment comparison v2 than to optimization — kept separate so the two don't get conflated.
- **Batching**: already implemented for fitness (`calculate_fitness_in_batches`, batch_size=200 with `gc.collect()`) and CSV loading (chunked above 50MB).
- **Multiprocessing**: wired but unexposed. `prox/conformance.py` already passes a `cores` parameter all the way through to PM4Py's alignment computation (`params = {'cores': max_cores, ...}`, conformance.py:145), using PM4Py's own internal multiprocessing pool — this is not GIL-blocked at all, since separate processes each get their own interpreter. It's simply never surfaced in `main.py`'s UI, so every run defaults to single-core.
- **CUDA: considered and rejected, not revisitable.** PRoX's expensive operations (Petri net discovery, alignment-based conformance, token replay) are combinatorial/graph algorithms, not the dense matrix math GPUs accelerate. The one place with real linear algebra (the alignment heuristic's LP relaxation) runs many small, independent per-trace solves — exactly the pattern where GPU kernel-launch/transfer overhead dominates and erases any benefit, short of research-grade batched-GPU-LP-solver work. Requiring CUDA would also mean requiring an NVIDIA GPU, directly contradicting the README's "designed to run locally on a standard laptop" goal and excluding every Mac user outright.

**Proposed scope**: (1) expose `cores` as a UI control, since the wiring already exists — **done**; (2) profile the pipeline against a realistically-large synthetic log to find actual bottlenecks empirically rather than guessing further — **done, see `dev_optimization.md`**; (3) check whether precision calculation would benefit from the same variant-dedup trick alignments already use — **resolved by (2)'s profiling data: deprioritized, precision/token-replay conformance is not the bottleneck at scale (discovery and visualisation are).**

## Phase 5 — Incremental analysis (flagged, not scoped)

Deferred from Phase 4: an incremental/cached analysis mode for recurring large logs, so re-running PRoX on a growing dataset doesn't reprocess everything from scratch each time. Flagged rather than scoped because there's no concrete pain signal yet — no evidence of repeat-large-log usage in this project so far — and it's the most architecturally invasive item under discussion (would touch caching, log diffing, and pipeline re-entry points that don't exist today). Revisit once a real use case actually hits this.
