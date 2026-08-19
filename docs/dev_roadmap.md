# PRoX Development Roadmap

**This is the leading document for PRoX development status.** Every phase —
completed, in progress, or roadmapped — is tracked here first. Where a
phase has enough detail to warrant its own file, this document carries an
accurate summary and links out to it: `dev_phase2.md` for the phase-by-phase
execution log (tests/CI/extensibility/capability work), `dev_optimization.md`
for performance work, `ML_roadmap.md` for predictive/statistical ML ideas,
`AI_summary_roadmap.md` for generative-AI ideas. Keep the summaries here
current even when the detail lives elsewhere — this is the one page meant
to answer "where does PRoX development actually stand?" without opening
five files.

Last assessed 2026-08-19, against `main` @ `a566875` — 90 tests passing,
`pyflakes` clean.

---

## Status at a glance

| Phase | Status | Detail |
|---|---|---|
| Phase 1 — Core correctness | Complete | `dev_phase2.md` |
| Phase 2 — Safety net (tests, CI, housekeeping) | Complete | `dev_phase2.md` |
| Phase 3 — Extensibility (registries) | Complete | `dev_phase2.md` |
| Phase 4 — Expand capability | Complete* | `dev_phase2.md` |
| Phase 4b — Optimization | Complete | `dev_optimization.md` |
| Phase 5 — Incremental analysis | Flagged, not scoped | `dev_phase2.md` |
| ML layer (conversion propensity + drivers) | Roadmapped | `ML_roadmap.md` |
| AI-assisted recommendations (optional, Gemini) | Roadmapped | `AI_summary_roadmap.md` |
| BigQuery live data source | Roadmapped | below |
| Product development suggestions | Roadmapped | below |

\* One sub-item — segment comparison v2 (automated golden-path diffing) —
is deliberately deferred; see "Medium bets" below. Everything else under
Phase 4 is shipped.

---

## Completed phases (summary)

### Phase 1 — Core correctness
Token-based replay genuinely implemented (`pm4py.fitness_token_based_replay`),
`create_analysis_config()` fully parameterized, DFG discovery exposed in the
UI alongside Inductive and Heuristics Miner. Caught and fixed a real pm4py
API-drift bug in the DFG-to-Petri-net conversion along the way. Full detail
in `dev_phase2.md`.

### Phase 2 — Safety net
Test suite (`tests/`, now 74 tests across the whole engine), CI
(`.github/workflows/ci.yml` — pyflakes then pytest on every PR/push to
`main`), `.gitignore`, and pinned dependency upper bounds. Caught and fixed
a real silent bug in `optimize_dataframe_memory()` while writing its test
(pandas 2.x+/3.x's `str` dtype wasn't matched by the old `== 'object'`
check). Full detail in `dev_phase2.md`.

### Phase 3 — Extensibility
Discovery, conformance, and filter dispatch replaced with registries
(`DISCOVERY_ALGORITHMS`, `CONFORMANCE_METHODS`, `FILTER_HANDLERS`); the UI
now derives its selectbox options and help text from the same registries
instead of a second hardcoded list, so adding a new algorithm or filter type
means one registry entry, not an "edit two files" seam. Full detail in
`dev_phase2.md`.

### Phase 4 — Expand capability
- **HTML report export**, later overhauled with a plain-language Executive
  Summary (health verdict, translated fitness/precision, most common
  journey, biggest bottleneck, business/funnel highlights), embedded
  business-insight charts, and a click-to-zoom lightbox for the
  process-map diagrams. `generate_segment_comparison_report()` brings the
  same treatment to segment comparison, with its own download button.
- **Segment comparison v1** — compare health score, fitness, precision,
  repeat rate, and happy path across the top-N values of any column.
  Parallelized in Phase 4b.
- **Business insights** — fixed three correctness bugs in
  `analyze_repeat_purchases()` (revenue/price values alone were treated as
  purchase evidence, causing cart-abandoners to be counted as buyers;
  order value used `max(price)` across a whole case instead of the actual
  purchase event; three different, inconsistent `purchase_values` defaults
  existed across the codebase). Added cart abandonment rate, average
  order value, category-level revenue breakdown, and a revenue-over-time
  trend.
- **Funnel analysis** — `analyze_conversion_funnel()` plus a dedicated
  **Funnel** tab, letting the user define their own funnel from any
  activities in the log, in any order (industry-agnostic by design — not
  just e-commerce), with auto-derivation from the data as a fallback
  starting point.
- **Deferred**: segment comparison v2 (automated golden-path diffing) —
  see "Medium bets" below.

Full detail in `dev_phase2.md`.

### Phase 4b — Optimization
`compare_segments()` parallelization (~2x wall-clock on a 4-core machine),
pipeline profiling against realistic synthetic logs (found discovery and
visualisation dominate at scale, not conformance — conformance is capped by
sampling), and Streamlit-layer caching (~200x on a repeat run with
unchanged inputs). Found and fixed a real correctness bug along the way:
`optimize_dataframe_memory()` was converting `case:concept:name`/
`concept:name` to `category` dtype, which `pm4py.convert_to_event_log()`
rejects — silently breaking discovery on real event logs. Full detail in
`dev_optimization.md`.

---

## In progress

Nothing currently in progress.

---

## Roadmapped (not yet scheduled)

### Phase 5 — Incremental analysis (flagged, not scoped)

Deferred from Phase 4: an incremental/cached analysis mode for recurring
large logs, so re-running PRoX on a growing dataset doesn't reprocess
everything from scratch each time. Flagged rather than scoped because
there's no concrete pain signal yet — no evidence of repeat-large-log usage
in this project so far — and it's the most architecturally invasive item
under discussion (would touch caching, log diffing, and pipeline
re-entry points that don't exist today). Revisit once a real use case
actually hits this. Full detail in `dev_phase2.md`.

### Machine learning layer

A colleague-suggested direction: layering ML on top of the event log.
Scoped into a concrete first candidate — conversion propensity prediction
paired with a plain-language root-cause driver analysis — in
`ML_roadmap.md`, including feature engineering, model choice, validation
approach, and the open questions (minimum data volume, leakage risk,
overlap with the Funnel tab) to resolve before building it. Kept as its
own file since ML output is probabilistic and needs a different kind of
trust framing than PRoX's otherwise-deterministic metrics.

### AI-assisted recommendations (optional, Gemini)

A response to competitive pressure to have some AI-branded capability,
scoped narrowly on purpose: an opt-in, off-by-default feature that sends
an already-aggregated summary of the analysis (never the event log or any
row-level data) to Gemini and displays the generated recommendations in a
clearly-labeled, separate section — distinct from the deterministic
recommendations already in the Executive Summary. Full detail, including
the exact data-allowlist design that keeps case/user IDs from ever leaving
the machine, in `AI_summary_roadmap.md`. Kept as its own file since sending
data to a third-party API raises data-handling questions neither
`ML_roadmap.md`'s locally-trained models nor any of PRoX's other features
do.

### Product development suggestions

A working list of where the product could go next, given everything
shipped so far (discovery, conformance, bottlenecks, variants, business
insights, funnel analysis, segment comparison, HTML reporting). Roughly
ranked by effort vs. payoff — none of these are committed, scoped, or
sequenced yet.

#### Quick wins (small, builds on what already exists)

**Shipped:**
- **Funnel x Segment cross-analysis — done.** The Funnel tab has an optional
  "Split by segment" selector (same 2-20-unique-value column candidates as
  Segment Comparison). `analyze_funnel_by_segment()` in `prox/analytics.py`
  reuses the overall funnel's stage order for every segment, so results are
  directly comparable - "does mobile drop off earlier than desktop?" is now
  a chart and a table, not a manual cross-reference between two tabs.
- **Data-quality pre-check — done.** A "2. Data Quality Check" step now runs
  right after upload, before filtering/analysis. `check_data_quality()` in
  `prox/data_manager.py` flags exact duplicate events, single-event cases
  (no transitions to analyse), and events logged out of chronological order
  within a case - the log-shape problems `load_and_validate_csv()`'s own
  null/timestamp checks don't catch.

**Still open:**
- **Config presets.** Save/load the sidebar configuration (discovery algo,
  sample size, filters, funnel definition) as a small JSON file. Removes the
  "reconfigure everything every session" friction for a repeat analyst —
  directly serves the "runs on a laptop, used repeatedly" use case.

#### Medium bets (real feature work, clear value)

- **Segment comparison v2 — automated golden-path diffing.** Already scoped
  and deliberately deferred (`dev_phase2.md`'s Phase 4 segment-comparison
  entry: "segment A visits checkout, segment B doesn't"). Worth revisiting
  now that v1 has real usage patterns (parallel execution, its own report
  export).
- **Cohort/retention view.** Current loyalty metrics (repeat rate,
  days-between-purchases) are transaction-level. A cohort retention curve
  (% of users from cohort week N still active in week N+1, N+2, ...) is a
  different, complementary lens that product/growth stakeholders
  specifically look for and PRoX doesn't have yet.
- **Two-log comparison.** Compare this week's export vs. last week's, or
  this cohort's export vs. last month's — a temporal/version diff, distinct
  from segment comparison's categorical split of a single upload.

#### Longer-term (bigger, already flagged elsewhere in these docs)

- **Phase 5 — incremental analysis** for recurring large logs (above).
  Still explicitly "no pain signal yet" — worth doing once someone is
  actually re-running PRoX on a growing dataset regularly, not before.
- **BigQuery live data source** (below). Removes the export-clean-upload
  cycle for GA4-in-BigQuery users. Biggest lift of anything on this list —
  only worth it if CSV upload is a genuine recurring friction point.

### BigQuery live data source (via Google OAuth)

**Idea**: instead of requiring the user to export, clean, and upload a CSV,
let PRoX connect directly to BigQuery and query event data live. Positioned
as a second, separate data-source workflow alongside the existing CSV
upload — not a replacement.

#### Why

Today's flow (`main.py`) gates the entire app behind a single upload widget:
`uploaded_file = st.file_uploader(...)` in the sidebar, then `if not
uploaded_file: st.stop()` before anything else renders. For a user whose
event data already lives in BigQuery (a common case for GA4/web-analytics
exports), that means an export-clean-upload round trip every time they want
to look at a different date range or dataset. A live connection removes
that round trip entirely.

#### Proposed shape

- **A data-source choice, shown first** — before the existing sidebar
  controls become interactive, similar to a landing step: "Upload CSV" or
  "Connect to BigQuery." This matches the existing gating pattern (`st.stop()`
  until data is ready) but adds a fork before it, rather than replacing it.
- **BigQuery path**:
  1. "Sign in with Google" — OAuth 2.0, scoped to `bigquery.readonly` only
     (PRoX is analysis-only; no write access should ever be requested).
  2. Once authenticated: either (a) a project/dataset/table picker that
     auto-generates a query, or (b) a free-form SQL editor for full
     flexibility. Recommend starting with (a) for v1 — safer, no risk of
     an accidentally-expensive or destructive query — and offering (b) later
     once there's a real need for arbitrary queries.
  3. Run a `dry_run` query first to estimate bytes scanned and warn the user
     before executing anything that could be costly, given BigQuery's
     pay-per-byte-scanned pricing.
  4. Execute, get a DataFrame back (via `google-cloud-bigquery`'s
     `to_dataframe()`), and feed it into the same validation/cleaning path
     the CSV upload already uses.
- **Shared validation logic**: `load_and_validate_csv()` currently mixes
  CSV-specific concerns (file-size checks, chunked reading) with genuinely
  reusable logic (column auto-mapping against `COLUMN_MAPPINGS`, composite
  case-ID creation, timestamp parsing, critical-column validation). This is
  a real refactor opportunity: split it into `_load_csv_source(...)` (CSV-only)
  and a shared `validate_and_clean_dataframe(df, ...)` that both the CSV path
  and a new BigQuery path call. Avoids duplicating the column-mapping and
  cleaning logic in two places.

#### Dependencies and scope

- New dependency: `google-cloud-bigquery` plus an OAuth flow library
  (e.g. `google-auth-oauthlib`). This should be an **optional extra**
  (e.g. `pip install prox[bigquery]`), not a hard requirement — most users
  running the CSV-only workflow shouldn't need to install or configure
  Google auth libraries at all. Consistent with the "runs on a standard
  laptop" design goal already documented in the README.
- Credentials live only in `st.session_state` for the session; never
  persisted to disk.
- **Out of scope for v1**: writing back to BigQuery (not needed — PRoX is
  read-only by design), scheduled/incremental refresh (this is the same
  territory as Phase 5's incremental-analysis idea above and should stay
  deferred alongside it), multi-account switching, and query-cost
  governance beyond the basic dry-run estimate.

#### Open questions to resolve before implementation

- Does the target BigQuery table already look like an event log (one row
  per event, with case/activity/timestamp-ish columns `COLUMN_MAPPINGS` can
  match), or would the query need to reshape data before it reaches PRoX?
  Likely varies per user/dataset — may need a lightweight "preview + confirm
  column mapping" step for the BigQuery path specifically, since there's no
  file to eyeball beforehand the way there is with a CSV.
- Where does OAuth client registration (GCP project, redirect URI) live —
  is this a per-deployment config the person running PRoX sets up once, or
  something each end user configures themselves? Affects how "opt-in" this
  really is in practice.
