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

Last assessed 2026-08-21, against `main` @ `52b8275` — 140 tests passing,
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
| Phase 5 — BigQuery live data source (via `foe.data`) | Complete | below |
| Phase 6 — Session insight, reporting & data controls | Complete | below |
| Phase 6b — Full-pipeline correctness pass | Complete | below |
| Incremental analysis | Flagged, not scoped | `dev_phase2.md` |
| ML layer (conversion propensity + drivers) | Roadmapped | `ML_roadmap.md` |
| AI-assisted recommendations (optional, Gemini) | Roadmapped | `AI_summary_roadmap.md` |
| Process mining capability gaps (5 items, by effort) | Roadmapped, not scoped | below |
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

### Phase 5 — BigQuery live data source (via `first-order-engine`'s `foe.data`)

**Shipped 2026-08-20** (PR #28, `bigquery_source.py`) — kept in full below
as the design record; see the "Resolved" note under Open Questions at the
end for how those were actually settled.

**Idea**: instead of requiring the user to export, clean, and upload a CSV,
let PRoX connect directly to BigQuery and query GA4 event data live.
Positioned as a second, separate data-source workflow alongside the
existing CSV upload — not a replacement.

**Update 2026-08-20**: most of what this section originally scoped from
scratch (OAuth flow, BigQuery client/session handling, dry-run cost
estimation, GA4 SQL builders) already exists, built and tested, in a
sibling project: [`first-order-engine`](https://github.com/BasLinders/first-order-engine)'s
`foe.data` package. PRoX should consume it rather than reimplement it —
this turns the item below from "build a BigQuery integration" into "wire
an existing engine into the Streamlit UI."

#### Why

Today's flow (`main.py`) gates the entire app behind a single upload widget:
`uploaded_file = st.file_uploader(...)` in the sidebar, then `if not
uploaded_file: st.stop()` before anything else renders. For a user whose
event data already lives in BigQuery (a common case for GA4/web-analytics
exports), that means an export-clean-upload round trip every time they want
to look at a different date range or dataset. A live connection removes
that round trip entirely.

#### What `foe.data.DataEngine` already provides

- **Framework-free OAuth** (`build_auth_url` / `exchange_code`) — the PKCE
  verifier and any caller state are encoded into the OAuth `state` param
  itself, so no server-side session storage is needed to survive the
  redirect. This is a good match for Streamlit specifically, whose reruns
  don't give you a durable server-side session the way a typical web
  framework does. `refresh_if_expired`, `credentials_to_dict` /
  `credentials_from_dict` round out token lifecycle handling.
- **Discovery**: `list_projects()`, `list_datasets(project)` for a
  project/dataset picker — no free-form SQL editor needed for v1.
- **Cost safety**: `dry_run(sql)` estimates bytes scanned before executing;
  `monthly_usage(dataset)` tracks the 1TB/month free tier.
- **`extract_event_log(EventLogExtractionParams, limit=0)`** — this is the
  key piece: it already returns exactly the shape PRoX's engine expects,
  one row per `(case_id, activity, timestamp)`, sourced from a GA4
  `events_*` export. `case_id_col` defaults to `user_pseudo_id`,
  `activity_col` to `event_name`; both are overridable to any top-level or
  dotted struct column. Optional `event_names` restricts which events are
  pulled, `attribute_params` unnests extra `event_params` keys as columns
  (e.g. `page_location`), and an optional user-scoping filter
  (`UserFilterType.CONTAINS` / `REGEX` / `EVENT`) narrows to a subset of
  users. The returned DataFrame can go straight into PRoX's existing
  validation path.
- Gated behind the `foe[bigquery]` extra — importing `foe.data` itself
  never requires `google-cloud-bigquery`, only instantiating `DataEngine`
  does. Same "don't force Google auth deps on CSV-only users" property
  this section originally asked for, already built in.

#### Proposed shape

- **A data-source choice, shown first** — before the existing sidebar
  controls become interactive, similar to a landing step: "Upload CSV" or
  "Connect to BigQuery." This matches the existing gating pattern (`st.stop()`
  until data is ready) but adds a fork before it, rather than replacing it.
- **BigQuery path**:
  1. "Sign in with Google" — `DataEngine.build_auth_url(client_id,
     client_secret, redirect_uri)` using OAuth client credentials read from
     `st.secrets["bigquery"]` (see `.streamlit/secrets.toml`, prepared
     below). Store the returned `verifier` in `st.session_state` as a
     belt-and-suspenders measure (it's also recoverable from `state`
     alone, per the module's own design). On the callback, call
     `DataEngine.exchange_code(...)` and stash the resulting `Credentials`
     via `credentials_to_dict()` in `st.session_state` — never on disk.
  2. Build a `DataEngine` via `DataEngine.from_credentials(...)`, then
     `list_projects()` / `list_datasets(project)` for a picker rather than
     free-form SQL, for the same "no accidentally-expensive or destructive
     query" reasons the original scoping called for.
  3. Construct `EventLogExtractionParams` (connection, date range,
     case/activity column choice, optional event-name/attribute filters),
     call `dry_run()` on the generated SQL first and show the estimated
     bytes/free-tier percentage before running anything.
  4. Call `extract_event_log(params)`, get a DataFrame back, and feed it
     into the same validation/cleaning path the CSV upload already uses.
- **Shared validation logic**: `load_and_validate_csv()` currently mixes
  CSV-specific concerns (file-size checks, chunked reading) with genuinely
  reusable logic (column auto-mapping against `COLUMN_MAPPINGS`, composite
  case-ID creation, timestamp parsing, critical-column validation). This is
  a real refactor opportunity: split it into `_load_csv_source(...)` (CSV-only)
  and a shared `validate_and_clean_dataframe(df, ...)` that both the CSV path
  and the new BigQuery path call. Avoids duplicating the column-mapping and
  cleaning logic in two places. `extract_event_log`'s output already uses
  `case_id`/`activity`/`timestamp` as column names, which simplifies the
  mapping step considerably versus a raw GA4 table.

#### Dependencies and scope

- New dependency: `first-order-engine[bigquery]` (which itself pulls in
  `google-cloud-bigquery`, `google-auth-oauthlib`, and
  `google-cloud-resourcemanager`). Kept as an **optional extra** in
  `requirements.txt`/`setup.py` — most users running the CSV-only workflow
  shouldn't need to install or configure Google auth libraries at all.
  Consistent with the "runs on a standard laptop" design goal already
  documented in the README.
- OAuth `client_id` / `client_secret` / `redirect_uri` live in
  `.streamlit/secrets.toml` (gitignored, per-deployment) — see below.
  Exchanged `Credentials` live only in `st.session_state` for the session;
  never persisted to disk.
- **Out of scope for v1**: writing back to BigQuery (not needed — PRoX is
  read-only by design, and `DataEngine` itself never issues DDL/DML for
  this recipe), scheduled/incremental refresh (this is the same territory
  as Phase 5's incremental-analysis idea above and should stay deferred
  alongside it), multi-account switching, and query-cost governance beyond
  the basic dry-run estimate.

#### Secrets scaffold (prepared)

`.streamlit/secrets.toml` (gitignored) and a checked-in
`.streamlit/secrets.toml.example` now exist with the `[bigquery]` keys
`DataEngine`'s OAuth flow needs (`client_id`, `client_secret`,
`redirect_uri`) plus the default `BQConnectionConfig` fields (`project`,
`dataset`, optional `location`) so a picker has sane defaults before the
user has authenticated. Values are placeholders — fill in from the GCP
OAuth client used for this deployment.

#### Utility assessment — resolved (2026-08-20)

`foe.data.sql.event_log.build_event_log()` was reassessed against PRoX's
actual consumption path (`load_and_validate_csv()`'s `CRITICAL_COLS`,
`COLUMN_MAPPINGS`, and `analytics.py`'s revenue/user-column resolution).
Three real gaps were found and have since been fixed upstream in
`first-order-engine`:

- **No separate `user_id` column** (needed for PRoX's composite key) —
  fixed via `include_user_id` (default `True`), which emits
  `user_pseudo_id AS user_id` alongside `case_id`.
- **No session-level case granularity** (only user-for-the-whole-range) —
  fixed via `session_id_param`: pass `'ga_session_id'` and the query pulls
  the nested int-valued event_params key via a correlated subquery, cast
  to STRING, instead of a flat `case_id_col`. Paired with `include_user_id`,
  this produces exactly the (user_id, session_id) pair PRoX's composite key
  expects — mutually exclusive with a non-default `case_id_col` (enforced
  by a model validator).
- **No numeric revenue** (`attribute_params` only reads `string_value`,
  numeric params come back NULL) — fixed via `include_purchase_revenue`
  (adds `ecommerce.purchase_revenue AS revenue`, a flat typed `FLOAT64`
  column) and the more general `numeric_attribute_params` for other numeric
  event params.

**Recommended `EventLogExtractionParams` defaults for the PRoX
integration**: `session_id_param="ga_session_id"`, `include_user_id=True`,
`include_purchase_revenue=True`, `activity_col="event_name"` (default).
With these, `extract_event_log()`'s output needs zero PRoX-side column
mapping changes — `case_id`/`activity`/`timestamp`/`user_id`/`revenue` all
land on existing `COLUMN_MAPPINGS` entries.

**New consideration surfaced during this check**: `foe`'s `pyproject.toml`
lists `prophet`, `statsmodels`, `pingouin`, and `patsy` as unconditional
base dependencies (not gated behind the `bigquery` extra), since
`foe.data` is a subpackage of the whole `foe` library — installing
`foe[bigquery]` for just `DataEngine` also pulls in Prophet, which
typically needs a compiled Stan backend on first install. Worth stating
plainly in install docs as the known cost of this optional path, since it
cuts against PRoX's "no compilation step, runs on a standard laptop"
positioning. Also: `foe` requires Python ≥3.10 vs. PRoX's README-stated
3.9+ base requirement — noted directly in `requirements.txt`'s `[bigquery]`
comment as shipped, so this doesn't need a separate doc update.

#### Open questions to resolve before implementation

- Where does OAuth client registration (GCP project, redirect URI) live —
  is this a per-deployment config the person running PRoX sets up once
  (the `.streamlit/secrets.toml` approach above assumes this), or something
  each end user configures themselves? The secrets-file approach only
  covers the former; a multi-tenant deployment would need a different
  answer.
- `event_names`/`attribute_params`/`numeric_attribute_params` need a UI
  decision: expose as advanced/overridable fields, or hardcode sane
  defaults for v1 and revisit if a real dataset needs otherwise.

**Resolved (2026-08-20, as shipped):** OAuth client registration is a
per-deployment `.streamlit/secrets.toml` config the person running PRoX
sets up once - the multi-tenant case was out of scope. `event_names` is
exposed as an advanced, comma-separated text field ("restrict to specific
events"); `attribute_params`/`numeric_attribute_params` were left
hardcoded to sane v1 defaults rather than exposed, since no real dataset
has needed otherwise yet.

### Phase 6 — Session insight, reporting & data controls

- **Case grouping default switched to user-level** - `case:concept:name`
  now defaults to `user_id` (previously always a `user_id + session_id`
  composite), so a case can span a user's whole session history instead of
  just one session. A per-user-scoped `session_id` column is always kept
  regardless, and a UI toggle switches back to the previous per-session
  grouping.
- **Session-level intent classification** - `classify_sessions()` /
  `summarize_user_journeys()` in `prox/analytics.py`: a transparent,
  priority-ordered rule set (Buying > Cart Abandonment > Researching >
  Browsing) labels every session from its activities, then rolls a user's
  session labels into a chronological journey string (e.g. "Browsing ->
  Researching -> Buying"). Surfaced in a new **Session Insights** tab.
- **Configurable process end point** - a "Process end point" selector in
  the Filter Events step anchors analysis to a chosen activity (defaulting
  to `purchase` when present) via the existing `crop` filter, instead of
  that only being settable in code.
- **Modular, opt-in PDF report builder** (`pdf_builder.py`) - a separate
  tool from `prox/report.py`'s all-in-one `generate_html_report()`: checks
  per available results tab (Process Maps, Variants, Bottlenecks,
  Conformance, Funnel, Business Insights, Session Insights, Segment
  Comparison), builds a PDF containing only what's checked, via `reportlab`
  (pure Python, no system rendering dependency - no wkhtmltopdf binary, no
  Cairo/Pango).
- **Smoothly ticking progress percentage** - the analysis progress bar
  previously only updated at 6 coarse pipeline-stage boundaries, sitting
  frozen for long stretches (especially during State Equation A*
  conformance). `run_full_analysis` now executes on a background thread
  while the main thread drives the bar from elapsed time via an asymptotic
  curve (capped at 95% until actually done), independent of the real
  per-stage callback, which still supplies the stage label text.
- **Sampling stratification exposed in the UI** - `strata_col` /
  `max_priority_ratio` (stratified sampling that reserves part of the
  conformance sample for cases where a chosen column = 1, e.g. purchases,
  so they aren't sampled away) existed in the pipeline but were hardcoded
  to `'purchase'` and never surfaced. Now a "Prioritise a column when
  sampling" selector (binary 0/1-style columns only) plus a max-priority-
  share slider in the Sampling step.
- **Opt-in revenue/price winsorization** - `prox.winsorize_series()` (same
  technique as first-order-engine's
  `ContinuousMetricEngine.winsorize_series`: cap at mean +/- N std devs, or
  a percentile band) applied to the revenue/price column right after data
  is loaded/cached, before any filtering/sampling/analysis reads it - caps
  outlier values instead of dropping the rows, so Average Order
  Value/revenue trend/category breakdown aren't diluted by a handful of
  extreme orders.

### Phase 6b — Full-pipeline correctness pass

A deliberate line-by-line review across `prox/discovery.py`,
`conformance.py`, `analytics.py`, and `data_manager.py`, prompted by the
Phase 6 work above. Each finding was verified empirically or by direct
reproduction before fixing, with a regression test added per fix:

- Inductive Miner discovery called `inductive_miner.apply()` without
  `variant=Variants.IMf`, so the Noise Threshold slider had been a
  complete no-op - noise filtering only exists under `IMf`, not the
  default `IM` variant.
- `_fitness_state_equation_alignments()` accepted `initial_marking`/
  `final_marking` but never used them, instead guessing markings from net
  topology - usually harmless for a discovered net, but capable of
  silently corrupting fitness/alignments for reference-model topologies
  where the real start/end doesn't coincide with sourceless/sinkless
  places. Now uses the real markings, matching the sibling
  `_fitness_token_replay()`.
- Purchase/cart/research-keyword matching built a regex via
  `'|'.join(values)` with no guard for an empty list - `'|'.join([])` is
  `''`, and `str.contains('')` matches every row, silently misclassifying
  everything as a match instead of nothing. Fixed at all 5 call sites via
  a shared `_contains_any()` helper.
- `refine_activity_labels()` decided whether to apply URL-cleaning based
  only on the first matched row's value, applying that one decision to the
  whole column - a column mixing plain and URL-like values leaked raw
  query strings/slashes into activity names for every row that didn't
  match the first row's style. Cleaning is now applied per row.

---

## In progress

Nothing currently in progress.

---

## Roadmapped (not yet scheduled)

### Incremental analysis (flagged, not scoped)

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

### Process mining capability gaps

Surfaced from an honest sophistication assessment (2026-08-21): PRoX is a
real discovery/conformance engine (Inductive Miner, Heuristics Miner, DFG,
Token Replay, State Equation A\* alignments, reference-model conformance)
with a genuine e-commerce/CRO-specific analytical layer on top - ahead of
a hobby pm4py script, behind an enterprise platform like Celonis or Disco.
The gap is concentrated in five capabilities those platforms have that
PRoX doesn't yet. Listed in order of engineering effort, smallest first -
none of these are scoped or scheduled.

1. **Organizational/resource-perspective mining.** `analyze_process_
   performance()` already reports basic per-resource event counts when a
   resource column exists, but stops there. A handover-of-work network
   (who hands cases to whom, and how often) and per-resource workload/
   throughput metrics are a natural, contained extension of that existing
   code path - no new data requirements, no new UI paradigm, just deeper
   aggregation on a dimension PRoX already partially reads.

2. **Interactive process explorer.** Process Maps and Segment Comparison
   render static matplotlib/Graphviz images today, not a clickable,
   filterable, animated flow view. Real lift, but bounded: an existing
   interactive graph component (or a custom vis.js/d3 embed) replacing the
   current image-based rendering, with click-to-filter wired back into the
   existing filter/config state Streamlit already manages.

3. **Decision-point (data-aware) mining.** Explaining *why* a case took
   one branch over another at a choice point - e.g. "cases with
   `device=mobile` skip the comparison step 80% of the time" - needs new
   machinery: identifying XOR choice points in the discovered process
   tree/Petri net, then correlating case/event attribute values observed
   before each choice with which branch was actually taken (a decision
   tree per choice point is the standard approach). No equivalent code
   exists yet to build on.

4. **Time-perspective prediction (remaining-time/SLA forecasting).**
   Today's timing analysis is descriptive (bottleneck durations, lead
   time) - not "given a case is currently at step X, when will it
   finish, and is it at risk of breaching an SLA?" That needs a
   trained-per-process-state predictor and a validation methodology, put
   this in the same effort class as the already-roadmapped ML layer
   (`ML_roadmap.md`) - plausibly an extension of it rather than a fully
   separate build.

5. **Multi-tenant / hosted deployment layer.** Authentication, session
   isolation, persisted (not just `st.session_state`) analysis storage,
   and audit logging - the architecture change from "single local
   Streamlit process on one analyst's laptop" (today's explicit design
   goal, per the README) to a shared, hosted, multi-user service. By far
   the largest lift here: it's not a new analytical capability but a
   different deployment model for the whole application, touching
   caching, storage, and access control throughout. Deliberately out of
   scope unless that positioning itself changes.

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

- **Incremental analysis** for recurring large logs (above). Still
  explicitly "no pain signal yet" — worth doing once someone is actually
  re-running PRoX on a growing dataset regularly, not before.
- ~~BigQuery live data source~~ — shipped as Phase 5 (above).

