# AI Summary Roadmap

*See `dev_roadmap.md` for the current status of all phases at a glance —
that file is the leading document; this one is the detail.*

Generative-AI feature ideas for PRoX, kept separate from `ML_roadmap.md`
(predictive/statistical ML) since the risk profile is different again:
this isn't about probabilistic predictions, it's about sending data to a
third-party API and getting back generated prose. That raises data-handling
and privacy questions `ML_roadmap.md`'s locally-trained models don't, on
top of the same "how much do we trust generated content next to
deterministic numbers" question. Entries here are scoped for discussion,
not committed to a phase number yet.

## Status: shipped as "AI Conclusion" (2026-09-29, #51)

The section below is kept as the design record. This is what was actually
built, and where it differs from that design:

- **Scope: a conclusion, not only recommendations.** An "AI Conclusion"
  expander under the results tabs asks Gemini for a plain-language
  summary, key findings and next steps. It sits apart from the
  deterministic Executive Summary and doesn't replace its rule-based
  recommendations. It is labelled "AI-generated - check it against the tabs
  above before sharing it".
- **Modules live in `utility/`, not `prox/`.** `utility/ai_payload.py`
  (`build_ai_payload()`) builds the digest, and `utility/ai_client.py`
  (`generate_conclusion()`) calls Gemini. The core engine makes no network
  calls, as planned.
- **Payload.** Aggregate only. It never includes case, user or session IDs,
  and it leaves out per-resource rows because resource names are often
  people. Ranked lists are capped at the top 5 entries. The payload carries a
  `context` block (case grouping, sampling, filters, algorithms) and covers
  the follow-up analyses the user has run in their tabs: reference-model
  conformance, funnel by segment, segment comparison and the propensity
  model. Analyses that weren't run are listed under `not_run`, so the model
  doesn't guess at them. The UI shows which sections are included and which
  aren't, and the exact JSON payload (collapsed) before anything is sent,
  with a note that labels from the event log go out as they are.
- **Prompt.** It tells the model to use only figures that appear in the
  data. It also explains that discovered-model conformance is a
  self-consistency check, not compliance, and asks the model to call a
  propensity model with ROC AUC below 0.65 weak. Every label from the event
  log is treated as untrusted data, never as instructions, to guard against
  prompt injection.
- **Model and SDK.** It uses `google-genai` (not `google-generativeai`),
  with `pip install -e ".[ai]"`. The default model is
  `gemini-3.5-flash-lite`, with `gemini-flash-latest` as a fallback on 503
  or 429 responses, and 2 retries with backoff for each model. The
  conclusion can be written in Dutch (the default) or English.
- **API key.** It is read from `GEMINI_API_KEY` in `.streamlit/secrets.toml`
  rather than typed into a password field. Without a key, the button is
  disabled and a hint is shown. A note under it says the user's own key and
  Google's Gemini API pricing and quota apply.
- **Reports.** The conclusion can be added as a section of the custom PDF
  report, and the "Download Full Report" HTML includes it in its own
  "AI Conclusion" box after the Executive Summary, frozen in as escaped
  plain text (the file never calls the API). Each conclusion is
  fingerprinted by its payload: if the results change afterwards, the UI
  flags it as out of date and leaves it out of both reports until it is
  generated again. Generation only runs on a button click, so reruns don't use API
  quota.

**Open questions below: resolved.** The allowlist is the set of section
builders in `build_ai_payload()`. The leakage regression test exists as
`tests/test_ai_payload.py::test_payload_never_contains_case_or_user_ids`.
The deterministic recommendations and the AI conclusion are shown in
separate places.

**Follow-ups from the design: shipped 2026-09-30.** The payload preview,
the cost note and the HTML report section (above). Nothing from the design
is still open.

## AI-assisted recommendations (optional, Gemini)

**Idea**: an opt-in, off-by-default feature that sends an aggregated,
already-computed summary of the analysis (not the event log) to Gemini and
displays the generated recommendations in a clearly-labeled, separate
section — distinct from the deterministic, threshold-based recommendations
`_generate_performance_recommendations()` already produces.

### Why

Market pressure to have *some* AI-branded capability is real, independent
of whether a given AI feature is the most useful thing to build next.
Rather than an LLM-generated Executive Summary (which would put a
paraphrasing model in charge of numbers that currently have to be exactly
right — PRoX's whole non-technical-stakeholder pitch depends on that),
narrowing the AI surface to *recommendations* keeps every number in the
report deterministic and auditable, and confines "AI could get this
slightly wrong" to a section that's inherently interpretive advice, not a
measured fact. Smallest surface area of the options considered, cheapest
to build, easiest to demo, and the one place in the report where "AI adds
color a fixed template can't" is actually true rather than aspirational.

### The core constraint: PII

The payload sent to Gemini must be a **hand-curated allowlist of
already-aggregated fields** — never a blanket serialization of the
`results` dict. That distinction matters concretely:
`results['conformance']['case_analysis']['cases']` contains a `case_id`
per entry, which is the composite `user_id + session_id` key
`load_and_validate_csv()` builds — a naive "just send the results dict"
implementation would leak that straight to Google's servers on every call.

**Allowlisted for the payload** (all aggregate, zero per-case/per-user data):
- `log_summary` — counts only (cases, events, activities, duration)
- `performance.summary_statistics` — health score, variability, bottleneck ratio
- `conformance.overall_summary` — fitness/precision/quality scores
- `performance.bottlenecks.summary` — bottleneck *activity names* (process
  labels, not personal data)
- `funnel_analysis.stages` — stage names + aggregate counts/percentages
- `repeat_purchase_analysis.metrics` — repeat rate, AOV, cart abandonment
  (all aggregate)

**Explicitly excluded**: `conformance.case_analysis.cases` (has case IDs),
any raw DataFrame, anything at row-level.

**Residual risk, stated honestly**: the allowlist protects against
*structural* leakage (IDs, timestamps, per-row data) — it can't protect
against a client's own activity or category names happening to contain
something sensitive (e.g. a badly-named event or category string). Worth
a one-line disclaimer in the UI rather than a false promise of zero risk.

**Best mitigation beyond the allowlist**: a "Preview payload" expander that
shows the *exact* JSON before anything is sent, so a cautious client can
verify what leaves their machine instead of trusting a privacy claim in
prose.

### Proposed shape

- New isolated module, `prox/ai_summary.py` — keeps network calls
  entirely out of the core deterministic engine, same boundary discipline
  proposed for the BigQuery data source in `dev_roadmap.md`.
- `generate_ai_recommendations(results, api_key, model=...)` builds the
  allowlisted payload (a dedicated function, unit-testable in isolation
  from the network call) and calls the Gemini API. Gemini-only for v1;
  structured behind a thin wrapper so swapping providers later is a small
  change, not a rewrite.
- **UI** — a collapsed "AI Recommendations (optional)" section, off by
  default, nothing sent anywhere unless a client actively opens it:
  1. Password-style API key field, stored only in `st.session_state`,
     never written to disk (same handling as the BigQuery credential
     precedent).
  2. "Preview data to be sent" — renders the literal JSON payload before
     anything is transmitted.
  3. "Generate" button, only live once a key is entered.
  4. Output rendered in a visually distinct block (e.g. "AI-generated, not
     independently verified") — never mixed into the deterministic
     Executive Summary text, preserving that trust boundary.
  5. Session-cached by payload hash, so re-rendering the UI doesn't burn
     API quota regenerating the same recommendations.
- **Static HTML report interaction**: the downloadable report is meant to
  be a shareable, fully offline file. If AI recommendations were generated
  in-session before download, they get **frozen into the HTML as plain
  text at export time** — the report itself must never make a live API
  call when opened later; a downloaded file phoning home on open would be
  its own privacy problem, separate from the one this feature is trying
  to solve.
- **Failure handling**: network/auth/rate-limit failures degrade
  gracefully — inline error, everything else in the app keeps working.
  This is a bolt-on layer, never a dependency for anything core.

### Dependencies and scope

- New dependency: `google-generativeai` (Gemini SDK). Optional extra
  (e.g. `pip install -e ".[ai]"`), not a hard requirement — consistent with
  how `ML_roadmap.md`'s scikit-learn dependency and `dev_roadmap.md`'s
  BigQuery dependency are both scoped as opt-in.
- Cost transparency: Gemini API calls cost money (a free tier exists but
  isn't unlimited) — the UI should say plainly that the client's own key
  and Google's standard pricing apply, so nobody is surprised.
- **Out of scope for v1**: provider abstraction beyond the thin wrapper
  (Gemini only), streaming responses, multi-turn refinement/chat, and
  including AI recommendations in the segment-comparison report (start
  with the single-run report only).

### Open questions to resolve before implementation

- **Exact payload allowlist.** The list above is a first pass — needs
  review against every key actually present in `results` to make sure
  nothing case-level or row-level is reachable through a nested field
  that wasn't considered (e.g. would `category_breakdown`'s keys ever be
  something other than a product category?).
- **A payload-leakage regression test.** A test asserting the payload
  builder's output never contains `case_id`, `user_id`, or any row-level
  key, in the same spirit as the existing XSS-escape tests on
  `generate_html_report()`. This should exist before the feature ships,
  not after.
- **Where this sits relative to the deterministic recommendations
  already in the Executive Summary.** Shown side by side? Replacing the
  rule-based list when a key is present? Needs a decision that keeps the
  distinction between "measured" and "generated" visually unambiguous.
