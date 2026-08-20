"""
bigquery_source.py

Streamlit UI for PRoX's BigQuery data source: OAuth sign-in, project/dataset
picker, and GA4 event-log extraction via first-order-engine's
foe.data.DataEngine (https://github.com/BasLinders/first-order-engine).
Kept out of main.py so the OAuth/session-state bookkeeping this needs stays
isolated from the rest of the UI.

Returns extracted data as CSV bytes so it flows through the exact same
load_and_validate_csv() path the CSV-upload source already uses -- no
prox/ engine changes needed. extract_event_log() is called with
session_id_param/include_user_id/include_purchase_revenue set below so its
output (case_id, activity, timestamp, user_id, revenue) lands directly on
PRoX's existing COLUMN_MAPPINGS (see docs/dev_roadmap.md's "BigQuery live
data source" section for the full compatibility assessment).

Requires the optional `foe[bigquery]` extra and a filled-in
.streamlit/secrets.toml -- both degrade to a clear on-screen message rather
than a crash when missing, since most PRoX users run CSV-only.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd
import streamlit as st

try:
    from foe.data import DataEngine
    from foe.data.sql.event_log import build_event_log
    from foe.core.models import BQConnectionConfig, DateRange, EventLogExtractionParams

    _FOE_AVAILABLE = True
except ImportError:
    _FOE_AVAILABLE = False

_SESSION_KEYS = (
    "bq_credentials", "bq_oauth_verifier", "bq_extracted_csv", "bq_extracted_label",
)


def _secrets_ok() -> bool:
    if not hasattr(st, "secrets") or "bigquery" not in st.secrets:
        return False
    bq = st.secrets["bigquery"]
    return bool(bq.get("client_id") and bq.get("client_secret") and bq.get("redirect_uri"))


def render_bigquery_source() -> Optional[bytes]:
    """Renders the BigQuery connect/extract UI in the main area. Returns CSV
    bytes once an extraction has completed, else None."""
    if not _FOE_AVAILABLE:
        st.error(
            "The BigQuery data source needs the optional `foe[bigquery]` extra "
            "(first-order-engine). Install it with:\n\n"
            '`pip install "foe[bigquery] @ '
            'git+https://github.com/BasLinders/first-order-engine.git"`\n\n'
            "Note: this pulls in first-order-engine's core dependencies "
            "(including Prophet), which may need a compiled Stan backend on "
            "first install."
        )
        return None

    if not _secrets_ok():
        st.error(
            "BigQuery OAuth isn't configured yet. Fill in `client_id`, "
            "`client_secret`, and `redirect_uri` under `[bigquery]` in "
            "`.streamlit/secrets.toml` (see `.streamlit/secrets.toml.example` "
            "for where each value comes from)."
        )
        return None

    secrets = st.secrets["bigquery"]
    client_id = secrets["client_id"]
    client_secret = secrets["client_secret"]
    redirect_uri = secrets["redirect_uri"]

    creds_dict = st.session_state.get("bq_credentials")
    query_params = st.query_params

    if not creds_dict and "code" in query_params:
        try:
            credentials, _ = DataEngine.exchange_code(
                code=query_params["code"],
                state=query_params.get("state", ""),
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                verifier=st.session_state.get("bq_oauth_verifier"),
            )
            st.session_state["bq_credentials"] = DataEngine.credentials_to_dict(credentials)
            st.session_state.pop("bq_oauth_verifier", None)
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Google sign-in failed: {e}")
            st.query_params.clear()
            return None
        creds_dict = st.session_state.get("bq_credentials")

    if not creds_dict:
        auth_url, verifier = DataEngine.build_auth_url(client_id, client_secret, redirect_uri)
        st.session_state["bq_oauth_verifier"] = verifier
        st.link_button("Sign in with Google", auth_url, width='stretch')
        st.caption(
            "Read-only access to BigQuery (bigquery + cloud-platform.read-only "
            "scopes) - PRoX never writes to your data."
        )
        return None

    credentials = DataEngine.credentials_from_dict(creds_dict)
    credentials = DataEngine.refresh_if_expired(credentials)
    st.session_state["bq_credentials"] = DataEngine.credentials_to_dict(credentials)
    engine = DataEngine.from_credentials(credentials)

    if st.button("Sign out of Google"):
        for key in _SESSION_KEYS:
            st.session_state.pop(key, None)
        st.rerun()

    default_project = secrets.get("project", "")
    default_dataset = secrets.get("dataset", "")
    default_location = secrets.get("location") or None

    try:
        projects = engine.list_projects()
    except Exception as e:
        st.error(f"Couldn't list BigQuery projects: {e}")
        return st.session_state.get("bq_extracted_csv")

    project_options = list(projects.keys()) or ([default_project] if default_project else [])
    if not project_options:
        st.warning("No BigQuery projects visible to this Google account.")
        return st.session_state.get("bq_extracted_csv")
    project_index = project_options.index(default_project) if default_project in project_options else 0
    project = st.selectbox(
        "Project", project_options, index=project_index,
        format_func=lambda p: f"{p} ({projects[p]})" if projects.get(p) else p,
    )

    try:
        datasets = engine.list_datasets(project)
    except Exception as e:
        st.error(f"Couldn't list datasets in {project}: {e}")
        return st.session_state.get("bq_extracted_csv")

    dataset_options = list(datasets.keys()) or ([default_dataset] if default_dataset else [])
    if not dataset_options:
        st.warning(f"No datasets visible in {project}.")
        return st.session_state.get("bq_extracted_csv")
    dataset_index = dataset_options.index(default_dataset) if default_dataset in dataset_options else 0
    dataset = st.selectbox("Dataset (GA4 export)", dataset_options, index=dataset_index)

    today = dt.date.today()
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input("Start date", value=today - dt.timedelta(days=7), max_value=today)
    with col_end:
        end_date = st.date_input("End date", value=today - dt.timedelta(days=1), max_value=today)

    if start_date > end_date:
        st.error("Start date must be before end date.")
        return st.session_state.get("bq_extracted_csv")

    with st.expander("Advanced: restrict to specific events"):
        event_names_raw = st.text_input(
            "Event names (comma-separated, blank = all events in range)", value="",
        )
    event_names = [e.strip() for e in event_names_raw.split(",") if e.strip()] if event_names_raw else []

    # session_id_param/include_user_id/include_purchase_revenue give
    # session-level cases with a real user_id + revenue column - see the
    # roadmap doc's "Recommended EventLogExtractionParams defaults" note.
    connection = BQConnectionConfig(project=project, dataset=dataset, location=default_location)
    date_range = DateRange(start_date=start_date, end_date=end_date)
    params = EventLogExtractionParams(
        connection=connection,
        date_range=date_range,
        session_id_param="ga_session_id",
        include_user_id=True,
        include_purchase_revenue=True,
        event_names=event_names,
    )

    if st.button("Estimate cost (dry run)"):
        sql = build_event_log(params)
        estimate = engine.dry_run(sql)
        if estimate.error:
            st.error(f"Dry run failed: {estimate.error}")
        else:
            st.info(f"Estimated scan: {estimate.display} ({estimate.free_tier_pct}% of the 1TB free tier).")

    extract_clicked = st.button("Extract Data", type="primary", width='stretch')

    if extract_clicked:
        with st.spinner("Querying BigQuery..."):
            try:
                df = engine.extract_event_log(params)
            except Exception as e:
                st.error(f"Extraction failed: {e}")
                return st.session_state.get("bq_extracted_csv")
        if df.empty:
            st.warning("Query returned no rows for this project/dataset/date range.")
            return st.session_state.get("bq_extracted_csv")
        # BigQuery TIMESTAMP columns come back tz-aware (UTC); CSV-sourced
        # logs are naive. Strip the offset (keeping the UTC wall-clock
        # value) so duration/bottleneck calculations behave identically
        # regardless of data source.
        if isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype):
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
        st.session_state["bq_extracted_csv"] = df.to_csv(index=False).encode("utf-8")
        st.session_state["bq_extracted_label"] = (
            f"bigquery_{project}_{dataset}_{start_date}_{end_date}.csv"
        )
        st.success(f"Extracted {len(df):,} events across {df['case_id'].nunique():,} sessions.")

    extracted = st.session_state.get("bq_extracted_csv")
    if extracted:
        st.caption(
            f"Using: {st.session_state.get('bq_extracted_label', 'BigQuery extract')} "
            "- click Extract Data again to refresh."
        )
    return extracted
