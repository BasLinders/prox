import io
import os

import pytest
from streamlit.testing.v1 import AppTest

from prox import generate_mock_csv_bytes, load_and_validate_csv
from prox.config import create_analysis_config
from prox.incremental import DEFAULT_CACHE_DIR, save_cached_dataset
from prox.saved_runs import DEFAULT_SAVED_RUNS_DIR, save_run

MAIN_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    # The cache and saved-run dirs are relative to the working directory, so
    # run from an empty one - the app sees no cached datasets or saved runs
    # beyond what a test adds, and never touches the real ones.
    monkeypatch.chdir(tmp_path)
    return AppTest.from_file(MAIN_PY, default_timeout=180)


def _headers(at):
    return [h.value for h in at.header]


def _data_loaded(at):
    return "2. Incremental Analysis" in _headers(at)


def _mock_event_log():
    df, _, _ = load_and_validate_csv(io.BytesIO(generate_mock_csv_bytes(n_sessions=50, seed=1)))
    return df


def _cache_mock_dataset(dataset_id):
    save_cached_dataset(_mock_event_log(), dataset_id, "user", cache_dir=DEFAULT_CACHE_DIR)


def test_app_starts_without_loading_data(app):
    app.run()

    assert not app.exception
    assert app.radio(key="data_source_choice").value is None
    assert not _data_loaded(app)


def test_chosen_upload_source_survives_reruns(app):
    app.run()
    app.radio(key="data_source_choice").set_value("Upload CSV").run()
    assert not app.exception

    # Generating mock data reruns the script - the chosen source has to carry
    # over for the data to load.
    app.number_input(key="mock_sessions").set_value(50)
    next(b for b in app.button if b.label == "Generate Mock Data").click().run()
    assert not app.exception
    assert _data_loaded(app)

    app.run()
    assert not app.exception
    assert _data_loaded(app)


def test_cached_dataset_is_only_loaded_once_picked(app):
    _cache_mock_dataset("first")
    _cache_mock_dataset("second")
    app.run()

    app.radio(key="data_source_choice").set_value("Load cached dataset").run()
    assert not app.exception
    assert app.selectbox[0].value is None
    assert not _data_loaded(app)

    app.selectbox[0].set_value("second").run()
    assert not app.exception
    assert _data_loaded(app)
    assert any("'second'" in c.value for c in app.caption)

    # Switching to another source and back brings back the picked dataset,
    # not the first one in the list.
    app.radio(key="data_source_choice").set_value("Upload CSV").run()
    app.radio(key="data_source_choice").set_value("Load cached dataset").run()
    assert not app.exception
    assert app.selectbox[0].value == "second"
    assert _data_loaded(app)


def test_saved_run_is_only_loaded_once_picked(app):
    save_run(_mock_event_log(), "Plain run", "user", save_dir=DEFAULT_SAVED_RUNS_DIR)
    # A run saved with its config makes the app rerun once to rebuild the
    # config widgets under that run's settings.
    configured = save_run(
        _mock_event_log(), "Configured run", "user",
        save_dir=DEFAULT_SAVED_RUNS_DIR, config=create_analysis_config(),
    )
    app.run()

    app.radio(key="data_source_choice").set_value("Load saved run").run()
    assert not app.exception
    assert app.selectbox[0].value is None
    assert not _data_loaded(app)

    app.selectbox[0].set_value(configured["run_id"]).run()
    assert not app.exception
    assert _data_loaded(app)
    assert app.selectbox[0].value == configured["run_id"]
    assert app.session_state["restored_run_id"] == configured["run_id"]


def _run_mock_analysis(app):
    app.run()
    app.radio(key="data_source_choice").set_value("Upload CSV").run()
    app.number_input(key="mock_sessions").set_value(50)
    next(b for b in app.button if b.label == "Generate Mock Data").click().run()
    next(b for b in app.button if b.label == "Run Analysis").click().run()
    assert not app.exception


def test_ai_conclusion_and_pdf_expanders_follow_the_results(app):
    _run_mock_analysis(app)

    expander_labels = [e.label for e in app.expander]
    assert "AI Conclusion" in expander_labels
    assert "Build a Custom PDF Report" in expander_labels
    # No GEMINI_API_KEY in this empty working directory's secrets.
    generate = next(b for b in app.button if b.label == "Generate AI Conclusion")
    assert generate.disabled
    # The exact payload is shown before anything is sent, with a cost note.
    assert app.json and '"log_summary"' in app.json[0].proto.body
    assert any("Gemini API pricing" in c.value for c in app.caption)


def test_generated_ai_conclusion_is_shown_and_offered_in_the_pdf(app, monkeypatch):
    from utility import ai_client

    sent = {}

    def fake_generate_conclusion(data, language, on_progress=None, **kwargs):
        sent["data"], sent["language"] = data, language
        return {
            "ok": True, "error": None, "error_kind": None, "model_used": None,
            "model_requested": "fake-model", "models_tried": ["fake-model"],
            "text": '{"summary": "All good.", "key_findings": ["F1"], "next_steps": ["Fix view_item_list"]}',
        }

    monkeypatch.setattr(ai_client, "is_configured", lambda: True)
    monkeypatch.setattr(ai_client, "generate_conclusion", fake_generate_conclusion)
    _run_mock_analysis(app)

    next(b for b in app.button if b.label == "Generate AI Conclusion").click().run()
    assert not app.exception

    assert sent["language"] == ai_client.DEFAULT_LANGUAGE
    assert sent["data"]["log_summary"]["cases"] > 0
    assert app.session_state["ai_conclusion"]["next_steps"] == ["Fix view_item_list"]
    # Underscores escaped, so activity names don't render as italics.
    assert any("view\\_item\\_list" in m.value for m in app.markdown)
    assert any(c.label == "AI Summary & Next Steps" for c in app.checkbox)

    # A follow-up analysis changes the payload: the conclusion is flagged as
    # out of date and no longer offered in the PDF.
    app.session_state["segment_result"] = {
        "segment_col": "device", "segments": {}, "errors": [],
        "comparison_table": {"mobile": {"cases": 5, "health_score": 50.0, "fitness_score": 0.9,
                                        "precision_score": 0.8, "repeat_rate": 0.0, "top_variant": "a"}},
    }
    app.run()
    assert not app.exception
    assert any("changed since this conclusion" in w.value for w in app.warning)
    assert not any(c.label == "AI Summary & Next Steps" for c in app.checkbox)
