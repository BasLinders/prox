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
