import os

import pytest
from streamlit.testing.v1 import AppTest

MAIN_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    # The cache and saved-run dirs are relative to the working directory, so
    # run from an empty one - the app sees no cached datasets or saved runs,
    # and never touches the real ones.
    monkeypatch.chdir(tmp_path)
    return AppTest.from_file(MAIN_PY, default_timeout=180)


def _headers(at):
    return [h.value for h in at.header]


def test_app_starts_without_loading_data(app):
    app.run()

    assert not app.exception
    assert any("Confirm data source" in i.value for i in app.info)
    assert "2. Incremental Analysis" not in _headers(app)


def test_confirmed_data_source_survives_reruns(app):
    app.run()
    app.radio(key="data_source_pending").set_value("Upload CSV")
    next(b for b in app.button if b.label == "Confirm data source").click().run()
    assert not app.exception

    # Generating mock data reruns the script without the Confirm button being
    # pressed - the confirmed source has to carry over for the data to load.
    app.number_input(key="mock_sessions").set_value(50)
    next(b for b in app.button if b.label == "Generate Mock Data").click().run()
    assert not app.exception
    assert "2. Incremental Analysis" in _headers(app)

    app.run()
    assert not app.exception
    assert "2. Incremental Analysis" in _headers(app)
