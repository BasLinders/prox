import sys
import types

import pandas as pd

from prox.config import create_analysis_config
from prox.segments import compare_segments

from conftest import make_event_log


def make_segmented_log():
    """3 mobile cases (a->b->c), 2 desktop cases (a->c), 1 tablet case (a->b->c)."""
    rows = []
    base = pd.Timestamp('2024-01-01')
    for i in range(3):
        case_id = f'm{i}'
        for j, act in enumerate(['a', 'b', 'c']):
            rows.append((case_id, act, base + pd.Timedelta(minutes=j), 'mobile'))
    for i in range(2):
        case_id = f'd{i}'
        for j, act in enumerate(['a', 'c']):
            rows.append((case_id, act, base + pd.Timedelta(minutes=j), 'desktop'))
    for i in range(1):
        case_id = f't{i}'
        for j, act in enumerate(['a', 'b', 'c']):
            rows.append((case_id, act, base + pd.Timedelta(minutes=j), 'tablet'))

    df = make_event_log([(r[0], r[1], r[2]) for r in rows])
    df['device'] = [r[3] for r in rows]
    return df


def test_compare_segments_produces_one_result_per_segment(tmp_path):
    df = make_segmented_log()
    config = create_analysis_config(filter_steps=[], sample_size=10)

    result = compare_segments(df, segment_col='device', config=config, output_folder=str(tmp_path))

    assert result['errors'] == []
    assert set(result['segments'].keys()) == {'mobile', 'desktop', 'tablet'}
    assert set(result['comparison_table'].keys()) == {'mobile', 'desktop', 'tablet'}
    assert result['comparison_table']['mobile']['cases'] == 3
    assert result['comparison_table']['desktop']['cases'] == 2
    assert result['comparison_table']['tablet']['cases'] == 1


def test_compare_segments_respects_top_n_segments(tmp_path):
    df = make_segmented_log()
    config = create_analysis_config(filter_steps=[], sample_size=10)

    result = compare_segments(
        df, segment_col='device', config=config, top_n_segments=2, output_folder=str(tmp_path)
    )

    # Ranked by case count: mobile (3), desktop (2) should win over tablet (1).
    assert set(result['comparison_table'].keys()) == {'mobile', 'desktop'}


def test_compare_segments_unknown_column_returns_error(tmp_path):
    df = make_segmented_log()
    config = create_analysis_config(filter_steps=[], sample_size=10)

    result = compare_segments(df, segment_col='does_not_exist', config=config, output_folder=str(tmp_path))

    assert result['segments'] == {}
    assert any('not found' in e.lower() for e in result['errors'])


def test_compare_segments_writes_to_distinct_output_folders(tmp_path):
    df = make_segmented_log()
    config = create_analysis_config(filter_steps=[], sample_size=10)

    compare_segments(df, segment_col='device', config=config, output_folder=str(tmp_path))

    written_folders = {p.name for p in tmp_path.iterdir() if p.is_dir()}
    assert any('mobile' in name for name in written_folders)
    assert any('desktop' in name for name in written_folders)


def test_compare_segments_sequential_matches_parallel_shape(tmp_path):
    df = make_segmented_log()
    config = create_analysis_config(filter_steps=[], sample_size=10)

    result = compare_segments(
        df, segment_col='device', config=config, parallel=False, output_folder=str(tmp_path)
    )

    assert result['errors'] == []
    assert set(result['comparison_table'].keys()) == {'mobile', 'desktop', 'tablet'}
    assert result['comparison_table']['mobile']['cases'] == 3
    assert result['comparison_table']['desktop']['cases'] == 2
    assert result['comparison_table']['tablet']['cases'] == 1


def test_compare_segments_parallel_and_sequential_agree_on_comparison_table(tmp_path):
    df = make_segmented_log()
    config = create_analysis_config(filter_steps=[], sample_size=10)

    parallel_result = compare_segments(
        df, segment_col='device', config=config, parallel=True, output_folder=str(tmp_path / 'p')
    )
    sequential_result = compare_segments(
        df, segment_col='device', config=config, parallel=False, output_folder=str(tmp_path / 's')
    )

    assert parallel_result['comparison_table'] == sequential_result['comparison_table']


def test_compare_segments_parallel_survives_unsafe_main_module(tmp_path):
    """Regression test for a real bug: under `streamlit run`, sys.modules['__main__']
    points at the user's Streamlit script, which is full of top-level st.* calls
    that raise outside a live session. Windows/macOS spawn workers reimport
    whatever __main__ points to in the parent to rebuild their environment, so
    every worker crashed on startup, surfacing to users as "A process in the
    process pool was terminated abruptly" for every single segment. Simulates
    that by pointing __main__ at a script that raises if ever reimported, and
    asserts compare_segments still succeeds and restores __main__ afterward.
    """
    unsafe_script = tmp_path / "unsafe_streamlit_app.py"
    unsafe_script.write_text(
        "raise RuntimeError('boom: this must never be reimported by a worker')\n"
    )
    fake_main = types.ModuleType('__main__')
    fake_main.__file__ = str(unsafe_script)

    original_main = sys.modules.get('__main__')
    sys.modules['__main__'] = fake_main
    try:
        df = make_segmented_log()
        config = create_analysis_config(filter_steps=[], sample_size=10)

        result = compare_segments(
            df, segment_col='device', config=config, parallel=True, output_folder=str(tmp_path / 'out')
        )

        assert result['errors'] == []
        assert set(result['comparison_table'].keys()) == {'mobile', 'desktop', 'tablet'}
        # __main__ must be restored to the caller's module once the pool is done,
        # not left pointing at the internal spawn-safe stub.
        assert sys.modules['__main__'] is fake_main
    finally:
        if original_main is not None:
            sys.modules['__main__'] = original_main


def test_compare_segments_single_segment_skips_pool(tmp_path):
    """With only one segment there's nothing to parallelize; should not touch a worker pool."""
    df = make_segmented_log()
    config = create_analysis_config(filter_steps=[], sample_size=10)

    result = compare_segments(
        df, segment_col='device', config=config, top_n_segments=1, parallel=True, output_folder=str(tmp_path)
    )

    assert result['errors'] == []
    assert set(result['comparison_table'].keys()) == {'mobile'}
