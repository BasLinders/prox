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
