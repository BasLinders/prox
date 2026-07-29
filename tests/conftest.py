import pandas as pd
import pytest


def make_event_log(rows):
    """rows: list of (case_id, activity, timestamp) tuples -> XES-standard DataFrame."""
    df = pd.DataFrame(rows, columns=['case:concept:name', 'concept:name', 'time:timestamp'])
    df['case:concept:name'] = df['case:concept:name'].astype(str)
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    return df


def make_simple_variant_log(n_cases=3, case_prefix=''):
    """n_cases identical traces of a -> b -> c, one minute apart."""
    rows = []
    base = pd.Timestamp('2024-01-01 00:00:00')
    for i in range(n_cases):
        case_id = f'{case_prefix}{i + 1}'
        for j, act in enumerate(['a', 'b', 'c']):
            rows.append((case_id, act, base + pd.Timedelta(minutes=j)))
    return make_event_log(rows)


@pytest.fixture
def simple_event_log():
    return make_simple_variant_log(n_cases=3)
