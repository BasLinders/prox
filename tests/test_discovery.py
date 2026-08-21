import io

import pm4py
import pytest

from prox.discovery import perform_process_discovery, _discover_inductive_miner
from prox.data_manager import load_and_validate_csv
from prox.mock_data import generate_mock_event_log

from conftest import make_simple_variant_log


@pytest.mark.parametrize("algo", ["inductive_miner", "heuristics_miner", "dfg"])
def test_perform_process_discovery_produces_valid_model(algo):
    """Regression test: dfg previously crashed due to a pm4py API-drift bug
    (dfg_converter.Variants.TO_PETRI_NET no longer exists in current pm4py)."""
    df = make_simple_variant_log(n_cases=3)
    model, errors, messages = perform_process_discovery(df, discovery_algo=algo)

    assert errors == []
    assert model is not None
    net, im, fm = model
    assert len(net.places) > 0
    assert len(net.transitions) > 0


def test_perform_process_discovery_unknown_algorithm():
    df = make_simple_variant_log(n_cases=3)
    model, errors, messages = perform_process_discovery(df, discovery_algo='not_a_real_algo')
    assert model is None
    assert any('unknown' in e.lower() for e in errors)


def test_perform_process_discovery_empty_log():
    df = make_simple_variant_log(n_cases=3).iloc[0:0]
    model, errors, messages = perform_process_discovery(df, discovery_algo='inductive_miner')
    assert model is None
    assert any('empty' in e.lower() for e in errors)


def test_perform_process_discovery_missing_columns():
    import pandas as pd
    model, errors, messages = perform_process_discovery(pd.DataFrame({'foo': [1]}))
    assert model is None
    assert any('missing' in e.lower() for e in errors)


def test_inductive_miner_noise_threshold_actually_affects_the_model():
    """Regression test: inductive_miner.apply() defaults to the plain IM
    variant, which silently ignores noise_threshold entirely - only the IMf
    variant applies it. Without variant=Variants.IMf, a noisy log discovered
    at noise_threshold=0.0 and noise_threshold=0.9 produced byte-identical
    models despite the UI's Noise Threshold slider suggesting otherwise."""
    raw = generate_mock_event_log(n_sessions=60, seed=1)
    csv_bytes = raw.to_csv(index=False).encode()
    log_df, messages, has_category = load_and_validate_csv(io.BytesIO(csv_bytes), case_grouping='user')
    log = pm4py.convert_to_event_log(log_df)

    net_low, im_low, fm_low, _ = _discover_inductive_miner(log, noise_threshold=0.0)
    net_high, im_high, fm_high, _ = _discover_inductive_miner(log, noise_threshold=0.9)

    assert (len(net_low.places), len(net_low.transitions)) != (len(net_high.places), len(net_high.transitions))
