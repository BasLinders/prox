import pytest

from prox.discovery import perform_process_discovery

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
