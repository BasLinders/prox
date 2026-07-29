import pm4py
import pytest

from prox.discovery import perform_process_discovery
from prox.conformance import run_conformance_checking, calculate_fitness_in_batches

from conftest import make_simple_variant_log


@pytest.fixture
def perfect_model():
    """A Petri net discovered directly from the log it will be checked against,
    so fitness should be at (or very near) 1.0 for every conformance method."""
    df = make_simple_variant_log(n_cases=3)
    model, errors, messages = perform_process_discovery(df, discovery_algo='inductive_miner', noise_threshold=0.0)
    assert errors == []
    return df, model


def test_token_replay_fitness_is_high_for_perfectly_fitting_log(perfect_model):
    """Regression test for the previously-fake 'Token Replay' option: this must
    actually run pm4py's token-based replay and produce a real, non-zero fitness."""
    df, (net, im, fm) = perfect_model
    result = run_conformance_checking(
        df, net, im, fm,
        alignment_variant='token_replay',
        perform_sampling=False
    )
    assert result['errors'] == []
    assert result['fitness']['log_fitness'] > 0.9
    assert result['fitness']['note'] == 'Calculated via token-based replay'


def test_state_equation_alignment_produces_per_trace_deviations(perfect_model):
    df, (net, im, fm) = perfect_model
    result = run_conformance_checking(
        df, net, im, fm,
        alignment_variant='state_equation_a_star',
        perform_sampling=False
    )
    assert result['errors'] == []
    assert result['fitness']['log_fitness'] > 0.9
    assert len(result['case_analysis']['cases']) == 3


def test_run_conformance_checking_with_sampling_enabled(perfect_model):
    df, (net, im, fm) = perfect_model
    result = run_conformance_checking(
        df, net, im, fm,
        alignment_variant='token_replay',
        perform_sampling=True,
        strata_col=None
    )
    assert result['errors'] == []
    assert result['fitness']['log_fitness'] > 0.9


def test_run_conformance_checking_unknown_method_reports_error(perfect_model):
    df, (net, im, fm) = perfect_model
    result = run_conformance_checking(
        df, net, im, fm,
        alignment_variant='not_a_real_method',
        perform_sampling=False
    )
    assert any('unknown conformance method' in e.lower() for e in result['errors'])
    assert result['fitness']['log_fitness'] == 0


def test_calculate_fitness_in_batches_matches_expected_range(perfect_model):
    df, (net, im, fm) = perfect_model
    log = pm4py.convert_to_event_log(df)
    fitness = calculate_fitness_in_batches(log, net, im, fm)
    assert 0.0 <= fitness <= 1.0
    assert fitness > 0.9
