import pm4py
import pytest

from prox.discovery import perform_process_discovery
from prox.conformance import (
    run_conformance_checking,
    calculate_fitness_in_batches,
    build_structured_reference_model,
    import_reference_model_bpmn,
    diff_reference_model_coverage,
)

from conftest import make_event_log, make_simple_variant_log


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


# ---------------------------------------------------------------------------
# build_structured_reference_model
# ---------------------------------------------------------------------------

def _transition_labels(net):
    return {t.label for t in net.transitions if t.label is not None}


def test_build_structured_reference_model_required_stage():
    model, errors = build_structured_reference_model([
        {'activities': ['a'], 'type': 'required'},
        {'activities': ['b'], 'type': 'required'},
    ])
    assert errors == []
    net, im, fm = model
    assert _transition_labels(net) == {'a', 'b'}


def test_build_structured_reference_model_choice_stage():
    model, errors = build_structured_reference_model([
        {'activities': ['a'], 'type': 'required'},
        {'activities': ['b', 'c'], 'type': 'choice'},
    ])
    assert errors == []
    net, im, fm = model
    assert _transition_labels(net) == {'a', 'b', 'c'}


def test_build_structured_reference_model_optional_stage():
    model, errors = build_structured_reference_model([
        {'activities': ['a'], 'type': 'required'},
        {'activities': ['b'], 'type': 'optional'},
    ])
    assert errors == []
    net, im, fm = model
    assert _transition_labels(net) == {'a', 'b'}

    # A case that skips the optional step and one that includes it both fit perfectly.
    log = pm4py.convert_to_event_log(make_event_log([
        ('1', 'a', '2024-01-01 00:00:00'),
        ('2', 'a', '2024-01-01 00:00:00'),
        ('2', 'b', '2024-01-01 00:01:00'),
    ]))
    fitness = calculate_fitness_in_batches(log, net, im, fm)
    assert fitness == pytest.approx(1.0)


def test_build_structured_reference_model_repeatable_stage():
    model, errors = build_structured_reference_model([
        {'activities': ['a'], 'type': 'required'},
        {'activities': ['b'], 'type': 'repeatable'},
        {'activities': ['c'], 'type': 'required'},
    ])
    assert errors == []
    net, im, fm = model
    assert _transition_labels(net) == {'a', 'b', 'c'}

    log = pm4py.convert_to_event_log(make_event_log([
        ('1', 'a', '2024-01-01 00:00:00'),
        ('1', 'c', '2024-01-01 00:01:00'),
        ('2', 'a', '2024-01-01 00:00:00'),
        ('2', 'b', '2024-01-01 00:01:00'),
        ('2', 'c', '2024-01-01 00:02:00'),
        ('3', 'a', '2024-01-01 00:00:00'),
        ('3', 'b', '2024-01-01 00:01:00'),
        ('3', 'b', '2024-01-01 00:02:00'),
        ('3', 'c', '2024-01-01 00:03:00'),
    ]))
    fitness = calculate_fitness_in_batches(log, net, im, fm)
    assert fitness == pytest.approx(1.0)


def test_build_structured_reference_model_parallel_stage():
    model, errors = build_structured_reference_model([
        {'activities': ['a', 'b'], 'type': 'parallel'},
        {'activities': ['c'], 'type': 'required'},
    ])
    assert errors == []
    net, im, fm = model
    assert _transition_labels(net) == {'a', 'b', 'c'}

    log = pm4py.convert_to_event_log(make_event_log([
        ('1', 'a', '2024-01-01 00:00:00'),
        ('1', 'b', '2024-01-01 00:01:00'),
        ('1', 'c', '2024-01-01 00:02:00'),
        ('2', 'b', '2024-01-01 00:00:00'),
        ('2', 'a', '2024-01-01 00:01:00'),
        ('2', 'c', '2024-01-01 00:02:00'),
    ]))
    fitness = calculate_fitness_in_batches(log, net, im, fm)
    assert fitness == pytest.approx(1.0)


def test_build_structured_reference_model_full_mixed_stage_end_to_end():
    model, errors = build_structured_reference_model([
        {'activities': ['start'], 'type': 'required'},
        {'activities': ['view', 'search'], 'type': 'choice'},
        {'activities': ['filter'], 'type': 'optional'},
        {'activities': ['tag', 'compare'], 'type': 'parallel'},
        {'activities': ['review'], 'type': 'repeatable'},
        {'activities': ['end'], 'type': 'required'},
    ])
    assert errors == []
    net, im, fm = model
    assert _transition_labels(net) == {
        'start', 'view', 'search', 'filter', 'tag', 'compare', 'review', 'end'
    }


def test_build_structured_reference_model_errors_empty_stages_list():
    model, errors = build_structured_reference_model([])
    assert model is None
    assert errors


def test_build_structured_reference_model_errors_stage_with_no_activities():
    model, errors = build_structured_reference_model([
        {'activities': [], 'type': 'required'},
        {'activities': ['b'], 'type': 'required'},
    ])
    assert model is None
    assert errors


# ---------------------------------------------------------------------------
# Regression tests for the two verified spikes behind this feature
# ---------------------------------------------------------------------------

def test_plain_sequence_reference_model_flags_cases_that_skip_a_required_step():
    model, errors = build_structured_reference_model([
        {'activities': ['a'], 'type': 'required'},
        {'activities': ['b'], 'type': 'required'},
        {'activities': ['c'], 'type': 'required'},
        {'activities': ['d'], 'type': 'required'},
    ])
    assert errors == []
    net, im, fm = model

    df = make_event_log([
        ('1', 'a', '2024-01-01 00:00:00'),
        ('1', 'b', '2024-01-01 00:01:00'),
        ('1', 'c', '2024-01-01 00:02:00'),
        ('1', 'd', '2024-01-01 00:03:00'),
        ('2', 'a', '2024-01-01 00:00:00'),
        ('2', 'c', '2024-01-01 00:01:00'),
        ('2', 'd', '2024-01-01 00:02:00'),
    ])
    result = run_conformance_checking(
        df, net, im, fm,
        alignment_variant='state_equation_a_star',
        perform_sampling=False
    )
    assert result['errors'] == []
    cases = {c['case_id']: c['fitness'] for c in result['case_analysis']['cases']}
    assert cases['1'] == pytest.approx(1.0)
    assert cases['2'] < 1.0


def test_optional_step_scores_perfectly_whether_included_or_skipped():
    model, errors = build_structured_reference_model([
        {'activities': ['a'], 'type': 'required'},
        {'activities': ['coupon'], 'type': 'optional'},
        {'activities': ['b'], 'type': 'required'},
    ])
    assert errors == []
    net, im, fm = model

    df = make_event_log([
        ('1', 'a', '2024-01-01 00:00:00'),
        ('1', 'b', '2024-01-01 00:01:00'),
        ('2', 'a', '2024-01-01 00:00:00'),
        ('2', 'coupon', '2024-01-01 00:01:00'),
        ('2', 'b', '2024-01-01 00:02:00'),
    ])
    result = run_conformance_checking(
        df, net, im, fm,
        alignment_variant='state_equation_a_star',
        perform_sampling=False
    )
    assert result['errors'] == []
    cases = {c['case_id']: c['fitness'] for c in result['case_analysis']['cases']}
    assert cases['1'] == pytest.approx(1.0)
    assert cases['2'] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# import_reference_model_bpmn
# ---------------------------------------------------------------------------

def test_import_reference_model_bpmn_valid_file():
    model, _ = build_structured_reference_model([
        {'activities': ['a'], 'type': 'required'},
        {'activities': ['b'], 'type': 'required'},
    ])
    net, im, fm = model
    bpmn_graph = pm4py.convert_to_bpmn(net, im, fm)

    import tempfile
    import os
    fd, path = tempfile.mkstemp(suffix='.bpmn')
    os.close(fd)
    try:
        # auto_layout=False: skip Graphviz-based diagram layout, which isn't
        # installed in CI and isn't needed just to round-trip the BPMN XML.
        pm4py.write_bpmn(bpmn_graph, path, auto_layout=False)
        with open(path, 'rb') as f:
            file_bytes = f.read()
    finally:
        os.remove(path)

    imported_model, errors = import_reference_model_bpmn(file_bytes)
    assert errors == []
    imported_net, imported_im, imported_fm = imported_model
    assert _transition_labels(imported_net) == {'a', 'b'}


def test_import_reference_model_bpmn_malformed_xml_reports_clean_error():
    model, errors = import_reference_model_bpmn(b"not xml at all <<<")
    assert model is None
    assert errors
    assert all(isinstance(e, str) for e in errors)


# ---------------------------------------------------------------------------
# diff_reference_model_coverage
# ---------------------------------------------------------------------------

def test_diff_reference_model_coverage_both_directions():
    model, _ = build_structured_reference_model([
        {'activities': ['a'], 'type': 'required'},
        {'activities': ['b'], 'type': 'required'},
        {'activities': ['never_happens'], 'type': 'required'},
    ])
    net, im, fm = model

    df = make_event_log([
        ('1', 'a', '2024-01-01 00:00:00'),
        ('1', 'b', '2024-01-01 00:01:00'),
        ('1', 'unexpected_activity', '2024-01-01 00:02:00'),
    ])

    diff = diff_reference_model_coverage(net, df)
    assert diff['unexpected_in_data'] == ['unexpected_activity']
    assert diff['never_observed'] == ['never_happens']


def test_diff_reference_model_coverage_empty_diff_when_log_and_reference_match():
    model, _ = build_structured_reference_model([
        {'activities': ['a'], 'type': 'required'},
        {'activities': ['b'], 'type': 'required'},
    ])
    net, im, fm = model

    df = make_event_log([
        ('1', 'a', '2024-01-01 00:00:00'),
        ('1', 'b', '2024-01-01 00:01:00'),
    ])

    diff = diff_reference_model_coverage(net, df)
    assert diff['unexpected_in_data'] == []
    assert diff['never_observed'] == []
