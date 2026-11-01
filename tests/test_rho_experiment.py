"""Rho configurations differ only in rho, and the recorded curves stay valid."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pytest

from sis6116 import mmas
from sis6116.experiments import REPO_ROOT
from sis6116.instance import generate_distance_matrix, matrix_fingerprint
from sis6116.mmas import MMASConfig, bounds_from_length, run_mmas
from sis6116.plotting import mean_and_sample_std, plot_mean_comparison
from sis6116.rho_experiment import (
    SHARED_FIELDS,
    assert_configurations_differ_only_in_rho,
    assert_history_invariants,
    assert_rho_output_dir,
    experiment_distance_matrix,
    rho_configurations,
    summarise_configuration,
    to_mmas_config,
)
from sis6116.validation import ValidationError


def test_docstring_says_shorter_tours_increase_both_bounds():
    text = mmas.__doc__ or ""
    assert "tightens" not in text
    assert "increases both bounds" in text
    assert "tau_max = 1 / (rho * L_best_so_far)" in text
    assert "tau_min = tau_max / (2 * N)" in text


def test_configurations_differ_only_in_rho_and_its_bounds():
    configs = rho_configurations()
    assert_configurations_differ_only_in_rho(configs)
    assert [config.config_id for config in configs] == [
        "C0_baseline",
        "C1_low_evaporation",
        "C2_high_evaporation",
    ]
    assert [config.rho for config in configs] == [0.20, 0.05, 0.50]
    mmas_configs = [to_mmas_config(config) for config in configs]
    reference = mmas_configs[0]
    length = 766
    n = 47
    for config, mmas_config in zip(configs, mmas_configs):
        assert mmas_config.rho == config.rho
        assert mmas_config.alpha == reference.alpha == 1.0
        assert mmas_config.beta == reference.beta == 2.0
        assert mmas_config.n_ants == reference.n_ants == 47
        assert mmas_config.n_iterations == reference.n_iterations == 100
        assert mmas_config.deposit_policy == "iteration_best"
        assert mmas_config.acs_local_update is False
        assert mmas_config.local_search is None
        assert mmas_config.stagnation_restart is False
        assert mmas_config.seed_incumbent_with_nn is True
        assert mmas_config.nn_start_city == 0
        tau_min, tau_max = bounds_from_length(length, config.rho, n)
        assert tau_max == pytest.approx(1.0 / (config.rho * length))
        assert tau_min == pytest.approx(tau_max / (2.0 * n))
    baseline_max = bounds_from_length(length, 0.20, n)[1]
    low_max = bounds_from_length(length, 0.05, n)[1]
    high_max = bounds_from_length(length, 0.50, n)[1]
    assert low_max == pytest.approx(baseline_max * (0.20 / 0.05))
    assert high_max == pytest.approx(baseline_max * (0.20 / 0.50))
    for name in SHARED_FIELDS:
        assert len({getattr(config, name) for config in configs}) == 1


def test_configurations_share_one_distance_matrix_and_budgets():
    distance = experiment_distance_matrix()
    generated = generate_distance_matrix(4616)
    assert matrix_fingerprint(distance) == matrix_fingerprint(generated)
    configs = rho_configurations()
    assert {config.instance_seed for config in configs} == {4616}
    assert {config.n_ants for config in configs} == {47}
    assert {config.n_iterations for config in configs} == {100}
    assert {config.optimiser_seeds for config in configs} == {tuple(range(10))}


def test_rho_experiment_refuses_the_preserved_baseline_directory():
    with pytest.raises(ValidationError, match="preserved baseline"):
        assert_rho_output_dir(REPO_ROOT / "outputs")
    allowed = assert_rho_output_dir(REPO_ROOT / "outputs" / "rho_experiment")
    assert allowed.name == "rho_experiment"


def test_history_invariants_cover_iteration_zero_through_the_budget():
    assert_history_invariants([10, 9, 9], 9, 2)
    with pytest.raises(ValidationError, match="iteration 0"):
        assert_history_invariants([10, 9], 9, 2)
    with pytest.raises(ValidationError, match="increased"):
        assert_history_invariants([10, 11], 11, 1)
    with pytest.raises(ValidationError, match="final convergence"):
        assert_history_invariants([10, 8], 7, 1)


def test_summary_uses_sample_standard_deviation():
    configs = rho_configurations()
    runs = []
    costs = [10, 12, 11]
    for cost in costs:
        run = run_mmas(
            np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int64),
            0,
            MMASConfig(n_iterations=1, n_ants=1, rho=configs[0].rho),
        )
        run.final_cost = cost
        run.last_improvement_iteration = cost - 10
        run.runtime_seconds = float(cost)
        runs.append(run)
    summary = summarise_configuration(configs[0], runs)
    assert summary["sample_std_ddof"] == 1
    assert summary["sample_std_final_cost"] == pytest.approx(float(np.std(costs, ddof=1)))
    assert summary["mean_final_cost"] == pytest.approx(float(np.mean(costs)))
    assert summary["best_cost"] == 10
    assert summary["worst_cost"] == 12
    assert summary["mean_last_improvement_iteration"] == pytest.approx(
        float(np.mean([0, 2, 1]))
    )
    assert math.isclose(
        float(summary["sample_std_final_cost"]),
        math.sqrt(((10 - 11) ** 2 + (12 - 11) ** 2 + (11 - 11) ** 2) / 2),
    )


def test_diagnostics_do_not_change_a_repeated_trajectory():
    distance = np.array(
        [
            [0, 5, 11, 18, 18, 9],
            [5, 0, 9, 24, 18, 20],
            [11, 9, 0, 12, 17, 27],
            [18, 24, 12, 0, 20, 17],
            [18, 18, 17, 20, 0, 7],
            [9, 20, 27, 17, 7, 0],
        ],
        dtype=np.int64,
    )
    config = MMASConfig(n_iterations=2, n_ants=2, rho=0.2)
    first = run_mmas(distance, 0, config)
    second = run_mmas(distance, 0, config)
    assert first.history == second.history
    assert first.history[0] == 68
    assert first.final_cost == second.final_cost == first.history[-1]
    assert len(first.diagnostics) == len(first.history) == 3
    assert first.diagnostics[0]["distinct_tours"] is None
    assert first.diagnostics[0]["fraction_at_tau_max"] == pytest.approx(1.0)
    assert first.diagnostics[0]["fraction_at_tau_min"] == pytest.approx(0.0)
    last = 0
    expected_since = []
    for iteration, cost in enumerate(first.history):
        if iteration > 0 and cost < first.history[iteration - 1]:
            last = iteration
        expected_since.append(iteration - last)
    assert [row["iterations_since_last_improvement"] for row in first.diagnostics] == expected_since
    for record in first.diagnostics[1:]:
        assert 1 <= int(record["distinct_tours"]) <= 2
        assert 0.0 <= float(record["fraction_at_tau_min"]) <= 1.0
        assert 0.0 <= float(record["fraction_at_tau_max"]) <= 1.0
    assert [row["distinct_tours"] for row in first.diagnostics] == [
        row["distinct_tours"] for row in second.diagnostics
    ]
    assert [row["fraction_at_tau_min"] for row in first.diagnostics] == pytest.approx(
        [row["fraction_at_tau_min"] for row in second.diagnostics]
    )


def test_comparison_plot_uses_the_measured_means(tmp_path):
    series = [
        ("C0_baseline (rho=0.20)", [[10, 8, 8], [12, 10, 9]]),
        ("C1_low_evaporation (rho=0.05)", [[9, 9, 7], [11, 8, 8]]),
        ("C2_high_evaporation (rho=0.50)", [[8, 6, 6], [8, 5, 4]]),
    ]
    figure = plot_mean_comparison(series, tmp_path / "comparison.png")
    try:
        mean_lines = [
            line
            for line in figure.axes[0].lines
            if not str(line.get_label()).startswith("NN")
        ]
        assert [line.get_label() for line in mean_lines] == [label for label, _ in series]
        for line, (_label, histories) in zip(mean_lines, series):
            mean, _std = mean_and_sample_std(histories)
            np.testing.assert_allclose(line.get_ydata(), mean)
        assert len(figure.axes[0].collections) == 3
        assert figure.axes[0].get_xlabel() == "Iteration (0 = initial incumbent)"
        assert figure.axes[0].get_ylabel() == "Best-so-far tour cost"
    finally:
        plt.close(figure)
