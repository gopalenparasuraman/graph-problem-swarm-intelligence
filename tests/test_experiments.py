"""Experiment outputs are computed from the shared seed-4616 instance."""

import csv
import json

import numpy as np
import pytest

from sis6116.experiments import (
    ExperimentSettings,
    baseline_settings,
    format_audit,
    render_summary_table,
    run_experiment,
    summarise_mmas,
)
from sis6116.instance import PERSONAL_SEED, generate_distance_matrix, matrix_fingerprint
from sis6116.mmas import MMASConfig, MMASResult
from sis6116.nearest_neighbor import best_of_all_starts, nearest_neighbor_tour
from sis6116.validation import ValidationError


def test_baseline_settings_are_fixed_and_match_the_default_config():
    settings = baseline_settings()
    assert settings == ExperimentSettings()
    assert settings.instance_seed == 4616
    assert settings.alpha == 1.0
    assert settings.beta == 2.0
    assert settings.rho == 0.2
    assert settings.n_ants is None
    assert settings.n_iterations == 100
    assert settings.optimiser_seeds == tuple(range(10))
    assert settings.nn_start_city == 0
    assert settings.seed_incumbent_with_nn is True
    assert settings.deposit_policy == "iteration_best"
    assert settings.acs_local_update is False
    assert settings.local_search is None
    assert settings.stagnation_restart is False
    config = MMASConfig()
    assert config.alpha == settings.alpha
    assert config.beta == settings.beta
    assert config.rho == settings.rho
    assert config.n_ants is None
    assert config.n_iterations == settings.n_iterations
    assert config.deposit_policy == "iteration_best"


def test_audit_text_states_the_variant_and_the_update_order():
    distance = generate_distance_matrix(PERSONAL_SEED)
    text = format_audit(baseline_settings(), distance)
    assert "iteration-best" in text
    assert "acs_local_update: false" in text
    assert "all_ant_deposit: false" in text
    assert "two_opt: false" in text
    assert "stagnation_restart: false" in text
    assert "return edge" in text
    assert "instance_seed: 4616" in text
    assert "optimiser_seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9" in text
    assert "nn_start_city: 0" in text
    assert "separate" in text
    assert "53104" in text
    assert text.index("recompute") < text.index("Evaporate")
    assert text.index("Evaporate") < text.index("Reinforce only the iteration-best")
    assert text.index("Reinforce only the iteration-best") < text.index("Clip off-diagonal")


def test_summary_uses_the_sample_standard_deviation():
    runs = [
        _stub_run(10, 1, 1.0),
        _stub_run(12, 2, 3.0),
        _stub_run(14, 3, 5.0),
    ]
    summary = summarise_mmas(runs)
    assert summary["mean_final_cost"] == pytest.approx(12.0)
    assert summary["sample_std_final_cost"] == pytest.approx(2.0)
    assert summary["mean_last_improvement_iteration"] == pytest.approx(2.0)
    assert summary["mean_runtime_seconds"] == pytest.approx(3.0)
    text = render_summary_table(
        [
            {
                "method": "nn_city0",
                "n_runs": 1,
                "mean_final_cost": 10.0,
                "sample_std_final_cost": None,
                "mean_last_improvement_iteration": None,
                "mean_runtime_seconds": 0.1,
            },
            summary,
        ]
    )
    assert "n/a" in text
    assert "12.000000" in text
    assert "2.000000" in text


def test_other_instance_seeds_and_too_few_runs_are_rejected(tmp_path):
    with pytest.raises(ValidationError, match="4616"):
        run_experiment(ExperimentSettings(instance_seed=1), tmp_path)
    with pytest.raises(ValidationError, match="at least 2"):
        run_experiment(
            ExperimentSettings(n_iterations=1, n_ants=1, optimiser_seeds=(0,)),
            tmp_path,
        )


def test_fixed_instance_is_shared_and_outputs_are_computed(monkeypatch, tmp_path):
    calls = []
    real = generate_distance_matrix

    def spy(seed=PERSONAL_SEED):
        calls.append(seed)
        return real(seed)

    monkeypatch.setattr("sis6116.experiments.generate_distance_matrix", spy)
    settings = ExperimentSettings(
        n_iterations=2,
        n_ants=2,
        optimiser_seeds=(4, 5, 6),
    )
    result = run_experiment(settings, tmp_path)
    distance = real(PERSONAL_SEED)
    fingerprint = matrix_fingerprint(distance)

    assert calls == [PERSONAL_SEED]
    assert result.instance_fingerprint == fingerprint
    assert len(result.runs) == 3
    assert {run.instance_fingerprint for run in result.runs} == {fingerprint}
    for run in result.runs:
        assert run.history[0] == result.nn_city0.cost
        assert run.final_cost == run.history[-1]
        assert run.final_cost <= run.history[0]
        assert len(run.history) == 3
        assert sorted(run.best_tour.tolist()) == list(range(47))

    fresh_city0 = nearest_neighbor_tour(distance, 0)
    fresh_best = best_of_all_starts(distance)
    assert result.nn_city0.cost == fresh_city0.cost
    assert result.nn_best.cost == fresh_best.cost
    assert result.nn_best.cost <= result.nn_city0.cost

    payload = json.loads((tmp_path / "experiment.json").read_text(encoding="utf-8"))
    assert payload["instance"]["N"] == 47
    assert payload["instance"]["strict_upper_triangle_sum"] == 53104
    assert payload["instance"]["fingerprint"] == fingerprint
    assert payload["instance"]["upper_left_5x5"][0] == [0, 56, 85, 16, 40]
    assert [run["metadata"]["instance_fingerprint"] for run in payload["runs"]] == [
        fingerprint,
        fingerprint,
        fingerprint,
    ]
    assert payload["summary"][2]["sample_std_final_cost"] == pytest.approx(
        float(np.std([run.final_cost for run in result.runs], ddof=1))
    )
    assert payload["summary"][0]["sample_std_final_cost"] is None
    assert payload["summary"][1]["mean_last_improvement_iteration"] is None

    with (tmp_path / "convergence.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 * 3
    assert [row["iteration"] for row in rows[:3]] == ["0", "1", "2"]
    assert int(rows[0]["best_so_far_cost"]) == result.nn_city0.cost

    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows[0]["method"] == "nn_city0"
    assert summary_rows[0]["sample_std_final_cost"] == ""
    assert summary_rows[2]["method"] == "mmas"
    assert float(summary_rows[2]["mean_final_cost"]) == pytest.approx(
        float(np.mean([run.final_cost for run in result.runs]))
    )
    assert float(summary_rows[2]["sample_std_final_cost"]) == pytest.approx(
        float(np.std([run.final_cost for run in result.runs], ddof=1))
    )

    for name in (
        "audit.txt",
        "runs.csv",
        "baselines.csv",
        "summary.txt",
        "convergence_mean_std.png",
        "convergence_overlay.png",
    ):
        path = tmp_path / name
        assert path.is_file()
        assert path.stat().st_size > 0
    audit = (tmp_path / "audit.txt").read_text(encoding="utf-8")
    assert "n_iterations: 2" in audit
    assert "n_ants: 2" in audit


def _stub_run(cost: int, last: int, runtime: float) -> MMASResult:
    return MMASResult(
        optimiser_seed=0,
        final_cost=cost,
        best_tour=np.array([0, 1, 2]),
        last_improvement_iteration=last,
        runtime_seconds=runtime,
        history=[cost],
        initial_nn_cost=cost,
        initial_nn_tour=np.array([0, 1, 2]),
        initial_tau_min=0.1,
        initial_tau_max=1.0,
        final_tau_min=0.1,
        final_tau_max=1.0,
        final_pheromone=np.zeros((3, 3)),
        instance_fingerprint="stub",
        parameters={},
    )
