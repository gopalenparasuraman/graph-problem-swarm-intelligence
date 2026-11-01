"""Evaporation-rate runs on the fixed personalised instance.

C0, C1, and C2 share every setting except rho. Rho also changes the
pheromone bounds through tau_max = 1 / (rho * L). This module does not
write the preserved baseline directory outputs/.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sis6116.experiments import REPO_ROOT
from sis6116.instance import (
    EXPECTED_N,
    EXPECTED_UPPER_TRIANGLE_SUM,
    GENERATION_RULES,
    PERSONAL_SEED,
    generate_distance_matrix,
    matrix_fingerprint,
    strict_upper_triangle_sum,
    verify_personal_instance,
)
from sis6116.mmas import (
    MMASConfig,
    MMASResult,
    PHEROMONE_BOUND_ATOL,
    PHEROMONE_BOUND_RTOL,
    run_mmas,
)
from sis6116.nearest_neighbor import best_of_all_starts, nearest_neighbor_tour
from sis6116.objective import tour_cost
from sis6116.plotting import mean_and_sample_std, plot_mean_comparison, plot_mean_std, plot_overlay
from sis6116.validation import ValidationError, last_strict_improvement, require_int

RHO_EXPERIMENT_VERSION = "rho-sensitivity-1"
PRESERVED_BASELINE_JSON = REPO_ROOT / "outputs" / "experiment.json"
SAMPLE_STD_DDOF = 1

SHARED_FIELDS = (
    "alpha",
    "beta",
    "n_ants",
    "n_iterations",
    "optimiser_seeds",
    "nn_start_city",
    "seed_incumbent_with_nn",
    "deposit_policy",
    "acs_local_update",
    "local_search",
    "stagnation_restart",
    "instance_seed",
)


@dataclass(frozen=True)
class RhoConfiguration:
    """One evaporation setting. Other fields are the shared fixed settings."""

    config_id: str
    rho: float
    alpha: float = 1.0
    beta: float = 2.0
    n_ants: int = 47
    n_iterations: int = 100
    optimiser_seeds: tuple[int, ...] = tuple(range(10))
    nn_start_city: int = 0
    seed_incumbent_with_nn: bool = True
    deposit_policy: str = "iteration_best"
    acs_local_update: bool = False
    local_search: str | None = None
    stagnation_restart: bool = False
    instance_seed: int = PERSONAL_SEED


def rho_configurations() -> tuple[RhoConfiguration, ...]:
    return (
        RhoConfiguration("C0_baseline", 0.20),
        RhoConfiguration("C1_low_evaporation", 0.05),
        RhoConfiguration("C2_high_evaporation", 0.50),
    )


def assert_configurations_differ_only_in_rho(
    configs: Sequence[RhoConfiguration],
) -> None:
    """Raise unless the three named configurations differ only in rho."""
    expected = (
        ("C0_baseline", 0.20),
        ("C1_low_evaporation", 0.05),
        ("C2_high_evaporation", 0.50),
    )
    if len(configs) != len(expected):
        raise ValidationError("rho experiment requires C0, C1, and C2")
    for config, (config_id, rho) in zip(configs, expected):
        if not isinstance(config, RhoConfiguration):
            raise ValidationError("configuration must be a RhoConfiguration")
        if config.config_id != config_id or config.rho != rho:
            raise ValidationError(
                f"expected {config_id} with rho={rho}, "
                f"got {config.config_id} with rho={config.rho}"
            )
    reference = configs[0]
    for config in configs[1:]:
        for name in SHARED_FIELDS:
            if getattr(config, name) != getattr(reference, name):
                raise ValidationError(
                    f"{config.config_id} differs from {reference.config_id} in {name}"
                )
        if config.rho == reference.rho:
            raise ValidationError("configurations must use different rho values")


def to_mmas_config(config: RhoConfiguration) -> MMASConfig:
    return MMASConfig(
        alpha=config.alpha,
        beta=config.beta,
        rho=config.rho,
        n_ants=config.n_ants,
        n_iterations=config.n_iterations,
        seed_incumbent_with_nn=config.seed_incumbent_with_nn,
        nn_start_city=config.nn_start_city,
        deposit_policy=config.deposit_policy,
        acs_local_update=config.acs_local_update,
        local_search=config.local_search,
        stagnation_restart=config.stagnation_restart,
    )


def assert_rho_output_dir(output_dir: Path | str) -> Path:
    """Allow outputs/rho_experiment and refuse the preserved outputs/ directory."""
    output = Path(output_dir).resolve()
    baseline = (REPO_ROOT / "outputs").resolve()
    if output == baseline:
        raise ValidationError(
            "rho experiment refuses to write the preserved baseline directory outputs/"
        )
    return output


def experiment_distance_matrix() -> np.ndarray:
    distance = generate_distance_matrix(PERSONAL_SEED)
    verify_personal_instance(distance, PERSONAL_SEED)
    return distance


def assert_history_invariants(
    history: Sequence[int], final_cost: int, n_iterations: int
) -> None:
    iterations = require_int("n_iterations", n_iterations, minimum=1)
    if len(history) != iterations + 1:
        raise ValidationError(
            "convergence curve must record iteration 0 through the final iteration"
        )
    if any(int(later) > int(earlier) for earlier, later in zip(history, history[1:])):
        raise ValidationError("best-so-far cost increased")
    if int(history[-1]) != int(final_cost):
        raise ValidationError(
            "final tour cost does not equal the final convergence value"
        )


def assert_diagnostic_invariants(run: MMASResult, n_ants: int) -> None:
    if len(run.diagnostics) != len(run.history):
        raise ValidationError("diagnostics do not cover every logged iteration")
    last = 0
    for record in run.diagnostics:
        iteration = int(record["iteration"])
        if iteration > 0 and int(run.history[iteration]) < int(run.history[iteration - 1]):
            last = iteration
        if int(record["iterations_since_last_improvement"]) != iteration - last:
            raise ValidationError(
                "iterations since last improvement disagree with the history"
            )
        fraction_min = float(record["fraction_at_tau_min"])
        fraction_max = float(record["fraction_at_tau_max"])
        if not 0.0 <= fraction_min <= 1.0 or not 0.0 <= fraction_max <= 1.0:
            raise ValidationError("pheromone bound fraction is outside [0, 1]")
        if float(record["bound_rtol"]) != PHEROMONE_BOUND_RTOL:
            raise ValidationError("diagnostic bound rtol does not match the definition")
        if float(record["bound_atol"]) != PHEROMONE_BOUND_ATOL:
            raise ValidationError("diagnostic bound atol does not match the definition")
        if iteration == 0:
            if record["distinct_tours"] is not None or record["n_tours"] is not None:
                raise ValidationError("iteration 0 must not record constructed tours")
        else:
            distinct = record["distinct_tours"]
            if distinct is None or not 1 <= int(distinct) <= n_ants:
                raise ValidationError("distinct tour count is outside the ant budget")
            if int(record["n_tours"]) != n_ants:
                raise ValidationError("diagnostic tour count does not match n_ants")
    if last != run.last_improvement_iteration:
        raise ValidationError("last improvement iteration disagrees with the history")
    if last_strict_improvement(list(run.history)) != run.last_improvement_iteration:
        raise ValidationError("last improvement iteration disagrees with the history")


def summarise_configuration(
    config: RhoConfiguration, runs: Sequence[MMASResult]
) -> dict[str, object]:
    if len(runs) < 2:
        raise ValidationError("sample standard deviation requires at least 2 runs")
    costs = np.asarray([run.final_cost for run in runs], dtype=np.float64)
    improvements = np.asarray(
        [run.last_improvement_iteration for run in runs], dtype=np.float64
    )
    runtimes = np.asarray([run.runtime_seconds for run in runs], dtype=np.float64)
    sample_std = float(costs.std(ddof=SAMPLE_STD_DDOF))
    if not np.isfinite(sample_std):
        raise ValidationError("sample standard deviation is not finite")
    return {
        "configuration_id": config.config_id,
        "rho": float(config.rho),
        "n_runs": len(runs),
        "mean_final_cost": float(costs.mean()),
        "sample_std_final_cost": sample_std,
        "sample_std_ddof": SAMPLE_STD_DDOF,
        "mean_last_improvement_iteration": float(improvements.mean()),
        "mean_runtime_seconds": float(runtimes.mean()),
        "best_cost": int(costs.min()),
        "worst_cost": int(costs.max()),
    }


def measured_curve(histories: list[list[int]]) -> dict[str, object]:
    mean, std = mean_and_sample_std(histories)
    recomputed = float(np.std(np.asarray(histories, dtype=np.float64), axis=0, ddof=1)[0])
    if not np.isclose(recomputed, float(std[0])):
        raise ValidationError("sample standard deviation does not use ddof=1")
    return {
        "mean_best_so_far": [float(value) for value in mean.tolist()],
        "sample_std_best_so_far": [float(value) for value in std.tolist()],
        "sample_std_ddof": SAMPLE_STD_DDOF,
    }


def assert_plotted_means(
    figure: plt.Figure, series: Sequence[tuple[str, list[list[int]]]]
) -> None:
    """The comparison lines must be the means of the histories passed in."""
    mean_lines = [
        line
        for line in figure.axes[0].lines
        if not str(line.get_label()).startswith("NN")
    ]
    if len(mean_lines) != len(series):
        raise ValidationError(
            "comparison figure does not have one curve per configuration"
        )
    if len(figure.axes[0].collections) < len(series):
        raise ValidationError("comparison figure is missing a sample-std band")
    for line, (label, histories) in zip(mean_lines, series):
        if line.get_label() != label:
            raise ValidationError("comparison legend does not match the configuration")
        mean, _std = mean_and_sample_std(histories)
        plotted = np.asarray(line.get_ydata(), dtype=np.float64)
        if plotted.shape != mean.shape or not np.allclose(plotted, mean):
            raise ValidationError("comparison curve is not the measured mean")


def compare_c0_to_preserved_baseline(
    runs: Sequence[MMASResult], path: Path = PRESERVED_BASELINE_JSON
) -> dict[str, object]:
    """Compare C0 search results with outputs/experiment.json.

    Wall-clock runtime is recorded in both files and is not required to match.
    """
    compared = [
        "history",
        "final_cost",
        "best_tour",
        "last_improvement_iteration",
        "instance_fingerprint",
        "optimiser_seed",
    ]
    report: dict[str, object] = {
        "preserved_path": str(path),
        "compared_fields": compared,
        "not_compared": ["runtime_seconds"],
        "mismatches": [],
    }
    if not path.is_file():
        report["status"] = "not_compared"
        report["reason"] = "preserved baseline file is absent"
        return report
    payload = json.loads(path.read_text(encoding="utf-8"))
    preserved_runs = payload.get("runs", [])
    report["preserved_rho"] = payload.get("settings", {}).get("rho")
    report["preserved_git_revision"] = payload.get("environment", {}).get("git_revision")
    mismatches: list[str] = []
    if len(preserved_runs) != len(runs):
        mismatches.append(
            f"run count preserved={len(preserved_runs)} c0={len(runs)}"
        )
    for index, (saved, run) in enumerate(zip(preserved_runs, runs)):
        seed = run.optimiser_seed
        if int(saved.get("optimiser_seed")) != seed:
            mismatches.append(f"seed index {index}")
        if list(saved.get("history", [])) != list(run.history):
            mismatches.append(f"history seed {seed}")
        if int(saved.get("final_cost")) != int(run.final_cost):
            mismatches.append(f"final_cost seed {seed}")
        if list(saved.get("best_tour", [])) != [int(city) for city in run.best_tour.tolist()]:
            mismatches.append(f"best_tour seed {seed}")
        if int(saved.get("last_improvement_iteration")) != int(run.last_improvement_iteration):
            mismatches.append(f"last_improvement_iteration seed {seed}")
        saved_fingerprint = saved.get("metadata", {}).get("instance_fingerprint")
        if saved_fingerprint != run.instance_fingerprint:
            mismatches.append(f"instance_fingerprint seed {seed}")
    report["mismatches"] = mismatches
    report["status"] = "reproduced" if not mismatches else "not_reproduced"
    return report


def _git_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    revision = completed.stdout.strip()
    return revision or "unavailable"


def _git_status() -> str:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return "clean" if not completed.stdout.strip() else "dirty"


def _environment() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "git_revision": _git_revision(),
        "git_status": _git_status(),
        "implementation_version": RHO_EXPERIMENT_VERSION,
    }


def _csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".12g")
    if isinstance(value, (list, tuple)):
        return " ".join(str(int(item)) for item in value)
    return str(value)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _series_label(config: RhoConfiguration) -> str:
    return f"{config.config_id} (rho={config.rho:.2f})"


def _validate_group(
    config: RhoConfiguration,
    runs: Sequence[MMASResult],
    distance: np.ndarray,
    fingerprint: str,
) -> None:
    if len(runs) != len(config.optimiser_seeds):
        raise ValidationError(f"{config.config_id} did not record every optimiser seed")
    if [run.optimiser_seed for run in runs] != list(config.optimiser_seeds):
        raise ValidationError(f"{config.config_id} optimiser seeds differ from the fixed list")
    for run in runs:
        if run.instance_fingerprint != fingerprint:
            raise ValidationError("runs do not share the personalised distance matrix")
        if int(run.parameters["n_ants"]) != int(config.n_ants):
            raise ValidationError(f"{config.config_id} changed the ant budget")
        if int(run.parameters["n_iterations"]) != int(config.n_iterations):
            raise ValidationError(f"{config.config_id} changed the iteration budget")
        if float(run.parameters["rho"]) != float(config.rho):
            raise ValidationError(f"{config.config_id} recorded a different rho")
        if float(run.parameters["alpha"]) != float(config.alpha):
            raise ValidationError(f"{config.config_id} changed alpha")
        if float(run.parameters["beta"]) != float(config.beta):
            raise ValidationError(f"{config.config_id} changed beta")
        if run.parameters["deposit_policy"] != config.deposit_policy:
            raise ValidationError(f"{config.config_id} changed the deposit policy")
        if run.parameters["local_search"] is not None or config.local_search is not None:
            raise ValidationError(f"{config.config_id} enabled local search")
        if bool(run.parameters["acs_local_update"]) or bool(run.parameters["stagnation_restart"]):
            raise ValidationError(f"{config.config_id} enabled a disabled v1 policy")
        if tour_cost(distance, run.best_tour) != run.final_cost:
            raise ValidationError("final tour cost does not equal the recorded final cost")
        assert_history_invariants(run.history, run.final_cost, config.n_iterations)
        assert_diagnostic_invariants(run, int(config.n_ants))
        initial_tau_min = float(run.initial_tau_min)
        initial_tau_max = float(run.initial_tau_max)
        if not np.isclose(initial_tau_max, 1.0 / (config.rho * float(run.initial_nn_cost))):
            raise ValidationError("initial tau_max does not follow 1/(rho*L_NN)")
        if not np.isclose(initial_tau_min, initial_tau_max / (2.0 * distance.shape[0])):
            raise ValidationError("initial tau_min does not follow tau_max/(2*N)")


def validate_rho_results(
    groups: Sequence[tuple[RhoConfiguration, list[MMASResult]]],
    distance: np.ndarray,
    fingerprint: str,
    summaries: Sequence[dict[str, object]],
    plotted: Sequence[dict[str, object]],
) -> list[str]:
    checks = [
        "three configurations differ only in rho",
        "one shared distance matrix",
        "n_ants and n_iterations match the fixed budgets",
        "convergence records iteration 0 through iteration 100",
        "best-so-far curves are non-increasing",
        "final tour costs equal the final convergence values",
        "sample standard deviation uses ddof=1",
        "comparison curves are the measured means",
    ]
    configs = [config for config, _runs in groups]
    assert_configurations_differ_only_in_rho(configs)
    if len({fingerprint}) != 1:
        raise ValidationError("instance fingerprint is missing")
    if int(distance.shape[0]) != EXPECTED_N:
        raise ValidationError("distance matrix does not have the personalised city count")
    verify_personal_instance(distance, PERSONAL_SEED)
    if len(summaries) != len(groups) or len(plotted) != len(groups):
        raise ValidationError("summary or plotted series does not match the configurations")
    nn_costs = {int(run.initial_nn_cost) for _config, runs in groups for run in runs}
    if len(nn_costs) != 1:
        raise ValidationError("runs do not share the city-0 nearest-neighbour incumbent")
    for (config, runs), summary, curve in zip(groups, summaries, plotted):
        _validate_group(config, runs, distance, fingerprint)
        expected = summarise_configuration(config, runs)
        for key, value in expected.items():
            if summary[key] != value:
                raise ValidationError(f"{config.config_id} summary disagrees with its runs")
        histories = [list(run.history) for run in runs]
        measured = measured_curve(histories)
        if curve["configuration_id"] != config.config_id:
            raise ValidationError("plotted series is not in configuration order")
        if curve["mean_best_so_far"] != measured["mean_best_so_far"]:
            raise ValidationError("stored comparison mean is not the measured mean")
        if curve["sample_std_best_so_far"] != measured["sample_std_best_so_far"]:
            raise ValidationError("stored comparison band is not the measured sample std")
        if int(curve["sample_std_ddof"]) != SAMPLE_STD_DDOF:
            raise ValidationError("plotted band does not declare ddof=1")
    return checks


@dataclass
class RhoExperimentResult:
    output_dir: Path
    paths: dict[str, Path]
    summaries: list[dict[str, object]]
    validation_checks: list[str]
    c0_reproduction: dict[str, object]
    instance_fingerprint: str


def _audit_text(
    distance: np.ndarray,
    fingerprint: str,
    environment: dict[str, str],
    summaries: list[dict[str, object]],
    reproduction: dict[str, object],
    checks: list[str],
) -> str:
    upper_left = np.array2string(np.asarray(distance[:5, :5]), separator=", ")
    lines = [
        "RHO EXPERIMENT",
        f"implementation_version: {RHO_EXPERIMENT_VERSION}",
        f"git_revision: {environment['git_revision']}",
        f"git_status: {environment['git_status']}",
        "core_update: unchanged from sis6116.mmas.run_mmas",
        "difference_between_configurations: rho only",
        "dependent_bounds: tau_max = 1 / (rho * L_best_so_far); tau_min = tau_max / (2 * N)",
        "initial_bounds: L is the city-0 nearest-neighbour length",
        "C0_baseline rho: 0.20",
        "C1_low_evaporation rho: 0.05",
        "C2_high_evaporation rho: 0.50",
        "alpha: 1.0",
        "beta: 2.0",
        "n_ants: 47",
        "n_iterations: 100",
        "n_runs: 10",
        "optimiser_seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9",
        "optimiser_rng: np.random.default_rng(optimiser_seed)",
        "handbook_rng: the handbook asks for an unseeded optimiser RNG. Seeds 0..9 are retained so C0 can be compared with the preserved baseline. This experiment does not switch to an unseeded stream.",
        "nn_start_city: 0",
        "deposit_policy: iteration_best",
        "acs_local_update: false",
        "all_ant_deposit: false",
        "local_search: none",
        "two_opt: false",
        "stagnation_restart: false",
        "output_directory: outputs/rho_experiment",
        "preserved_baseline_directory: outputs/",
        "",
        "DIAGNOSTICS",
        "effect_on_search: none. Counts and fractions are read after construction and after the pheromone update. They do not draw random numbers or change tours, pheromone, or stopping.",
        "distinct_tours: number of unique city sequences constructed in that iteration. Iteration 0 constructs no tours, so the value is null.",
        "iterations_since_last_improvement: iteration index minus the index of the last strict best-so-far decrease. The value is 0 at iteration 0 and on an improvement iteration.",
        f"pheromone_bound_rtol: {PHEROMONE_BOUND_RTOL}",
        f"pheromone_bound_atol: {PHEROMONE_BOUND_ATOL}",
        "pheromone_bound_rule: numpy.isclose with the rtol and atol above. Off-diagonal entries only. Iterations 1..100 are measured after clipping. Iteration 0 uses the initial matrix.",
        f"sample_std_ddof: {SAMPLE_STD_DDOF}",
        "",
        "INSTANCE",
        f"instance_seed: {PERSONAL_SEED}",
        f"N: {distance.shape[0]}",
        f"fingerprint: {fingerprint}",
        f"strict_upper_triangle_sum: {strict_upper_triangle_sum(distance)}",
        f"expected_upper_triangle_sum: {EXPECTED_UPPER_TRIANGLE_SUM}",
        "upper_left_5x5:",
        upper_left,
        "generation:",
        *[f"  {rule}" for rule in GENERATION_RULES],
        "",
        "C0 REPRODUCTION",
        f"status: {reproduction['status']}",
        f"preserved_path: {reproduction['preserved_path']}",
        "compared: history, final_cost, best_tour, last_improvement_iteration, instance_fingerprint, optimiser_seed",
        "not_compared: runtime_seconds",
        f"mismatch_count: {len(reproduction['mismatches'])}",
        "",
        "VALIDATION",
        "result: passed",
        *[f"  {check}" for check in checks],
        "",
        "MEASURED SUMMARY",
        *[
            (
                f"{row['configuration_id']}: rho={row['rho']} "
                f"mean_final_cost={row['mean_final_cost']} "
                f"sample_std_final_cost={row['sample_std_final_cost']} "
                f"mean_last_improvement_iteration={row['mean_last_improvement_iteration']} "
                f"mean_runtime_seconds={row['mean_runtime_seconds']} "
                f"best_cost={row['best_cost']} worst_cost={row['worst_cost']}"
            )
            for row in summaries
        ],
        "",
        "RUNTIME ENVIRONMENT",
        *[f"{key}: {value}" for key, value in environment.items()],
        "",
    ]
    return "\n".join(lines)


def run_rho_experiment(
    output_dir: Path | str = REPO_ROOT / "outputs" / "rho_experiment",
) -> RhoExperimentResult:
    """Run C0, C1, and C2 on one shared distance matrix."""
    output = assert_rho_output_dir(output_dir)
    configs = rho_configurations()
    assert_configurations_differ_only_in_rho(configs)
    distance = experiment_distance_matrix()
    fingerprint = matrix_fingerprint(distance)
    n = int(distance.shape[0])
    nn_city0 = nearest_neighbor_tour(distance, 0)
    nn_best = best_of_all_starts(distance)

    groups: list[tuple[RhoConfiguration, list[MMASResult]]] = []
    for config in configs:
        mmas_config = to_mmas_config(config)
        runs: list[MMASResult] = []
        for seed in config.optimiser_seeds:
            result = run_mmas(distance, seed, mmas_config)
            if result.history[0] != nn_city0.cost or result.initial_nn_cost != nn_city0.cost:
                raise ValidationError(
                    "MMAS initial incumbent does not match the city-0 nearest neighbour"
                )
            runs.append(result)
            print(
                f"{config.config_id} rho={config.rho:.2f} "
                f"optimiser_seed={result.optimiser_seed} "
                f"final_cost={result.final_cost} "
                f"last_improvement_iteration={result.last_improvement_iteration} "
                f"runtime_seconds={result.runtime_seconds:.6f}",
                flush=True,
            )
        groups.append((config, runs))

    summaries = [summarise_configuration(config, runs) for config, runs in groups]
    plotted_rows = []
    series: list[tuple[str, list[list[int]]]] = []
    for config, runs in groups:
        histories = [list(run.history) for run in runs]
        curve = measured_curve(histories)
        plotted_rows.append({"configuration_id": config.config_id, "label": _series_label(config), **curve})
        series.append((_series_label(config), histories))
    checks = validate_rho_results(groups, distance, fingerprint, summaries, plotted_rows)
    reproduction = compare_c0_to_preserved_baseline(groups[0][1])
    environment = _environment()

    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "audit": output / "audit.txt",
        "runs_csv": output / "runs.csv",
        "summary_csv": output / "summary.csv",
        "diagnostics_csv": output / "diagnostics.csv",
        "experiment_json": output / "experiment.json",
        "comparison_plot": output / "convergence_comparison.png",
    }
    for config, _runs in groups:
        paths[f"{config.config_id}_mean_std"] = output / f"{config.config_id}_mean_std.png"
        paths[f"{config.config_id}_runs"] = output / f"{config.config_id}_runs.png"

    figures: list[plt.Figure] = []
    try:
        for config, runs in groups:
            histories = [list(run.history) for run in runs]
            figures.append(
                plot_mean_std(
                    histories,
                    paths[f"{config.config_id}_mean_std"],
                    nn_city0=float(nn_city0.cost),
                    nn_best=float(nn_best.cost),
                    title=f"{_series_label(config)} mean best-so-far",
                    series_label="mean best-so-far",
                )
            )
            figures.append(
                plot_overlay(
                    histories,
                    [run.optimiser_seed for run in runs],
                    paths[f"{config.config_id}_runs"],
                    title=f"{_series_label(config)} best-so-far by run",
                )
            )
        comparison = plot_mean_comparison(
            series,
            paths["comparison_plot"],
            nn_city0=float(nn_city0.cost),
            nn_best=float(nn_best.cost),
            title="Mean best-so-far by evaporation rate",
        )
        figures.append(comparison)
        assert_plotted_means(comparison, series)
    except Exception:
        for figure in figures:
            plt.close(figure)
        raise
    else:
        for figure in figures:
            plt.close(figure)
    checks = [*checks, "comparison figure lines equal the measured means"]

    run_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    json_runs: list[dict[str, object]] = []
    for config, runs in groups:
        for run in runs:
            tour = [int(city) for city in run.best_tour.tolist()]
            run_rows.append(
                {
                    "configuration_id": config.config_id,
                    "rho": float(config.rho),
                    "optimiser_seed": run.optimiser_seed,
                    "final_cost": run.final_cost,
                    "last_improvement_iteration": run.last_improvement_iteration,
                    "runtime_seconds": run.runtime_seconds,
                    "best_tour": tour,
                }
            )
            json_runs.append(
                {
                    "configuration_id": config.config_id,
                    "rho": float(config.rho),
                    "optimiser_seed": run.optimiser_seed,
                    "final_cost": run.final_cost,
                    "last_improvement_iteration": run.last_improvement_iteration,
                    "runtime_seconds": run.runtime_seconds,
                    "best_tour": tour,
                    "history": list(run.history),
                    "initial_nn_cost": run.initial_nn_cost,
                    "initial_tau_min": run.initial_tau_min,
                    "initial_tau_max": run.initial_tau_max,
                    "final_tau_min": run.final_tau_min,
                    "final_tau_max": run.final_tau_max,
                    "diagnostics": run.diagnostics,
                    "parameters": run.parameters,
                    "instance_fingerprint": run.instance_fingerprint,
                }
            )
            for record in run.diagnostics:
                diagnostic_rows.append(
                    {
                        "configuration_id": config.config_id,
                        "rho": float(config.rho),
                        "optimiser_seed": run.optimiser_seed,
                        "iteration": record["iteration"],
                        "distinct_tours": record["distinct_tours"],
                        "fraction_at_tau_min": record["fraction_at_tau_min"],
                        "fraction_at_tau_max": record["fraction_at_tau_max"],
                        "iterations_since_last_improvement": record[
                            "iterations_since_last_improvement"
                        ],
                        "tau_min": record["tau_min"],
                        "tau_max": record["tau_max"],
                        "n_off_diagonal": record["n_off_diagonal"],
                        "n_at_tau_min": record["n_at_tau_min"],
                        "n_at_tau_max": record["n_at_tau_max"],
                        "n_at_both_bounds": record["n_at_both_bounds"],
                        "bound_rtol": record["bound_rtol"],
                        "bound_atol": record["bound_atol"],
                    }
                )

    _write_csv(
        paths["runs_csv"],
        [
            "configuration_id",
            "rho",
            "optimiser_seed",
            "final_cost",
            "last_improvement_iteration",
            "runtime_seconds",
            "best_tour",
        ],
        run_rows,
    )
    _write_csv(
        paths["summary_csv"],
        [
            "configuration_id",
            "rho",
            "mean_final_cost",
            "sample_std_final_cost",
            "mean_last_improvement_iteration",
            "mean_runtime_seconds",
            "best_cost",
            "worst_cost",
            "sample_std_ddof",
            "n_runs",
        ],
        summaries,
    )
    _write_csv(
        paths["diagnostics_csv"],
        [
            "configuration_id",
            "rho",
            "optimiser_seed",
            "iteration",
            "distinct_tours",
            "fraction_at_tau_min",
            "fraction_at_tau_max",
            "iterations_since_last_improvement",
            "tau_min",
            "tau_max",
            "n_off_diagonal",
            "n_at_tau_min",
            "n_at_tau_max",
            "n_at_both_bounds",
            "bound_rtol",
            "bound_atol",
        ],
        diagnostic_rows,
    )

    payload = {
        "experiment": "rho_sensitivity",
        "implementation_version": RHO_EXPERIMENT_VERSION,
        "handbook_optimiser_rng": (
            "Seeds 0..9 are retained for comparison with the preserved baseline. "
            "The handbook asks for an unseeded optimiser RNG. This experiment keeps the seeded stream."
        ),
        "diagnostic_definitions": {
            "distinct_tours": "Unique constructed city sequences. Null at iteration 0.",
            "fraction_at_bounds": (
                "Off-diagonal share within numpy.isclose of tau_min or tau_max "
                f"(rtol={PHEROMONE_BOUND_RTOL}, atol={PHEROMONE_BOUND_ATOL}) after clipping."
            ),
            "iterations_since_last_improvement": (
                "Iteration index minus the last strict best-so-far improvement index."
            ),
            "observational_only": True,
        },
        "shared_parameters": {
            "instance_seed": PERSONAL_SEED,
            "alpha": configs[0].alpha,
            "beta": configs[0].beta,
            "n_ants": configs[0].n_ants,
            "n_iterations": configs[0].n_iterations,
            "optimiser_seeds": list(configs[0].optimiser_seeds),
            "nn_start_city": configs[0].nn_start_city,
            "seed_incumbent_with_nn": configs[0].seed_incumbent_with_nn,
            "deposit_policy": configs[0].deposit_policy,
            "acs_local_update": configs[0].acs_local_update,
            "local_search": configs[0].local_search,
            "stagnation_restart": configs[0].stagnation_restart,
            "sample_std_ddof": SAMPLE_STD_DDOF,
            "pheromone_bound_rtol": PHEROMONE_BOUND_RTOL,
            "pheromone_bound_atol": PHEROMONE_BOUND_ATOL,
        },
        "configurations": [
            {"configuration_id": config.config_id, "rho": float(config.rho)}
            for config in configs
        ],
        "environment": environment,
        "instance": {
            "seed": PERSONAL_SEED,
            "N": n,
            "fingerprint": fingerprint,
            "strict_upper_triangle_sum": strict_upper_triangle_sum(distance),
            "upper_left_5x5": [
                [int(value) for value in row] for row in distance[:5, :5].tolist()
            ],
            "generation": list(GENERATION_RULES),
        },
        "baselines": {
            "nn_city0_cost": nn_city0.cost,
            "nn_city0_tour": [int(city) for city in nn_city0.tour.tolist()],
            "nn_best_of_all_starts_cost": nn_best.cost,
            "nn_best_of_all_starts_start_city": nn_best.start_city,
        },
        "c0_baseline_reproduction": reproduction,
        "validation": {"result": "passed", "checks": checks},
        "summary": summaries,
        "plotted_series": plotted_rows,
        "runs": json_runs,
    }
    paths["experiment_json"].write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    paths["audit"].write_text(
        _audit_text(distance, fingerprint, environment, summaries, reproduction, checks),
        encoding="utf-8",
    )

    print(f"wrote {output}", flush=True)
    for row in summaries:
        print(
            f"{row['configuration_id']} mean_final_cost={row['mean_final_cost']} "
            f"sample_std={row['sample_std_final_cost']} "
            f"best_cost={row['best_cost']} worst_cost={row['worst_cost']}",
            flush=True,
        )
    print(f"c0_reproduction: {reproduction['status']}", flush=True)
    return RhoExperimentResult(
        output_dir=output,
        paths=paths,
        summaries=summaries,
        validation_checks=checks,
        c0_reproduction=reproduction,
        instance_fingerprint=fingerprint,
    )


def main() -> None:
    run_rho_experiment()


if __name__ == "__main__":
    main()
