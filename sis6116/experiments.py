"""Fixed-baseline experiment: one personalised instance, many optimiser seeds.

The constants below are the baseline. They are printed at run time.
They are not a tuned configuration.
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

import numpy as np

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
from sis6116.mmas import MMASConfig, MMASResult, run_mmas, validate_mmas_config
from sis6116.nearest_neighbor import NearestNeighborResult, best_of_all_starts, nearest_neighbor_tour
from sis6116.plotting import plot_mean_std, plot_overlay
from sis6116.validation import ValidationError, require_int

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentSettings:
    """Defaults match baseline_settings()."""

    instance_seed: int = PERSONAL_SEED
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.2
    n_ants: int | None = None
    n_iterations: int = 100
    optimiser_seeds: tuple[int, ...] = tuple(range(10))
    seed_incumbent_with_nn: bool = True
    nn_start_city: int = 0
    deposit_policy: str = "iteration_best"
    acs_local_update: bool = False
    local_search: str | None = None
    stagnation_restart: bool = False


def baseline_settings() -> ExperimentSettings:
    """Fixed baseline. Not a tuned configuration."""
    return ExperimentSettings(
        instance_seed=PERSONAL_SEED,
        alpha=1.0,
        beta=2.0,
        rho=0.2,
        n_ants=None,
        n_iterations=100,
        optimiser_seeds=tuple(range(10)),
        seed_incumbent_with_nn=True,
        nn_start_city=0,
        deposit_policy="iteration_best",
        acs_local_update=False,
        local_search=None,
        stagnation_restart=False,
    )


@dataclass
class ExperimentResult:
    audit_text: str
    summary_text: str
    summary_rows: list[dict[str, object]]
    output_dir: Path
    paths: dict[str, Path]
    instance_fingerprint: str
    runs: list[MMASResult]
    nn_city0: NearestNeighborResult
    nn_best: NearestNeighborResult


def _config_from_settings(settings: ExperimentSettings) -> MMASConfig:
    return MMASConfig(
        alpha=settings.alpha,
        beta=settings.beta,
        rho=settings.rho,
        n_ants=settings.n_ants,
        n_iterations=settings.n_iterations,
        seed_incumbent_with_nn=settings.seed_incumbent_with_nn,
        nn_start_city=settings.nn_start_city,
        deposit_policy=settings.deposit_policy,
        acs_local_update=settings.acs_local_update,
        local_search=settings.local_search,
        stagnation_restart=settings.stagnation_restart,
    )


def _optimiser_seeds(settings: ExperimentSettings) -> tuple[int, ...]:
    raw = settings.optimiser_seeds
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValidationError("optimiser_seeds must be a sequence of integers")
    seeds = tuple(require_int("optimiser_seed", seed, minimum=0) for seed in raw)
    if len(seeds) < 2:
        raise ValidationError(
            "experiment requires at least 2 optimiser seeds so the sample "
            "standard deviation is defined"
        )
    return seeds


def format_audit(settings: ExperimentSettings, distance: np.ndarray) -> str:
    n = int(distance.shape[0])
    n_ants = n if settings.n_ants is None else int(settings.n_ants)
    ant_rule = "equal to N" if settings.n_ants is None else "fixed count"
    seeds = ", ".join(str(seed) for seed in settings.optimiser_seeds)
    local_search = "none" if settings.local_search is None else settings.local_search
    upper_left = np.array2string(np.asarray(distance[:5, :5]), separator=", ")
    lines = [
        "VARIANT MECHANICS",
        "algorithm: MAX-MIN Ant System (MMAS)",
        "problem: symmetric TSP",
        "tour: permutation of cities 0..N-1",
        "tour_cost: sum of the N cycle edges, including the return edge",
        "construction: each ant draws a start city uniformly, then samples unvisited cities",
        "transition_weight: (tau_ij ** alpha) * ((1 / d_ij) ** beta)",
        "probability_normalisation: weights divided by their sum; nonfinite, negative, or nonpositive mass raises",
        "sampling: inverse transform, one rng.random() draw per transition",
        "ant_order: ants 0..m-1 in that order, one tour at a time",
        "deposit_policy: iteration-best only",
        "deposit_amount: 1 / L_iteration_best",
        "deposit_symmetry: both directions of each used edge, including the return edge",
        "iteration_best_tie: lowest ant index",
        "best_so_far_rule: replace only on a strictly lower cost",
        "initial_trails: every off-diagonal entry equals tau_max; diagonal is 0",
        "initial_tau_max: 1 / (rho * L_NN)",
        "initial_tau_min: tau_max / (2 * N)",
        "L_NN: deterministic nearest neighbour from nn_start_city",
        "nn_tie_break: lowest city index",
        "best_of_all_starts_tie_break: lowest start-city index",
        "incumbent_initialisation: nearest-neighbour tour is the iteration-0 best-so-far",
        "adaptive_bounds: after a strict global-best improvement, tau_max = 1 / (rho * L_best_so_far) and tau_min = tau_max / (2 * N)",
        "update_order:",
        "  1. Construct all m tours on frozen pheromone (no ACS local update).",
        "  2. Select the iteration-best tour (lowest cost, then lowest ant index).",
        "  3. If its cost is strictly lower than best-so-far, replace best-so-far and recompute tau_max = 1/(rho*L_best_so_far), tau_min = tau_max/(2*N).",
        "  4. Evaporate trails by (1-rho).",
        "  5. Reinforce only the iteration-best tour by 1/L_iteration_best on both directions of each edge, including the return edge.",
        "  6. Clip off-diagonal pheromone to [tau_min, tau_max]. Diagonal stays 0.",
        "disabled_in_v1:",
        f"  acs_local_update: {str(settings.acs_local_update).lower()}",
        "  all_ant_deposit: false",
        f"  local_search: {local_search}",
        "  two_opt: false",
        f"  stagnation_restart: {str(settings.stagnation_restart).lower()}",
        "random_streams: the instance matrix is built once with np.random.default_rng(instance_seed). Each MMAS run has a separate np.random.default_rng(optimiser_seed) used only for start cities and transition sampling.",
        "",
        "EXPERIMENTAL SETTINGS",
        f"instance_seed: {settings.instance_seed}",
        f"N: {n}",
        f"alpha: {settings.alpha}",
        f"beta: {settings.beta}",
        f"rho: {settings.rho}",
        f"n_ants: {n_ants} ({ant_rule})",
        f"n_iterations: {settings.n_iterations}",
        f"n_runs: {len(tuple(settings.optimiser_seeds))}",
        f"optimiser_seeds: {seeds}",
        f"nn_start_city: {settings.nn_start_city}",
        f"seed_incumbent_with_nn: {str(settings.seed_incumbent_with_nn).lower()}",
        f"deposit_policy: {settings.deposit_policy}",
        "sample_std_ddof: 1",
        "convergence_log: iteration 0 is the initial nearest-neighbour incumbent; each later row is best-so-far after that iteration",
        "runtime: wall-clock seconds inside one MMAS run, including its nearest-neighbour initialisation and all iterations; excluding instance generation, the separate baseline calls, plotting, and file output",
        "nn_city0_runtime: wall-clock of the city-0 nearest-neighbour call",
        "nn_best_runtime: wall-clock of the best-of-all-starts call",
        "",
        "INSTANCE CHECKS",
        f"expected_N: {EXPECTED_N}",
        f"observed_N: {n}",
        f"expected_upper_triangle_sum: {EXPECTED_UPPER_TRIANGLE_SUM}",
        f"observed_upper_triangle_sum: {strict_upper_triangle_sum(distance)}",
        "upper_left_5x5:",
        upper_left,
        "generation:",
        *[f"  {rule}" for rule in GENERATION_RULES],
        "",
        "RUNTIME ENVIRONMENT",
        f"python: {sys.version.replace(chr(10), ' ')}",
        f"numpy: {np.__version__}",
        f"platform: {platform.platform()}",
        f"git_revision: {_git_revision()}",
        "",
    ]
    return "\n".join(lines)


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


def summarise_baseline(method: str, result: NearestNeighborResult) -> dict[str, object]:
    return {
        "method": method,
        "n_runs": 1,
        "mean_final_cost": float(result.cost),
        "sample_std_final_cost": None,
        "mean_last_improvement_iteration": None,
        "mean_runtime_seconds": float(result.runtime_seconds),
    }


def summarise_mmas(runs: list[MMASResult]) -> dict[str, object]:
    if len(runs) < 2:
        raise ValidationError(
            "sample standard deviation requires at least 2 runs"
        )
    costs = np.asarray([run.final_cost for run in runs], dtype=np.float64)
    improvements = np.asarray(
        [run.last_improvement_iteration for run in runs], dtype=np.float64
    )
    runtimes = np.asarray([run.runtime_seconds for run in runs], dtype=np.float64)
    return {
        "method": "mmas",
        "n_runs": len(runs),
        "mean_final_cost": float(costs.mean()),
        "sample_std_final_cost": float(costs.std(ddof=1)),
        "mean_last_improvement_iteration": float(improvements.mean()),
        "mean_runtime_seconds": float(runtimes.mean()),
    }


def render_summary_table(rows: list[dict[str, object]]) -> str:
    columns = (
        ("method", 28),
        ("n_runs", 8),
        ("mean_final_cost", 18),
        ("sample_std_final_cost", 26),
        ("mean_last_improvement_iteration", 34),
        ("mean_runtime_seconds", 22),
    )

    def cell(value: object, width: int) -> str:
        if value is None:
            text = "n/a"
        elif isinstance(value, float):
            text = f"{value:.6f}"
        else:
            text = str(value)
        return text.ljust(width)

    header = "".join(cell(name, width) for name, width in columns)
    rule = "".join("-" * width for _, width in columns)
    body = [
        "".join(cell(row[name], width) for name, width in columns) for row in rows
    ]
    notes = (
        "sample_std_final_cost is the sample standard deviation (ddof=1). "
        "It is undefined for a single nearest-neighbour run, so those cells "
        "are n/a. Nearest-neighbour rows are deterministic single calls, so "
        "mean_last_improvement_iteration is n/a for them as well."
    )
    return (
        "SUMMARY (computed from this run)\n"
        + header
        + "\n"
        + rule
        + "\n"
        + "\n".join(body)
        + "\n\n"
        + notes
        + "\n"
    )


def _csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".12g")
    if isinstance(value, (list, tuple)):
        return " ".join(str(int(item)) for item in value)
    if isinstance(value, np.ndarray):
        return " ".join(str(int(item)) for item in value.tolist())
    return str(value)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _environment() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_revision": _git_revision(),
    }


def _run_record(index: int, run: MMASResult, n: int) -> dict[str, object]:
    return {
        "run_index": index,
        "optimiser_seed": run.optimiser_seed,
        "final_cost": run.final_cost,
        "best_tour": [int(city) for city in run.best_tour.tolist()],
        "last_improvement_iteration": run.last_improvement_iteration,
        "runtime_seconds": run.runtime_seconds,
        "initial_nn_cost": run.initial_nn_cost,
        "initial_tau_min": run.initial_tau_min,
        "initial_tau_max": run.initial_tau_max,
        "final_tau_min": run.final_tau_min,
        "final_tau_max": run.final_tau_max,
        "history": list(run.history),
        "parameters": run.parameters,
        "metadata": {
            "instance_seed": PERSONAL_SEED,
            "instance_fingerprint": run.instance_fingerprint,
            "instance_n": n,
            "optimiser_seed": run.optimiser_seed,
            "optimiser_rng": "np.random.default_rng",
            **_environment(),
        },
    }


def _baseline_record(result: NearestNeighborResult, fingerprint: str) -> dict[str, object]:
    return {
        "start_city": result.start_city,
        "cost": result.cost,
        "tour": [int(city) for city in result.tour.tolist()],
        "runtime_seconds": result.runtime_seconds,
        "instance_fingerprint": fingerprint,
    }


def run_experiment(
    settings: ExperimentSettings, output_dir: Path | str
) -> ExperimentResult:
    """Run NN baselines and one MMAS run per optimiser seed on the fixed instance."""
    if not isinstance(settings, ExperimentSettings):
        raise ValidationError("settings must be an ExperimentSettings")
    instance_seed = require_int("instance_seed", settings.instance_seed, minimum=0)
    if instance_seed != PERSONAL_SEED:
        raise ValidationError(
            "experiments use the fixed personalised instance seed 4616"
        )
    seeds = _optimiser_seeds(settings)
    config = _config_from_settings(settings)
    validate_mmas_config(config)

    distance = generate_distance_matrix(PERSONAL_SEED)
    verify_personal_instance(distance, PERSONAL_SEED)
    fingerprint = matrix_fingerprint(distance)
    n = int(distance.shape[0])
    if int(settings.nn_start_city) < 0 or int(settings.nn_start_city) >= n:
        raise ValidationError("nn_start_city is outside the instance")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "audit": output / "audit.txt",
        "convergence_csv": output / "convergence.csv",
        "runs_csv": output / "runs.csv",
        "baselines_csv": output / "baselines.csv",
        "summary_csv": output / "summary.csv",
        "summary_txt": output / "summary.txt",
        "experiment_json": output / "experiment.json",
        "convergence_mean_std": output / "convergence_mean_std.png",
        "convergence_overlay": output / "convergence_overlay.png",
    }
    audit_text = format_audit(settings, distance)
    paths["audit"].write_text(audit_text, encoding="utf-8")
    print(audit_text, flush=True)

    nn_city0 = nearest_neighbor_tour(distance, 0)
    nn_best = best_of_all_starts(distance)
    if settings.nn_start_city == 0:
        nn_init = nn_city0
    else:
        nn_init = nearest_neighbor_tour(distance, int(settings.nn_start_city))

    runs: list[MMASResult] = []
    for seed in seeds:
        result = run_mmas(distance, seed, config)
        if result.instance_fingerprint != fingerprint:
            raise ValidationError(
                "MMAS run fingerprint does not match the shared instance"
            )
        if result.history[0] != nn_init.cost or result.initial_nn_cost != nn_init.cost:
            raise ValidationError(
                "MMAS initial incumbent does not match the configured nearest neighbour"
            )
        if len(result.history) != settings.n_iterations + 1:
            raise ValidationError(
                "convergence log is missing the initial state or an iteration"
            )
        runs.append(result)
        print(
            f"run optimiser_seed={result.optimiser_seed} "
            f"final_cost={result.final_cost} "
            f"last_improvement_iteration={result.last_improvement_iteration} "
            f"runtime_seconds={result.runtime_seconds:.6f}",
            flush=True,
        )

    if len({run.instance_fingerprint for run in runs}) != 1:
        raise ValidationError("runs do not share one instance fingerprint")

    summary_rows = [
        summarise_baseline("nn_city0", nn_city0),
        summarise_baseline("nn_best_of_all_starts", nn_best),
        summarise_mmas(runs),
    ]
    summary_text = render_summary_table(summary_rows)
    paths["summary_txt"].write_text(summary_text, encoding="utf-8")
    print(summary_text, flush=True)

    summary_fields = [
        "method",
        "n_runs",
        "mean_final_cost",
        "sample_std_final_cost",
        "mean_last_improvement_iteration",
        "mean_runtime_seconds",
    ]
    _write_csv(paths["summary_csv"], summary_fields, summary_rows)
    _write_csv(
        paths["baselines_csv"],
        ["method", "start_city", "cost", "tour", "runtime_seconds", "instance_fingerprint"],
        [
            {"method": "nn_city0", **_baseline_record(nn_city0, fingerprint)},
            {
                "method": "nn_best_of_all_starts",
                **_baseline_record(nn_best, fingerprint),
            },
        ],
    )
    run_fields = [
        "run_index",
        "optimiser_seed",
        "final_cost",
        "best_tour",
        "last_improvement_iteration",
        "runtime_seconds",
        "initial_nn_cost",
        "final_tau_min",
        "final_tau_max",
        "instance_fingerprint",
    ]
    run_rows = []
    convergence_rows = []
    for index, run in enumerate(runs):
        record = _run_record(index, run, n)
        run_rows.append(
            {
                "run_index": index,
                "optimiser_seed": run.optimiser_seed,
                "final_cost": run.final_cost,
                "best_tour": record["best_tour"],
                "last_improvement_iteration": run.last_improvement_iteration,
                "runtime_seconds": run.runtime_seconds,
                "initial_nn_cost": run.initial_nn_cost,
                "final_tau_min": run.final_tau_min,
                "final_tau_max": run.final_tau_max,
                "instance_fingerprint": fingerprint,
            }
        )
        for iteration, cost in enumerate(run.history):
            convergence_rows.append(
                {
                    "run_index": index,
                    "optimiser_seed": run.optimiser_seed,
                    "iteration": iteration,
                    "best_so_far_cost": cost,
                }
            )
    _write_csv(paths["runs_csv"], run_fields, run_rows)
    _write_csv(
        paths["convergence_csv"],
        ["run_index", "optimiser_seed", "iteration", "best_so_far_cost"],
        convergence_rows,
    )

    payload = {
        "variant_mechanics_text": audit_text,
        "settings": {
            "instance_seed": PERSONAL_SEED,
            "alpha": settings.alpha,
            "beta": settings.beta,
            "rho": settings.rho,
            "n_ants": n if settings.n_ants is None else settings.n_ants,
            "n_ants_rule": "equal to N" if settings.n_ants is None else "fixed count",
            "n_iterations": settings.n_iterations,
            "optimiser_seeds": list(seeds),
            "nn_start_city": settings.nn_start_city,
            "seed_incumbent_with_nn": settings.seed_incumbent_with_nn,
            "deposit_policy": settings.deposit_policy,
            "acs_local_update": settings.acs_local_update,
            "local_search": settings.local_search,
            "stagnation_restart": settings.stagnation_restart,
            "sample_std_ddof": 1,
        },
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
        "environment": _environment(),
        "baselines": {
            "nn_city0": _baseline_record(nn_city0, fingerprint),
            "nn_best_of_all_starts": _baseline_record(nn_best, fingerprint),
        },
        "runs": [_run_record(index, run, n) for index, run in enumerate(runs)],
        "summary": summary_rows,
    }
    paths["experiment_json"].write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    histories = [list(run.history) for run in runs]
    figures: list[object] = []
    try:
        figures.append(
            plot_mean_std(
                histories,
                paths["convergence_mean_std"],
                nn_city0=float(nn_city0.cost),
                nn_best=float(nn_best.cost),
            )
        )
        figures.append(plot_overlay(histories, list(seeds), paths["convergence_overlay"]))
    finally:
        import matplotlib.pyplot as plt

        for figure in figures:
            plt.close(figure)

    return ExperimentResult(
        audit_text=audit_text,
        summary_text=summary_text,
        summary_rows=summary_rows,
        output_dir=output,
        paths=paths,
        instance_fingerprint=fingerprint,
        runs=runs,
        nn_city0=nn_city0,
        nn_best=nn_best,
    )


def run_baseline_experiment(output_dir: Path | str = Path("outputs")) -> ExperimentResult:
    return run_experiment(baseline_settings(), output_dir)


def main() -> None:
    result = run_baseline_experiment(REPO_ROOT / "outputs")
    print(f"wrote {result.output_dir}", flush=True)


if __name__ == "__main__":
    main()
