"""MAX-MIN Ant System for symmetric TSP.

v1 reinforces only the iteration-best tour. It does not apply an ACS
local update, all-ant deposit, 2-opt, or a stagnation restart. Those
switches exist on MMASConfig and raise NotImplementedError if enabled.

Pheromone update order inside each iteration
--------------------------------------------
1. Construct all m tours while pheromone is frozen.
2. Select the iteration-best tour: lowest cost, then lowest ant index.
3. If that cost is strictly lower than best-so-far, replace best-so-far
   and recompute tau_max = 1 / (rho * L_best_so_far) and
   tau_min = tau_max / (2 * N). Ties keep the earlier tour and the
   previous bounds.
4. Evaporate every trail by multiplying by (1 - rho).
5. Add 1 / L_iteration_best on both directions of each iteration-best
   edge, including the return to the start. No other tour deposits.
6. Clip off-diagonal trails into [tau_min, tau_max]. The diagonal stays 0.

The deposit in step 5 uses L_iteration_best even when best-so-far is
shorter. The clip in step 6 uses the bounds from step 3, so a new
global best tightens the bounds in the same iteration.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from sis6116.instance import matrix_fingerprint
from sis6116.nearest_neighbor import nearest_neighbor_tour
from sis6116.objective import tour_cost
from sis6116.validation import (
    ValidationError,
    last_strict_improvement,
    require_bool,
    require_finite_float,
    require_int,
    validate_distance_matrix,
    validate_tour,
)

KNOWN_DEPOSIT_POLICIES = ("iteration_best", "all_ants", "global_best")


@dataclass(frozen=True)
class MMASConfig:
    """Baseline defaults. Not a tuned configuration."""

    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.2
    n_ants: int | None = None
    n_iterations: int = 100
    seed_incumbent_with_nn: bool = True
    nn_start_city: int = 0
    deposit_policy: str = "iteration_best"
    acs_local_update: bool = False
    local_search: str | None = None
    stagnation_restart: bool = False


@dataclass
class MMASResult:
    optimiser_seed: int
    final_cost: int
    best_tour: np.ndarray
    last_improvement_iteration: int
    runtime_seconds: float
    history: list[int]
    initial_nn_cost: int
    initial_nn_tour: np.ndarray
    initial_tau_min: float
    initial_tau_max: float
    final_tau_min: float
    final_tau_max: float
    final_pheromone: np.ndarray
    instance_fingerprint: str
    parameters: dict[str, object]


def assert_v1_policies(config: MMASConfig) -> None:
    """Reject v1-disabled behaviour instead of ignoring the switch."""
    if not isinstance(config, MMASConfig):
        raise ValidationError("config must be an MMASConfig")
    if not isinstance(config.deposit_policy, str):
        raise ValidationError("deposit_policy must be a string")
    if config.deposit_policy not in KNOWN_DEPOSIT_POLICIES:
        raise ValidationError(
            f"unknown deposit_policy {config.deposit_policy!r}"
        )
    if config.deposit_policy != "iteration_best":
        raise NotImplementedError(
            f"deposit_policy={config.deposit_policy!r} is configurable "
            "but not implemented in v1; the baseline deposits the "
            "iteration-best tour only"
        )
    if require_bool("acs_local_update", config.acs_local_update):
        raise NotImplementedError(
            "ACS local pheromone update is configurable but disabled in v1"
        )
    if config.local_search is not None:
        if not isinstance(config.local_search, str):
            raise ValidationError("local_search must be null or a search name")
        raise NotImplementedError(
            f"local search {config.local_search!r} is configurable but "
            "not implemented in v1"
        )
    if require_bool("stagnation_restart", config.stagnation_restart):
        raise NotImplementedError(
            "stagnation restart is configurable but disabled in v1"
        )
    if not require_bool("seed_incumbent_with_nn", config.seed_incumbent_with_nn):
        raise NotImplementedError(
            "v1 initialises best-so-far with the nearest-neighbour tour "
            "so iteration 0 has a cost"
        )


def validate_mmas_config(config: MMASConfig) -> dict[str, float | int | None]:
    assert_v1_policies(config)
    alpha = require_finite_float("alpha", config.alpha)
    beta = require_finite_float("beta", config.beta)
    rho = require_finite_float("rho", config.rho)
    if alpha < 0.0:
        raise ValidationError("alpha must be >= 0")
    if beta < 0.0:
        raise ValidationError("beta must be >= 0")
    if not 0.0 < rho < 1.0:
        raise ValidationError("rho must be in (0, 1)")
    n_iterations = require_int("n_iterations", config.n_iterations, minimum=1)
    n_ants: int | None
    if config.n_ants is None:
        n_ants = None
    else:
        n_ants = require_int("n_ants", config.n_ants, minimum=1)
    nn_start_city = require_int("nn_start_city", config.nn_start_city, minimum=0)
    return {
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "n_iterations": n_iterations,
        "n_ants": n_ants,
        "nn_start_city": nn_start_city,
    }


def bounds_from_length(length: int, rho: float, n: int) -> tuple[float, float]:
    """Return (tau_min, tau_max) = (tau_max / (2N), 1 / (rho * L))."""
    if length <= 0:
        raise ValidationError("tour length for pheromone bounds must be positive")
    rho = require_finite_float("rho", rho)
    n = require_int("n", n, minimum=1)
    if not 0.0 < rho < 1.0:
        raise ValidationError("rho must be in (0, 1)")
    tau_max = 1.0 / (rho * float(length))
    tau_min = tau_max / (2.0 * float(n))
    if (
        not math.isfinite(tau_max)
        or not math.isfinite(tau_min)
        or tau_min < 0.0
        or tau_min > tau_max
    ):
        raise ValidationError("nonfinite or inverted pheromone bounds")
    return tau_min, tau_max


def heuristic_matrix(distance: np.ndarray, beta: float) -> np.ndarray:
    """eta_ij = (1 / d_ij) ** beta off the diagonal, and 0 on the diagonal."""
    n = int(distance.shape[0])
    eta = np.zeros((n, n), dtype=np.float64)
    off = ~np.eye(n, dtype=bool)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        eta[off] = np.power(1.0 / distance[off].astype(np.float64), beta)
    if not np.isfinite(eta).all() or np.any(eta[off] < 0.0):
        raise ValidationError("nonfinite heuristic information")
    return eta


def _pheromone_factor(trail: float, alpha: float) -> float:
    try:
        if alpha == 0.0:
            value = 1.0
        elif alpha == 1.0:
            value = trail
        else:
            value = trail ** alpha
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise ValidationError("nonfinite transition weight") from exc
    if isinstance(value, complex) or not isinstance(value, (int, float)):
        raise ValidationError("nonfinite transition weight")
    value = float(value)
    if not math.isfinite(value):
        raise ValidationError("nonfinite transition weight")
    if value < 0.0:
        raise ValidationError("negative transition weight")
    return value


def weights_for_unvisited(
    tau_row: list[float],
    eta_row: list[float],
    visited: list[bool],
    alpha: float,
) -> tuple[list[int], list[float]]:
    """Weights for unvisited cities only. Visited cities are omitted."""
    if len(tau_row) != len(eta_row) or len(tau_row) != len(visited):
        raise ValidationError("pheromone, heuristic, and visited mask differ in length")
    cities: list[int] = []
    weights: list[float] = []
    for city, seen in enumerate(visited):
        if seen:
            continue
        pheromone = _pheromone_factor(float(tau_row[city]), alpha)
        heuristic = float(eta_row[city])
        if not math.isfinite(heuristic):
            raise ValidationError("nonfinite transition weight")
        weight = pheromone * heuristic
        if not math.isfinite(weight):
            raise ValidationError("nonfinite transition weight")
        if weight < 0.0:
            raise ValidationError("negative transition weight")
        cities.append(city)
        weights.append(weight)
    if not cities:
        raise ValidationError("no unvisited cities remain")
    return cities, weights


def probabilities_from_weights(weights: list[float]) -> list[float]:
    """Normalise weights. Raise on nonfinite, negative, or nonpositive mass."""
    if not weights:
        raise ValidationError("probability weights must be non-empty")
    for weight in weights:
        if not math.isfinite(weight):
            raise ValidationError("nonfinite transition weight")
        if weight < 0.0:
            raise ValidationError("negative transition weight")
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise ValidationError(
            "transition weights have nonpositive or nonfinite mass"
        )
    probs = [weight / total for weight in weights]
    for prob in probs:
        if not math.isfinite(prob) or prob < 0.0:
            raise ValidationError("nonfinite transition probability")
    mass = math.fsum(probs)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValidationError("nonfinite transition probability")
    if abs(mass - 1.0) > 1e-8:
        probs = [prob / mass for prob in probs]
        mass = math.fsum(probs)
    if not math.isfinite(mass) or abs(mass - 1.0) > 1e-6:
        raise ValidationError("transition probabilities do not sum to 1")
    if mass != 1.0:
        probs = [prob / mass for prob in probs]
    final_mass = math.fsum(probs)
    if not math.isfinite(final_mass) or abs(final_mass - 1.0) > 1e-6:
        raise ValidationError("transition probabilities do not sum to 1")
    return probs


def candidate_probabilities(
    tau: np.ndarray,
    eta: np.ndarray,
    current: int,
    visited: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (unvisited cities, probabilities) aligned with each other."""
    visited_list = [bool(flag) for flag in np.asarray(visited).tolist()]
    tau_row = [float(value) for value in np.asarray(tau[current], dtype=np.float64).tolist()]
    eta_row = [float(value) for value in np.asarray(eta[current], dtype=np.float64).tolist()]
    cities, weights = weights_for_unvisited(tau_row, eta_row, visited_list, float(alpha))
    probs = probabilities_from_weights(weights)
    return (
        np.asarray(cities, dtype=np.int64),
        np.asarray(probs, dtype=np.float64),
    )


def sample_categorical(probs: list[float] | np.ndarray, rng: object) -> int:
    """Inverse-transform sample. One rng.random() draw in [0, 1)."""
    values = [float(prob) for prob in list(probs)]
    if any(not math.isfinite(prob) or prob < 0.0 for prob in values):
        raise ValidationError("nonfinite transition probability")
    draw = getattr(rng, "random")
    threshold = float(draw())
    if not math.isfinite(threshold) or threshold < 0.0 or threshold >= 1.0:
        raise ValidationError("random draw outside [0, 1)")
    cumulative = 0.0
    for index, prob in enumerate(values):
        cumulative += prob
        if cumulative > threshold:
            return index
    for index in range(len(values) - 1, -1, -1):
        if values[index] > 0.0:
            return index
    raise ValidationError("failed to sample a transition")


def construct_tour(
    tau_rows: list[list[float]],
    eta_rows: list[list[float]],
    alpha: float,
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    """One tour. The start city consumes rng.integers; each step consumes rng.random."""
    start = int(rng.integers(0, n))
    visited = [False] * n
    visited[start] = True
    tour = np.empty(n, dtype=np.int64)
    tour[0] = start
    current = start
    for step in range(1, n):
        cities, weights = weights_for_unvisited(
            tau_rows[current], eta_rows[current], visited, alpha
        )
        if len(cities) != n - step:
            raise ValidationError(
                "visited-city mask does not match the number of remaining cities"
            )
        probs = probabilities_from_weights(weights)
        pick = sample_categorical(probs, rng)
        city = cities[pick]
        if visited[city]:
            raise ValidationError("sampled a visited city")
        tour[step] = city
        visited[city] = True
        current = city
    if not all(visited):
        raise ValidationError("constructed tour left a city unvisited")
    return tour


def construct_tours(
    tau: np.ndarray,
    eta_rows: list[list[float]],
    alpha: float,
    n_ants: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Build m tours on a snapshot of tau. Ants run in index order."""
    tau_rows = [[float(value) for value in row] for row in tau.tolist()]
    return [
        construct_tour(tau_rows, eta_rows, alpha, rng, tau.shape[0])
        for _ in range(n_ants)
    ]


def select_iteration_best(costs: list[int]) -> int:
    """Lowest cost, then the lowest ant index."""
    if not costs:
        raise ValidationError("iteration has no tours")
    best_index = 0
    best_cost = costs[0]
    for index in range(1, len(costs)):
        if costs[index] < best_cost:
            best_index = index
            best_cost = costs[index]
    return best_index


def evaporate(tau: np.ndarray, rho: float) -> None:
    tau *= 1.0 - rho
    np.fill_diagonal(tau, 0.0)


def deposit_iteration_best(tau: np.ndarray, tour: np.ndarray, cost: int) -> None:
    """Add 1/L on both directions of each cycle edge, including the return."""
    if cost <= 0:
        raise ValidationError("deposit amount requires a positive tour cost")
    delta = 1.0 / float(cost)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValidationError("deposit amount must be positive and finite")
    n = int(tour.shape[0])
    for index in range(n):
        start = int(tour[index])
        end = int(tour[(index + 1) % n])
        if start == end:
            raise ValidationError("tour has a self-loop")
        tau[start, end] += delta
        tau[end, start] += delta


def clip_pheromone(tau: np.ndarray, tau_min: float, tau_max: float) -> None:
    """Clip off-diagonal entries. The diagonal is set back to 0."""
    if not math.isfinite(tau_min) or not math.isfinite(tau_max):
        raise ValidationError("nonfinite pheromone bounds")
    if tau_min < 0.0 or tau_min > tau_max:
        raise ValidationError("pheromone bounds are inverted or negative")
    if tau.shape[0] != tau.shape[1]:
        raise ValidationError("pheromone matrix must be square")
    off = ~np.eye(tau.shape[0], dtype=bool)
    tau[off] = np.clip(tau[off], tau_min, tau_max)
    np.fill_diagonal(tau, 0.0)
    if not np.isfinite(tau).all():
        raise ValidationError("nonfinite pheromone after clipping")


def reinforce_iteration_best(
    tau: np.ndarray,
    tour: np.ndarray,
    cost: int,
    rho: float,
    tau_min: float,
    tau_max: float,
) -> None:
    """Evaporate, deposit the iteration-best tour, then clip.

    Bounds must already include any global-best update from this iteration.
    """
    evaporate(tau, rho)
    deposit_iteration_best(tau, tour, cost)
    clip_pheromone(tau, tau_min, tau_max)


def _as_float_rows(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix.tolist()]


def run_mmas(
    distance: object,
    optimiser_seed: int,
    config: MMASConfig | None = None,
) -> MMASResult:
    """Run one MMAS trajectory from a dedicated optimiser random stream."""
    if config is None:
        config = MMASConfig()
    checked = validate_mmas_config(config)
    matrix = validate_distance_matrix(distance)
    n = int(matrix.shape[0])
    seed = require_int("optimiser_seed", optimiser_seed, minimum=0)
    alpha = float(checked["alpha"])
    beta = float(checked["beta"])
    rho = float(checked["rho"])
    n_iterations = int(checked["n_iterations"])
    nn_start_city = int(checked["nn_start_city"])
    if nn_start_city >= n:
        raise ValidationError("nn_start_city is outside the instance")
    n_ants = n if checked["n_ants"] is None else int(checked["n_ants"])
    instance_fingerprint = matrix_fingerprint(matrix)

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    nn_result = nearest_neighbor_tour(matrix, nn_start_city)
    nn_tour = nn_result.tour.copy()
    nn_cost = int(nn_result.cost)
    tau_min, tau_max = bounds_from_length(nn_cost, rho, n)
    initial_tau_min = tau_min
    initial_tau_max = tau_max
    tau = np.full((n, n), tau_max, dtype=np.float64)
    np.fill_diagonal(tau, 0.0)
    eta_rows = _as_float_rows(heuristic_matrix(matrix, beta))

    best_tour = nn_tour.copy()
    best_cost = nn_cost
    last_improvement = 0
    history = [best_cost]

    for iteration in range(1, n_iterations + 1):
        # Frozen pheromone: construct_tours copies tau before any ant moves,
        # and this loop does not write tau until every tour has a cost.
        tours = construct_tours(tau, eta_rows, alpha, n_ants, rng)
        if len(tours) != n_ants:
            raise ValidationError("construction did not return one tour per ant")
        costs = [tour_cost(matrix, tour) for tour in tours]
        best_ant = select_iteration_best(costs)
        ib_tour = validate_tour(tours[best_ant], n)
        ib_cost = int(costs[best_ant])

        # Strict improvement updates the incumbent and the bounds before
        # evaporation, so the clip below uses the new [tau_min, tau_max].
        if ib_cost < best_cost:
            best_cost = ib_cost
            best_tour = ib_tour.copy()
            last_improvement = iteration
            tau_min, tau_max = bounds_from_length(best_cost, rho, n)

        # Evaporate, then deposit only this iteration-best tour, then clip.
        reinforce_iteration_best(tau, ib_tour, ib_cost, rho, tau_min, tau_max)
        if not np.array_equal(tau, tau.T):
            raise ValidationError("pheromone matrix lost symmetry")
        history.append(best_cost)

    runtime = time.perf_counter() - started
    if tour_cost(matrix, best_tour) != best_cost:
        raise ValidationError("recorded best tour does not match its cost")
    if last_strict_improvement(history) != last_improvement:
        raise ValidationError(
            "last improvement iteration does not match the convergence log"
        )
    if history[-1] != best_cost or len(history) != n_iterations + 1:
        raise ValidationError("convergence log does not match best-so-far")

    parameters: dict[str, object] = {
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "n_ants": n_ants,
        "n_iterations": n_iterations,
        "deposit_policy": config.deposit_policy,
        "acs_local_update": config.acs_local_update,
        "local_search": config.local_search,
        "stagnation_restart": config.stagnation_restart,
        "seed_incumbent_with_nn": config.seed_incumbent_with_nn,
        "nn_start_city": nn_start_city,
        "initial_nn_cost": nn_cost,
        "initial_tau_min": initial_tau_min,
        "initial_tau_max": initial_tau_max,
        "tau_max_formula": "1 / (rho * L)",
        "tau_min_formula": "tau_max / (2 * N)",
        "optimiser_seed": seed,
        "optimiser_rng": "np.random.default_rng",
    }
    return MMASResult(
        optimiser_seed=seed,
        final_cost=best_cost,
        best_tour=best_tour.copy(),
        last_improvement_iteration=last_improvement,
        runtime_seconds=runtime,
        history=list(history),
        initial_nn_cost=nn_cost,
        initial_nn_tour=nn_tour.copy(),
        initial_tau_min=initial_tau_min,
        initial_tau_max=initial_tau_max,
        final_tau_min=tau_min,
        final_tau_max=tau_max,
        final_pheromone=tau.copy(),
        instance_fingerprint=instance_fingerprint,
        parameters=parameters,
    )
