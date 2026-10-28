"""MMAS mechanics: probabilities, pheromone, bounds, monotonicity, reproducibility."""

import copy
import math

import numpy as np
import pytest

from sis6116 import mmas
from sis6116.instance import generate_distance_matrix
from sis6116.mmas import (
    MMASConfig,
    bounds_from_length,
    candidate_probabilities,
    clip_pheromone,
    construct_tours,
    heuristic_matrix,
    probabilities_from_weights,
    reinforce_iteration_best,
    run_mmas,
    sample_categorical,
    select_iteration_best,
)
from sis6116.nearest_neighbor import nearest_neighbor_tour
from sis6116.objective import tour_cost
from sis6116.validation import ValidationError, last_strict_improvement

TINY = np.array(
    [
        [0, 1, 9, 9],
        [1, 0, 1, 9],
        [9, 1, 0, 1],
        [9, 9, 1, 0],
    ],
    dtype=np.int64,
)

SIX = np.array(
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


class _Draw:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


def test_probabilities_sum_to_one_and_match_the_weight_formula():
    tau = np.array(
        [
            [0.0, 4.0, 1.0, 2.0],
            [4.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [2.0, 1.0, 1.0, 0.0],
        ]
    )
    eta = np.array(
        [
            [0.0, 1.0, 0.5, 0.25],
            [1.0, 0.0, 1.0, 1.0],
            [0.5, 1.0, 0.0, 1.0],
            [0.25, 1.0, 1.0, 0.0],
        ]
    )
    visited = np.array([True, False, False, False])
    cities, probs = candidate_probabilities(tau, eta, 0, visited, alpha=1.0)
    assert cities.tolist() == [1, 2, 3]
    np.testing.assert_allclose(probs, [0.8, 0.1, 0.1])
    assert float(probs.sum()) == pytest.approx(1.0)

    tiny = probabilities_from_weights([1e-18, 1e-18, 1e-18])
    assert math.fsum(tiny) == pytest.approx(1.0)
    assert tiny[0] == pytest.approx(1.0 / 3.0)


def test_alpha_zero_and_beta_zero_are_uniform_over_unvisited():
    distance = np.array(
        [
            [0, 2, 3, 4],
            [2, 0, 5, 6],
            [3, 5, 0, 7],
            [4, 6, 7, 0],
        ],
        dtype=np.int64,
    )
    eta = heuristic_matrix(distance, beta=0.0)
    tau = np.full((4, 4), 7.0)
    visited = np.array([True, False, False, False])
    cities, probs = candidate_probabilities(tau, eta, 0, visited, alpha=0.0)
    assert cities.tolist() == [1, 2, 3]
    np.testing.assert_allclose(probs, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


def test_precomputed_heuristic_matches_one_over_distance():
    distance = np.array(
        [
            [0, 1, 2, 4],
            [1, 0, 2, 2],
            [2, 2, 0, 2],
            [4, 2, 2, 0],
        ],
        dtype=np.int64,
    )
    tau = np.array(
        [
            [0.0, 4.0, 3.0, 1.0],
            [4.0, 0.0, 1.0, 1.0],
            [3.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    eta = heuristic_matrix(distance, beta=2.0)
    visited = np.array([True, False, False, False])
    cities, probs = candidate_probabilities(tau, eta, 0, visited, alpha=1.0)
    manual = np.array(
        [tau[0, city] * (1.0 / distance[0, city]) ** 2 for city in cities]
    )
    manual = manual / manual.sum()
    np.testing.assert_allclose(probs, manual)
    assert float(probs.sum()) == pytest.approx(1.0)


def test_visited_cities_are_excluded_even_if_their_weight_would_be_large():
    tau = np.array(
        [
            [0.0, 1.0, 1000.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1000.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    eta = np.ones((4, 4))
    np.fill_diagonal(eta, 0.0)
    visited = np.array([True, False, True, False])
    cities, probs = candidate_probabilities(tau, eta, 0, visited, alpha=1.0)
    assert 2 not in cities.tolist()
    assert 0 not in cities.tolist()
    assert cities.tolist() == [1, 3]
    np.testing.assert_allclose(probs, [0.5, 0.5])
    assert float(probs.sum()) == pytest.approx(1.0)


def test_nonfinite_or_nonpositive_probabilities_raise():
    with pytest.raises(ValidationError, match="nonfinite"):
        probabilities_from_weights([math.inf, 1.0])
    with pytest.raises(ValidationError, match="nonfinite"):
        probabilities_from_weights([math.nan, 1.0])
    with pytest.raises(ValidationError, match="nonpositive"):
        probabilities_from_weights([0.0, 0.0])
    with pytest.raises(ValidationError, match="negative"):
        probabilities_from_weights([-1.0, 2.0])

    tau = np.ones((4, 4))
    tau[0, 1] = np.nan
    eta = np.ones((4, 4))
    np.fill_diagonal(eta, 0.0)
    visited = np.array([True, False, False, False])
    with pytest.raises(ValidationError, match="nonfinite"):
        candidate_probabilities(tau, eta, 0, visited, alpha=1.0)

    tau[0, 1] = -2.0
    with pytest.raises(ValidationError, match="negative"):
        candidate_probabilities(tau, eta, 0, visited, alpha=1.0)

    tau[0, 1] = 1e300
    with pytest.raises(ValidationError, match="nonfinite"):
        candidate_probabilities(tau, eta, 0, visited, alpha=10.0)


def test_inverse_transform_sampling_follows_the_normalised_mass():
    probs = [0.25, 0.75]
    assert sample_categorical(probs, _Draw(0.0)) == 0
    assert sample_categorical(probs, _Draw(0.2)) == 0
    assert sample_categorical(probs, _Draw(0.25)) == 1
    assert sample_categorical(probs, _Draw(0.99)) == 1
    assert sample_categorical([0.0, 1.0], _Draw(0.0)) == 1
    with pytest.raises(ValidationError):
        sample_categorical([0.5, 0.5], _Draw(1.0))
    with pytest.raises(ValidationError):
        sample_categorical([math.nan, 0.0], _Draw(0.2))


def test_iteration_best_tie_keeps_the_lowest_ant_index():
    assert select_iteration_best([5, 4, 4]) == 1
    assert select_iteration_best([3, 3, 1]) == 2
    assert select_iteration_best([2, 2]) == 0


def test_evaporate_then_deposit_then_clip_order_and_symmetry():
    tau = np.full((4, 4), 10.0)
    np.fill_diagonal(tau, 0.0)
    tour = np.array([0, 1, 2, 3], dtype=np.int64)
    reinforce_iteration_best(tau, tour, cost=8, rho=0.2, tau_min=0.01, tau_max=100.0)
    # (1-rho)*10 + 1/8 = 8.125. Deposit-then-evaporate would be 8.1.
    np.testing.assert_allclose(tau[0, 1], 8.125)
    np.testing.assert_allclose(tau[1, 0], 8.125)
    np.testing.assert_allclose(tau[3, 0], 8.125)
    np.testing.assert_allclose(tau[0, 3], 8.125)
    np.testing.assert_allclose(tau[0, 2], 8.0)
    np.testing.assert_allclose(tau[2, 0], 8.0)
    np.testing.assert_allclose(tau[1, 3], 8.0)
    assert np.all(np.diag(tau) == 0.0)
    np.testing.assert_array_equal(tau, tau.T)


def test_clip_off_diagonal_bounds_and_leave_the_diagonal_at_zero():
    tau = np.array(
        [
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 100.0],
            [5.0, 100.0, 0.0],
        ]
    )
    clip_pheromone(tau, 1.0, 10.0)
    expected = np.array(
        [
            [0.0, 1.0, 5.0],
            [1.0, 0.0, 10.0],
            [5.0, 10.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(tau, expected)


def test_only_iteration_best_is_reinforced_including_the_return_edge(monkeypatch):
    bad = np.array([0, 1, 3, 2], dtype=np.int64)
    good = np.array([0, 1, 2, 3], dtype=np.int64)
    assert tour_cost(TINY, bad) == 20
    assert tour_cost(TINY, good) == 12

    def fake(*_args, **_kwargs):
        return [bad.copy(), good.copy()]

    monkeypatch.setattr(mmas, "construct_tours", fake)
    result = run_mmas(TINY, 0, MMASConfig(n_iterations=1, n_ants=2, rho=0.02))
    rho = 0.02
    tau_max = 1.0 / (rho * 12.0)
    assert result.initial_nn_cost == 12
    assert result.initial_tau_max == pytest.approx(tau_max)
    assert result.history == [12, 12]
    assert result.last_improvement_iteration == 0
    pheromone = result.final_pheromone
    np.testing.assert_allclose(pheromone[1, 2], tau_max)
    np.testing.assert_allclose(pheromone[2, 1], tau_max)
    np.testing.assert_allclose(pheromone[3, 0], tau_max)
    np.testing.assert_allclose(pheromone[0, 3], tau_max)
    unused = (1.0 - rho) * tau_max
    np.testing.assert_allclose(pheromone[0, 2], unused)
    np.testing.assert_allclose(pheromone[2, 0], unused)
    np.testing.assert_allclose(pheromone[1, 3], unused)
    assert pheromone[1, 2] > pheromone[0, 2]
    assert np.all(np.diag(pheromone) == 0.0)
    np.testing.assert_array_equal(pheromone, pheromone.T)


def test_repeated_tour_clips_unused_edges_to_tau_min(monkeypatch):
    tour = np.array([0, 1, 2, 3], dtype=np.int64)

    def fake(*_args, **_kwargs):
        return [tour.copy()]

    monkeypatch.setattr(mmas, "construct_tours", fake)
    result = run_mmas(
        TINY, 0, MMASConfig(n_iterations=6, n_ants=1, rho=0.5)
    )
    assert result.final_tau_min == pytest.approx(result.final_tau_max / (2 * 4))
    assert result.final_pheromone[0, 2] == pytest.approx(result.final_tau_min)
    assert result.final_pheromone[2, 0] == pytest.approx(result.final_tau_min)
    off = result.final_pheromone[~np.eye(4, dtype=bool)]
    assert np.all(off <= result.final_tau_max + 1e-12)
    assert np.all(off >= result.final_tau_min - 1e-12)
    assert result.final_pheromone[0, 1] > result.final_pheromone[0, 2]


def test_best_so_far_is_monotone_and_bounds_follow_strict_improvements(monkeypatch):
    worse = np.array([1, 0, 5, 4, 2, 3], dtype=np.int64)
    better = np.array([2, 1, 0, 5, 4, 3], dtype=np.int64)
    assert tour_cost(SIX, worse) == 74
    assert tour_cost(SIX, better) == 62
    assert nearest_neighbor_tour(SIX, 0).cost == 68
    scripted = iter(([worse], [better], [worse]))

    def fake(*_args, **_kwargs):
        batch = next(scripted)
        return [tour.copy() for tour in batch]

    monkeypatch.setattr(mmas, "construct_tours", fake)
    rho = 0.2
    result = run_mmas(
        SIX, 0, MMASConfig(n_iterations=3, n_ants=1, rho=rho, nn_start_city=0)
    )
    assert result.history == [68, 68, 62, 62]
    assert last_strict_improvement(result.history) == 2
    assert result.last_improvement_iteration == 2
    assert result.final_cost == 62
    assert result.best_tour.tolist() == better.tolist()
    assert result.initial_tau_max == pytest.approx(1.0 / (rho * 68.0))
    assert result.final_tau_max == pytest.approx(1.0 / (rho * 62.0))
    assert result.final_tau_min == pytest.approx(result.final_tau_max / (2.0 * 6.0))
    assert result.history[-1] <= result.history[0]


def test_tau_max_uses_the_configured_start_not_best_of_all_starts():
    rho = 0.02
    from_zero = run_mmas(SIX, 0, MMASConfig(n_iterations=1, n_ants=1, rho=rho))
    assert from_zero.initial_nn_cost == 68
    assert from_zero.history[0] == 68
    assert from_zero.initial_tau_max == pytest.approx(1.0 / (rho * 68.0))
    assert from_zero.initial_tau_min == pytest.approx(from_zero.initial_tau_max / (2.0 * 6.0))

    from_two = run_mmas(
        SIX, 0, MMASConfig(n_iterations=1, n_ants=1, rho=rho, nn_start_city=2)
    )
    assert from_two.initial_nn_cost == 62
    assert from_two.initial_tau_max == pytest.approx(1.0 / (rho * 62.0))


def test_construction_does_not_update_pheromone(symmetric_distance):
    distance = symmetric_distance(n=6, seed=4)
    eta_rows = heuristic_matrix(distance, 2.0).tolist()
    tau = np.full(distance.shape, 1.5)
    np.fill_diagonal(tau, 0.0)
    before = tau.copy()
    tours = construct_tours(tau, eta_rows, 1.0, 4, np.random.default_rng(0))
    np.testing.assert_array_equal(tau, before)
    assert len(tours) == 4
    for tour in tours:
        assert sorted(int(city) for city in tour.tolist()) == list(range(6))
        assert tour_cost(distance, tour) > 0


def test_each_iteration_builds_m_tours_without_touching_pheromone(monkeypatch, symmetric_distance):
    distance = symmetric_distance(n=5, seed=8)
    counts = []
    real = mmas.construct_tours

    def wrap(tau, eta_rows, alpha, n_ants, rng):
        counts.append(n_ants)
        snapshot = tau.copy()
        tours = real(tau, eta_rows, alpha, n_ants, rng)
        np.testing.assert_array_equal(tau, snapshot)
        return tours

    monkeypatch.setattr(mmas, "construct_tours", wrap)
    run_mmas(distance, 0, MMASConfig(n_iterations=4, n_ants=3))
    assert counts == [3, 3, 3, 3]

    counts.clear()
    monkeypatch.setattr(mmas, "construct_tours", wrap)
    run_mmas(distance, 1, MMASConfig(n_iterations=2))
    assert counts == [5, 5]


def test_same_seed_reproducibility_and_search_invariants(symmetric_distance):
    distance = symmetric_distance(n=8, seed=11)
    original = distance.copy()
    config = MMASConfig(n_iterations=5, n_ants=4, rho=0.1, alpha=1.0, beta=2.0)
    first = run_mmas(distance, 123, config)
    second = run_mmas(distance, 123, config)
    assert first.history == second.history
    assert first.final_cost == second.final_cost
    np.testing.assert_array_equal(first.best_tour, second.best_tour)
    np.testing.assert_array_equal(first.final_pheromone, second.final_pheromone)
    assert first.last_improvement_iteration == second.last_improvement_iteration
    np.testing.assert_array_equal(distance, original)

    assert first.history[-1] == first.final_cost
    assert first.history[0] == first.initial_nn_cost
    assert last_strict_improvement(first.history) == first.last_improvement_iteration
    assert sorted(first.best_tour.tolist()) == list(range(8))
    assert tour_cost(distance, first.best_tour) == first.final_cost
    assert first.final_cost <= first.history[0]
    off = ~np.eye(8, dtype=bool)
    assert np.all(first.final_pheromone[off] <= first.final_tau_max + 1e-9)
    assert np.all(first.final_pheromone[off] >= first.final_tau_min - 1e-9)
    np.testing.assert_array_equal(first.final_pheromone, first.final_pheromone.T)
    assert np.all(np.diag(first.final_pheromone) == 0.0)
    tau_min, tau_max = bounds_from_length(first.initial_nn_cost, 0.1, 8)
    assert tau_min == pytest.approx(first.initial_tau_min)
    assert tau_max == pytest.approx(first.initial_tau_max)


def test_instance_stream_is_not_advanced_by_building_the_matrix():
    optimiser = np.random.default_rng(123)
    state = copy.deepcopy(optimiser.bit_generator.state)
    generate_distance_matrix(4616)
    assert optimiser.bit_generator.state == state
    optimiser.random()
    assert optimiser.bit_generator.state != state


def test_invalid_parameters_and_disabled_policies_fail(symmetric_distance):
    distance = symmetric_distance(n=4, seed=2)
    with pytest.raises(ValidationError, match="rho"):
        run_mmas(distance, 0, MMASConfig(rho=0.0, n_iterations=1))
    with pytest.raises(ValidationError, match="rho"):
        run_mmas(distance, 0, MMASConfig(rho=1.0, n_iterations=1))
    with pytest.raises(ValidationError, match="rho"):
        run_mmas(distance, 0, MMASConfig(rho=-0.1, n_iterations=1))
    with pytest.raises(ValidationError, match="alpha"):
        run_mmas(distance, 0, MMASConfig(alpha=-1.0, n_iterations=1))
    with pytest.raises(ValidationError, match="beta"):
        run_mmas(distance, 0, MMASConfig(beta=-0.5, n_iterations=1))
    with pytest.raises(ValidationError, match="n_iterations"):
        run_mmas(distance, 0, MMASConfig(n_iterations=0))
    with pytest.raises(ValidationError, match="n_ants"):
        run_mmas(distance, 0, MMASConfig(n_ants=0, n_iterations=1))
    with pytest.raises(ValidationError, match="optimiser_seed"):
        run_mmas(distance, -1, MMASConfig(n_iterations=1))
    with pytest.raises(ValidationError, match="optimiser_seed"):
        run_mmas(distance, True, MMASConfig(n_iterations=1))
    with pytest.raises(NotImplementedError):
        run_mmas(distance, 0, MMASConfig(n_iterations=1, deposit_policy="all_ants"))
    with pytest.raises(NotImplementedError):
        run_mmas(distance, 0, MMASConfig(n_iterations=1, deposit_policy="global_best"))
    with pytest.raises(NotImplementedError):
        run_mmas(distance, 0, MMASConfig(n_iterations=1, acs_local_update=True))
    with pytest.raises(NotImplementedError):
        run_mmas(distance, 0, MMASConfig(n_iterations=1, local_search="2-opt"))
    with pytest.raises(NotImplementedError):
        run_mmas(distance, 0, MMASConfig(n_iterations=1, stagnation_restart=True))
    with pytest.raises(NotImplementedError):
        run_mmas(distance, 0, MMASConfig(n_iterations=1, seed_incumbent_with_nn=False))
    with pytest.raises(ValidationError, match="unknown deposit_policy"):
        run_mmas(distance, 0, MMASConfig(n_iterations=1, deposit_policy="none"))
