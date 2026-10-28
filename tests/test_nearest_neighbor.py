"""Nearest-neighbour start policy and tie breaks."""

import numpy as np
import pytest

from sis6116.nearest_neighbor import best_of_all_starts, nearest_neighbor_tour
from sis6116.objective import tour_cost
from sis6116.validation import ValidationError

TIE = np.array(
    [
        [0, 5, 5],
        [5, 0, 1],
        [5, 1, 0],
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


def test_tie_breaks_to_the_lowest_city_index():
    result = nearest_neighbor_tour(TIE, 0)
    assert result.tour.tolist() == [0, 1, 2]
    assert result.cost == 11
    assert result.cost == tour_cost(TIE, result.tour)


def test_city0_and_best_of_all_starts_are_reported_separately():
    city0 = nearest_neighbor_tour(SIX, 0)
    best = best_of_all_starts(SIX)
    assert city0.tour.tolist() == [0, 1, 2, 3, 5, 4]
    assert city0.cost == 68
    assert best.start_city == 2
    assert best.tour.tolist() == [2, 1, 0, 5, 4, 3]
    assert best.cost == 62
    assert best.cost < city0.cost


def test_best_of_all_starts_keeps_the_lowest_start_on_ties(symmetric_distance):
    distance = symmetric_distance(n=7, seed=3)
    by_start = [nearest_neighbor_tour(distance, start) for start in range(7)]
    best = best_of_all_starts(distance)
    assert best.cost == min(result.cost for result in by_start)
    expected_start = min(
        start for start, result in enumerate(by_start) if result.cost == best.cost
    )
    assert best.start_city == expected_start
    assert np.array_equal(best.tour, by_start[expected_start].tour)


def test_nearest_neighbor_does_not_mutate_the_matrix(symmetric_distance):
    distance = symmetric_distance()
    original = distance.copy()
    nearest_neighbor_tour(distance, 0)
    best_of_all_starts(distance)
    np.testing.assert_array_equal(distance, original)


def test_start_city_must_be_inside_the_instance(symmetric_distance):
    distance = symmetric_distance(n=4)
    with pytest.raises(ValidationError):
        nearest_neighbor_tour(distance, -1)
    with pytest.raises(ValidationError):
        nearest_neighbor_tour(distance, 4)
