"""Tour cost, closure, and a 4-city exhaustive check."""

import itertools

import numpy as np
import pytest

from sis6116.objective import tour_cost, tour_edges
from sis6116.validation import ValidationError

FOUR = np.array(
    [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ],
    dtype=np.int64,
)


def test_tour_cost_includes_return_edge():
    tour = [0, 1, 3, 2]
    path = int(FOUR[0, 1] + FOUR[1, 3] + FOUR[3, 2])
    return_edge = int(FOUR[2, 0])
    assert path == 65
    assert return_edge == 15
    assert tour_cost(FOUR, tour) == path + return_edge
    assert (2, 0) in tour_edges(np.asarray(tour))


def test_invalid_tours_raise():
    with pytest.raises(ValidationError):
        tour_cost(FOUR, [0, 1, 1, 2])
    with pytest.raises(ValidationError):
        tour_cost(FOUR, [0, 1, 2])
    with pytest.raises(ValidationError):
        tour_cost(FOUR, [0, 1, 2, 9])
    with pytest.raises(ValidationError):
        tour_cost(FOUR, [0, 1, 2, -1])


def test_four_city_exhaustive_optimum():
    costs = []
    for perm in itertools.permutations(range(4)):
        manual = sum(int(FOUR[perm[i], perm[(i + 1) % 4]]) for i in range(4))
        assert tour_cost(FOUR, perm) == manual
        costs.append(manual)
    assert len(costs) == 24
    assert min(costs) == 80
    assert tour_cost(FOUR, [0, 1, 3, 2]) == 80


def test_reversed_tour_has_the_same_cost_on_a_symmetric_matrix():
    forward = [0, 1, 3, 2]
    backward = [0, 2, 3, 1]
    assert tour_cost(FOUR, forward) == tour_cost(FOUR, backward) == 80


def test_distance_matrix_must_be_symmetric_zero_diagonal_and_positive():
    asymmetric = FOUR.copy()
    asymmetric[0, 1] = 3
    with pytest.raises(ValidationError, match="symmetric"):
        tour_cost(asymmetric, [0, 1, 2, 3])

    diagonal = FOUR.copy()
    diagonal[1, 1] = 4
    with pytest.raises(ValidationError, match="diagonal"):
        tour_cost(diagonal, [0, 1, 2, 3])

    zero_edge = FOUR.copy()
    zero_edge[0, 1] = 0
    zero_edge[1, 0] = 0
    with pytest.raises(ValidationError, match="positive"):
        tour_cost(zero_edge, [0, 1, 2, 3])

    with pytest.raises(ValidationError, match="square"):
        tour_cost(np.array([[0, 1, 2], [1, 0, 2]], dtype=np.int64), [0, 1])

    with pytest.raises(ValidationError, match="at least 3"):
        tour_cost(np.array([[0, 1], [1, 0]], dtype=np.int64), [0, 1])

    broken = FOUR.astype(np.float64)
    broken[0, 2] = np.nan
    broken[2, 0] = np.nan
    with pytest.raises(ValidationError, match="finite"):
        tour_cost(broken, [0, 1, 2, 3])
