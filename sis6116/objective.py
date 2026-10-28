"""Symmetric TSP tour length, including the return to the start city."""

from __future__ import annotations

import numpy as np

from sis6116.validation import validate_distance_matrix, validate_tour


def tour_cost(distance: object, tour: object) -> int:
    """Sum the N edges of a Hamiltonian cycle, including the return edge."""
    matrix = validate_distance_matrix(distance)
    n = int(matrix.shape[0])
    cities = validate_tour(tour, n)
    total = 0
    for index in range(n):
        start = int(cities[index])
        end = int(cities[(index + 1) % n])
        total += int(matrix[start, end])
    return total


def tour_edges(tour: np.ndarray) -> list[tuple[int, int]]:
    """Directed edges of the cycle, including the return edge."""
    cities = [int(city) for city in np.asarray(tour).tolist()]
    n = len(cities)
    return [(cities[index], cities[(index + 1) % n]) for index in range(n)]
