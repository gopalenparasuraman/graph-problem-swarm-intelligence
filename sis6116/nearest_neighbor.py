"""Deterministic nearest-neighbour tours.

The next city is the unvisited neighbour of minimum distance.
Ties break to the lowest city index. Best-of-all-starts keeps the
lowest start index when tour costs tie.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from sis6116.objective import tour_cost
from sis6116.validation import ValidationError, require_int, validate_distance_matrix


@dataclass
class NearestNeighborResult:
    tour: np.ndarray
    cost: int
    start_city: int
    runtime_seconds: float


def _build_tour(distance: np.ndarray, start_city: int) -> np.ndarray:
    n = int(distance.shape[0])
    visited = np.zeros(n, dtype=bool)
    tour = np.empty(n, dtype=np.int64)
    tour[0] = start_city
    visited[start_city] = True
    for step in range(1, n):
        current = int(tour[step - 1])
        row = distance[current].astype(np.float64, copy=True)
        row[visited] = np.inf
        nxt = int(np.argmin(row))
        if visited[nxt] or not np.isfinite(row[nxt]):
            raise ValidationError(
                "nearest neighbour could not move to an unvisited city"
            )
        tour[step] = nxt
        visited[nxt] = True
    return tour


def nearest_neighbor_tour(distance: object, start_city: int) -> NearestNeighborResult:
    """Nearest-neighbour tour from one explicit start city."""
    started = time.perf_counter()
    matrix = validate_distance_matrix(distance)
    n = int(matrix.shape[0])
    start = require_int("start_city", start_city, minimum=0)
    if start >= n:
        raise ValidationError("start_city is outside the instance")
    tour = _build_tour(matrix, start)
    cost = tour_cost(matrix, tour)
    return NearestNeighborResult(
        tour=tour,
        cost=cost,
        start_city=start,
        runtime_seconds=time.perf_counter() - started,
    )


def best_of_all_starts(distance: object) -> NearestNeighborResult:
    """Shortest nearest-neighbour tour over start cities 0..N-1.

    Equal costs keep the lowest start index.
    """
    started = time.perf_counter()
    matrix = validate_distance_matrix(distance)
    n = int(matrix.shape[0])
    best_tour: np.ndarray | None = None
    best_cost: int | None = None
    best_start: int | None = None
    for start in range(n):
        tour = _build_tour(matrix, start)
        cost = tour_cost(matrix, tour)
        if best_cost is None or cost < best_cost:
            best_tour = tour
            best_cost = cost
            best_start = start
    if best_tour is None or best_cost is None or best_start is None:
        raise ValidationError("nearest neighbour produced no tour")
    return NearestNeighborResult(
        tour=best_tour,
        cost=best_cost,
        start_city=best_start,
        runtime_seconds=time.perf_counter() - started,
    )
