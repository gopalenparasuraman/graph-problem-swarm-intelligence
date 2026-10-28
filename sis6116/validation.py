"""Input checks and invariant checks. Invalid states raise ValidationError."""

from __future__ import annotations

import math

import numpy as np

MIN_CITIES = 3


class ValidationError(ValueError):
    """Raised when an input, probability, or invariant is invalid."""


def require_int(name: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name} must be an integer")
    number = int(value)
    if minimum is not None and number < minimum:
        raise ValidationError(f"{name} must be >= {minimum}")
    return number


def require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{name} must be a bool")
    return value


def require_finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or isinstance(value, (int, np.integer)):
        number = float(value)
    elif isinstance(value, (float, np.floating)):
        number = float(value)
    else:
        raise ValidationError(f"{name} must be a real number")
    if not math.isfinite(number):
        raise ValidationError(f"{name} must be finite")
    return number


def validate_distance_matrix(distance: object) -> np.ndarray:
    """Return an owned int64 copy or raise ValidationError."""
    raw = np.asarray(distance)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValidationError("distance matrix must be square")
    n = int(raw.shape[0])
    if n < MIN_CITIES:
        raise ValidationError(f"distance matrix must have at least {MIN_CITIES} cities")
    if raw.dtype == np.bool_ or not np.issubdtype(raw.dtype, np.number):
        raise ValidationError("distance matrix must be numeric")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise ValidationError("distance matrix must be real")
    if not np.isfinite(raw).all():
        raise ValidationError("distance matrix must be finite")
    if np.any(np.abs(raw) > np.iinfo(np.int64).max):
        raise ValidationError("distance matrix values do not fit in int64")
    if np.issubdtype(raw.dtype, np.floating) and not np.all(raw == np.floor(raw)):
        raise ValidationError("distances must be integers")
    array = np.array(raw, dtype=np.int64, copy=True)
    if not np.array_equal(array, array.T):
        raise ValidationError("distance matrix must be symmetric")
    if np.any(np.diag(array) != 0):
        raise ValidationError("distance matrix diagonal must be 0")
    off = array[~np.eye(n, dtype=bool)]
    if np.any(off <= 0):
        raise ValidationError("off-diagonal distances must be positive")
    return array


def validate_tour(tour: object, n: int) -> np.ndarray:
    """Return the tour as int64 or raise if it is not a permutation of 0..n-1."""
    n = require_int("n", n, minimum=MIN_CITIES)
    raw = np.asarray(tour)
    if raw.ndim != 1 or int(raw.shape[0]) != n:
        raise ValidationError(f"tour must be a permutation of length {n}")
    if raw.dtype == np.bool_ or not np.issubdtype(raw.dtype, np.number):
        raise ValidationError("tour cities must be integers")
    if np.issubdtype(raw.dtype, np.floating):
        if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
            raise ValidationError("tour cities must be integers")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise ValidationError("tour cities must be integers")
    cities = np.array(raw, dtype=np.int64, copy=True)
    if np.any(cities < 0) or np.any(cities >= n):
        raise ValidationError("tour city is outside 0..N-1")
    if int(np.unique(cities).size) != n:
        raise ValidationError("tour must visit each city exactly once")
    return cities


def last_strict_improvement(history: list[int]) -> int:
    """Return the index of the last strict decrease. Index 0 is the initial state.

    Raises if the series increases.
    """
    if not history:
        raise ValidationError("convergence history is empty")
    last = 0
    for index in range(1, len(history)):
        previous = history[index - 1]
        current = history[index]
        if current > previous:
            raise ValidationError("best-so-far cost increased")
        if current < previous:
            last = index
    return last
