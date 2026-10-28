"""Personalised symmetric TSP instance.

Generation, for the fixed seed 4616, is::

    N = 40 + (seed % 11)
    instance_rng = np.random.default_rng(seed)
    matrix = instance_rng.integers(1, 100, size=(N, N))
    distance_matrix = (matrix + matrix.T) // 2
    np.fill_diagonal(distance_matrix, 0)
"""

from __future__ import annotations

import hashlib

import numpy as np

from sis6116.validation import ValidationError, require_int, validate_distance_matrix

PERSONAL_SEED = 4616
EXPECTED_N = 47
EXPECTED_UPPER_TRIANGLE_SUM = 53104
EXPECTED_UPPER_LEFT_5X5 = np.array(
    [
        [0, 56, 85, 16, 40],
        [56, 0, 27, 65, 26],
        [85, 27, 0, 41, 15],
        [16, 65, 41, 0, 66],
        [40, 26, 15, 66, 0],
    ],
    dtype=np.int64,
)

GENERATION_RULES = (
    "N = 40 + (seed % 11)",
    "instance_rng = np.random.default_rng(seed)",
    "matrix = instance_rng.integers(1, 100, size=(N, N))",
    "distance_matrix = (matrix + matrix.T) // 2",
    "np.fill_diagonal(distance_matrix, 0)",
)


def city_count(seed: int = PERSONAL_SEED) -> int:
    seed = require_int("seed", seed, minimum=0)
    return 40 + (seed % 11)


def generate_distance_matrix(seed: int = PERSONAL_SEED) -> np.ndarray:
    """Draw one symmetric integer distance matrix from its own random stream."""
    seed = require_int("seed", seed, minimum=0)
    n = city_count(seed)
    instance_rng = np.random.default_rng(seed)
    matrix = instance_rng.integers(1, 100, size=(n, n))
    distance_matrix = (matrix + matrix.T) // 2
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix


def strict_upper_triangle_sum(distance: np.ndarray) -> int:
    array = np.asarray(distance)
    return int(np.triu(array, k=1).sum())


def matrix_fingerprint(distance: np.ndarray) -> str:
    """Hash a canonical int64 copy of the matrix."""
    array = np.ascontiguousarray(np.asarray(distance), dtype=np.int64)
    hasher = hashlib.sha256()
    hasher.update(np.int64(array.shape[0]).tobytes())
    hasher.update(np.int64(array.shape[1]).tobytes())
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def verify_personal_instance(
    distance: np.ndarray, seed: int = PERSONAL_SEED
) -> None:
    """Fail if `distance` is not the seed-4616 instance described above."""
    seed = require_int("seed", seed, minimum=0)
    if seed != PERSONAL_SEED:
        raise ValidationError(
            "personal instance verification is defined for seed 4616 only"
        )
    array = validate_distance_matrix(distance)
    n = int(array.shape[0])
    if n != EXPECTED_N or city_count(seed) != EXPECTED_N:
        raise ValidationError(f"personal instance N is {n}, expected {EXPECTED_N}")
    upper = strict_upper_triangle_sum(array)
    if upper != EXPECTED_UPPER_TRIANGLE_SUM:
        raise ValidationError(
            "strict upper triangle sum is "
            f"{upper}, expected {EXPECTED_UPPER_TRIANGLE_SUM}"
        )
    observed = array[:5, :5]
    if not np.array_equal(observed, EXPECTED_UPPER_LEFT_5X5):
        raise ValidationError(
            "upper-left 5x5 does not match the personalised instance check:\n"
            f"observed:\n{observed}\nexpected:\n{EXPECTED_UPPER_LEFT_5X5}"
        )
    off = array[~np.eye(n, dtype=bool)]
    if int(off.min()) < 1 or int(off.max()) > 99:
        raise ValidationError("off-diagonal distances must lie in 1..99")
