"""Personalised instance checks. Expected values are the assignment spec."""

import numpy as np

from sis6116.instance import (
    PERSONAL_SEED,
    city_count,
    generate_distance_matrix,
    matrix_fingerprint,
    strict_upper_triangle_sum,
    verify_personal_instance,
)
from sis6116.validation import ValidationError

EXPECTED_UPPER_LEFT = np.array(
    [
        [0, 56, 85, 16, 40],
        [56, 0, 27, 65, 26],
        [85, 27, 0, 41, 15],
        [16, 65, 41, 0, 66],
        [40, 26, 15, 66, 0],
    ],
    dtype=np.int64,
)


def test_city_count_formula():
    assert 40 + (4616 % 11) == 47
    assert city_count(4616) == 47


def test_matrix_generation_matches_spec():
    distance = generate_distance_matrix(PERSONAL_SEED)
    assert distance.shape == (47, 47)
    np.testing.assert_array_equal(distance[:5, :5], EXPECTED_UPPER_LEFT)
    assert strict_upper_triangle_sum(distance) == 53104


def test_generation_is_deterministic():
    first = generate_distance_matrix(4616)
    second = generate_distance_matrix(4616)
    np.testing.assert_array_equal(first, second)


def test_seed_changes_the_matrix():
    personal = generate_distance_matrix(4616)
    other = generate_distance_matrix(4617)
    assert other.shape != personal.shape


def test_symmetry():
    distance = generate_distance_matrix(4616)
    np.testing.assert_array_equal(distance, distance.T)


def test_diagonal_is_zero():
    distance = generate_distance_matrix(4616)
    assert np.all(np.diag(distance) == 0)


def test_off_diagonal_positive_and_in_draw_range():
    distance = generate_distance_matrix(4616)
    off = distance[~np.eye(distance.shape[0], dtype=bool)]
    assert np.all(off > 0)
    assert int(off.min()) >= 1
    assert int(off.max()) <= 99


def test_verify_accepts_the_generated_instance():
    distance = generate_distance_matrix(4616)
    verify_personal_instance(distance, 4616)


def test_verify_rejects_a_tampered_matrix():
    distance = generate_distance_matrix(4616)
    distance = distance.copy()
    distance[0, 1] = 1
    distance[1, 0] = 1
    try:
        verify_personal_instance(distance, 4616)
    except ValidationError:
        return
    raise AssertionError("tampered instance was accepted")


def test_verify_rejects_a_different_seed():
    distance = generate_distance_matrix(4616)
    try:
        verify_personal_instance(distance, 1)
    except ValidationError:
        return
    raise AssertionError("seed 1 was accepted")


def test_fingerprint_tracks_matrix_contents():
    distance = generate_distance_matrix(4616)
    assert matrix_fingerprint(distance) == matrix_fingerprint(distance.copy())
    tampered = distance.copy()
    tampered[0, 1] = 1
    tampered[1, 0] = 1
    assert matrix_fingerprint(tampered) != matrix_fingerprint(distance)
