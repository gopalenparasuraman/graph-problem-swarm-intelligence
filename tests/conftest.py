import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest


@pytest.fixture
def symmetric_distance():
    def _make(n: int = 5, seed: int = 0, low: int = 1, high: int = 20) -> np.ndarray:
        rng = np.random.default_rng(seed)
        raw = rng.integers(low, high, size=(n, n))
        distance = (raw + raw.T) // 2
        np.fill_diagonal(distance, 0)
        return distance

    return _make
