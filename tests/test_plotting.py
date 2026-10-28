"""Mean curve and sample-std band are computed from the run histories."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pytest

from sis6116.plotting import mean_and_sample_std, plot_mean_std, plot_overlay
from sis6116.validation import ValidationError


def test_mean_and_sample_std_uses_ddof_one():
    histories = [[10, 8, 8], [12, 12, 9]]
    mean, std = mean_and_sample_std(histories)
    np.testing.assert_allclose(mean, [11.0, 10.0, 8.5])
    np.testing.assert_allclose(std, [math.sqrt(2.0), math.sqrt(8.0), math.sqrt(0.5)])
    with pytest.raises(ValidationError):
        mean_and_sample_std([[1, 2, 3]])


def test_mean_curve_and_overlay_follow_the_histories(tmp_path):
    histories = [[10, 8, 8], [12, 12, 9]]
    mean_path = tmp_path / "mean.png"
    overlay_path = tmp_path / "overlay.png"
    mean_fig = plot_mean_std(histories, mean_path, nn_city0=13, nn_best=11)
    overlay_fig = plot_overlay(histories, [4, 9], overlay_path)
    try:
        mean, _std = mean_and_sample_std(histories)
        plotted = mean_fig.axes[0].lines[0].get_ydata()
        np.testing.assert_allclose(plotted, mean)
        labels = [line.get_label() for line in mean_fig.axes[0].lines]
        assert "NN city 0" in labels
        assert "NN best of all starts" in labels
        assert len(mean_fig.axes[0].collections) >= 1
        assert len(overlay_fig.axes[0].lines) == 2
    finally:
        plt.close(mean_fig)
        plt.close(overlay_fig)
    assert mean_path.stat().st_size > 0
    assert overlay_path.stat().st_size > 0
