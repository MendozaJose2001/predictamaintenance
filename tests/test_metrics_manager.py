"""Tests for Metrics (new array-based interface).

All metrics now receive (y_pred, y_true, clipping_threshold) directly
as numpy arrays — no sklearn Pipeline dependency.
"""

import numpy as np
import pytest

from src.metrics_manager import Metrics


# ---------------------------------------------------------------------------
# TestClipBoth
# ---------------------------------------------------------------------------

class TestClipBoth:
    """Tests for the internal _clip_both helper."""

    def test_clips_y_pred_above_threshold(self):
        """Predictions above threshold must be clipped."""
        y_pred = np.array([115.0, 70.0])
        y_true = np.array([120.0, 80.0])
        _, _, y_pred_c = Metrics._clip_both(y_pred, y_true, 110)
        np.testing.assert_array_equal(y_pred_c, np.array([110.0, 70.0]))

    def test_clips_y_true_above_threshold(self):
        """Ground truth above threshold must be clipped."""
        y_pred = np.array([100.0, 80.0])
        y_true = np.array([120.0, 80.0])
        _, y_true_c, _ = Metrics._clip_both(y_pred, y_true, 110)
        np.testing.assert_array_equal(y_true_c, np.array([110.0, 80.0]))

    def test_diff_is_pred_minus_true(self):
        """diff must equal clipped y_pred minus clipped y_true."""
        y_pred = np.array([115.0, 70.0])
        y_true = np.array([120.0, 80.0])
        diff, _, _ = Metrics._clip_both(y_pred, y_true, 110)
        np.testing.assert_array_almost_equal(diff, np.array([0.0, -10.0]))

    def test_no_clip_needed(self):
        """Values below threshold must pass through unchanged."""
        y_pred = np.array([50.0, 30.0])
        y_true = np.array([60.0, 40.0])
        diff, y_true_c, y_pred_c = Metrics._clip_both(y_pred, y_true, 110)
        np.testing.assert_array_equal(y_pred_c, y_pred)
        np.testing.assert_array_equal(y_true_c, y_true)
        np.testing.assert_array_almost_equal(diff, np.array([-10.0, -10.0]))


# ---------------------------------------------------------------------------
# TestSScore
# ---------------------------------------------------------------------------

class TestSScore:
    """Tests for the NASA S-score metric."""

    def test_perfect_prediction_returns_zero(self):
        """When predictions exactly match ground truth, S-score must be 0."""
        y_pred = np.array([80.0, 50.0])
        y_true = np.array([80.0, 50.0])
        score = Metrics.s_score(y_pred, y_true, clipping_threshold=110)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_late_prediction_penalized_more_than_early(self):
        """A late prediction of +d must score higher than early of -d."""
        y_true = np.array([80.0])
        score_late  = Metrics.s_score(np.array([90.0]), y_true, 110)
        score_early = Metrics.s_score(np.array([70.0]), y_true, 110)
        assert score_late > score_early

    def test_known_values(self):
        """S-score must match manually computed values.

        y_pred=[115,70], y_true=[120,80], threshold=110
        After clip: y_pred=[110,70], y_true=[110,80]
        diff=[0,-10]
        s[0]=exp(0/10)-1=0, s[1]=exp(10/13)-1
        mean=(0 + exp(10/13)-1)/2
        """
        y_pred = np.array([115.0, 70.0])
        y_true = np.array([120.0, 80.0])
        expected = (0.0 + (np.exp(10.0 / 13.0) - 1)) / 2
        score = Metrics.s_score(y_pred, y_true, clipping_threshold=110)
        assert score == pytest.approx(expected, rel=1e-5)

    def test_returns_non_negative_float(self):
        """S-score must always be a non-negative float."""
        score = Metrics.s_score(
            np.array([80.0, 60.0, 100.0]),
            np.array([75.0, 65.0, 95.0]),
            clipping_threshold=110,
        )
        assert isinstance(score, float)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# TestCIndex
# ---------------------------------------------------------------------------

class TestCIndex:
    """Tests for the Concordance Index metric."""

    def test_perfect_ranking_returns_one(self):
        """Perfect ranking must return C-index of 1.0."""
        y_pred = np.array([100.0, 80.0, 60.0, 40.0])
        y_true = np.array([100.0, 80.0, 60.0, 40.0])
        assert Metrics.c_index(y_pred, y_true, 200) == pytest.approx(1.0, abs=1e-6)

    def test_reversed_ranking_returns_zero(self):
        """Completely reversed ranking must return C-index of 0.0."""
        y_pred = np.array([40.0, 60.0, 80.0, 100.0])
        y_true = np.array([100.0, 80.0, 60.0, 40.0])
        assert Metrics.c_index(y_pred, y_true, 200) == pytest.approx(0.0, abs=1e-6)

    def test_returns_float_in_valid_range(self):
        """C-index must be a float in [0.0, 1.0]."""
        c = Metrics.c_index(
            np.array([80.0, 50.0, 30.0]),
            np.array([90.0, 40.0, 35.0]),
            clipping_threshold=110,
        )
        assert isinstance(c, float)
        assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# TestMAE
# ---------------------------------------------------------------------------

class TestMAE:
    """Tests for the Mean Absolute Error metric."""

    def test_perfect_prediction_returns_zero(self):
        """Perfect predictions must yield MAE of 0."""
        y = np.array([80.0, 50.0])
        assert Metrics.mae(y, y, clipping_threshold=110) == pytest.approx(0.0, abs=1e-6)

    def test_known_values(self):
        """MAE must match manually computed value.

        y_pred=[115,70], y_true=[120,80], threshold=110
        After clip: diff=[0,-10] → abs=[0,10] → mean=5.0
        """
        assert Metrics.mae(
            np.array([115.0, 70.0]),
            np.array([120.0, 80.0]),
            clipping_threshold=110,
        ) == pytest.approx(5.0, abs=1e-6)

    def test_returns_non_negative_float(self):
        """MAE must always be a non-negative float."""
        mae = Metrics.mae(
            np.array([80.0, 60.0]),
            np.array([75.0, 65.0]),
            clipping_threshold=110,
        )
        assert isinstance(mae, float)
        assert mae >= 0.0


# ---------------------------------------------------------------------------
# TestRMSE
# ---------------------------------------------------------------------------

class TestRMSE:
    """Tests for the Root Mean Squared Error metric."""

    def test_perfect_prediction_returns_zero(self):
        """Perfect predictions must yield RMSE of 0."""
        y = np.array([80.0, 50.0])
        assert Metrics.rmse(y, y, clipping_threshold=110) == pytest.approx(0.0, abs=1e-6)

    def test_known_values(self):
        """RMSE must match manually computed value.

        y_pred=[115,70], y_true=[120,80], threshold=110
        After clip: diff=[0,-10] → sq=[0,100] → mean=50 → sqrt≈7.071
        """
        assert Metrics.rmse(
            np.array([115.0, 70.0]),
            np.array([120.0, 80.0]),
            clipping_threshold=110,
        ) == pytest.approx(np.sqrt(50.0), rel=1e-5)

    def test_rmse_greater_or_equal_mae(self):
        """RMSE must always be >= MAE for the same inputs."""
        y_pred = np.array([115.0, 70.0, 90.0])
        y_true = np.array([100.0, 80.0, 85.0])
        rmse = Metrics.rmse(y_pred, y_true, 110)
        mae  = Metrics.mae(y_pred,  y_true, 110)
        assert rmse >= mae

    def test_returns_non_negative_float(self):
        """RMSE must always be a non-negative float."""
        rmse = Metrics.rmse(
            np.array([80.0, 60.0]),
            np.array([75.0, 65.0]),
            clipping_threshold=110,
        )
        assert isinstance(rmse, float)
        assert rmse >= 0.0


# ---------------------------------------------------------------------------
# TestGetMetrics
# ---------------------------------------------------------------------------

class TestGetMetrics:
    """Tests for the get_metrics() factory method."""

    def test_returns_four_metrics(self):
        """get_metrics must return exactly four entries."""
        assert len(Metrics.get_metrics()) == 4

    def test_metric_keys(self):
        """get_metrics must contain S_score, C_index, MAE, RMSE."""
        keys = set(Metrics.get_metrics().keys())
        assert keys == {'S_score', 'C_index', 'MAE', 'RMSE'}

    def test_all_callable(self):
        """All metric values must be callable."""
        for name, func in Metrics.get_metrics().items():
            assert callable(func), f"{name} is not callable"