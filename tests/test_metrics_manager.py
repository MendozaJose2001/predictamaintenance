import numpy as np
import pytest
from unittest.mock import MagicMock
from src.metrics_manager import Metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_estimator(threshold: int, raw_predictions: np.ndarray) -> MagicMock:
    """Builds a mock sklearn Pipeline with a model step.

    Args:
        threshold: Clipping threshold to attach to the mock model step.
        raw_predictions: Raw values returned by estimator.predict().

    Returns:
        MagicMock that mimics a fitted sklearn Pipeline.
    """
    mock_model = MagicMock()
    mock_model.clipping_threshold = threshold

    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {'model': mock_model}
    mock_pipeline.predict.return_value = raw_predictions

    return mock_pipeline


_X_DUMMY: np.ndarray = np.array([])


# ---------------------------------------------------------------------------
# _comun_values
# ---------------------------------------------------------------------------

class TestComunValues:
    """Tests for the internal _comun_values helper."""

    def test_clips_y_pred_above_threshold(self):
        """Predictions above threshold must be clipped to threshold."""
        estimator = _make_estimator(
            threshold=110,
            raw_predictions=np.array([115.0, 70.0])
        )
        y_true = np.array([120.0, 80.0])

        _, _, y_pred = Metrics._comun_values(estimator, _X_DUMMY, y_true)

        np.testing.assert_array_equal(y_pred, np.array([110.0, 70.0]))

    def test_clips_y_true_above_threshold(self):
        """Ground truth above threshold must be clipped to threshold."""
        estimator = _make_estimator(
            threshold=110,
            raw_predictions=np.array([100.0, 80.0])
        )
        y_true = np.array([120.0, 80.0])

        _, y_true_piecewise, _ = Metrics._comun_values(estimator, _X_DUMMY, y_true)

        np.testing.assert_array_equal(y_true_piecewise, np.array([110.0, 80.0]))

    def test_diff_is_pred_minus_true(self):
        """diff must equal clipped y_pred minus clipped y_true.

        Setup: threshold=110, y_pred_raw=[115, 70], y_true=[120, 80]
            After clip: y_pred=[110, 70], y_true_pw=[110, 80]
            diff = [0, -10]
        """
        estimator = _make_estimator(
            threshold=110,
            raw_predictions=np.array([115.0, 70.0])
        )
        y_true = np.array([120.0, 80.0])

        diff, _, _ = Metrics._comun_values(estimator, _X_DUMMY, y_true)

        np.testing.assert_array_almost_equal(diff, np.array([0.0, -10.0]))

    def test_no_clip_needed(self):
        """Values below threshold must pass through unchanged."""
        estimator = _make_estimator(
            threshold=110,
            raw_predictions=np.array([50.0, 30.0])
        )
        y_true = np.array([60.0, 40.0])

        diff, y_true_piecewise, y_pred = Metrics._comun_values(estimator, _X_DUMMY, y_true)

        np.testing.assert_array_equal(y_pred, np.array([50.0, 30.0]))
        np.testing.assert_array_equal(y_true_piecewise, np.array([60.0, 40.0]))
        np.testing.assert_array_almost_equal(diff, np.array([-10.0, -10.0]))


# ---------------------------------------------------------------------------
# s_score_metric
# ---------------------------------------------------------------------------

class TestSScoreMetric:
    """Tests for the NASA S-score metric."""

    def test_perfect_prediction_returns_zero(self):
        """When predictions exactly match ground truth, S-score must be 0."""
        estimator = _make_estimator(
            threshold=110,
            raw_predictions=np.array([80.0, 50.0])
        )
        y_true = np.array([80.0, 50.0])

        score = Metrics.s_score_metric(estimator, _X_DUMMY, y_true)

        assert score == pytest.approx(0.0, abs=1e-6)

    def test_late_prediction_penalized_more_than_early(self):
        """A late prediction of +d must score higher than an early prediction of -d."""
        threshold = 110

        estimator_late = _make_estimator(
            threshold=threshold,
            raw_predictions=np.array([90.0])
        )
        estimator_early = _make_estimator(
            threshold=threshold,
            raw_predictions=np.array([70.0])
        )
        y_true = np.array([80.0])

        score_late = Metrics.s_score_metric(estimator_late, _X_DUMMY, y_true)
        score_early = Metrics.s_score_metric(estimator_early, _X_DUMMY, y_true)

        assert score_late > score_early

    def test_known_values(self):
        """S-score must match manually computed values for known inputs.

        Setup: threshold=110, y_pred_raw=[115, 70], y_true=[120, 80]
            After clip: y_pred=[110, 70], y_true_pw=[110, 80]
            diff = [0, -10]
            s[0] = exp(0/10) - 1 = 0.0
            s[1] = exp(10/13) - 1 ≈ 1.1572
            mean  ≈ 0.5786
        """
        estimator = _make_estimator(
            threshold=110,
            raw_predictions=np.array([115.0, 70.0])
        )
        y_true = np.array([120.0, 80.0])

        expected = (0.0 + (np.exp(10.0 / 13.0) - 1)) / 2

        score = Metrics.s_score_metric(estimator, _X_DUMMY, y_true)

        assert score == pytest.approx(expected, rel=1e-5)

    def test_returns_positive_float(self):
        """S-score must always be a non-negative float."""
        estimator = _make_estimator(
            threshold=110,
            raw_predictions=np.array([80.0, 60.0, 100.0])
        )
        y_true = np.array([75.0, 65.0, 95.0])

        score = Metrics.s_score_metric(estimator, _X_DUMMY, y_true)

        assert isinstance(score, float)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# c_index_metric
# ---------------------------------------------------------------------------

class TestCIndexMetric:
    """Tests for the Concordance Index metric."""

    def test_perfect_ranking_returns_one(self):
        """Perfect ranking must return C-index of 1.0."""
        estimator = _make_estimator(
            threshold=200,
            raw_predictions=np.array([100.0, 80.0, 60.0, 40.0])
        )
        y_true = np.array([100.0, 80.0, 60.0, 40.0])

        c_index = Metrics.c_index_metric(estimator, _X_DUMMY, y_true)

        assert c_index == pytest.approx(1.0, abs=1e-6)

    def test_reversed_ranking_returns_zero(self):
        """Completely reversed ranking must return C-index of 0.0."""
        estimator = _make_estimator(
            threshold=200,
            raw_predictions=np.array([40.0, 60.0, 80.0, 100.0])
        )
        y_true = np.array([100.0, 80.0, 60.0, 40.0])

        c_index = Metrics.c_index_metric(estimator, _X_DUMMY, y_true)

        assert c_index == pytest.approx(0.0, abs=1e-6)

    def test_returns_float_in_valid_range(self):
        """C-index must be a float in [0.0, 1.0]."""
        estimator = _make_estimator(
            threshold=110,
            raw_predictions=np.array([80.0, 50.0, 30.0])
        )
        y_true = np.array([90.0, 40.0, 35.0])

        c_index = Metrics.c_index_metric(estimator, _X_DUMMY, y_true)

        assert isinstance(c_index, float)
        assert 0.0 <= c_index <= 1.0


# ---------------------------------------------------------------------------
# mae_metric
# ---------------------------------------------------------------------------

class TestMaeMetric:
    """Tests for the Mean Absolute Error metric."""

    def test_perfect_prediction_returns_zero(self):
        """Perfect predictions must yield MAE of 0."""
        estimator = _make_estimator(
            threshold=110,
            raw_predictions=np.array([80.0, 50.0])
        )
        y_true = np.array([80.0, 50.0])

        mae = Metrics.mae_metric(estimator, _X_DUMMY, y_true)

        assert mae == pytest.approx(0.0, abs=1e-6)

    def test_known_values(self):
        """MAE must match manually computed value for known inputs.

        Setup: threshold=110, y_pred_raw=[115, 70], y_true=[120, 80]
            After clip: y_pred=[110, 70], y_true_pw=[110, 80]
            diff = [0, -10] → abs = [0, 10] → mean = 5.0
        """
        estimator = _make_estimator(
            threshold=110,
            raw_predictions=np.array([115.0, 70.0])
        )
        y_true = np.array([120.0, 80.0])

        mae = Metrics.mae_metric(estimator, _X_DUMMY, y_true)

        assert mae == pytest.approx(5.0, abs=1e-6)

    def test_returns_non_negative_float(self):
        """MAE must always be a non-negative float."""
        estimator = _make_estimator(
            threshold=110,
            raw_predictions=np.array([80.0, 60.0])
        )
        y_true = np.array([75.0, 65.0])

        mae = Metrics.mae_metric(estimator, _X_DUMMY, y_true)

        assert isinstance(mae, float)
        assert mae >= 0.0


# ---------------------------------------------------------------------------
# rmse_metric
# ---------------------------------------------------------------------------

class TestRmseMetric:
    """Tests for the Root Mean Squared Error metric."""

    def test_perfect_prediction_returns_zero(self):
        """Perfect predictions must yield RMSE of 0."""
        estimator = _make_estimator(
            threshold=110,
            raw_predictions=np.array([80.0, 50.0])
        )
        y_true = np.array([80.0, 50.0])

        rmse = Metrics.rmse_metric(estimator, _X_DUMMY, y_true)

        assert rmse == pytest.approx(0.0, abs=1e-6)

    def test_known_values(self):
        """RMSE must match manually computed value for known inputs.

        Setup: threshold=110, y_pred_raw=[115, 70], y_true=[120, 80]
            After clip: y_pred=[110, 70], y_true_pw=[110, 80]
            diff = [0, -10] → sq = [0, 100] → mean = 50 → sqrt ≈ 7.071
        """
        estimator = _make_estimator(
            threshold=110,
            raw_predictions=np.array([115.0, 70.0])
        )
        y_true = np.array([120.0, 80.0])

        rmse = Metrics.rmse_metric(estimator, _X_DUMMY, y_true)

        assert rmse == pytest.approx(np.sqrt(50.0), rel=1e-5)

    def test_rmse_greater_or_equal_mae(self):
        """RMSE must always be >= MAE for the same inputs."""
        estimator_rmse = _make_estimator(
            threshold=110,
            raw_predictions=np.array([115.0, 70.0, 90.0])
        )
        estimator_mae = _make_estimator(
            threshold=110,
            raw_predictions=np.array([115.0, 70.0, 90.0])
        )
        y_true = np.array([100.0, 80.0, 85.0])

        rmse = Metrics.rmse_metric(estimator_rmse, _X_DUMMY, y_true)
        mae = Metrics.mae_metric(estimator_mae, _X_DUMMY, y_true)

        assert rmse >= mae

    def test_returns_non_negative_float(self):
        """RMSE must always be a non-negative float."""
        estimator = _make_estimator(
            threshold=110,
            raw_predictions=np.array([80.0, 60.0])
        )
        y_true = np.array([75.0, 65.0])

        rmse = Metrics.rmse_metric(estimator, _X_DUMMY, y_true)

        assert isinstance(rmse, float)
        assert rmse >= 0.0