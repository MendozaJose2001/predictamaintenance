"""Tests for predictor_io.py — persistence layer for RULProductionPredictor.

Uses session-scoped fixtures from conftest.py (motor_1_pipeline_output,
motor_1_X_df) following the same scope hierarchy as the rest of the suite.

All file I/O uses pytest's tmp_path fixture — function-scoped, automatically
cleaned up after each test. No manual teardown required.

Coverage:
    TestBuildStem            — filename stem generation
    TestExtractModelParams   — model hyperparameter extraction
    TestPredictorSession     — dataclass fields and stem property
    TestSavePredictor        — serialization, JSON structure, return value
    TestLoadPredictor        — loading by path, by name, error handling
"""

import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.decision_tree import DecisionTreeModel
from src.pipeline.rul_pipeline import RULPipeline
from src.utils.production_predictor import RULProductionPredictor
from src.utils.predictor_io import (
    PredictorSession,
    _build_stem,
    _extract_model_params,
    load_predictor,
    save_predictor,
)


# ---------------------------------------------------------------------------
# Fixtures — session-scoped to match conftest.py hierarchy
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def fitted_pipeline(motor_1_pipeline_output: dict) -> RULPipeline:
    """Fitted RULPipeline from conftest session fixture."""
    return motor_1_pipeline_output['pipeline']


@pytest.fixture(scope='session')
def fitted_model(motor_1_pipeline_output: dict) -> DecisionTreeModel:
    """DecisionTreeModel fitted on motor 1 pipeline output."""
    model = DecisionTreeModel(max_depth=5, min_samples_leaf=10)
    model.fit(
        motor_1_pipeline_output['X'],
        motor_1_pipeline_output['y_rul'],
    )
    return model


@pytest.fixture(scope='session')
def predictor(
    fitted_pipeline: RULPipeline,
    fitted_model: DecisionTreeModel,
) -> RULProductionPredictor:
    """RULProductionPredictor built from fitted pipeline and model."""
    return RULProductionPredictor(
        pipeline=fitted_pipeline,
        model=fitted_model,
    )


@pytest.fixture(scope='session')
def sample_metrics() -> dict:
    """Sample cross-validation metrics for save_predictor tests."""
    return {
        'mean_S_score': 12.3,
        'mean_MAE':     8.5,
        'mean_RMSE':    11.2,
        'mean_C_index': 0.78,
    }


@pytest.fixture(scope='session')
def sample_pipeline_params() -> dict:
    return {
        'feature_set':        'B',
        'window_size':        20,
        'n_components':       5,
        'clipping_threshold': 125,
    }


# ---------------------------------------------------------------------------
# TestBuildStem
# ---------------------------------------------------------------------------

class TestBuildStem:

    def test_format_contains_model_name(self, sample_pipeline_params: dict):
        stem = _build_stem('RandomForestModel', sample_pipeline_params, '20250509_1423')
        assert 'RandomForestModel' in stem

    def test_format_contains_feature_set(self, sample_pipeline_params: dict):
        stem = _build_stem('RandomForestModel', sample_pipeline_params, '20250509_1423')
        assert '_B_' in stem

    def test_format_contains_window_size(self, sample_pipeline_params: dict):
        stem = _build_stem('RandomForestModel', sample_pipeline_params, '20250509_1423')
        assert '20w' in stem

    def test_format_contains_n_components(self, sample_pipeline_params: dict):
        stem = _build_stem('RandomForestModel', sample_pipeline_params, '20250509_1423')
        assert '5pc' in stem

    def test_format_contains_timestamp(self, sample_pipeline_params: dict):
        stem = _build_stem('RandomForestModel', sample_pipeline_params, '20250509_1423')
        assert '20250509_1423' in stem

    def test_full_format(self, sample_pipeline_params: dict):
        stem = _build_stem('DecisionTreeModel', sample_pipeline_params, '20250509_1423')
        assert stem == 'DecisionTreeModel_B_20w_5pc_20250509_1423'

    def test_deterministic(self, sample_pipeline_params: dict):
        s1 = _build_stem('RandomForestModel', sample_pipeline_params, '20250509_1423')
        s2 = _build_stem('RandomForestModel', sample_pipeline_params, '20250509_1423')
        assert s1 == s2


# ---------------------------------------------------------------------------
# TestExtractModelParams
# ---------------------------------------------------------------------------

class TestExtractModelParams:

    def test_excludes_clipping_threshold(self, predictor: RULProductionPredictor):
        params = _extract_model_params(predictor)
        assert 'clipping_threshold' not in params

    def test_contains_max_depth(self, predictor: RULProductionPredictor):
        params = _extract_model_params(predictor)
        assert 'max_depth' in params
        assert params['max_depth'] == 5

    def test_contains_min_samples_leaf(self, predictor: RULProductionPredictor):
        params = _extract_model_params(predictor)
        assert 'min_samples_leaf' in params
        assert params['min_samples_leaf'] == 10

    def test_returns_dict(self, predictor: RULProductionPredictor):
        assert isinstance(_extract_model_params(predictor), dict)

    def test_fallback_empty_dict_if_no_get_params(
        self,
        fitted_pipeline: RULPipeline,
        fitted_model: DecisionTreeModel,
    ):
        """Model without get_params() must return empty dict — not raise."""

        class _ModelWithoutGetParams:
            is_fitted_ = True

        # Bypass __init__ validation by patching is_fitted_ directly
        p = RULProductionPredictor.__new__(RULProductionPredictor)
        p.pipeline = fitted_pipeline
        p.pipeline_ = fitted_pipeline
        p.model = fitted_model
        p.model_ = _ModelWithoutGetParams()  # type: ignore
        p.feature_set = 'B'
        p.window_size = 20
        p.n_components = 5
        p.clipping_threshold = 125
        p.model_name = None
        p.model_name_ = '_ModelWithoutGetParams'

        result = _extract_model_params(p)
        assert result == {}


# ---------------------------------------------------------------------------
# TestPredictorSession
# ---------------------------------------------------------------------------

class TestPredictorSession:

    @pytest.fixture
    def session(self, tmp_path: Path, sample_pipeline_params: dict) -> PredictorSession:
        stem = 'DecisionTreeModel_B_20w_5pc_20250509_1423'
        return PredictorSession(
            model_name='DecisionTreeModel',
            timestamp='20250509_1423',
            pipeline_params=sample_pipeline_params,
            model_params={'max_depth': 5, 'min_samples_leaf': 10},
            metrics={'mean_S_score': 12.3},
            path_model=tmp_path / 'models' / f'{stem}.pkl',
            path_metadata=tmp_path / 'metadata' / f'{stem}.json',
        )

    def test_stem_matches_pkl_filename(self, session: PredictorSession):
        assert session.stem == 'DecisionTreeModel_B_20w_5pc_20250509_1423'

    def test_stem_consistent_between_paths(self, session: PredictorSession):
        assert session.path_model.stem == session.path_metadata.stem

    def test_model_name_stored(self, session: PredictorSession):
        assert session.model_name == 'DecisionTreeModel'

    def test_timestamp_stored(self, session: PredictorSession):
        assert session.timestamp == '20250509_1423'

    def test_pipeline_params_stored(
        self, session: PredictorSession, sample_pipeline_params: dict
    ):
        assert session.pipeline_params == sample_pipeline_params

    def test_model_params_stored(self, session: PredictorSession):
        assert session.model_params == {'max_depth': 5, 'min_samples_leaf': 10}

    def test_metrics_stored(self, session: PredictorSession):
        assert session.metrics == {'mean_S_score': 12.3}


# ---------------------------------------------------------------------------
# TestSavePredictor
# ---------------------------------------------------------------------------

class TestSavePredictor:

    def test_creates_pkl_file(
        self,
        predictor: RULProductionPredictor,
        sample_metrics: dict,
        tmp_path: Path,
    ):
        save_predictor(predictor, sample_metrics, base_dir=tmp_path)
        pkl_files = list((tmp_path / 'models').glob('*.pkl'))
        assert len(pkl_files) == 1

    def test_creates_json_file(
        self,
        predictor: RULProductionPredictor,
        sample_metrics: dict,
        tmp_path: Path,
    ):
        save_predictor(predictor, sample_metrics, base_dir=tmp_path)
        json_files = list((tmp_path / 'metadata').glob('*.json'))
        assert len(json_files) == 1

    def test_pkl_and_json_share_stem(
        self,
        predictor: RULProductionPredictor,
        sample_metrics: dict,
        tmp_path: Path,
    ):
        session = save_predictor(predictor, sample_metrics, base_dir=tmp_path)
        assert session.path_model.stem == session.path_metadata.stem

    def test_returns_predictor_session(
        self,
        predictor: RULProductionPredictor,
        sample_metrics: dict,
        tmp_path: Path,
    ):
        session = save_predictor(predictor, sample_metrics, base_dir=tmp_path)
        assert isinstance(session, PredictorSession)

    def test_session_model_name_correct(
        self,
        predictor: RULProductionPredictor,
        sample_metrics: dict,
        tmp_path: Path,
    ):
        session = save_predictor(predictor, sample_metrics, base_dir=tmp_path)
        assert session.model_name == predictor.model_name_

    def test_json_has_model_name(
        self,
        predictor: RULProductionPredictor,
        sample_metrics: dict,
        tmp_path: Path,
    ):
        session = save_predictor(predictor, sample_metrics, base_dir=tmp_path)
        with open(session.path_metadata) as f:
            meta = json.load(f)
        assert meta['model_name'] == predictor.model_name_

    def test_json_has_pipeline_params_block(
        self,
        predictor: RULProductionPredictor,
        sample_metrics: dict,
        tmp_path: Path,
    ):
        session = save_predictor(predictor, sample_metrics, base_dir=tmp_path)
        with open(session.path_metadata) as f:
            meta = json.load(f)
        assert 'pipeline_params' in meta
        pp = meta['pipeline_params']
        assert 'feature_set' in pp
        assert 'window_size' in pp
        assert 'n_components' in pp
        assert 'clipping_threshold' in pp

    def test_json_pipeline_params_values_correct(
        self,
        predictor: RULProductionPredictor,
        sample_metrics: dict,
        tmp_path: Path,
    ):
        session = save_predictor(predictor, sample_metrics, base_dir=tmp_path)
        with open(session.path_metadata) as f:
            meta = json.load(f)
        pp = meta['pipeline_params']
        assert pp['feature_set'] == predictor.feature_set
        assert pp['window_size'] == predictor.window_size
        assert pp['n_components'] == predictor.n_components
        assert pp['clipping_threshold'] == predictor.clipping_threshold

    def test_json_has_model_params_block(
        self,
        predictor: RULProductionPredictor,
        sample_metrics: dict,
        tmp_path: Path,
    ):
        session = save_predictor(predictor, sample_metrics, base_dir=tmp_path)
        with open(session.path_metadata) as f:
            meta = json.load(f)
        assert 'model_params' in meta

    def test_json_model_params_excludes_clipping_threshold(
        self,
        predictor: RULProductionPredictor,
        sample_metrics: dict,
        tmp_path: Path,
    ):
        session = save_predictor(predictor, sample_metrics, base_dir=tmp_path)
        with open(session.path_metadata) as f:
            meta = json.load(f)
        assert 'clipping_threshold' not in meta['model_params']

    def test_json_has_metrics_block(
        self,
        predictor: RULProductionPredictor,
        sample_metrics: dict,
        tmp_path: Path,
    ):
        session = save_predictor(predictor, sample_metrics, base_dir=tmp_path)
        with open(session.path_metadata) as f:
            meta = json.load(f)
        assert 'metrics' in meta
        for key in sample_metrics:
            assert key in meta['metrics']

    def test_json_no_top_level_pipeline_keys(
        self,
        predictor: RULProductionPredictor,
        sample_metrics: dict,
        tmp_path: Path,
    ):
        """Pipeline params must not appear at top level — Option B hierarchy."""
        session = save_predictor(predictor, sample_metrics, base_dir=tmp_path)
        with open(session.path_metadata) as f:
            meta = json.load(f)
        for key in ('feature_set', 'window_size', 'n_components'):
            assert key not in meta

    def test_creates_subdirectories(
        self,
        predictor: RULProductionPredictor,
        sample_metrics: dict,
        tmp_path: Path,
    ):
        save_predictor(predictor, sample_metrics, base_dir=tmp_path)
        assert (tmp_path / 'models').exists()
        assert (tmp_path / 'metadata').exists()


# ---------------------------------------------------------------------------
# TestLoadPredictor
# ---------------------------------------------------------------------------

class TestLoadPredictor:

    def test_load_by_absolute_path(
        self,
        predictor: RULProductionPredictor,
        sample_metrics: dict,
        tmp_path: Path,
    ):
        session = save_predictor(predictor, sample_metrics, base_dir=tmp_path)
        loaded = load_predictor(session.path_model, base_dir=tmp_path)
        assert isinstance(loaded, RULProductionPredictor)

    def test_load_by_filename_with_extension(
        self,
        predictor: RULProductionPredictor,
        sample_metrics: dict,
        tmp_path: Path,
    ):
        session = save_predictor(predictor, sample_metrics, base_dir=tmp_path)
        loaded = load_predictor(
            session.path_model.name,
            base_dir=tmp_path,
        )
        assert isinstance(loaded, RULProductionPredictor)

    def test_load_by_filename_without_extension(
        self,
        predictor: RULProductionPredictor,
        sample_metrics: dict,
        tmp_path: Path,
    ):
        session = save_predictor(predictor, sample_metrics, base_dir=tmp_path)
        loaded = load_predictor(
            session.path_model.stem,
            base_dir=tmp_path,
        )
        assert isinstance(loaded, RULProductionPredictor)

    def test_loaded_predictions_identical(
        self,
        predictor: RULProductionPredictor,
        sample_metrics: dict,
        motor_1_X_df: pd.DataFrame,
        tmp_path: Path,
    ):
        """Round-trip must produce bit-identical predictions."""
        session = save_predictor(predictor, sample_metrics, base_dir=tmp_path)
        loaded = load_predictor(session.path_model, base_dir=tmp_path)
        original = predictor.predict(motor_1_X_df, return_mode='all')
        restored = loaded.predict(motor_1_X_df, return_mode='all')
        np.testing.assert_array_equal(original, restored)

    def test_file_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_predictor('nonexistent_model.pkl', base_dir=tmp_path)

    def test_wrong_object_type_raises(self, tmp_path: Path):
        """Loading a non-RULProductionPredictor object must raise ValueError."""
        (tmp_path / 'models').mkdir(parents=True, exist_ok=True)
        fake_path = tmp_path / 'models' / 'fake_model.pkl'
        joblib.dump({'not': 'a predictor'}, fake_path)
        with pytest.raises(ValueError, match="RULProductionPredictor"):
            load_predictor(fake_path, base_dir=tmp_path)

    def test_loaded_params_preserved(
        self,
        predictor: RULProductionPredictor,
        sample_metrics: dict,
        tmp_path: Path,
    ):
        session = save_predictor(predictor, sample_metrics, base_dir=tmp_path)
        loaded = load_predictor(session.path_model, base_dir=tmp_path)
        assert loaded.feature_set == predictor.feature_set
        assert loaded.window_size == predictor.window_size
        assert loaded.n_components == predictor.n_components
        assert loaded.clipping_threshold == predictor.clipping_threshold

    def test_loaded_fit_still_raises(
        self,
        predictor: RULProductionPredictor,
        sample_metrics: dict,
        tmp_path: Path,
    ):
        """Immutability must survive save/load round-trip."""
        session = save_predictor(predictor, sample_metrics, base_dir=tmp_path)
        loaded = load_predictor(session.path_model, base_dir=tmp_path)
        with pytest.raises(NotImplementedError):
            loaded.fit()