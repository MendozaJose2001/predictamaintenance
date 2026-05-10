"""Tests for RULProductionPredictor (src/utils/production_predictor.py).

Uses session-scoped fixtures from conftest.py directly — motor_1_X_df,
motor_1_y_df, and motor_1_pipeline_output — following the same scope
hierarchy as the rest of the test suite.

All local fixtures use scope='session' to stay compatible with the
session-scoped conftest fixtures they depend on.

Coverage:
    TestInit              — constructor, attribute extraction, validation errors
    TestFromConfig        — factory method, param splitting, training
    TestFit               — NotImplementedError contract
    TestPredict           — shape, dtype, clipping, modes, edge cases
    TestGetParams         — sklearn introspection
    TestSerialization     — joblib round-trip, identical predictions, cleanup

Serialization tests write a temporary .pkl to pytest's tmp_path fixture
— function-scoped, automatically cleaned up after each test function.
"""

import numpy as np
import pandas as pd
import pytest
import joblib

from src.models.decision_tree import DecisionTreeModel
from src.models.negative_binomial import NegativeBinomialPiecewise
from src.pipeline.rul_pipeline import RULPipeline
from src.utils.production_predictor import RULProductionPredictor


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
def sample_params() -> dict:
    """Flat param dict for from_config tests."""
    return {
        'feature_set':        'B',
        'window_size':        20,
        'n_components':       5,
        'clipping_threshold': 125,
        'max_depth':          5,
        'min_samples_leaf':   10,
    }


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:

    def test_pipeline_attr_set(
        self,
        predictor: RULProductionPredictor,
        fitted_pipeline: RULPipeline,
    ):
        assert predictor.pipeline_ is fitted_pipeline

    def test_model_attr_set(
        self,
        predictor: RULProductionPredictor,
        fitted_model: DecisionTreeModel,
    ):
        assert predictor.model_ is fitted_model
        assert getattr(predictor.model_, 'is_fitted_', False) is True

    def test_hyperparams_extracted_from_pipeline(
        self,
        predictor: RULProductionPredictor,
        fitted_pipeline: RULPipeline,
    ):
        """Hyperparams must be extracted from pipeline — not passed manually."""
        assert predictor.feature_set        == fitted_pipeline.feature_set
        assert predictor.window_size        == fitted_pipeline.window_size
        assert predictor.n_components       == fitted_pipeline.n_components
        assert predictor.clipping_threshold == fitted_pipeline.clipping_threshold

    def test_model_name_inferred_if_none(
        self,
        fitted_pipeline: RULPipeline,
        fitted_model: DecisionTreeModel,
    ):
        p = RULProductionPredictor(pipeline=fitted_pipeline, model=fitted_model)
        assert p.model_name_ == 'DecisionTreeModel'

    def test_model_name_explicit(
        self,
        fitted_pipeline: RULPipeline,
        fitted_model: DecisionTreeModel,
    ):
        p = RULProductionPredictor(
            pipeline=fitted_pipeline,
            model=fitted_model,
            model_name='MyCustomName',
        )
        assert p.model_name_ == 'MyCustomName'

    def test_raises_if_pipeline_not_fitted(
        self,
        fitted_model: DecisionTreeModel,
    ):
        unfitted_pipeline = RULPipeline(window_size=20, n_components=5)
        with pytest.raises(ValueError, match="is_fitted_"):
            RULProductionPredictor(
                pipeline=unfitted_pipeline,
                model=fitted_model,
            )

    def test_raises_if_model_not_fitted(
        self,
        fitted_pipeline: RULPipeline,
    ):
        unfitted_model = DecisionTreeModel()
        with pytest.raises(ValueError, match="is_fitted_"):
            RULProductionPredictor(
                pipeline=fitted_pipeline,
                model=unfitted_model,
            )


# ---------------------------------------------------------------------------
# TestFromConfig
# ---------------------------------------------------------------------------

class TestFromConfig:

    def test_returns_rul_production_predictor(
        self,
        sample_params: dict,
        motor_1_X_df: pd.DataFrame,
        motor_1_y_df: pd.DataFrame,
    ):
        p = RULProductionPredictor.from_config(
            model_class=DecisionTreeModel,
            params=sample_params,
            X_df=motor_1_X_df,
            y_df=motor_1_y_df,
        )
        assert isinstance(p, RULProductionPredictor)

    def test_pipeline_is_fitted(
        self,
        sample_params: dict,
        motor_1_X_df: pd.DataFrame,
        motor_1_y_df: pd.DataFrame,
    ):
        p = RULProductionPredictor.from_config(
            model_class=DecisionTreeModel,
            params=sample_params,
            X_df=motor_1_X_df,
            y_df=motor_1_y_df,
        )
        assert p.pipeline_.is_fitted_ is True

    def test_model_is_fitted(
        self,
        sample_params: dict,
        motor_1_X_df: pd.DataFrame,
        motor_1_y_df: pd.DataFrame,
    ):
        p = RULProductionPredictor.from_config(
            model_class=DecisionTreeModel,
            params=sample_params,
            X_df=motor_1_X_df,
            y_df=motor_1_y_df,
        )
        assert getattr(p.model_, 'is_fitted_', False) is True

    def test_hyperparams_extracted_correctly(
        self,
        sample_params: dict,
        motor_1_X_df: pd.DataFrame,
        motor_1_y_df: pd.DataFrame,
    ):
        p = RULProductionPredictor.from_config(
            model_class=DecisionTreeModel,
            params=sample_params,
            X_df=motor_1_X_df,
            y_df=motor_1_y_df,
        )
        assert p.feature_set        == sample_params['feature_set']
        assert p.window_size        == sample_params['window_size']
        assert p.n_components       == sample_params['n_components']
        assert p.clipping_threshold == sample_params['clipping_threshold']

    def test_model_name_inferred(
        self,
        sample_params: dict,
        motor_1_X_df: pd.DataFrame,
        motor_1_y_df: pd.DataFrame,
    ):
        p = RULProductionPredictor.from_config(
            model_class=DecisionTreeModel,
            params=sample_params,
            X_df=motor_1_X_df,
            y_df=motor_1_y_df,
        )
        assert p.model_name_ == 'DecisionTreeModel'

    def test_model_name_explicit(
        self,
        sample_params: dict,
        motor_1_X_df: pd.DataFrame,
        motor_1_y_df: pd.DataFrame,
    ):
        p = RULProductionPredictor.from_config(
            model_class=DecisionTreeModel,
            params=sample_params,
            X_df=motor_1_X_df,
            y_df=motor_1_y_df,
            model_name='MiModelo',
        )
        assert p.model_name_ == 'MiModelo'

    def test_predict_works_after_from_config(
        self,
        sample_params: dict,
        motor_1_X_df: pd.DataFrame,
        motor_1_y_df: pd.DataFrame,
    ):
        p = RULProductionPredictor.from_config(
            model_class=DecisionTreeModel,
            params=sample_params,
            X_df=motor_1_X_df,
            y_df=motor_1_y_df,
        )
        result = p.predict(motor_1_X_df)
        assert result.shape == (1,)
        assert not np.isnan(result).any()

    def test_from_config_loads_data_automatically(
        self,
        sample_params: dict,
        motor_1_X_df: pd.DataFrame,
    ):
        """from_config must work without X_df and y_df — loads data internally."""
        p = RULProductionPredictor.from_config(
            model_class=DecisionTreeModel,
            params=sample_params,
        )
        assert isinstance(p, RULProductionPredictor)
        assert p.pipeline_.is_fitted_ is True
        assert getattr(p.model_, 'is_fitted_', False) is True
        result = p.predict(motor_1_X_df)
        assert result.shape == (1,)


# ---------------------------------------------------------------------------
# TestFit
# ---------------------------------------------------------------------------

class TestFit:

    def test_raises_not_implemented(self, predictor: RULProductionPredictor):
        with pytest.raises(NotImplementedError):
            predictor.fit()

    def test_error_message_mentions_ggs(self, predictor: RULProductionPredictor):
        with pytest.raises(NotImplementedError, match="GGSTrainingManager"):
            predictor.fit()

    def test_raises_with_args(
        self,
        predictor: RULProductionPredictor,
        motor_1_X_df: pd.DataFrame,
    ):
        with pytest.raises(NotImplementedError):
            predictor.fit(motor_1_X_df)


# ---------------------------------------------------------------------------
# TestPredict
# ---------------------------------------------------------------------------

class TestPredict:

    def test_default_mode_returns_shape_1(
        self,
        predictor: RULProductionPredictor,
        motor_1_X_df: pd.DataFrame,
    ):
        result = predictor.predict(motor_1_X_df)
        assert result.shape == (1,)

    def test_last_mode_returns_shape_1(
        self,
        predictor: RULProductionPredictor,
        motor_1_X_df: pd.DataFrame,
    ):
        result = predictor.predict(motor_1_X_df, return_mode='last')
        assert result.shape == (1,)

    def test_all_mode_returns_n_windows(
        self,
        predictor: RULProductionPredictor,
        motor_1_X_df: pd.DataFrame,
        motor_1_pipeline_output: dict,
    ):
        result = predictor.predict(motor_1_X_df, return_mode='all')
        n_windows = motor_1_pipeline_output['X'].shape[0]
        assert result.shape == (n_windows,)

    def test_last_equals_final_element_of_all(
        self,
        predictor: RULProductionPredictor,
        motor_1_X_df: pd.DataFrame,
    ):
        last = predictor.predict(motor_1_X_df, return_mode='last')
        all_ = predictor.predict(motor_1_X_df, return_mode='all')
        assert last[0] == all_[-1]

    def test_output_non_negative(
        self,
        predictor: RULProductionPredictor,
        motor_1_X_df: pd.DataFrame,
    ):
        result = predictor.predict(motor_1_X_df, return_mode='all')
        assert (result >= 0.0).all()

    def test_output_clipped(
        self,
        predictor: RULProductionPredictor,
        motor_1_X_df: pd.DataFrame,
    ):
        result = predictor.predict(motor_1_X_df, return_mode='all')
        assert (result <= predictor.clipping_threshold).all()

    def test_output_is_float(
        self,
        predictor: RULProductionPredictor,
        motor_1_X_df: pd.DataFrame,
    ):
        result = predictor.predict(motor_1_X_df, return_mode='all')
        assert np.issubdtype(result.dtype, np.floating)

    def test_output_no_nan(
        self,
        predictor: RULProductionPredictor,
        motor_1_X_df: pd.DataFrame,
    ):
        result = predictor.predict(motor_1_X_df, return_mode='all')
        assert not np.isnan(result).any()

    def test_invalid_return_mode_raises(
        self,
        predictor: RULProductionPredictor,
        motor_1_X_df: pd.DataFrame,
    ):
        with pytest.raises(ValueError, match="return_mode"):
            predictor.predict(motor_1_X_df, return_mode='invalid')

    def test_insufficient_history_raises(
        self,
        predictor: RULProductionPredictor,
        motor_1_X_df: pd.DataFrame,
    ):
        short_df = motor_1_X_df.iloc[:predictor.window_size - 1].reset_index(drop=True)
        with pytest.raises(ValueError):
            predictor.predict(short_df)

    def test_no_rul_column_required(
        self,
        predictor: RULProductionPredictor,
        motor_1_X_df: pd.DataFrame,
    ):
        assert 'RUL' not in motor_1_X_df.columns
        result = predictor.predict(motor_1_X_df)
        assert result.shape == (1,)

    def test_without_unit_number_column(
        self,
        predictor: RULProductionPredictor,
        motor_1_X_df: pd.DataFrame,
    ):
        df_no_unit = motor_1_X_df.drop(columns=['unit_number'])
        result = predictor.predict(df_no_unit)
        assert result.shape == (1,)


# ---------------------------------------------------------------------------
# TestGetParams
# ---------------------------------------------------------------------------

class TestGetParams:

    def test_returns_dict(self, predictor: RULProductionPredictor):
        params = predictor.get_params()
        assert isinstance(params, dict)

    def test_contains_pipeline_and_model(self, predictor: RULProductionPredictor):
        params = predictor.get_params()
        assert 'pipeline' in params
        assert 'model' in params

    def test_hyperparams_consistent_with_pipeline(
        self,
        predictor: RULProductionPredictor,
        fitted_pipeline: RULPipeline,
    ):
        assert predictor.feature_set        == fitted_pipeline.feature_set
        assert predictor.window_size        == fitted_pipeline.window_size
        assert predictor.n_components       == fitted_pipeline.n_components
        assert predictor.clipping_threshold == fitted_pipeline.clipping_threshold


# ---------------------------------------------------------------------------
# TestSerialization
# ---------------------------------------------------------------------------

class TestSerialization:

    def test_joblib_dump_and_load(
        self,
        predictor: RULProductionPredictor,
        tmp_path,
    ):
        path = tmp_path / 'predictor_test.pkl'
        joblib.dump(predictor, path)
        loaded = joblib.load(path)
        assert loaded is not None

    def test_loaded_predictor_is_same_type(
        self,
        predictor: RULProductionPredictor,
        tmp_path,
    ):
        path = tmp_path / 'predictor_test.pkl'
        joblib.dump(predictor, path)
        loaded = joblib.load(path)
        assert isinstance(loaded, RULProductionPredictor)

    def test_loaded_predictor_identical_predictions(
        self,
        predictor: RULProductionPredictor,
        motor_1_X_df: pd.DataFrame,
        tmp_path,
    ):
        path = tmp_path / 'predictor_test.pkl'
        joblib.dump(predictor, path)
        loaded = joblib.load(path)
        original = predictor.predict(motor_1_X_df, return_mode='all')
        restored = loaded.predict(motor_1_X_df, return_mode='all')
        np.testing.assert_array_equal(original, restored)

    def test_loaded_predictor_preserves_hyperparams(
        self,
        predictor: RULProductionPredictor,
        tmp_path,
    ):
        path = tmp_path / 'predictor_test.pkl'
        joblib.dump(predictor, path)
        loaded = joblib.load(path)
        assert loaded.feature_set        == predictor.feature_set
        assert loaded.window_size        == predictor.window_size
        assert loaded.n_components       == predictor.n_components
        assert loaded.clipping_threshold == predictor.clipping_threshold

    def test_loaded_predictor_fit_still_raises(
        self,
        predictor: RULProductionPredictor,
        tmp_path,
    ):
        path = tmp_path / 'predictor_test.pkl'
        joblib.dump(predictor, path)
        loaded = joblib.load(path)
        with pytest.raises(NotImplementedError):
            loaded.fit()


# ---------------------------------------------------------------------------
# TestSave
# ---------------------------------------------------------------------------

class TestSave:

    def test_save_returns_predictor_session(
        self,
        predictor: RULProductionPredictor,
        tmp_path,
    ):
        from src.utils.predictor_io import PredictorSession
        session = predictor.save(base_dir=tmp_path)
        assert isinstance(session, PredictorSession)

    def test_save_creates_pkl(
        self,
        predictor: RULProductionPredictor,
        tmp_path,
    ):
        predictor.save(base_dir=tmp_path)
        pkl_files = list((tmp_path / 'models').glob('*.pkl'))
        assert len(pkl_files) == 1

    def test_save_creates_json(
        self,
        predictor: RULProductionPredictor,
        tmp_path,
    ):
        predictor.save(base_dir=tmp_path)
        json_files = list((tmp_path / 'metadata').glob('*.json'))
        assert len(json_files) == 1

    def test_save_with_metrics(
        self,
        predictor: RULProductionPredictor,
        tmp_path,
    ):
        import json
        metrics = {'mean_S_score': 12.3, 'mean_MAE': 8.5}
        session = predictor.save(metrics=metrics, base_dir=tmp_path)
        with open(session.path_metadata) as f:
            meta = json.load(f)
        assert meta['metrics']['mean_S_score'] == 12.3
        assert meta['metrics']['mean_MAE'] == 8.5

    def test_save_without_metrics_stores_empty_dict(
        self,
        predictor: RULProductionPredictor,
        tmp_path,
    ):
        import json
        session = predictor.save(base_dir=tmp_path)
        with open(session.path_metadata) as f:
            meta = json.load(f)
        assert meta['metrics'] == {}