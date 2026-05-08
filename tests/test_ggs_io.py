"""Tests for src/utils/ggs_io.py — GGS session management and I/O.

Tests are isolated from the full GGS pipeline — no model training,
no feature extraction. Uses tmp_path fixtures for file I/O isolation.

Coverage:
    - compute_param_grid_hash: determinism, sensitivity to changes
    - config_key: determinism, order independence
    - resolve_ggs_session: new session, resume detection, timestamp parsing
    - save_metadata / save_checkpoint / save_results / delete_checkpoint
    - GGSSession: stem property, field integrity
    - _extract_params: key filtering
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.utils.ggs_io import (
    GGSSession,
    _extract_params,
    compute_param_grid_hash,
    config_key,
    delete_checkpoint,
    resolve_ggs_session,
    save_checkpoint,
    save_metadata,
    save_results,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeModel:
    """Minimal fake model class for testing — no training needed."""
    pass

class _FakeModelWith_Underscore:
    """Model class whose name contains underscores."""
    pass


@pytest.fixture
def param_grid() -> dict:
    return {
        'feature_set':  ['A', 'B'],
        'window_size':  [20, 30],
        'n_components': [5],
        'alpha':        [1.0],
    }


@pytest.fixture
def sample_results() -> list[dict]:
    """Two minimal result dicts simulating completed GGS configs."""
    return [
        {
            'feature_set': 'A', 'window_size': 20, 'n_components': 5,
            'alpha': 1.0, 'mean_S_score': 10.0, 'mean_MAE': 8.0,
            'mean_RMSE': 12.0, 'mean_C_index': 0.7,
        },
        {
            'feature_set': 'B', 'window_size': 20, 'n_components': 5,
            'alpha': 1.0, 'mean_S_score': 9.0, 'mean_MAE': 7.5,
            'mean_RMSE': 11.0, 'mean_C_index': 0.75,
        },
    ]


@pytest.fixture
def new_session(tmp_path: Path, param_grid: dict) -> GGSSession:
    """Returns a fresh GGSSession with no prior checkpoint."""
    return resolve_ggs_session(
        model_class=_FakeModel,
        param_grid=param_grid,
        base_dir=tmp_path,
    )


# ---------------------------------------------------------------------------
# TestComputeParamGridHash
# ---------------------------------------------------------------------------

class TestComputeParamGridHash:

    def test_returns_8_char_hex(self, param_grid: dict):
        h = compute_param_grid_hash(param_grid)
        assert len(h) == 8
        assert all(c in '0123456789abcdef' for c in h)

    def test_deterministic(self, param_grid: dict):
        assert compute_param_grid_hash(param_grid) == compute_param_grid_hash(param_grid)

    def test_same_content_different_key_order(self, param_grid: dict):
        """Hash must be identical regardless of dict key order."""
        reordered = {k: param_grid[k] for k in reversed(list(param_grid))}
        assert compute_param_grid_hash(param_grid) == compute_param_grid_hash(reordered)

    def test_different_values_produce_different_hash(self, param_grid: dict):
        modified = {**param_grid, 'window_size': [15, 25]}
        assert compute_param_grid_hash(param_grid) != compute_param_grid_hash(modified)

    def test_different_keys_produce_different_hash(self, param_grid: dict):
        modified = {**param_grid, 'new_param': [1, 2]}
        assert compute_param_grid_hash(param_grid) != compute_param_grid_hash(modified)


# ---------------------------------------------------------------------------
# TestConfigKey
# ---------------------------------------------------------------------------

class TestConfigKey:

    def test_returns_string(self):
        assert isinstance(config_key({'a': 1, 'b': 2}), str)

    def test_deterministic(self):
        params = {'feature_set': 'A', 'window_size': 20, 'alpha': 1.0}
        assert config_key(params) == config_key(params)

    def test_order_independent(self):
        """config_key must be identical regardless of dict key order."""
        p1 = {'feature_set': 'A', 'window_size': 20}
        p2 = {'window_size': 20, 'feature_set': 'A'}
        assert config_key(p1) == config_key(p2)

    def test_different_values_produce_different_key(self):
        p1 = {'feature_set': 'A', 'window_size': 20}
        p2 = {'feature_set': 'B', 'window_size': 20}
        assert config_key(p1) != config_key(p2)


# ---------------------------------------------------------------------------
# TestGGSSession
# ---------------------------------------------------------------------------

class TestGGSSession:

    def test_stem_format(self, new_session: GGSSession):
        """stem must be '{model_name}_{param_hash}_{timestamp}'."""
        stem = new_session.stem
        assert new_session.model_name in stem
        assert new_session.param_hash in stem
        assert new_session.timestamp in stem

    def test_is_resume_false_for_new_session(self, new_session: GGSSession):
        assert new_session.is_resume is False

    def test_completed_keys_empty_for_new_session(self, new_session: GGSSession):
        assert len(new_session.completed_keys) == 0

    def test_prior_results_empty_for_new_session(self, new_session: GGSSession):
        assert len(new_session.prior_results) == 0

    def test_paths_under_correct_subdirs(self, new_session: GGSSession, tmp_path: Path):
        assert new_session.path_metadata.parent == tmp_path / 'metadata'
        assert new_session.path_checkpoint.parent == tmp_path / 'checkpoints'
        assert new_session.path_results.parent == tmp_path / 'results'

    def test_all_paths_share_same_stem(self, new_session: GGSSession):
        stems = {
            new_session.path_metadata.stem,
            new_session.path_checkpoint.stem,
            new_session.path_results.stem,
        }
        assert len(stems) == 1


# ---------------------------------------------------------------------------
# TestResolveGGSSession — new session
# ---------------------------------------------------------------------------

class TestResolveGGSSessionNew:

    def test_creates_subdirectories(self, tmp_path: Path, param_grid: dict):
        resolve_ggs_session(_FakeModel, param_grid, tmp_path)
        assert (tmp_path / 'metadata').exists()
        assert (tmp_path / 'checkpoints').exists()
        assert (tmp_path / 'results').exists()

    def test_no_checkpoint_file_created(self, new_session: GGSSession):
        """resolve_ggs_session must NOT create a checkpoint file."""
        assert not new_session.path_checkpoint.exists()

    def test_model_name_from_class(self, new_session: GGSSession):
        assert new_session.model_name == '_FakeModel'

    def test_param_hash_8_chars(self, new_session: GGSSession):
        assert len(new_session.param_hash) == 8

    def test_timestamp_format(self, new_session: GGSSession):
        """Timestamp must match YYYYMMDD_HHMM format."""
        ts = new_session.timestamp
        assert len(ts) == 13  # '20260507_1430'
        assert ts[8] == '_'
        assert ts[:8].isdigit()
        assert ts[9:].isdigit()

    def test_different_param_grids_different_sessions(
        self, tmp_path: Path, param_grid: dict
    ):
        s1 = resolve_ggs_session(_FakeModel, param_grid, tmp_path)
        modified = {**param_grid, 'window_size': [99]}
        s2 = resolve_ggs_session(_FakeModel, modified, tmp_path)
        assert s1.param_hash != s2.param_hash
        assert s1.path_checkpoint != s2.path_checkpoint

    def test_model_with_underscore_in_name(self, tmp_path: Path, param_grid: dict):
        """Model names with underscores must not break timestamp parsing."""
        session = resolve_ggs_session(_FakeModelWith_Underscore, param_grid, tmp_path)
        assert session.model_name == '_FakeModelWith_Underscore'
        assert len(session.param_hash) == 8
        assert session.timestamp  # non-empty


# ---------------------------------------------------------------------------
# TestResolveGGSSession — resume
# ---------------------------------------------------------------------------

class TestResolveGGSSessionResume:

    @pytest.fixture
    def resumed_session(
        self,
        tmp_path: Path,
        param_grid: dict,
        sample_results: list[dict],
    ) -> GGSSession:
        """Creates a checkpoint then resolves a new session (resume)."""
        # First run — create and save checkpoint
        first_session = resolve_ggs_session(_FakeModel, param_grid, tmp_path)
        save_checkpoint(first_session, sample_results)
        # Second resolve — should detect checkpoint and resume
        return resolve_ggs_session(_FakeModel, param_grid, tmp_path)

    def test_is_resume_true(self, resumed_session: GGSSession):
        assert resumed_session.is_resume is True

    def test_inherits_timestamp(
        self, tmp_path: Path, param_grid: dict, sample_results: list[dict]
    ):
        first = resolve_ggs_session(_FakeModel, param_grid, tmp_path)
        original_timestamp = first.timestamp
        save_checkpoint(first, sample_results)
        second = resolve_ggs_session(_FakeModel, param_grid, tmp_path)
        assert second.timestamp == original_timestamp

    def test_prior_results_loaded(
        self, resumed_session: GGSSession, sample_results: list[dict]
    ):
        assert len(resumed_session.prior_results) == len(sample_results)

    def test_completed_keys_populated(
        self, resumed_session: GGSSession, sample_results: list[dict],
        param_grid: dict
    ):
        assert len(resumed_session.completed_keys) == len(sample_results)

    def test_completed_keys_match_results(
        self, resumed_session: GGSSession, sample_results: list[dict],
        param_grid: dict
    ):
        expected_keys = {
            config_key(_extract_params(r, param_grid))
            for r in sample_results
        }
        assert resumed_session.completed_keys == expected_keys

    def test_model_with_underscore_resume(
        self, tmp_path: Path, param_grid: dict, sample_results: list[dict]
    ):
        """Resume must work correctly for models with underscores in name."""
        first = resolve_ggs_session(_FakeModelWith_Underscore, param_grid, tmp_path)
        save_checkpoint(first, sample_results)
        second = resolve_ggs_session(_FakeModelWith_Underscore, param_grid, tmp_path)
        assert second.is_resume is True
        assert second.timestamp == first.timestamp


# ---------------------------------------------------------------------------
# TestSaveMetadata
# ---------------------------------------------------------------------------

class TestSaveMetadata:

    def test_creates_json_file(self, new_session: GGSSession, param_grid: dict):
        save_metadata(new_session, param_grid, n_folds=5, total_configs=8)
        assert new_session.path_metadata.exists()

    def test_json_contains_model_name(
        self, new_session: GGSSession, param_grid: dict
    ):
        save_metadata(new_session, param_grid, n_folds=5, total_configs=8)
        with open(new_session.path_metadata) as f:
            meta = json.load(f)
        assert meta['model'] == '_FakeModel'

    def test_json_contains_param_grid(
        self, new_session: GGSSession, param_grid: dict
    ):
        save_metadata(new_session, param_grid, n_folds=5, total_configs=8)
        with open(new_session.path_metadata) as f:
            meta = json.load(f)
        assert 'param_grid' in meta
        assert set(meta['param_grid'].keys()) == set(param_grid.keys())

    def test_json_contains_n_folds_and_total(
        self, new_session: GGSSession, param_grid: dict
    ):
        save_metadata(new_session, param_grid, n_folds=5, total_configs=8)
        with open(new_session.path_metadata) as f:
            meta = json.load(f)
        assert meta['n_folds'] == 5
        assert meta['total_configs'] == 8


# ---------------------------------------------------------------------------
# TestSaveCheckpoint
# ---------------------------------------------------------------------------

class TestSaveCheckpoint:

    def test_creates_csv_file(
        self, new_session: GGSSession, sample_results: list[dict]
    ):
        save_checkpoint(new_session, sample_results)
        assert new_session.path_checkpoint.exists()

    def test_csv_has_correct_n_rows(
        self, new_session: GGSSession, sample_results: list[dict]
    ):
        save_checkpoint(new_session, sample_results)
        df = pd.read_csv(new_session.path_checkpoint)
        assert len(df) == len(sample_results)

    def test_csv_contains_metric_columns(
        self, new_session: GGSSession, sample_results: list[dict]
    ):
        save_checkpoint(new_session, sample_results)
        df = pd.read_csv(new_session.path_checkpoint)
        for col in ['mean_S_score', 'mean_MAE', 'mean_RMSE', 'mean_C_index']:
            assert col in df.columns

    def test_overwrites_on_second_call(
        self, new_session: GGSSession, sample_results: list[dict]
    ):
        """Second save_checkpoint must overwrite, not append."""
        save_checkpoint(new_session, sample_results[:1])
        save_checkpoint(new_session, sample_results)
        df = pd.read_csv(new_session.path_checkpoint)
        assert len(df) == len(sample_results)

    def test_combines_prior_and_new_results(
        self, tmp_path: Path, param_grid: dict, sample_results: list[dict]
    ):
        """Checkpoint must include prior_results + new results."""
        first = resolve_ggs_session(_FakeModel, param_grid, tmp_path)
        save_checkpoint(first, sample_results)
        # Resume and save more results
        second = resolve_ggs_session(_FakeModel, param_grid, tmp_path)
        extra = [sample_results[0]]  # one more result
        save_checkpoint(second, extra)
        df = pd.read_csv(second.path_checkpoint)
        assert len(df) == len(sample_results) + len(extra)


# ---------------------------------------------------------------------------
# TestSaveResults
# ---------------------------------------------------------------------------

class TestSaveResults:

    def test_creates_results_csv(
        self, new_session: GGSSession, sample_results: list[dict]
    ):
        save_results(new_session, sample_results)
        assert new_session.path_results.exists()

    def test_removes_checkpoint(
        self, new_session: GGSSession, sample_results: list[dict]
    ):
        save_checkpoint(new_session, sample_results)
        assert new_session.path_checkpoint.exists()
        save_results(new_session, sample_results)
        assert not new_session.path_checkpoint.exists()

    def test_results_csv_has_correct_n_rows(
        self, new_session: GGSSession, sample_results: list[dict]
    ):
        save_results(new_session, sample_results)
        df = pd.read_csv(new_session.path_results)
        assert len(df) == len(sample_results)

    def test_no_checkpoint_no_error(
        self, new_session: GGSSession, sample_results: list[dict]
    ):
        """save_results must not fail if no checkpoint exists."""
        assert not new_session.path_checkpoint.exists()
        save_results(new_session, sample_results)  # should not raise


# ---------------------------------------------------------------------------
# TestDeleteCheckpoint
# ---------------------------------------------------------------------------

class TestDeleteCheckpoint:

    def test_removes_existing_checkpoint(
        self, new_session: GGSSession, sample_results: list[dict]
    ):
        save_checkpoint(new_session, sample_results)
        assert new_session.path_checkpoint.exists()
        delete_checkpoint(new_session)
        assert not new_session.path_checkpoint.exists()

    def test_no_error_if_no_checkpoint(self, new_session: GGSSession):
        """delete_checkpoint must not raise if file does not exist."""
        assert not new_session.path_checkpoint.exists()
        delete_checkpoint(new_session)  # should not raise


# ---------------------------------------------------------------------------
# TestExtractParams
# ---------------------------------------------------------------------------

class TestExtractParams:

    def test_returns_only_param_grid_keys(self, param_grid: dict):
        result = {
            'feature_set': 'A', 'window_size': 20,
            'n_components': 5, 'alpha': 1.0,
            'mean_S_score': 10.0, 'mean_MAE': 8.0,  # metric keys — should be excluded
        }
        extracted = _extract_params(result, param_grid)
        assert set(extracted.keys()) == set(param_grid.keys())

    def test_values_preserved(self, param_grid: dict):
        result = {
            'feature_set': 'A', 'window_size': 20,
            'n_components': 5, 'alpha': 1.0,
            'mean_S_score': 10.0,
        }
        extracted = _extract_params(result, param_grid)
        assert extracted['feature_set'] == 'A'
        assert extracted['window_size'] == 20

    def test_missing_keys_ignored(self, param_grid: dict):
        """Keys in param_grid but missing from result are silently ignored."""
        partial_result = {'feature_set': 'A'}  # missing window_size etc.
        extracted = _extract_params(partial_result, param_grid)
        assert set(extracted.keys()) == {'feature_set'}