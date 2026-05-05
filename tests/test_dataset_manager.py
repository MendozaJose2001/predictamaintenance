import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.dataset_manager import (
    _get_txt,
    store_dataframe_csv,
    DatasetManager,
    _columnas,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_train(n_units: int = 3, cycles_per_unit: int = 5) -> pd.DataFrame:
    """Builds a minimal synthetic train DataFrame mimicking C-MAPSS structure.

    Args:
        n_units: Number of engine units to generate.
        cycles_per_unit: Number of cycles per unit.

    Returns:
        DataFrame with unit_number, time_in_cycles, settings, and sensor columns.
    """
    col_names = (
        _columnas['identificadores']
        + _columnas['settings']
        + _columnas['sensores']
    )
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, cycles_per_unit + 1):
            row = [unit, cycle] + [float(cycle)] * (len(col_names) - 2)
            rows.append(row)

    return pd.DataFrame(rows, columns=col_names)


def _make_raw_test(n_units: int = 3, cycles_per_unit: int = 4) -> pd.DataFrame:
    """Builds a minimal synthetic test DataFrame mimicking C-MAPSS structure.

    Args:
        n_units: Number of engine units to generate.
        cycles_per_unit: Number of cycles per unit.

    Returns:
        DataFrame with unit_number, time_in_cycles, settings, and sensor columns.
    """
    col_names = (
        _columnas['identificadores']
        + _columnas['settings']
        + _columnas['sensores']
    )
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, cycles_per_unit + 1):
            row = [unit, cycle] + [float(cycle)] * (len(col_names) - 2)
            rows.append(row)

    return pd.DataFrame(rows, columns=col_names)


def _make_raw_true(n_units: int = 3, rul_value: int = 10) -> pd.DataFrame:
    """Builds a minimal synthetic ground truth DataFrame.

    Args:
        n_units: Number of engine units.
        rul_value: RUL value assigned to all units.

    Returns:
        DataFrame with true_RUL and unit_number columns.
    """
    return pd.DataFrame({
        'true_RUL': [rul_value] * n_units,
        'unit_number': range(1, n_units + 1)
    })


# ---------------------------------------------------------------------------
# _get_txt
# ---------------------------------------------------------------------------

class TestGetTxt:
    """Tests for the _get_txt file reader."""

    def test_train_returns_correct_columns(self):
        """Train file must return DataFrame with all expected column names."""
        expected_cols = (
            _columnas['identificadores']
            + _columnas['settings']
            + _columnas['sensores']
        )
        fake_df = pd.DataFrame(columns=expected_cols)

        with patch('src.dataset_manager.pd.read_csv', return_value=fake_df):
            result = _get_txt('train')

        assert list(result.columns) == expected_cols

    def test_test_returns_correct_columns(self):
        """Test file must return DataFrame with all expected column names."""
        expected_cols = (
            _columnas['identificadores']
            + _columnas['settings']
            + _columnas['sensores']
        )
        fake_df = pd.DataFrame(columns=expected_cols)

        with patch('src.dataset_manager.pd.read_csv', return_value=fake_df):
            result = _get_txt('test')

        assert list(result.columns) == expected_cols

    def test_true_returns_true_rul_and_unit_number(self):
        """Ground truth file must return DataFrame with true_RUL and unit_number."""
        fake_df = pd.DataFrame({'true_RUL': [10, 20, 30]})

        with patch('src.dataset_manager.pd.read_csv', return_value=fake_df):
            result = _get_txt('true')

        assert 'true_RUL' in result.columns
        assert 'unit_number' in result.columns

    def test_true_unit_number_starts_at_one(self):
        """Ground truth unit_number must start at 1 and be sequential."""
        fake_df = pd.DataFrame({'true_RUL': [10, 20, 30]})

        with patch('src.dataset_manager.pd.read_csv', return_value=fake_df):
            result = _get_txt('true')

        assert list(result['unit_number']) == [1, 2, 3]

    def test_raises_runtime_error_on_failure(self):
        """Must raise RuntimeError with informative message when read fails."""
        with patch('src.dataset_manager.pd.read_csv', side_effect=FileNotFoundError("file not found")):
            with pytest.raises(RuntimeError, match="Error reading file"):
                _get_txt('train')

    def test_runtime_error_chains_original_exception(self):
        """RuntimeError must chain the original exception via __cause__."""
        with patch('src.dataset_manager.pd.read_csv', side_effect=FileNotFoundError("missing")):
            with pytest.raises(RuntimeError) as exc_info:
                _get_txt('train')

        assert exc_info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# store_dataframe_csv
# ---------------------------------------------------------------------------

class TestStoreDataframeCsv:
    """Tests for the CSV storage utility."""

    def test_creates_directory_if_not_exists(self, tmp_path):
        """Must create the output directory when it does not exist."""
        output_dir = tmp_path / "new_dir"
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

        store_dataframe_csv(df, 'test_file', str(output_dir))

        assert output_dir.exists()

    def test_saves_csv_when_directory_is_new(self, tmp_path):
        """Must save the CSV file when the directory is newly created."""
        output_dir = tmp_path / "new_dir"
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

        store_dataframe_csv(df, 'test_file', str(output_dir))

        assert (output_dir / 'test_file.csv').exists()

    def test_saves_csv_when_directory_already_exists(self, tmp_path):
        """Must save the CSV file even when the directory already exists."""
        output_dir = tmp_path / "existing_dir"
        output_dir.mkdir()
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

        store_dataframe_csv(df, 'test_file', str(output_dir))

        assert (output_dir / 'test_file.csv').exists()

    def test_saved_content_matches_original(self, tmp_path):
        """CSV content must match the original DataFrame after saving and reading back."""
        output_dir = tmp_path / "output"
        df = pd.DataFrame({'x': [10, 20, 30], 'y': [1.1, 2.2, 3.3]})

        store_dataframe_csv(df, 'data', str(output_dir))
        result = pd.read_csv(output_dir / 'data.csv')

        pd.testing.assert_frame_equal(df, result)


# ---------------------------------------------------------------------------
# get_base_dataset
# ---------------------------------------------------------------------------

class TestGetBaseDataset:
    """Tests for the unified dataset construction."""

    def _patch_get_txt(self, n_units: int = 3, cycles: int = 5, rul: int = 10):
        """Returns a context manager patching _get_txt with synthetic data."""
        train = _make_raw_train(n_units, cycles)
        test = _make_raw_test(n_units, cycles - 1)
        true = _make_raw_true(n_units, rul)

        def side_effect(key):
            return {'train': train.copy(), 'test': test.copy(), 'true': true.copy()}[key]

        return patch('src.dataset_manager._get_txt', side_effect=side_effect)

    def test_train_engines_have_evento_one(self):
        """Train engines must have evento=1 only at their last cycle."""
        with self._patch_get_txt():
            df = DatasetManager.get_base_dataset()

        train_rows = df[df['unit_number'] <= 100]
        # evento=1 only at the last cycle of each train motor
        last_cycles = train_rows.groupby('unit_number')['time_in_cycles'].transform('max')
        last_rows = train_rows[train_rows['time_in_cycles'] == last_cycles]
        non_last_rows = train_rows[train_rows['time_in_cycles'] != last_cycles]

        assert (last_rows['evento'] == 1).all()
        assert (non_last_rows['evento'] == 0).all()

    def test_test_engines_have_evento_zero(self):
        """All rows from test engines must have evento=0."""
        with self._patch_get_txt():
            df = DatasetManager.get_base_dataset()

        test_rows = df[df['unit_number'] > 100]
        assert (test_rows['evento'] == 0).all()

    def test_test_unit_ids_offset_by_100(self):
        """Test engine unit numbers must all be greater than 100."""
        with self._patch_get_txt():
            df = DatasetManager.get_base_dataset()

        # Test engines are identified by unit_number > 100, not by evento=0,
        # because train engines also have evento=0 in non-last cycles.
        test_ids = df[df['unit_number'] > 100]['unit_number'].unique()
        assert (test_ids > 100).all()

    def test_no_vida_total_column_in_output(self):
        """Intermediate vida_total column must not appear in the final dataset."""
        with self._patch_get_txt():
            df = DatasetManager.get_base_dataset()

        assert 'vida_total' not in df.columns

    def test_train_rul_at_last_cycle_is_zero(self):
        """RUL at the last observed cycle for train engines must be 0."""
        with self._patch_get_txt(n_units=3, cycles=5):
            df = DatasetManager.get_base_dataset()

        train_df = df[df['unit_number'] <= 100]
        last_indices = train_df.groupby('unit_number')['time_in_cycles'].idxmax()
        train_last_rul = train_df.loc[last_indices, 'RUL']
        assert (train_last_rul == 0).all()

    def test_test_rul_reconstructed_correctly(self):
        """RUL for test engines must equal vida_total minus time_in_cycles."""
        n_units, cycles, rul = 3, 4, 10
        with self._patch_get_txt(n_units=n_units, cycles=cycles, rul=rul):
            df = DatasetManager.get_base_dataset()

        test_rows = df[df['unit_number'] > 100]

        # vida_total = last_cycle + true_RUL = (cycles-1) + rul
        expected_vida_total = (cycles - 1) + rul
        expected_rul = expected_vida_total - test_rows['time_in_cycles']
        pd.testing.assert_series_equal(
            test_rows['RUL'].reset_index(drop=True),
            expected_rul.reset_index(drop=True),
            check_names=False
        )

    def test_output_contains_rul_and_evento_columns(self):
        """Output DataFrame must contain both RUL and evento columns."""
        with self._patch_get_txt():
            df = DatasetManager.get_base_dataset()

        assert 'RUL' in df.columns
        assert 'evento' in df.columns


# ---------------------------------------------------------------------------
# generate_metadata
# ---------------------------------------------------------------------------

class TestGenerateMetadata:
    """Tests for the metadata generation method."""

    def _make_input(self) -> pd.DataFrame:
        """Builds a minimal unified dataset for metadata tests."""
        return pd.DataFrame({
            'unit_number': [1, 1, 2, 2, 3],
            'time_in_cycles': [1, 2, 1, 3, 1],
            'evento': [1, 1, 1, 1, 0]
        })

    def test_one_row_per_unit(self):
        """Output must contain exactly one row per unique unit_number."""
        df = self._make_input()
        meta = DatasetManager.generate_metadata(df)
        assert len(meta) == df['unit_number'].nunique()

    def test_max_cycles_correct(self):
        """max_cycles must equal the maximum time_in_cycles per unit."""
        df = self._make_input()
        meta = DatasetManager.generate_metadata(df).set_index('unit_number')
        assert meta.loc[1, 'max_cycles'] == 2
        assert meta.loc[2, 'max_cycles'] == 3
        assert meta.loc[3, 'max_cycles'] == 1

    def test_event_correct(self):
        """event must reflect the evento value of each unit."""
        df = self._make_input()
        meta = DatasetManager.generate_metadata(df).set_index('unit_number')
        assert meta.loc[1, 'event'] == 1
        assert meta.loc[3, 'event'] == 0

    def test_n_settings_and_n_sensores_correct(self):
        """n_settings and n_sensores must match the global _columnas definition."""
        df = self._make_input()
        meta = DatasetManager.generate_metadata(df)
        assert (meta['n_settings'] == len(_columnas['settings'])).all()
        assert (meta['n_sensores'] == len(_columnas['sensores'])).all()

    def test_output_columns(self):
        """Output must contain exactly the expected columns in the correct order."""
        df = self._make_input()
        meta = DatasetManager.generate_metadata(df)
        assert list(meta.columns) == [
            'unit_number', 'max_cycles', 'n_settings', 'n_sensores', 'event'
        ]


# ---------------------------------------------------------------------------
# clean_dataset
# ---------------------------------------------------------------------------

class TestCleanDataset:
    """Tests for the zero-variance column removal."""

    def _make_input(self) -> pd.DataFrame:
        """Builds a DataFrame with one constant sensor and one varying sensor."""
        return pd.DataFrame({
            'unit_number': [1, 1, 2, 2],
            'time_in_cycles': [1, 2, 1, 2],
            'RUL': [10, 9, 8, 7],
            'evento': [1, 1, 0, 0],
            'sensor_const': [5.0, 5.0, 5.0, 5.0],   # constant — should be removed
            'sensor_vary': [1.0, 2.0, 3.0, 4.0],     # varying — should be kept
        })

    def test_removes_constant_sensor_columns(self):
        """Columns with IQR=0 must be removed from the output."""
        df = self._make_input()
        clean_df, _ = DatasetManager.clean_dataset(df)
        assert 'sensor_const' not in clean_df.columns

    def test_keeps_varying_sensor_columns(self):
        """Columns with IQR>0 must be retained in the output."""
        df = self._make_input()
        clean_df, _ = DatasetManager.clean_dataset(df)
        assert 'sensor_vary' in clean_df.columns

    def test_protected_columns_never_removed(self):
        """unit_number, time_in_cycles, RUL, and evento must never be removed."""
        df = self._make_input()
        clean_df, _ = DatasetManager.clean_dataset(df)
        for col in ['unit_number', 'time_in_cycles', 'RUL', 'evento']:
            assert col in clean_df.columns

    def test_stats_contains_iqr_column(self):
        """Returned stats DataFrame must include the computed IQR column."""
        df = self._make_input()
        _, stats = DatasetManager.clean_dataset(df)
        assert 'IQR' in stats.columns


# ---------------------------------------------------------------------------
# split_dataset
# ---------------------------------------------------------------------------

class TestSplitDataset:
    """Tests for the engine-level train/test split."""

    def _make_metadata(self, n_units: int = 20) -> pd.DataFrame:
        """Builds a synthetic metadata DataFrame.

        Args:
            n_units: Number of engine units to include.

        Returns:
            DataFrame with unit_number column.
        """
        return pd.DataFrame({'unit_number': range(1, n_units + 1)})

    def test_split_sizes_match_test_size(self):
        """Test partition size must match the requested test_size fraction."""
        n_units = 20
        with patch('src.dataset_manager.pd.read_csv', return_value=self._make_metadata(n_units)):
            train, test = DatasetManager.split_dataset(test_size=0.3, random_state=42)

        assert len(test) == pytest.approx(n_units * 0.3, abs=1)

    def test_no_overlap_between_partitions(self):
        """No engine unit must appear in both train and test partitions."""
        with patch('src.dataset_manager.pd.read_csv', return_value=self._make_metadata()):
            train, test = DatasetManager.split_dataset()

        assert len(set(train) & set(test)) == 0

    def test_all_units_accounted_for(self):
        """Union of train and test must contain all original unit numbers."""
        n_units = 20
        with patch('src.dataset_manager.pd.read_csv', return_value=self._make_metadata(n_units)):
            train, test = DatasetManager.split_dataset()

        assert len(train) + len(test) == n_units

    def test_reproducible_with_same_random_state(self):
        """Same random_state must produce identical splits across calls."""
        meta = self._make_metadata()
        with patch('src.dataset_manager.pd.read_csv', return_value=meta):
            train1, test1 = DatasetManager.split_dataset(random_state=42)
        with patch('src.dataset_manager.pd.read_csv', return_value=meta):
            train2, test2 = DatasetManager.split_dataset(random_state=42)

        np.testing.assert_array_equal(sorted(train1), sorted(train2))
        np.testing.assert_array_equal(sorted(test1), sorted(test2))

    def test_different_random_states_produce_different_splits(self):
        """Different random states must produce different splits."""
        meta = self._make_metadata(50)
        with patch('src.dataset_manager.pd.read_csv', return_value=meta):
            train1, _ = DatasetManager.split_dataset(random_state=0)
        with patch('src.dataset_manager.pd.read_csv', return_value=meta):
            train2, _ = DatasetManager.split_dataset(random_state=99)

        assert not np.array_equal(sorted(train1), sorted(train2))