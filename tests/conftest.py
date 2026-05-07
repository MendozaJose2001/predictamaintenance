"""Shared pytest fixtures for the PredictaMaintenance test suite.

This module provides session-scoped fixtures that are automatically
available to all test modules without explicit imports. Using
scope='session' ensures that expensive operations (Nodo 3 feature
extraction) run only once per test session regardless of how many
test modules request the fixture.

Fixture hierarchy:
    motor_1_raw_df          — raw DataFrame for motor 1
    motor_1_X_df            — X_df without RUL
    motor_1_y_df            — y_df with RUL
    motor_1_pipeline_output — full Nodos 2-4 output for motor 1
    multi_motor_data        — motors 1,2,3 for GroupKFold tests
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.rul_pipeline import RULPipeline


# ---------------------------------------------------------------------------
# Motor 1 — base fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def motor_1_raw_df() -> pd.DataFrame:
    """Loads motor 1 CSV with unit_number inserted."""
    df = pd.read_csv('data/clean/data_motor_1.csv')
    df.insert(0, 'unit_number', 1)
    return df


@pytest.fixture(scope='session')
def motor_1_X_df(motor_1_raw_df: pd.DataFrame) -> pd.DataFrame:
    """X_df for motor 1 — all columns except RUL."""
    return motor_1_raw_df.drop(columns=['RUL'])


@pytest.fixture(scope='session')
def motor_1_y_df(motor_1_raw_df: pd.DataFrame) -> pd.DataFrame:
    """y_df for motor 1 — unit_number, time_in_cycles, RUL."""
    return motor_1_raw_df[['unit_number', 'time_in_cycles', 'RUL']].copy()


# ---------------------------------------------------------------------------
# Motor 1 — full pipeline output (Nodos 2-4)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def motor_1_pipeline_output(
    motor_1_X_df: pd.DataFrame,
    motor_1_y_df: pd.DataFrame,
) -> dict:
    """Runs RULPipeline (Nodos 2-4) on motor 1.

    Session-scoped — Nodo 3 (feature extraction) runs only once
    per test session regardless of how many test modules use this fixture.

    Returns:
        Dict with keys:
            X:      (n_windows, n_components) feature matrix
            y_rul:  (n_windows,) clipped RUL
            t_stop: (n_windows,) last cycle per window
            evento: (n_windows,) event indicator
            groups: (n_windows,) motor_id per window
            X_df:   original X DataFrame
            y_df:   original y DataFrame
            pipeline: fitted RULPipeline instance
    """
    pipeline = RULPipeline(
        window_size=20,
        clipping_threshold=125,
        n_components=5,
    )
    X, y_rul, t_stop, evento, groups = pipeline.fit_transform(
        motor_1_X_df, motor_1_y_df
    )

    return {
        'X':        X,
        'y_rul':    y_rul,
        't_stop':   t_stop,
        'evento':   evento,
        'groups':   groups,
        'X_df':     motor_1_X_df,
        'y_df':     motor_1_y_df,
        'pipeline': pipeline,
    }


# ---------------------------------------------------------------------------
# Multi-motor — for GroupKFold tests (needs >= 2 groups)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def multi_motor_data() -> dict:
    """Loads motors 1, 2, 3 for GroupKFold compatibility.

    GroupKFold(n_splits=2) requires at least 2 groups. This fixture
    provides 3 motors to ensure stable splits across folds.

    Returns:
        Dict with keys:
            X_df:   feature DataFrame (3 motors, no RUL)
            y_df:   target DataFrame (unit_number, time_in_cycles, RUL)
            groups: motor_id per row
    """
    dfs = []
    for motor_id in [1, 2, 3]:
        df = pd.read_csv(f'data/clean/data_motor_{motor_id}.csv')
        df.insert(0, 'unit_number', motor_id)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    return {
        'X_df':   df_all.drop(columns=['RUL']),
        'y_df':   df_all[['unit_number', 'time_in_cycles', 'RUL']].copy(),
        'groups': df_all['unit_number'].to_numpy(),
    }