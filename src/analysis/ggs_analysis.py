"""GGS results analysis functions for PredictaMaintenance.

Provides utilities to load, summarise, and inspect the output CSV files
produced by the Group Grid Search training pipeline. All functions operate
on a single model key at a time and print results directly to stdout for
use in Jupyter notebooks.

Typical usage::

    from src.analysis.ggs_analysis import get_results_resume, view_top_results

    get_results_resume('xgb')
    view_top_results('xgb')
"""

import pandas as pd

from src.analysis.config import GGS_RESULTS_PATHS, GGS_COLS, MODEL_NAMES

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_model_key(model_key: str) -> None:
    """Validates that model_key is a registered GGS model identifier.

    Args:
        model_key: Short model identifier (e.g. 'xgb', 'nb').

    Raises:
        ValueError: If model_key is not a string or not in the registry.
    """
    if not isinstance(model_key, str):
        raise ValueError(
            f"model_key must be a string, got {type(model_key).__name__}."
        )
    if model_key not in GGS_RESULTS_PATHS:
        valid = list(GGS_RESULTS_PATHS.keys())
        raise ValueError(
            f"Unknown model key '{model_key}'. Valid keys: {valid}."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_results(model_key: str) -> pd.DataFrame:
    """Loads the GGS results CSV for the given model.

    Args:
        model_key: Short model identifier (e.g. 'xgb', 'nb', 'cox').

    Returns:
        DataFrame containing all evaluated configurations and their
        cross-validated metrics.

    Raises:
        ValueError: If model_key is not a valid registered identifier.
    """
    _validate_model_key(model_key)
    return pd.read_csv(GGS_RESULTS_PATHS[model_key])


def get_results_resume(model_key: str) -> None:
    """Prints a summary of GGS results for the given model.

    Reports total configurations, successful evaluations, NaN failures,
    and duplicate counts. Duplicate detection uses the full set of
    hyperparameter columns defined in GGS_COLS for the model.

    Args:
        model_key: Short model identifier (e.g. 'xgb', 'nb', 'cox').

    Raises:
        ValueError: If model_key is not a valid registered identifier.
    """
    _validate_model_key(model_key)
    df = read_results(model_key)

    total        = len(df)
    exitosas     = int(df['mean_S_score'].notna().sum())
    fallidas     = int(df['mean_S_score'].isna().sum())
    n_duplicados = int(df.duplicated(subset=GGS_COLS[model_key]).sum())

    model_display = MODEL_NAMES.get(model_key, model_key)
    print(f"Model:                 {model_display}")
    print(f"Total configuraciones: {total}")
    print(f"Exitosas:              {exitosas}")
    print(f"Fallidas (NaN):        {fallidas}")
    print(f"Duplicados:            {n_duplicados}")
    print(f"Únicas:                {total - n_duplicados}")


def view_top_results(model_key: str, top_n: int = 10) -> None:
    """Displays the top-N configurations ranked by mean S-Score.

    Drops NaN rows before ranking. Output is rendered via IPython display
    for notebook compatibility.

    Args:
        model_key: Short model identifier (e.g. 'xgb', 'nb', 'cox').
        top_n: Number of top configurations to display. Defaults to 10.

    Raises:
        ValueError: If model_key is not a valid registered identifier.
    """
    _validate_model_key(model_key)
    df = read_results(model_key)

    df_top = (
        df
        .dropna(subset=['mean_S_score'])
        .sort_values('mean_S_score', ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )

    from IPython.display import display  # noqa: PLC0415
    display(df_top[GGS_COLS[model_key]])


def get_failed_configs(model_key: str) -> pd.DataFrame:
    """Returns all configurations that produced NaN metrics.

    Useful for diagnosing systematic failure patterns in the GGS —
    for example, identifying which hyperparameter combinations caused
    numerical failures or non-convergence.

    Args:
        model_key: Short model identifier.

    Returns:
        DataFrame containing only the failed configurations with all
        their hyperparameter columns.

    Raises:
        ValueError: If model_key is not a valid registered identifier.
    """
    _validate_model_key(model_key)
    df = read_results(model_key)
    return df[df['mean_S_score'].isna()].reset_index(drop=True)