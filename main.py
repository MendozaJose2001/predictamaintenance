"""PredictaMaintenance — CLI entry point for GGS model training.

Trains a RUL estimation model using Group Grid Search with GroupKFold
cross-validation over the C-MAPSS FD001 dataset.

Usage:
    python main.py --model nb     # NegativeBinomialPiecewise
    python main.py --model svr    # SVRModel
    python main.py --model dt     # DecisionTreeModel
    python main.py --model rf     # RandomForestModel

Results are saved automatically to outputs/ggs/results/.
If a previous run was interrupted, the GGS resumes from the last checkpoint.

Output files (outputs/ggs/):
    metadata/   — param_grid and run configuration
    results/    — final CSV with all configurations and metrics
    checkpoints/ — temporary (deleted on completion)
"""

import argparse
from pathlib import Path

import pandas as pd

from src.dataset_manager import DatasetManager
from src.ggs_training_manager import GGSTrainingManager
from src.models.negative_binomial import NegativeBinomialPiecewise
from src.models.svr_model import SVRModel
from src.models.decision_tree import DecisionTreeModel
from src.models.random_forest import RandomForestModel


# ---------------------------------------------------------------------------
# Param grids — one per model
# ---------------------------------------------------------------------------

_PARAM_GRID_NB = {
    'feature_set':        ['A', 'B', 'C', 'D'],
    'window_size':        [15, 20, 25, 30],
    'n_components':       [10, 15, 20],
    'clipping_threshold': [115, 120, 125, 130],
    'link_type':          ['log'],
    'alpha':              [0.1, 0.5, 1.0, 1.5],
    'alpha_reg':          [0.0, 0.1, 0.5],
    'l1_ratio':           [0.0, 0.5, 1.0],
}

_PARAM_GRID_SVR = {
    'feature_set':        ['A', 'B', 'C', 'D'],
    'window_size':        [15, 20, 25, 30],
    'n_components':       [10, 15, 20],
    'clipping_threshold': [115, 120, 125, 130],
    'kernel':             ['rbf', 'linear', 'poly'],
    'C':                  [0.1, 1.0, 10.0],
    'epsilon':            [0.01, 0.1],
    'gamma':              ['scale', 0.01],
    'degree':             [2, 3],
}

_PARAM_GRID_DT = {
    'feature_set':        ['A', 'B', 'C', 'D'],
    'window_size':        [15, 20, 25, 30],
    'n_components':       [10, 15, 20],
    'clipping_threshold': [115, 120, 125, 130],
    'max_depth':          [5, 10, None],
    'min_samples_leaf':   [1, 5, 10],
    'min_samples_split':  [2, 10],
    'max_features':       ['sqrt', 1.0],
}

_PARAM_GRID_RF = {
    'feature_set':        ['A', 'B', 'C', 'D'],
    'window_size':        [15, 20, 25, 30],
    'n_components':       [10, 15, 20],
    'clipping_threshold': [115, 120, 125, 130],
    'n_estimators':       [50, 100, 200],
    'max_depth':          [5, 10, None],
    'min_samples_leaf':   [1, 5, 10],
    'max_features':       ['sqrt', 1.0],
}

_MODELS: dict = {
    'nb':  (NegativeBinomialPiecewise, _PARAM_GRID_NB),
    'svr': (SVRModel,                  _PARAM_GRID_SVR),
    'dt':  (DecisionTreeModel,         _PARAM_GRID_DT),
    'rf':  (RandomForestModel,         _PARAM_GRID_RF),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='PredictaMaintenance — GGS model training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model nb
  python main.py --model svr
  python main.py --model dt --jobs 2
  python main.py --model rf --jobs 3
  python main.py --model nb --folds 3 --top 15
        """,
    )
    parser.add_argument(
        '--model',
        choices=list(_MODELS.keys()),
        required=True,
        help='Model to train: nb, svr, dt, rf',
    )
    parser.add_argument(
        '--folds',
        type=int,
        default=5,
        help='Number of GroupKFold folds (default: 5)',
    )
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='Number of top results to display (default: 10)',
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=1,
        help='Parallel workers: 1=sequential (default), -1=all cores, 2=two cores',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run with reduced param_grid (2 configs) for quick validation',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data() -> tuple:
    """Loads all 200 motors and splits into train/test via DatasetManager."""
    print("\nLoading dataset...")
    m_train, m_test = DatasetManager.split_dataset()

    dfs = []
    for idx in m_train:
        df = pd.read_csv(f'data/clean/data_motor_{idx}.csv')
        df.insert(0, 'unit_number', idx)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    X_df   = df_all.drop(columns=['RUL'])
    y_df   = df_all[['unit_number', 'time_in_cycles', 'RUL']].copy()
    groups = df_all['unit_number'].to_numpy()

    print(f"  Training motors: {X_df['unit_number'].nunique()}")
    print(f"  Total rows:      {len(X_df)}")
    return X_df, y_df, groups


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    model_class, param_grid = _MODELS[args.model]

    # Debug mode — minimal grid for quick validation
    if args.debug:
        param_grid = {k: v[:1] if isinstance(v, list) else [v]
                      for k, v in param_grid.items()}
        param_grid['feature_set'] = ['A', 'B']
        args.folds = 2
        print("\n  ⚡ DEBUG MODE — reduced grid, 2 folds")

    # Summary
    n_configs = 1
    for v in param_grid.values():
        n_configs *= len(v)

    print(f"\n{'='*55}")
    print(f"  PredictaMaintenance — GGS Training")
    print(f"{'='*55}")
    print(f"  Model:   {model_class.__name__}")
    print(f"  Configs: {n_configs} × {args.folds} folds = {n_configs * args.folds} fits")
    print(f"  Output:  outputs/ggs/results/")

    # Load data
    X_df, y_df, groups = _load_data()

    # Run GGS
    manager = GGSTrainingManager(
        model_class=model_class,
        X_df=X_df,
        y_df=y_df,
        groups=groups,
    )

    manager.group_grid_search(
        param_grid=param_grid,
        n_folds=args.folds,
        silence=False,
        checkpoint_every=1,
        base_dir=Path('outputs/ggs'),
        n_jobs=args.jobs,
    )

    # Display top results
    results_df = manager.get_ggs_results(top_n=args.top)

    print(f"\n{'='*55}")
    print(f"  TOP {args.top} CONFIGURATIONS")
    print(f"{'='*55}")
    print(results_df.to_string(index=False))
    print()


if __name__ == '__main__':
    main()