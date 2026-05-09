#./src/utils/validate_regressor.py

"""Reusable validation script for RUL regressor models.

Validates any BaseRULModel regressor under two scenarios:

    Scenario A — Full training set (no CV):
        Fits the model on all training motors and reports predictions
        on the same set. Useful for checking convergence and basic
        behaviour before running a full GGS.

    Scenario B — GroupKFold (n_folds=5):
        Simulates the exact GGS evaluation loop. Fits on train folds
        and evaluates on val folds. Reports per-fold and mean metrics.

Supported models (--model):
    dt   → DecisionTreeModel
    rf   → RandomForestModel  (when implemented)
    nb   → NegativeBinomialPiecewise
    svr  → SVRModel

Usage (from project root):
    python -m src.utils.validate_regressor --model dt
    python -m src.utils.validate_regressor --model dt --kfold
    python -m src.utils.validate_regressor --model dt --kfold --folds 3
    python -m src.utils.validate_regressor --model dt --window-size 30 --feature-set B
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.dataset_manager import DatasetManager
from src.pipeline.rul_pipeline import RULPipeline
from src.metrics_manager import Metrics


# ---------------------------------------------------------------------------
# Model registry — add new models here
# ---------------------------------------------------------------------------

def _get_model(name: str, clipping_threshold: int):
    """Returns an unfitted model instance for the given name."""
    if name == 'rf':
        from src.models.random_forest import RandomForestModel
        return RandomForestModel(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            clipping_threshold=clipping_threshold,
        )
    if name == 'dt':
        from src.models.decision_tree import DecisionTreeModel
        return DecisionTreeModel(
            max_depth=5,
            min_samples_leaf=10,
            clipping_threshold=clipping_threshold,
        )
    if name == 'nb':
        from src.models.negative_binomial import NegativeBinomialPiecewise
        return NegativeBinomialPiecewise(
            clipping_threshold=clipping_threshold,
        )
    if name == 'svr':
        from src.models.svr_model import SVRModel
        return SVRModel(
            clipping_threshold=clipping_threshold,
        )
    raise ValueError(f"Modelo desconocido: '{name}'. Disponibles: dt, nb, svr")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Validación de regresor RUL — un motor o GroupKFold',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python -m src.utils.validate_regressor --model dt
  python -m src.utils.validate_regressor --model dt --kfold
  python -m src.utils.validate_regressor --model nb --kfold --folds 3
        """,
    )
    parser.add_argument(
        '--model', required=True,
        choices=['dt', 'rf', 'nb', 'svr'],
        help='Modelo a validar',
    )
    parser.add_argument(
        '--kfold', action='store_true',
        help='Correr GroupKFold en lugar de entrenamiento completo',
    )
    parser.add_argument(
        '--folds', type=int, default=5,
        help='Número de folds GroupKFold (default: 5)',
    )
    parser.add_argument(
        '--window-size', type=int, default=30,
        help='Tamaño de ventana (default: 30)',
    )
    parser.add_argument(
        '--n-components', type=int, default=10,
        help='Componentes PCA (default: 10)',
    )
    parser.add_argument(
        '--clipping-threshold', type=int, default=125,
        help='Umbral de clipping RUL (default: 125)',
    )
    parser.add_argument(
        '--feature-set', default='B',
        choices=['A', 'B', 'C', 'D'],
        help='Conjunto de features (default: B)',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Carga los motores de entrenamiento."""
    m_train, _ = DatasetManager.split_dataset()
    dfs = []
    for idx in m_train:
        df = pd.read_csv(f'data/clean/data_motor_{idx}.csv')
        df.insert(0, 'unit_number', idx)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    X_df   = df_all.drop(columns=['RUL'])
    y_df   = df_all[['unit_number', 'time_in_cycles', 'RUL']].copy()
    groups = df_all['unit_number'].to_numpy()
    return X_df, y_df, groups


def _report_metrics(y_pred: np.ndarray, y_true: np.ndarray, clip: int) -> dict:
    """Calcula y muestra las 4 métricas estándar."""
    results = {}
    for name, func in Metrics.get_metrics().items():
        try:
            val = float(func(y_pred, y_true, clip))
        except Exception:
            val = np.nan
        results[name] = val
        print(f"    {name:<12} {val:.4f}")
    return results


# ---------------------------------------------------------------------------
# Scenario A — Full training set
# ---------------------------------------------------------------------------

def _validate_full(args: argparse.Namespace) -> None:
    _sep("Scenario A — Entrenamiento completo (140 motores)")

    X_df, y_df, _ = _load_data()
    print(f"  Motores: {X_df['unit_number'].nunique()} | Filas: {len(X_df)}")

    pipeline = RULPipeline(
        window_size=args.window_size,
        n_components=args.n_components,
        clipping_threshold=args.clipping_threshold,
        feature_set=args.feature_set,
    )
    X, y_rul, t_stop, evento, _ = pipeline.fit_transform(X_df, y_df)
    print(f"  Ventanas: {len(X)} | PCA components: {X.shape[1]}")

    model = _get_model(args.model, args.clipping_threshold)
    print(f"\n  Fitting {model.__class__.__name__}...")
    model.fit(X, y_rul, t_stop=t_stop, evento=evento)
    print(f"  is_fitted_: {model.is_fitted_}")

    if not model.is_fitted_:
        print("  ❌ El modelo no convergió.")
        return

    y_pred = model.predict(X)
    n_nan = int(np.isnan(y_pred).sum())

    print(f"\n  Predicciones NaN: {n_nan}/{len(y_pred)} ({100*n_nan/len(y_pred):.1f}%)")
    print(f"\n  Métricas (train set completo):")
    _report_metrics(y_pred, y_rul, args.clipping_threshold)


# ---------------------------------------------------------------------------
# Scenario B — GroupKFold
# ---------------------------------------------------------------------------

def _validate_kfold(args: argparse.Namespace) -> None:
    _sep(f"Scenario B — GroupKFold ({args.folds} folds)")

    X_df, y_df, groups = _load_data()
    print(f"  Motores: {X_df['unit_number'].nunique()} | Filas: {len(X_df)}")

    gkf = GroupKFold(n_splits=args.folds)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(X_df, groups=groups), start=1
    ):
        print(f"\n  --- Fold {fold_idx}/{args.folds} ---")

        X_tr_df = X_df.iloc[train_idx].reset_index(drop=True)
        X_vl_df = X_df.iloc[val_idx].reset_index(drop=True)
        y_tr_df = y_df.iloc[train_idx].reset_index(drop=True)
        y_vl_df = y_df.iloc[val_idx].reset_index(drop=True)

        print(f"  Train: {X_tr_df['unit_number'].nunique()} motores | "
              f"Val: {X_vl_df['unit_number'].nunique()} motores")

        pipeline = RULPipeline(
            window_size=args.window_size,
            n_components=args.n_components,
            clipping_threshold=args.clipping_threshold,
            feature_set=args.feature_set,
        )
        X_tr, y_rul_tr, t_stop_tr, evento_tr, _ = pipeline.fit_transform(
            X_tr_df, y_tr_df
        )
        X_vl, y_rul_vl, t_stop_vl, evento_vl, _ = pipeline.transform(
            X_vl_df, y_vl_df
        )

        model = _get_model(args.model, args.clipping_threshold)
        model.fit(X_tr, y_rul_tr, t_stop=t_stop_tr, evento=evento_tr)

        if not model.is_fitted_:
            print(f"  ❌ Fold {fold_idx} no convergió")
            fold_results.append({
                'fold': fold_idx, 'converged': False,
                'MAE': np.nan, 'RMSE': np.nan,
                'S_score': np.nan, 'C_index': np.nan,
            })
            continue

        y_pred = model.predict(X_vl)
        print(f"  Métricas fold {fold_idx}:")
        metrics = _report_metrics(y_pred, y_rul_vl, args.clipping_threshold)
        fold_results.append({'fold': fold_idx, 'converged': True, **metrics})

    # Resumen
    _sep("Resumen GroupKFold")
    df_res = pd.DataFrame(fold_results)
    print(df_res.to_string(index=False))

    df_ok = df_res[df_res['converged']]
    if len(df_ok) > 0:
        print(f"\n  Folds convergidos: {len(df_ok)}/{args.folds}")
        print(f"\n  Medias:")
        for col in ['MAE', 'RMSE', 'S_score', 'C_index']:
            print(f"    {col:<12} {df_ok[col].mean():.4f} ± {df_ok[col].std():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    print(f"\n  Modelo:          {args.model.upper()}")
    print(f"  Feature set:     {args.feature_set}")
    print(f"  Window size:     {args.window_size}")
    print(f"  N components:    {args.n_components}")
    print(f"  Clipping:        {args.clipping_threshold}")

    if args.kfold:
        _validate_kfold(args)
    else:
        _validate_full(args)

    print()


if __name__ == '__main__':
    main()