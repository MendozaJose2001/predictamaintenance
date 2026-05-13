"""Sanity check script for CoxFrailty — motor único y simulación de fold GGS.

Verifica el comportamiento real del modelo CoxFrailty con datos reales
del pipeline, replicando las mismas pruebas realizadas para CoxPHModel
y WeibullAFTModel.

Usage:
    python scripts/test_cox_frailty_sanity.py

Output:
    - Resultados impresos en consola
    - Gráficas de S(t|X) y F(t|X) para motor único y fold GGS
    - Métricas del fold con comparación vs CoxPH estándar
"""

import warnings
import sys
from pathlib import Path

# Añadir raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')

from src.pipeline.rul_pipeline import RULPipeline
from src.models.cox_frailty import CoxFrailty
from src.dataset_manager import DatasetManager
from src.metrics_manager import Metrics


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

MOTOR_ID     = 1
WINDOW_SIZE  = 20
N_COMP       = 10
CLIP         = 125
CONF_THRESH  = 0.5
DISTRIBUTION = 'gamma'
METHOD       = 'em'
TDF          = 5
N_MOTORES    = 20
N_FOLDS      = 3

SEPARATOR = '=' * 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print(f'\n{SEPARATOR}')
    print(f'  {title}')
    print(SEPARATOR)


def plot_survival_curves(
    sf: list[tuple[np.ndarray, np.ndarray]],
    dc: list[tuple[np.ndarray, np.ndarray]],
    t_stop_sel: np.ndarray,
    y_rul_sel: np.ndarray,
    labels: list[str],
    title: str,
    conf_thresh: float,
) -> None:
    """Plots S(t|X) and F(t|X) for selected windows."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, label in enumerate(labels):
        times_sf, probs_sf = sf[i]
        times_dc, probs_dc = dc[i]
        lbl = f'{label} | t={t_stop_sel[i]:.0f}, RUL={y_rul_sel[i]:.0f}'
        axes[0].plot(times_sf, probs_sf, label=lbl)
        axes[1].plot(times_dc, probs_dc, label=lbl)

    axes[0].axhline(1 - conf_thresh, color='red', linestyle='--',
                    label=f'1-conf={1-conf_thresh:.1f}')
    axes[1].axhline(conf_thresh, color='red', linestyle='--',
                    label=f'conf={conf_thresh:.1f}')

    for ax, ylabel, subtitle in zip(
        axes,
        ['S(t)', 'F(t)'],
        ['S(t|X)', 'F(t|X) = 1 - S(t|X)'],
    ):
        ax.set_title(subtitle)
        ax.set_xlabel('Ciclo (t)')
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_rul_trajectory(
    t_stop: np.ndarray,
    y_rul: np.ndarray,
    rul_pred: np.ndarray,
    title: str,
) -> None:
    """Plots predicted vs real RUL trajectory."""
    valid = ~np.isnan(rul_pred)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_stop, y_rul, 'k-', linewidth=2, label='RUL real (clipped)')

    if valid.any():
        ax.plot(t_stop[valid], rul_pred[valid], 'b--',
                linewidth=1.5, label='RUL predicho (CoxFrailty)')
    else:
        ax.text(0.5, 0.5,
                'S(t) ≈ 1 siempre\nF(t) nunca alcanza conf_thresh\n'
                '(resultado negativo esperado)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='red',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax.set_title(title)
    ax.set_xlabel('Ciclo (t_stop)')
    ax.set_ylabel('RUL (ciclos)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Bloque 1 — Motor único
# ---------------------------------------------------------------------------

def run_single_motor() -> None:
    print_section(f'BLOQUE 1 — Motor único (Motor {MOTOR_ID})')

    # Cargar datos
    df = pd.read_csv(f'data/clean/data_motor_{MOTOR_ID}.csv')
    df.insert(0, 'unit_number', MOTOR_ID)
    X_df = df.drop(columns=['RUL'])
    y_df = df[['unit_number', 'time_in_cycles', 'RUL']].copy()

    print(f"\nMotor {MOTOR_ID}: {len(df)} ciclos, "
          f"RUL final = {df['RUL'].iloc[-1]}")
    print(f"Eventos (fallos): {df['evento'].sum()}")

    # Pipeline
    pipeline = RULPipeline(
        window_size=WINDOW_SIZE,
        clipping_threshold=CLIP,
        n_components=N_COMP,
        feature_set='B',
    )
    X, y_rul, t_stop, evento, groups = pipeline.fit_transform(X_df, y_df)

    print(f"\nPipeline output:")
    print(f"  Ventanas:      {X.shape[0]}")
    print(f"  Componentes:   {X.shape[1]}")
    print(f"  Eventos:       {evento.sum()} ({100*evento.mean():.2f}%)")
    print(f"  t_stop rango:  [{t_stop.min():.0f}, {t_stop.max():.0f}]")
    print(f"  groups únicos: {np.unique(groups)}")

    # Fit
    print(f"\nFitting CoxFrailty(distribution='{DISTRIBUTION}', "
          f"method='{METHOD}', tdf={TDF})...")
    model = CoxFrailty(
        distribution=DISTRIBUTION,
        method=METHOD,
        tdf=TDF,
        maxit=300,
        confidence_threshold=CONF_THRESH,
        clipping_threshold=CLIP,
    )
    model.fit(X, y_rul, t_stop=t_stop, evento=evento, groups=groups)

    print(f"is_fitted_:  {model.is_fitted_}")
    print(f"model_name_: {model.model_name_}")

    if not model.is_fitted_:
        print("\n[RESULTADO ESPERADO] Motor único no converge — "
              "frailty requiere varianza entre grupos (≥2 motores).")
        return

    # Curvas de supervivencia
    n = X.shape[0]
    idx_plot  = [0, n//4, n//2, 3*n//4, n-1]
    labels    = ['Temprana', '25%', '50%', '75%', 'Última (fallo)']
    X_sel     = X[idx_plot]
    t_sel     = t_stop[idx_plot]
    y_sel     = y_rul[idx_plot]

    sf = model.predict_survival_function(X_sel)
    dc = model.predict_death_curve(X_sel)

    if sf is None or dc is None:
        print("[WARN] predict_survival_function retornó None.")
        return

    all_probs_sf = np.concatenate([p for _, p in sf])
    all_probs_dc = np.concatenate([p for _, p in dc])
    print(f"\nRango S(t): [{all_probs_sf.min():.4f}, {all_probs_sf.max():.4f}]")
    print(f"Rango F(t): [{all_probs_dc.min():.4f}, {all_probs_dc.max():.4f}]")
    print(f"¿F(t) alcanza conf_thresh={CONF_THRESH}? "
          f"{(all_probs_dc >= CONF_THRESH).any()}")

    plot_survival_curves(
        sf, dc, t_sel, y_sel, labels,
        title=(f'CoxFrailty — Motor {MOTOR_ID} | '
               f'{DISTRIBUTION} | method={METHOD} | conf={CONF_THRESH}'),
        conf_thresh=CONF_THRESH,
    )

    # Trayectoria RUL
    rul_pred = model.predict_with_time(X, t_stop)
    valid    = ~np.isnan(rul_pred)
    print(f"\nPredicciones válidas: {valid.sum()}/{len(rul_pred)}")
    if valid.any():
        print(f"RUL predicho rango: [{rul_pred[valid].min():.1f}, "
              f"{rul_pred[valid].max():.1f}]")

    plot_rul_trajectory(
        t_stop, y_rul, rul_pred,
        title=(f'Trayectoria RUL — Motor {MOTOR_ID} | '
               f'CoxFrailty {DISTRIBUTION} | conf={CONF_THRESH}'),
    )


# ---------------------------------------------------------------------------
# Bloque 2 — Simulación de fold GGS
# ---------------------------------------------------------------------------

def run_fold_simulation() -> None:
    print_section(f'BLOQUE 2 — Simulación fold GGS ({N_MOTORES} motores)')

    # Cargar datos
    m_train, _ = DatasetManager.split_dataset()
    motores    = m_train[:N_MOTORES]

    dfs = []
    for idx in motores:
        df = pd.read_csv(f'data/clean/data_motor_{idx}.csv')
        df.insert(0, 'unit_number', idx)
        dfs.append(df)

    df_all   = pd.concat(dfs, ignore_index=True)
    X_df_all = df_all.drop(columns=['RUL'])
    y_df_all = df_all[['unit_number', 'time_in_cycles', 'RUL']].copy()
    groups   = df_all['unit_number'].to_numpy()

    print(f"\nMotores cargados: {df_all['unit_number'].nunique()}")
    print(f"Total filas:      {len(df_all)}")
    print(f"Eventos totales:  {df_all['evento'].sum()} "
          f"({100*df_all['evento'].mean():.2f}%)")

    # Fold 0
    gkf = GroupKFold(n_splits=N_FOLDS)
    train_idx, val_idx = list(gkf.split(X_df_all, groups=groups))[0]

    X_train = X_df_all.iloc[train_idx].reset_index(drop=True)
    X_val   = X_df_all.iloc[val_idx].reset_index(drop=True)
    y_train = y_df_all.iloc[train_idx].reset_index(drop=True)
    y_val   = y_df_all.iloc[val_idx].reset_index(drop=True)

    pipeline = RULPipeline(
        window_size=WINDOW_SIZE,
        clipping_threshold=CLIP,
        n_components=N_COMP,
        feature_set='B',
    )
    X_tr, y_rul_tr, t_stop_tr, evento_tr, groups_tr = (
        pipeline.fit_transform(X_train, y_train)
    )
    X_vl, y_rul_vl, t_stop_vl, evento_vl, groups_vl = (
        pipeline.transform(X_val, y_val)
    )

    print(f"\nFold 0:")
    print(f"  Train: {X_train['unit_number'].nunique()} motores, "
          f"{X_tr.shape[0]} ventanas, {evento_tr.sum()} eventos "
          f"({100*evento_tr.mean():.2f}%)")
    print(f"  Val:   {X_val['unit_number'].nunique()} motores, "
          f"{X_vl.shape[0]} ventanas, {evento_vl.sum()} eventos "
          f"({100*evento_vl.mean():.2f}%)")

    # Fit
    print(f"\nFitting CoxFrailty(distribution='{DISTRIBUTION}', "
          f"method='{METHOD}', tdf={TDF})...")
    model = CoxFrailty(
        distribution=DISTRIBUTION,
        method=METHOD,
        tdf=TDF,
        maxit=300,
        confidence_threshold=CONF_THRESH,
        clipping_threshold=CLIP,
    )
    model.fit(
        X_tr, y_rul_tr,
        t_stop=t_stop_tr,
        evento=evento_tr,
        groups=groups_tr,
    )
    print(f"is_fitted_: {model.is_fitted_}")

    if not model.is_fitted_:
        print("\n[RESULTADO NEGATIVO] Modelo no convergió en fold 0.")
        return

    # Curvas en validación — primer motor de val
    motor_val = X_val['unit_number'].unique()[0]
    mask      = groups_vl == motor_val
    X_m       = X_vl[mask]
    t_m       = t_stop_vl[mask]
    y_m       = y_rul_vl[mask]

    print(f"\nMotor de validación: {motor_val} | {mask.sum()} ventanas")

    n_m      = X_m.shape[0]
    idx_plot = [0, n_m//4, n_m//2, 3*n_m//4, n_m-1]
    labels   = ['Temprana', '25%', '50%', '75%', 'Última']

    sf_v = model.predict_survival_function(X_m[idx_plot])
    dc_v = model.predict_death_curve(X_m[idx_plot])

    if sf_v is None or dc_v is None:
        print("[WARN] predict_survival_function retornó None.")
        return

    all_dc = np.concatenate([p for _, p in dc_v])
    print(f"Rango F(t) val: [{all_dc.min():.4f}, {all_dc.max():.4f}]")
    print(f"¿F(t) alcanza conf={CONF_THRESH}? {(all_dc >= CONF_THRESH).any()}")

    plot_survival_curves(
        sf_v, dc_v,
        t_stop_sel=t_m[idx_plot],
        y_rul_sel=y_m[idx_plot],
        labels=labels,
        title=(f'CoxFrailty Fold 0 — Motor val {motor_val} | '
               f'{DISTRIBUTION} | method={METHOD} | conf={CONF_THRESH}'),
        conf_thresh=CONF_THRESH,
    )

    # Métricas
    rul_pred_vl = model.predict_with_time(X_vl, t_stop_vl)
    valid_mask  = ~np.isnan(rul_pred_vl)

    print(f"\nPredicciones válidas: {valid_mask.sum()}/{len(rul_pred_vl)}")

    if valid_mask.sum() > 1:
        mae     = Metrics.mae(rul_pred_vl[valid_mask], y_rul_vl[valid_mask], CLIP)
        rmse    = Metrics.rmse(rul_pred_vl[valid_mask], y_rul_vl[valid_mask], CLIP)
        s_score = Metrics.s_score(rul_pred_vl[valid_mask], y_rul_vl[valid_mask], CLIP)
        c_index = Metrics.c_index(rul_pred_vl[valid_mask], y_rul_vl[valid_mask], CLIP)

        print(f"\nMétricas fold 0 ({valid_mask.sum()} ventanas válidas):")
        print(f"  MAE:     {mae:.2f}")
        print(f"  RMSE:    {rmse:.2f}")
        print(f"  S-Score: {s_score:.2f}")
        print(f"  C-Index: {c_index:.4f}")
        print(f"\n  Referencia CoxPH estándar: MAE≈25, S-Score≈459, C-Index≈0.66")
        print(f"  Referencia CoxFrailty (gamma/em): MAE≈24, S-Score≈430, C-Index≈0.67")
        print(f"  Configuración actual: distribution='{DISTRIBUTION}', method='{METHOD}', tdf={TDF}")
    else:
        print("\n[RESULTADO NEGATIVO] S(t)≈1 siempre — F(t) no alcanza umbral.")
        print("  Consistente con CoxPHModel y WeibullAFTModel en C-MAPSS FD001.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"\n{'#'*60}")
    print(f"  CoxFrailty Sanity Check")
    print(f"  distribution={DISTRIBUTION} | method={METHOD} | tdf={TDF}")
    print(f"  conf_thresh={CONF_THRESH} | window_size={WINDOW_SIZE}")
    print(f"{'#'*60}")

    run_single_motor()
    run_fold_simulation()

    print(f"\n{'#'*60}")
    print(f"  Sanity check completado")
    print(f"{'#'*60}\n")