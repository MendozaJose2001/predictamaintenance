"""Visualisation functions for test set analysis in PredictaMaintenance.

Provides trajectory plots, error plots, and per-motor boxplots for the
comparative evaluation of eligible production models on the C-MAPSS FD001
test set. All functions retrieve data internally via test_analysis helpers,
keeping notebook cells free of data preparation logic.

Typical usage::

    from src.analysis.plot_analysis import (
        plot_per_motor_boxplots,
        plot_trayectorias_absolutas,
        plot_trayectorias_error,
    )

    plot_per_motor_boxplots()
    plot_trayectorias_absolutas()
    plot_trayectorias_error()
"""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.artist import Artist
import matplotlib.pyplot as plt

from src.analysis.config import ELIGIBLE_MODELS, MODEL_NAMES, MODEL_STYLE
from src.analysis.test_analysis import (
    get_trajectory_predict,
    get_test_motors,
    test_per_motor,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_representative_motors(
    df_test: pd.DataFrame,
    m_test: list[int],
    quantiles: tuple[float, float, float] = (0.10, 0.50, 0.90),
) -> dict[str, int]:
    """Selects three representative motors by RUL_max within their test cluster.

    Only motors with evento=1 (observed failure) are considered, ensuring
    that y_true is fully known rather than reconstructed from the ground
    truth file under uncertainty.

    RUL_max is defined as the maximum RUL value within the motor's test
    cluster — i.e., the RUL at the first observed window. This captures
    how much life remained when the motor entered the test set, which
    determines the difficulty of the prediction task.

    Args:
        df_test: Concatenated test set DataFrame with unit_number column.
        m_test: Ordered list of test motor IDs.
        quantiles: Tuple of (low, mid, high) quantiles for motor selection.

    Returns:
        Dict mapping descriptive label to motor ID for the three selected
        motors (short, medium, long life).
    """
    motors_with_failure = [
        mid for mid in m_test
        if df_test[df_test['unit_number'] == mid]['evento'].max() == 1
    ]

    rul_max: dict[int, float] = {
        mid: float(df_test[df_test['unit_number'] == mid]['RUL'].max())
        for mid in motors_with_failure
    }
    rul_values = pd.Series(rul_max, dtype=float)

    q_low, q_mid, q_high = quantiles
    motor_corto = int(rul_values.index[
        int((rul_values - float(rul_values.quantile(q_low))).abs().argmin())
    ])
    motor_medio = int(rul_values.index[
        int((rul_values - float(rul_values.quantile(q_mid))).abs().argmin())
    ])
    motor_largo = int(rul_values.index[
        int((rul_values - float(rul_values.quantile(q_high))).abs().argmin())
    ])

    return {
        f'Motor corto  (RUL_max={rul_max[motor_corto]:.0f})': motor_corto,
        f'Motor medio  (RUL_max={rul_max[motor_medio]:.0f})': motor_medio,
        f'Motor largo  (RUL_max={rul_max[motor_largo]:.0f})': motor_largo,
    }


def _load_plot_data(
    debug: bool = False,
) -> tuple[dict, pd.DataFrame, dict[str, int]]:
    """Loads results, test data, and selects representative motors.

    Args:
        debug: If True, prints selected motor details.

    Returns:
        Tuple of (results, df_test, motores_plot).
    """
    results, _, m_test = get_trajectory_predict(debug=False)
    df_test, _         = get_test_motors(debug=False)
    motores_plot       = _select_representative_motors(df_test, m_test)

    if debug:
        print(f"Motores con fallo observado seleccionados:")
        for label, mid in motores_plot.items():
            print(f"  {label} — ID: {mid}")

    return results, df_test, motores_plot


def _add_shared_legend(
    fig: Figure,
    axes: np.ndarray,
    ncol: int = 6,
    bbox: tuple[float, float] = (0.5, -0.08),
) -> None:
    """Adds a deduplicated shared legend below the figure.

    Args:
        fig: The matplotlib Figure object.
        axes: Array of Axes from which to extract handles and labels.
        ncol: Number of columns in the legend. Defaults to 6.
        bbox: Anchor position for the legend. Defaults to (0.5, -0.08).
    """
    handles, labels = axes[0].get_legend_handles_labels()
    seen: dict[str, Artist] = {}
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen[label] = handle  # type: ignore[assignment]
    fig.legend(
        seen.values(), seen.keys(),
        loc='lower center', ncol=ncol,
        fontsize=10, framealpha=0.9,
        bbox_to_anchor=bbox,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_per_motor_boxplots(debug: bool = False) -> None:
    """Renders S-Score and C-Index boxplots across the 60 test motors.

    Calls test_per_motor to compute per-motor metrics, then renders
    side-by-side boxplots for S-Score and C-Index. NaN values (motors
    with undefined C-Index due to constant y_true) are filtered before
    plotting. Models are ordered by ascending S-Score median.

    Args:
        debug: Passed to get_trajectory_predict for verbose loading output.
    """
    df_per_motor, ss_per_motor, ci_per_motor = test_per_motor(debug)

    order     = df_per_motor['Modelo'].tolist()
    color_map = {
        MODEL_NAMES[k]: MODEL_STYLE[k]['color']
        for k in ELIGIBLE_MODELS
    }

    keys_ordered = sorted(
        ss_per_motor.keys(),
        key=lambda k: df_per_motor[
            df_per_motor['Modelo'] == MODEL_NAMES[k]
        ].index[0],
    )
    labels_ordered = [MODEL_NAMES[k] for k in keys_ordered]
    cols_ordered   = [color_map[l] for l in labels_ordered]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # S-Score boxplot
    data_ss = [
        [v for v in ss_per_motor[k] if not np.isnan(v)]
        for k in keys_ordered
    ]
    bp1 = axes[0].boxplot(
        data_ss, patch_artist=True, notch=False,
        medianprops=dict(color='black', linewidth=2),
    )
    for patch, col in zip(bp1['boxes'], cols_ordered):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    axes[0].set_xticks(range(1, len(labels_ordered) + 1))
    axes[0].set_xticklabels(labels_ordered, rotation=20, ha='right', fontsize=9)
    axes[0].set_title('S-Score por motor (↓ mejor)', fontweight='bold')
    axes[0].set_ylabel('S-Score')
    axes[0].grid(True, alpha=0.3, axis='y')

    # C-Index boxplot
    data_ci = [
        [v for v in ci_per_motor[k] if not np.isnan(v)]
        for k in keys_ordered
    ]
    bp2 = axes[1].boxplot(
        data_ci, patch_artist=True, notch=False,
        medianprops=dict(color='black', linewidth=2),
    )
    for patch, col in zip(bp2['boxes'], cols_ordered):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    axes[1].set_xticks(range(1, len(labels_ordered) + 1))
    axes[1].set_xticklabels(labels_ordered, rotation=20, ha='right', fontsize=9)
    axes[1].set_title('C-Index por motor (↑ mejor)', fontweight='bold')
    axes[1].set_ylabel('C-Index')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        'Subanálisis 3 — Distribución del error por motor | Test set (60 motores)',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    plt.show()


def plot_trayectorias_absolutas(
    debug: bool = False,
    log_scale: bool = False,
) -> None:
    """Renders absolute RUL trajectories for three representative test motors.

    Plots predicted vs real RUL over relative cycle index for each eligible
    model. Only motors with observed failure (evento=1) are considered for
    selection, ensuring y_true is fully known.

    Args:
        debug: If True, prints selected motor details.
        log_scale: If True, applies logarithmic scale to the Y axis.
            Useful when predicted curves overlap heavily in linear scale.
    """
    results, df_test, motores_plot = _load_plot_data(debug)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, (titulo, motor_id) in zip(axes, motores_plot.items()):
        mask_ref = results['xgb']['groups'] == motor_id
        y_true_m = results['xgb']['y_true'][mask_ref]
        n_win    = mask_ref.sum()
        ciclos   = np.arange(n_win)

        ax.plot(ciclos, y_true_m,
                color='black', lw=2.5, ls='-', label='RUL real', zorder=5)

        for key, style in MODEL_STYLE.items():
            mask_k   = results[key]['groups'] == motor_id
            y_pred_k = results[key]['y_pred'][mask_k]
            ax.plot(ciclos, y_pred_k,
                    color=style['color'], ls=style['ls'],
                    lw=style['lw'], label=style['label'], zorder=4)

        if log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('RUL (ciclos, escala log)')
        else:
            ax.set_ylim(bottom=0)
            ax.set_ylabel('RUL (ciclos)')

        ax.set_title(titulo, fontsize=10, fontweight='bold')
        ax.set_xlabel('Ciclo relativo (ventana)')
        ax.grid(True, alpha=0.3)

    _add_shared_legend(fig, axes, ncol=6)
    plt.suptitle(
        'Trayectorias de RUL — Test set | Todos los modelos vs RUL real\n'
        'Solo motores con fallo observado (evento=1)',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    plt.show()


def plot_trayectorias_error(
    debug: bool = False,
    log_scale: bool = False,
) -> None:
    """Renders prediction error trajectories for three representative motors.

    Plots error = y_pred - y_true over relative cycle index for each model.
    Positive error indicates overestimation (dangerous — motor closer to
    failure than predicted). A ±10-cycle reference band is shown.

    Args:
        debug: If True, prints selected motor details.
        log_scale: If True, applies symmetric logarithmic scale to the Y axis.
            Useful when error curves overlap heavily in linear scale.
    """
    results, df_test, motores_plot = _load_plot_data(debug)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, (titulo, motor_id) in zip(axes, motores_plot.items()):
        mask_ref = results['xgb']['groups'] == motor_id
        n_win    = mask_ref.sum()
        ciclos   = np.arange(n_win)

        ax.axhline(0, color='black', lw=2.0, ls='-',
                   label='Error = 0 (perfecto)', zorder=5)
        ax.axhspan(-10, 10, alpha=0.06, color='green',
                   label='Zona ±10 ciclos')

        for key, style in MODEL_STYLE.items():
            mask_k   = results[key]['groups'] == motor_id
            y_pred_k = results[key]['y_pred'][mask_k]
            y_true_k = results[key]['y_true'][mask_k]
            error_k  = y_pred_k - y_true_k
            ax.plot(ciclos, error_k,
                    color=style['color'], ls=style['ls'],
                    lw=style['lw'], label=style['label'], zorder=4)

        if log_scale:
            ax.set_yscale('symlog', linthresh=10)
            ax.set_ylabel('Error (ciclos, escala symlog)')
        else:
            ax.set_ylabel('Error (ciclos)  [+ sobreestima | − subestima]')

        ax.set_title(titulo, fontsize=10, fontweight='bold')
        ax.set_xlabel('Ciclo relativo (ventana)')
        ax.grid(True, alpha=0.3)

    _add_shared_legend(fig, axes, ncol=7, bbox=(0.5, -0.10))
    plt.suptitle(
        'Error de predicción por ciclo relativo — Test set\n'
        'Error = RUL predicho − RUL real  |  Positivo = sobreestima (peligroso)',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    plt.show()