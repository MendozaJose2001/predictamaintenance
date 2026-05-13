"""R repository module for Cox proportional hazards with shared frailty.

This module is the sole point of contact between Python and R for the
CoxFrailty model family. It handles all rpy2 conversions, R function calls,
and result extraction, returning plain Python/numpy objects to the caller.

No knowledge of BaseRULModel, sklearn pipelines, or metric computation
should exist in this module — only R translation logic.

Design pattern:
    To avoid rpy2 type annotation issues (R objects have no Python type hints),
    the fitted model is stored in R's global environment under a unique name.
    fit_cox_frailty returns that name as a plain Python str. All subsequent
    operations reference the model by name via ro.r() string evaluation,
    keeping all Python-visible types as native Python/numpy objects.

R implementation:
    Uses survival::coxph with frailty(motor_id, distribution, method, tdf)
    for shared frailty estimation. This replaces the original frailtyPenal
    implementation which was numerically unstable on C-MAPSS cross-validation
    folds (istop=2 failures). coxph is the base R survival package and provides
    robust convergence with the counting process format Surv(t_start, t_stop).

frailty() parameters exposed to GGS:
    distribution: 'gamma', 'gaussian', or 't' — frailty distribution family.
        gamma:    standard, EM-based estimation. Default for frailty models.
        gaussian: REML-based estimation. Symmetric heterogeneity assumption.
        t:        robust to outliers. Uses tdf degrees of freedom.
    method: estimation method for the frailty variance θ.
        'em':   EM algorithm. Default for gamma. Not available for gaussian.
        'aic':  minimize AIC to select θ. Available for all distributions.
        'reml': REML. Default for gaussian. Not meaningful for gamma/t.
        'df':   fix equivalent degrees of freedom. Requires df argument.
        'fixed': fix θ externally. Requires theta argument.
        GGS explores 'em' and 'aic' — the two data-driven methods compatible
        with both gamma and t distributions. When method='em' is passed to
        gaussian frailty, R silently falls back to 'reml' — consistent with
        the project convention of passing all parameters regardless of
        conditional applicability (e.g. degree in SVR, n_baseline_knots in Cox).
    tdf: degrees of freedom for the t frailty distribution. Only active when
        distribution='t'. Silently ignored by R for gamma and gaussian —
        consistent with the project convention for conditionally active params.

Data format:
    Uses the Andersen-Gill counting process format with Surv(t_start, t_stop,
    evento). The time axis represents absolute cycle count (time_in_cycles),
    so the survival function S(t|X) gives P(T > t) where t is cycles since
    motor start. In production, RUL = t_failure - t_current requires knowing
    the current cycle, which is passed as t_stop in fit() kwargs.

Data conversion strategy:
    pandas2ri is intentionally avoided. Benchmarking revealed that pandas2ri
    produces subtle type differences that cause convergence failures. Instead,
    data is written to a temporary CSV and loaded in R via read.csv(), which
    replicates the native R type system.

Prediction strategy:
    survfit.coxph does not accept newdata when the model has frailty terms.
    Individual survival functions are therefore computed directly using the
    proportional hazards formula:

        S(t|X) = S0(t)^exp(β'X)

    where S0(t) = exp(-H0(t)) is the baseline survival function derived from
    the cumulative baseline hazard H0(t) via basehaz(fit, centered=FALSE),
    times are the corresponding event times, and β are the regression
    coefficients stored in fit$coefficients. This marginalizes over the
    frailty distribution, representing the expected survival for a motor
    with average frailty.
"""

import os
import tempfile
import uuid
import contextlib
import io

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.rinterface_lib import callbacks as r_callbacks

# R packages — loaded once at import time
_survival = importr('survival')


@contextlib.contextmanager
def _silence_r():
    """Suppresses R console output (stdout and stderr/warnings).

    rpy2 routes R's console output through two callbacks:
        consolewrite_print   — R stdout (print, cat)
        consolewrite_warnerror — R stderr (warnings, messages)

    Both are redirected to /dev/null during the context. This silences
    the 'Inner loop failed to converge' warnings from coxpenal.fit()
    and the library path warnings from importr(), which are expected
    and documented but clutter the GGS progress output.
    """
    _original_print    = r_callbacks.consolewrite_print
    _original_warnerr  = r_callbacks.consolewrite_warnerror

    def _devnull(s: str) -> None:
        pass

    r_callbacks.consolewrite_print      = _devnull
    r_callbacks.consolewrite_warnerror  = _devnull
    try:
        yield
    finally:
        r_callbacks.consolewrite_print      = _original_print
        r_callbacks.consolewrite_warnerror  = _original_warnerr


def fit_cox_frailty(
    X: pd.DataFrame,
    t_start: np.ndarray,
    t_stop: np.ndarray,
    evento: np.ndarray,
    motor_id: np.ndarray,
    distribution: str = 'gamma',
    maxit: int = 300,
    method: str = 'em',
    tdf: int = 5,
) -> str:
    """Fits a Cox proportional hazards model with shared frailty via R.

    Uses survival::coxph with a frailty() term and the Andersen-Gill counting
    process format Surv(t_start, t_stop, evento). The frailty term captures
    unobserved motor-level heterogeneity:

        h(t|X, ω_i) = h_0(t) · ω_i · exp(β'X)

    The fitted model is stored in R's global environment under a unique name
    and that name is returned as a plain Python str.

    frailty parameter interactions:
        distribution='gamma',   method='em':   standard EM estimation (recommended)
        distribution='gamma',   method='aic':  AIC-based θ selection
        distribution='gaussian',method='em':   mapped to 'reml' internally —
                                               frailty.gaussian does not accept 'em'
        distribution='gaussian',method='aic':  AIC-based θ selection
        distribution='t',       method='em':   EM with tdf degrees of freedom
        distribution='t',       method='aic':  AIC-based θ with tdf d.f.
        tdf is silently ignored by R when distribution != 't'.

    Args:
        X: Feature matrix of shape (n_samples, n_features). Must be pre-scaled.
            Column names are used to build the R formula dynamically.
        t_start: Start of the time interval for each row, shape (n_samples,).
        t_stop: End of the time interval for each row, shape (n_samples,).
        evento: Event indicator array of shape (n_samples,).
        motor_id: Motor unit identifier array of shape (n_samples,).
        distribution: Frailty distribution. One of 'gamma', 'gaussian', 't'.
            Defaults to 'gamma'.
        maxit: Maximum number of outer iterations for coxph. Defaults to 300.
        method: Estimation method for frailty variance θ. One of 'em', 'aic',
            'reml', 'df', 'fixed'. GGS explores 'em' and 'aic'. Defaults to 'em'.
        tdf: Degrees of freedom for t frailty distribution. Only active when
            distribution='t'. Silently ignored otherwise. Defaults to 5.

    Returns:
        Name of the fitted model stored in R's global environment.

    Raises:
        RuntimeError: If the R model fails to fit (zero iterations completed).
    """
    df_name = _build_r_dataframe(X, t_start, t_stop, evento, motor_id)

    feature_names = list(X.columns)
    covariates = ' + '.join(feature_names)
    model_name = f'cox_frailty_{uuid.uuid4().hex}'

    # Method compatibility mapping per frailty distribution:
    #   frailty.gamma:    accepts 'em', 'aic', 'df', 'fixed'  → no mapping needed
    #   frailty.gaussian: accepts 'reml', 'aic', 'df', 'fixed' → map 'em' to 'reml'
    #   frailty.t:        accepts 'aic', 'df', 'fixed'         → map 'em' to 'aic'
    # Consistent with the project convention for conditionally active parameters
    # (e.g. degree in SVR, n_baseline_knots in Cox PH).
    if distribution == 'gaussian' and method == 'em':
        effective_method = 'reml'
    elif distribution == 't' and method == 'em':
        effective_method = 'aic'
    else:
        effective_method = method

    ro.r(f'''
        {model_name} <- coxph(
            Surv(t_start, t_stop, evento) ~ {covariates} +
                frailty(motor_id,
                        distribution="{distribution}",
                        method="{effective_method}",
                        tdf={tdf}),
            data={df_name},
            control=coxph.control(iter.max={maxit})
        )
    ''')

    # Verify that at least one iteration completed
    iter_val: int = int(np.array(ro.r(f'{model_name}$iter[1]'))[0])
    if iter_val == 0:
        ro.r(f'rm({model_name}, {df_name})')
        raise RuntimeError(
            f"coxph did not complete any iterations for "
            f"distribution='{distribution}', method='{method}'. "
            f"Check the data or try different hyperparameters."
        )

    # Extract and store the baseline hazard immediately while the dataframe
    # is still in R globalenv. basehaz() requires the original data frame to
    # re-evaluate the formula — storing the result now avoids keeping the data
    # frame in memory for the lifetime of the model.
    ro.r(f'{model_name}_bh <- basehaz({model_name}, centered=FALSE)')

    # Now safe to remove the temporary dataframe from R globalenv
    ro.r(f'rm({df_name})')

    return model_name


def predict_survival_functions(
    model_name: str,
    X_new: pd.DataFrame,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Predicts individual survival functions using the proportional hazards formula.

    Computes S(t|X) = S0(t)^exp(β'X) for each row in X_new, where S0(t) is
    derived from the cumulative baseline hazard stored at fit time:

        S0(t) = exp(-H0(t))

    The baseline hazard is extracted during fit_cox_frailty and stored in R
    globalenv as {model_name}_bh, avoiding the need to call basehaz() at
    prediction time (which would require the original data frame).

    survfit.coxph is not used because it does not accept newdata when the
    model contains frailty terms. The formula marginalizes over the frailty
    distribution, yielding the expected survival for a motor with average frailty.

    X_new must already be scaled with the same RobustScaler+PCA used during
    training, as the pipeline handles scaling before calling predict().

    Args:
        model_name: Name of the fitted model in R's global environment, as
            returned by fit_cox_frailty.
        X_new: Feature matrix of shape (n_samples, n_features). Must contain
            the same columns as the training X and be pre-scaled by the pipeline.

    Returns:
        List of (times, survival_probs) tuples, one per row in X_new, where:
            times: numpy array of time points at which S(t) is evaluated.
            survival_probs: numpy array of survival probabilities S(t|X) at
                each time point, clipped to [0, 1].
    """
    # Use pre-computed baseline hazard stored at fit time
    cum_hazard = np.array(ro.r(f'{model_name}_bh$hazard'))
    times = np.array(ro.r(f'{model_name}_bh$time'))
    S0 = np.exp(-cum_hazard)

    # Extract coefficients — coxph stores them in $coefficients
    coef = np.array(ro.r(f'{model_name}$coefficients'))
    coef_names_arr = np.array(ro.r(f'names({model_name}$coefficients)'))
    coef_names: list[str] = [str(name) for name in coef_names_arr]

    # Align X_new columns with the coefficient order from the fitted model
    X_aligned = X_new[coef_names].to_numpy()

    # Compute linear predictor β'X for each row
    linear_predictors = X_aligned @ coef

    # Compute individual survival functions S(t|X) = S0(t)^exp(β'X)
    # np.clip handles numerical overflow when exp(lp) is very large
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for lp in linear_predictors:
        S_individual = np.clip(S0 ** np.exp(lp), 0.0, 1.0)
        result.append((times, S_individual))

    return result


def remove_model(model_name: str) -> None:
    """Removes a fitted model and its baseline hazard from R's global environment.

    Cleans up both the coxph model object and the pre-computed baseline hazard
    stored as {model_name}_bh. Silently ignored if either object does not exist.

    Args:
        model_name: Name of the model in R's global environment, as returned
            by fit_cox_frailty.
    """
    bh_name = f'{model_name}_bh'
    ro.r(f'rm(list=intersect(ls(), c("{model_name}", "{bh_name}")))')


def _build_r_dataframe(
    X: pd.DataFrame,
    t_start: np.ndarray,
    t_stop: np.ndarray,
    evento: np.ndarray,
    motor_id: np.ndarray,
) -> str:
    """Builds an R dataframe via CSV to avoid pandas2ri conversion issues.

    pandas2ri conversion produces subtle type differences that cause
    convergence failures even when the underlying data is identical to a
    manually loaded CSV. Writing to a temporary CSV and loading via read.csv()
    replicates the native R type system and guarantees consistent behavior.

    The temporary CSV is removed from disk after loading into R.

    Args:
        X: Feature matrix of shape (n_samples, n_features). Must be pre-scaled.
        t_start: Start of each time interval, shape (n_samples,).
        t_stop: End of each time interval, shape (n_samples,).
        evento: Event indicator array of shape (n_samples,).
        motor_id: Motor unit identifier array of shape (n_samples,).

    Returns:
        Name of the R variable containing the loaded dataframe in R globalenv.
    """
    df_combined = X.copy().reset_index(drop=True)
    df_combined['t_start']  = t_start.astype(float)
    df_combined['t_stop']   = t_stop.astype(float)
    df_combined['evento']   = evento.astype(int)
    df_combined['motor_id'] = motor_id.astype(int)

    tmp_path = tempfile.mktemp(suffix='.csv')
    df_combined.to_csv(tmp_path, index=False)

    df_name = f'df_cox_{uuid.uuid4().hex[:8]}'
    ro.r(f'{df_name} <- read.csv("{tmp_path}")')
    os.remove(tmp_path)

    return df_name