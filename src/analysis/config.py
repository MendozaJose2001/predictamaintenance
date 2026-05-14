"""Shared configuration constants for GGS and test analysis modules.

This module centralises all path references, model identifiers, column
definitions, and visual style constants used across the analysis package.
Importing from this module guarantees that any change to a path or model
key is reflected consistently across all analysis functions.
"""

# ---------------------------------------------------------------------------
# Model identifiers and display names
# ---------------------------------------------------------------------------

MODEL_NAMES: dict[str, str] = {
    'nb'     : 'Negative Binomial',
    'dt'     : 'Decision Tree',
    'rf'     : 'Random Forest',
    'xgb'    : 'XGBoost',
    'svr'    : 'SVR',
    'cox'    : 'CoxPH',
    'aft'    : 'WeibullAFT',
    'frailty': 'CoxFrailty',
}

#: Models eligible for production — survival models excluded due to
#: structural incompatibility with the sliding window pipeline.
ELIGIBLE_MODELS: list[str] = ['nb', 'dt', 'rf', 'xgb', 'svr']

# ---------------------------------------------------------------------------
# Production model paths
# ---------------------------------------------------------------------------

PREDICTOR_PATHS: dict[str, str] = {
    'nb' : 'outputs/production/models/NegativeBinomialPiecewise_B_30w_20pc_20260509_1811.pkl',
    'dt' : 'outputs/production/models/DecisionTreeModel_A_30w_10pc_20260510_1148.pkl',
    'rf' : 'outputs/production/models/RandomForestModel_A_30w_20pc_20260510_2159.pkl',
    'xgb': 'outputs/production/models/XGBModel_A_30w_10pc_20260513_1317.pkl',
    'svr': 'outputs/production/models/SVRModel_A_30w_10pc_20260510_2152.pkl',
}

# ---------------------------------------------------------------------------
# GGS results paths
# ---------------------------------------------------------------------------

GGS_RESULTS_PATHS: dict[str, str] = {
    'nb'     : 'outputs/ggs/results/NegativeBinomialPiecewise_13f6c147_20260507_2337.csv',
    'dt'     : 'outputs/ggs/results/DecisionTreeModel_c30866d4_20260509_0941.csv',
    'rf'     : 'outputs/ggs/results/RandomForestModel_467bdc30_20260509_1125.csv',
    'xgb'    : 'outputs/ggs/results/XGBModel_43704765_20260511_1444.csv',
    'svr'    : 'outputs/ggs/results/SVRModel_c5ba7e02_20260508_1026.csv',
    'cox'    : 'outputs/ggs/results/CoxPHModel_59d4a8b5_20260512_1106.csv',
    'aft'    : 'outputs/ggs/results/WeibullAFTModel_dd4bcf5e_20260512_1116.csv',
    'frailty': 'outputs/ggs/results/CoxFrailty_3da8809d_20260512_2005.csv',
}

# ---------------------------------------------------------------------------
# GGS result column definitions per model
# ---------------------------------------------------------------------------

#: Columns used for duplicate detection and top-N display per model.
GGS_COLS: dict[str, list[str]] = {
    'nb': [
        'feature_set', 'window_size', 'n_components', 'clipping_threshold',
        'alpha', 'alpha_reg', 'l1_ratio',
        'mean_S_score', 'mean_MAE', 'mean_RMSE', 'mean_C_index',
    ],
    'dt': [
        'feature_set', 'window_size', 'n_components', 'clipping_threshold',
        'max_depth', 'min_samples_leaf', 'min_samples_split', 'max_features',
        'mean_S_score', 'mean_C_index', 'mean_MAE', 'mean_RMSE',
    ],
    'rf': [
        'feature_set', 'window_size', 'n_components', 'clipping_threshold',
        'n_estimators', 'max_depth', 'min_samples_leaf', 'max_features',
        'mean_S_score', 'mean_C_index', 'mean_MAE', 'mean_RMSE',
    ],
    'xgb': [
        'feature_set', 'window_size', 'n_components', 'clipping_threshold',
        'n_estimators', 'learning_rate', 'max_depth', 'subsample',
        'colsample_bytree', 'reg_lambda', 'min_child_weight',
        'mean_S_score', 'mean_C_index', 'mean_MAE', 'mean_RMSE',
    ],
    'svr': [
        'feature_set', 'window_size', 'n_components', 'clipping_threshold',
        'kernel', 'C', 'epsilon', 'gamma', 'degree',
        'mean_S_score', 'mean_C_index', 'mean_MAE', 'mean_RMSE',
    ],
    'cox': [
        'feature_set', 'window_size', 'n_components', 'clipping_threshold',
        'baseline_estimation_method', 'n_baseline_knots', 'penalizer',
        'l1_ratio', 'confidence_threshold',
        'mean_S_score', 'mean_C_index', 'mean_MAE', 'mean_RMSE',
    ],
    'aft': [
        'feature_set', 'window_size', 'n_components', 'clipping_threshold',
        'confidence_threshold', 'penalizer', 'l1_ratio', 'fit_intercept',
        'mean_S_score', 'mean_C_index', 'mean_MAE', 'mean_RMSE',
    ],
    'frailty': [
        'feature_set', 'window_size', 'n_components', 'clipping_threshold',
        'distribution', 'method', 'tdf', 'confidence_threshold',
        'mean_S_score', 'mean_C_index', 'mean_MAE', 'mean_RMSE',
    ],
}

# ---------------------------------------------------------------------------
# Visual style constants
# ---------------------------------------------------------------------------

#: Per-model matplotlib style definitions for trajectory plots.
#: Keys match MODEL_NAMES keys for eligible models only.
MODEL_STYLE: dict[str, dict] = {
    'nb' : {'color': '#F44336', 'ls': '--',  'lw': 1.2, 'label': 'NB'},
    'dt' : {'color': '#9C27B0', 'ls': ':',   'lw': 1.2, 'label': 'DT'},
    'rf' : {'color': '#FF9800', 'ls': '-.',  'lw': 1.2, 'label': 'RF'},
    'svr': {'color': '#2196F3', 'ls': '-',   'lw': 1.8, 'label': 'SVR'},
    'xgb': {'color': '#4CAF50', 'ls': '-',   'lw': 2.0, 'label': 'XGB ★'},
}

#: Colors assigned to eligible models for boxplot and bar charts.
#: Order matches ELIGIBLE_MODELS.
MODEL_COLORS: list[str] = [
    '#4CAF50',  # xgb
    '#2196F3',  # svr
    '#FF9800',  # rf
    '#9C27B0',  # dt
    '#F44336',  # nb
]