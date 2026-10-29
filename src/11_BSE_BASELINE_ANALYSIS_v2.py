# ======================================================================================
# 11_BSE_BASELINE_ANALYSIS_v2
#========================================================

import os
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ======================================================================================
# 1. PATHS
# ======================================================================================

BASE_DIR = r"PATH_TO_DATA_DIRECTORY"
)

FEATURE_CSV = os.path.join(
    BASE_DIR,
    "bse_baseline_results",
    "bse_baseline_features_originals.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "bse_baseline_results",
    "analysis_v2_locked_densenet_test"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# Historical DenseNet121 test predictions used for comparison.
DENSENET_PREDICTIONS_CSV = r"PATH_TO_DENSENET_TEST_PREDICTIONS_CSV"
)


# ======================================================================================
# 2. SETTINGS
# ======================================================================================

TARGET = "H_norm_atomic5"

SEED = 42

CV_FOLDS = 5



INTENSITY_FEATURES = [
    "Mean",
    "Std",
    "CV",
    "Median",
    "IQR",
    "P10",
    "P90",
    "Skewness",
    "Kurtosis",
    "BSE_Entropy",
]


RIDGE_ALPHAS = [
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
]


# ======================================================================================
# 3. HELPERS
# ======================================================================================

def split_stem_aug(pid):
    """
    Convert Particle_ID to:
        Base_ID
        augmentation tag

    Examples:
        Particle_12             -> Particle_12, original
        Particle_12__aug_rot90  -> Particle_12, aug_rot90
    """

    s = os.path.splitext(
        str(pid)
    )[0]

    m = re.search(
        r"^(.+?)__aug_([A-Za-z0-9]+)$",
        s
    )

    if m:
        return (
            m.group(1),
            f"aug_{m.group(2).lower()}"
        )

    return s, "original"


def regression_metrics(y_true, y_pred):
    """
    MAE, RMSE and R².
    """

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    return {
        "MAE": float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        ),

        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred
                )
            )
        ),

        "R2": float(
            r2_score(
                y_true,
                y_pred
            )
        ),
    }


def safe_pearson(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    good = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[good]
    y = y[good]

    if len(x) < 3 or np.std(x) == 0:
        return np.nan, np.nan

    r, p = pearsonr(
        x,
        y
    )

    return float(r), float(p)


def safe_spearman(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    good = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[good]
    y = y[good]

    if len(x) < 3 or np.std(x) == 0:
        return np.nan, np.nan

    rho, p = spearmanr(
        x,
        y
    )

    return float(rho), float(p)


# ======================================================================================
# 4. LOAD CODE-1 BSE FEATURE TABLE
# ======================================================================================

print("=" * 100)
print(" BSE BASELINE ANALYSIS — LOCKED DENSENET TEST SET")
print("=" * 100)


print("\nBSE feature table:")
print(FEATURE_CSV)


if not os.path.isfile(FEATURE_CSV):

    raise FileNotFoundError(
        f"\nFeature CSV not found:\n{FEATURE_CSV}"
    )


df = pd.read_csv(
    FEATURE_CSV
)


print(
    f"\nPhysical particles loaded: "
    f"{len(df):,}"
)

print(
    f"Unique Base_IDs: "
    f"{df['Base_ID'].nunique():,}"
)


# ======================================================================================
# 5. FEATURE TABLE QC
# ======================================================================================

required_columns = [
    "Particle_ID",
    "Base_ID",
    TARGET,
    *INTENSITY_FEATURES,
]


missing_columns = [
    c
    for c in required_columns
    if c not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing required columns:\n"
        f"{missing_columns}"
    )


duplicate_count = (
    df["Base_ID"]
    .duplicated()
    .sum()
)


print(
    f"Duplicate Base_ID rows: "
    f"{duplicate_count}"
)


if duplicate_count != 0:

    raise RuntimeError(
        "Feature table must contain exactly "
        "ONE row per physical Base_ID."
    )


numeric_cols = [
    TARGET,
    *INTENSITY_FEATURES,
]


if df[numeric_cols].isna().any().any():

    bad = (
        df[numeric_cols]
        .isna()
        .sum()
    )

    raise RuntimeError(
        "NaNs detected:\n"
        f"{bad[bad > 0]}"
    )


if not np.isfinite(
    df[numeric_cols]
    .to_numpy(dtype=float)
).all():

    raise RuntimeError(
        "Infinite values detected."
    )


# ======================================================================================
# 6. LOAD HISTORICAL DENSENET TEST PREDICTIONS FIRST
#
# THIS FILE DEFINES THE TEST SET.
# ======================================================================================

print("\n" + "=" * 100)
print("READING HISTORICAL DENSENET TEST SET")
print("=" * 100)


print(
    "\nDenseNet prediction file:"
)

print(
    DENSENET_PREDICTIONS_CSV
)


if not os.path.isfile(
    DENSENET_PREDICTIONS_CSV
):

    raise FileNotFoundError(
        "\nDenseNet test prediction CSV not found:\n"
        f"{DENSENET_PREDICTIONS_CSV}"
    )


dense_raw = pd.read_csv(
    DENSENET_PREDICTIONS_CSV
)


dense_raw.columns = [
    str(c).strip()
    for c in dense_raw.columns
]


print(
    f"\nRows in DenseNet prediction CSV: "
    f"{len(dense_raw):,}"
)

print(
    f"Columns: "
    f"{list(dense_raw.columns)}"
)


if "Particle_ID" not in dense_raw.columns:

    raise ValueError(
        "DenseNet prediction CSV does not contain Particle_ID."
    )


# ======================================================================================
# 7. IDENTIFY TRUE/PREDICTION COLUMNS
# ======================================================================================

true_candidates = [
    "True",
    "y_true",
    "True_H_norm",
    "H_norm_atomic5",
]


pred_candidates = [
    "Pred",
    "Prediction",
    "y_pred",
    "Pred_H_norm",
]


true_col = next(
    (
        c
        for c in true_candidates
        if c in dense_raw.columns
    ),
    None
)


pred_col = next(
    (
        c
        for c in pred_candidates
        if c in dense_raw.columns
    ),
    None
)


if true_col is None:

    raise ValueError(
        "Could not identify TRUE-target column "
        "in DenseNet prediction CSV."
    )


if pred_col is None:

    raise ValueError(
        "Could not identify PREDICTION column "
        "in DenseNet prediction CSV."
    )


print(
    f"\nDenseNet true column: {true_col}"
)

print(
    f"DenseNet prediction column: {pred_col}"
)


# ======================================================================================
# 8. IDENTIFY ORIGINAL DENSENET TEST PARTICLES
# ======================================================================================

parsed = (
    dense_raw["Particle_ID"]
    .astype(str)
    .map(split_stem_aug)
)


dense_raw["Base_ID"] = [
    x[0]
    for x in parsed
]


dense_raw["Aug_Tag"] = [
    x[1]
    for x in parsed
]


print(
    "\nDenseNet prediction augmentation counts:"
)

print(
    dense_raw["Aug_Tag"]
    .value_counts()
    .sort_index()
)


dense_original = (
    dense_raw[
        dense_raw["Aug_Tag"] == "original"
    ]
    .copy()
    .reset_index(drop=True)
)


print(
    f"\nDenseNet ORIGINAL test rows: "
    f"{len(dense_original):,}"
)

print(
    f"DenseNet ORIGINAL unique Base_IDs: "
    f"{dense_original['Base_ID'].nunique():,}"
)


if dense_original["Base_ID"].duplicated().any():

    duplicates = (
        dense_original[
            dense_original["Base_ID"]
            .duplicated(keep=False)
        ]
        .sort_values("Base_ID")
    )

    print(
        duplicates[
            [
                "Particle_ID",
                "Base_ID"
            ]
        ]
    )

    raise RuntimeError(
        "Multiple original DenseNet predictions "
        "exist for the same Base_ID."
    )


# ======================================================================================
# 9. LOCK THE DENSENET TEST SET
# ======================================================================================

dense_test_ids = set(
    dense_original["Base_ID"]
)


baseline_ids = set(
    df["Base_ID"]
)


missing_from_baseline = sorted(
    dense_test_ids
    - baseline_ids
)


print("\n" + "-" * 100)

print(
    f"DenseNet original test Base_IDs: "
    f"{len(dense_test_ids):,}"
)

print(
    f"Available BSE baseline Base_IDs: "
    f"{len(baseline_ids):,}"
)

print(
    f"DenseNet test IDs missing from baseline table: "
    f"{len(missing_from_baseline):,}"
)


if missing_from_baseline:

    print(
        "\nMissing DenseNet test IDs:"
    )

    print(
        missing_from_baseline
    )

    raise RuntimeError(
        "\nSTOPPING.\n"
        "At least one historical DenseNet test particle "
        "does not exist in the BSE baseline feature table.\n"
        "We will not silently drop test particles."
    )


print(
    "\nPASS: Every historical DenseNet original test particle "
    "exists in the BSE baseline table."
)


# ======================================================================================
# 10. DEFINE LOCKED TEST + DEVELOPMENT SETS
# ======================================================================================

df["Split"] = np.where(
    df["Base_ID"].isin(
        dense_test_ids
    ),
    "locked_test",
    "development"
)


test_df = (
    df[
        df["Split"] == "locked_test"
    ]
    .copy()
    .reset_index(drop=True)
)


dev_df = (
    df[
        df["Split"] == "development"
    ]
    .copy()
    .reset_index(drop=True)
)


print("\n" + "=" * 100)
print("FINAL DATA PARTITION")
print("=" * 100)


print(
    f"\nDevelopment physical particles: "
    f"{len(dev_df):,}"
)

print(
    f"Locked test physical particles: "
    f"{len(test_df):,}"
)

print(
    f"Total physical particles: "
    f"{len(df):,}"
)


if len(test_df) != len(dense_test_ids):

    raise RuntimeError(
        "Locked test-set size does not match "
        "DenseNet original test-set size."
    )


if set(dev_df["Base_ID"]) & set(test_df["Base_ID"]):

    raise RuntimeError(
        "Development/test leakage detected."
    )


split_csv = os.path.join(
    OUTPUT_DIR,
    "locked_densenet_test_split_assignments.csv"
)


df[
    [
        "Particle_ID",
        "Base_ID",
        TARGET,
        "Split",
    ]
].to_csv(
    split_csv,
    index=False
)


# ======================================================================================
# 11. DESCRIPTIVE CORRELATION ANALYSIS
#
# These correlations use all 806 physical particles as a descriptive
# dataset-level analysis.
#
# They are NOT predictive test metrics.
# ======================================================================================

print("\n" + "=" * 100)
print("DESCRIPTIVE CORRELATIONS WITH H_norm")
print("=" * 100)


correlation_records = []


for feature in INTENSITY_FEATURES:

    x = df[feature].to_numpy(
        dtype=float
    )

    y = df[TARGET].to_numpy(
        dtype=float
    )


    pearson_r, pearson_p = safe_pearson(
        x,
        y
    )


    spearman_rho, spearman_p = safe_spearman(
        x,
        y
    )


    correlation_records.append(
        {
            "Feature": feature,
            "Pearson_r": pearson_r,
            "Pearson_p": pearson_p,
            "Spearman_rho": spearman_rho,
            "Spearman_p": spearman_p,
            "Abs_Pearson_r": (
                abs(pearson_r)
                if np.isfinite(pearson_r)
                else np.nan
            ),
            "Abs_Spearman_rho": (
                abs(spearman_rho)
                if np.isfinite(spearman_rho)
                else np.nan
            ),
        }
    )


corr_df = pd.DataFrame(
    correlation_records
)


corr_df = (
    corr_df
    .sort_values(
        "Abs_Spearman_rho",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    "\nCorrelation table "
    "(sorted by |Spearman rho|):\n"
)


print(
    corr_df[
        [
            "Feature",
            "Pearson_r",
            "Pearson_p",
            "Spearman_rho",
            "Spearman_p",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


corr_csv = os.path.join(
    OUTPUT_DIR,
    "bse_feature_correlations_with_hnorm.csv"
)


corr_df.to_csv(
    corr_csv,
    index=False,
    float_format="%.10f"
)


entropy_corr = (
    corr_df[
        corr_df["Feature"] == "BSE_Entropy"
    ]
    .iloc[0]
)


print("\n----------------------------------------")
print("BSE ENTROPY vs EDS H_norm")
print("----------------------------------------")


print(
    f"Pearson r   = "
    f"{entropy_corr['Pearson_r']:.6f}"
)

print(
    f"Pearson p   = "
    f"{entropy_corr['Pearson_p']:.6e}"
)

print(
    f"Spearman ρ  = "
    f"{entropy_corr['Spearman_rho']:.6f}"
)

print(
    f"Spearman p  = "
    f"{entropy_corr['Spearman_p']:.6e}"
)


# ======================================================================================
# 12. CORRELATION FIGURE
# ======================================================================================

plot_corr = (
    corr_df
    .sort_values(
        "Spearman_rho",
        ascending=True
    )
    .reset_index(drop=True)
)


y_pos = np.arange(
    len(plot_corr)
)


fig, ax = plt.subplots(
    figsize=(8.5, 6.0)
)


bar_height = 0.36


ax.barh(
    y_pos - bar_height / 2,
    plot_corr["Pearson_r"],
    height=bar_height,
    label="Pearson r"
)


ax.barh(
    y_pos + bar_height / 2,
    plot_corr["Spearman_rho"],
    height=bar_height,
    label="Spearman ρ"
)


ax.axvline(
    0,
    linewidth=1
)


ax.set_yticks(
    y_pos
)


ax.set_yticklabels(
    plot_corr["Feature"]
)


ax.set_xlabel(
    "Correlation with EDS-derived $H_{norm}$"
)


ax.set_ylabel(
    "BSE intensity descriptor"
)


ax.set_title(
    "Global BSE intensity descriptors vs "
    "EDS-derived compositional homogeneity"
)


ax.legend(
    frameon=False
)


ax.grid(
    axis="x",
    alpha=0.25
)


fig.tight_layout()


corr_fig_path = os.path.join(
    OUTPUT_DIR,
    "bse_feature_correlations.png"
)


fig.savefig(
    corr_fig_path,
    dpi=600,
    bbox_inches="tight"
)


plt.show()


# ======================================================================================
# 13. PREPARE DEVELOPMENT AND LOCKED TEST ARRAYS
# ======================================================================================

y_dev = (
    dev_df[TARGET]
    .to_numpy(dtype=float)
)


y_test = (
    test_df[TARGET]
    .to_numpy(dtype=float)
)


X_dev_entropy = (
    dev_df[
        ["BSE_Entropy"]
    ]
    .to_numpy(dtype=float)
)


X_test_entropy = (
    test_df[
        ["BSE_Entropy"]
    ]
    .to_numpy(dtype=float)
)


X_dev_multi = (
    dev_df[
        INTENSITY_FEATURES
    ]
    .to_numpy(dtype=float)
)


X_test_multi = (
    test_df[
        INTENSITY_FEATURES
    ]
    .to_numpy(dtype=float)
)


# ======================================================================================
# 14. BASELINE A:
#     BSE HISTOGRAM ENTROPY ONLY -> LINEAR REGRESSION
#
# No hyperparameter exists.
#
# Fit ONLY on development particles.
# Evaluate ONLY on locked DenseNet test particles.
# ======================================================================================

print("\n" + "=" * 100)
print("BASELINE A: BSE HISTOGRAM ENTROPY ONLY")
print("=" * 100)


entropy_model = LinearRegression()


entropy_model.fit(
    X_dev_entropy,
    y_dev
)


entropy_test_pred = entropy_model.predict(
    X_test_entropy
)


entropy_test_metrics = regression_metrics(
    y_test,
    entropy_test_pred
)


print(
    f"\nFitted using "
    f"{len(dev_df):,} development particles."
)


print(
    "\nLinear equation:"
)


print(
    f"H_norm_pred = "
    f"{entropy_model.intercept_:.6f} "
    f"+ ({entropy_model.coef_[0]:.6f}) "
    f"* BSE_Entropy"
)


print(
    "\nLOCKED TEST metrics:"
)


for k, v in entropy_test_metrics.items():

    print(
        f"  {k}: {v:.6f}"
    )


# ======================================================================================
# 15. BASELINE B:
#     GLOBAL BSE INTENSITY STATISTICS -> RIDGE
#
# Alpha is selected using 5-fold CV INSIDE DEVELOPMENT SET ONLY.
#
# The locked DenseNet test set is never touched during model selection.
# ======================================================================================

print("\n" + "=" * 100)
print("BASELINE B: GLOBAL BSE STATISTICS + RIDGE")
print("=" * 100)


cv = KFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=SEED
)


ridge_search_records = []


for alpha in RIDGE_ALPHAS:

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "ridge",
                Ridge(
                    alpha=alpha
                )
            ),
        ]
    )


    # sklearn returns NEGATIVE MAE because larger score = better.
    neg_mae_scores = cross_val_score(
        model,
        X_dev_multi,
        y_dev,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )


    fold_mae = -neg_mae_scores


    ridge_search_records.append(
        {
            "alpha": alpha,
            "CV_MAE_mean": float(
                np.mean(fold_mae)
            ),
            "CV_MAE_std": float(
                np.std(
                    fold_mae,
                    ddof=1
                )
            ),
        }
    )


ridge_search_df = pd.DataFrame(
    ridge_search_records
)


ridge_search_df = (
    ridge_search_df
    .sort_values(
        [
            "CV_MAE_mean",
            "CV_MAE_std",
        ],
        ascending=[
            True,
            True,
        ]
    )
    .reset_index(drop=True)
)


print(
    "\n5-fold DEVELOPMENT-SET CV:\n"
)


print(
    ridge_search_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


best_alpha = float(
    ridge_search_df.iloc[0]["alpha"]
)


print(
    f"\nSelected Ridge alpha: "
    f"{best_alpha}"
)


print(
    "Selection criterion: lowest mean "
    "development-set CV MAE."
)


# ----------------------------------------------------------------------
# Now refit selected Ridge model using ALL development particles.
# ----------------------------------------------------------------------

ridge_model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(
                alpha=best_alpha
            )
        ),
    ]
)


ridge_model.fit(
    X_dev_multi,
    y_dev
)


ridge_test_pred = ridge_model.predict(
    X_test_multi
)


ridge_test_metrics = regression_metrics(
    y_test,
    ridge_test_pred
)


print(
    f"\nFinal Ridge fitted using ALL "
    f"{len(dev_df):,} development particles."
)


print(
    "\nLOCKED TEST metrics:"
)


for k, v in ridge_test_metrics.items():

    print(
        f"  {k}: {v:.6f}"
    )


ridge_search_path = os.path.join(
    OUTPUT_DIR,
    "ridge_development_cv_alpha_search.csv"
)


ridge_search_df.to_csv(
    ridge_search_path,
    index=False,
    float_format="%.10f"
)


# ======================================================================================
# 16. ALIGN DENSENET PREDICTIONS WITH EXACT SAME LOCKED TEST PARTICLES
# ======================================================================================

dense_keep = (
    dense_original[
        [
            "Base_ID",
            true_col,
            pred_col,
        ]
    ]
    .copy()
)


dense_keep = dense_keep.rename(
    columns={
        true_col: "DenseNet_True",
        pred_col: "DenseNet_Pred",
    }
)


comparison = (
    test_df[
        [
            "Particle_ID",
            "Base_ID",
            TARGET,
            *INTENSITY_FEATURES,
        ]
    ]
    .copy()
)


comparison["EntropyLinear_Pred"] = (
    entropy_test_pred
)


comparison["Ridge_Pred"] = (
    ridge_test_pred
)


comparison = comparison.merge(
    dense_keep,
    on="Base_ID",
    how="left",
    validate="one_to_one"
)


if comparison[
    [
        "DenseNet_True",
        "DenseNet_Pred"
    ]
].isna().any().any():

    raise RuntimeError(
        "Missing DenseNet predictions after merge."
    )


# ======================================================================================
# 17. VERIFY TARGETS ARE IDENTICAL
#
# The EDS H_norm stored in Code 1 must match the target stored in the
# historical DenseNet prediction CSV for every test particle.
# ======================================================================================

target_difference = np.abs(
    comparison[TARGET].to_numpy(
        dtype=float
    )
    -
    comparison["DenseNet_True"].to_numpy(
        dtype=float
    )
)


max_target_difference = float(
    np.max(
        target_difference
    )
)


print("\n" + "=" * 100)
print("TARGET CONSISTENCY CHECK")
print("=" * 100)


print(
    f"\nMaximum |Code1 H_norm - DenseNet True| = "
    f"{max_target_difference:.12f}"
)


if max_target_difference > 1e-6:

    bad_mask = (
        target_difference > 1e-6
    )

    bad_targets = comparison.loc[
        bad_mask,
        [
            "Base_ID",
            TARGET,
            "DenseNet_True"
        ]
    ].copy()


    bad_targets[
        "AbsoluteDifference"
    ] = target_difference[
        bad_mask
    ]


    print(
        "\nMismatched targets:"
    )

    print(
        bad_targets.to_string(
            index=False
        )
    )


    raise RuntimeError(
        "\nSTOPPING.\n"
        "Target values disagree between the baseline table "
        "and the historical DenseNet prediction file."
    )


print(
    "\nPASS: EDS-derived target values match exactly."
)


# ======================================================================================
# 18. DENSENET METRICS — ORIGINAL TEST IMAGES ONLY
# ======================================================================================

dense_metrics = regression_metrics(
    comparison[TARGET],
    comparison["DenseNet_Pred"]
)


print("\n" + "=" * 100)
print("DENSENET121 — ORIGINAL LOCKED TEST PARTICLES")
print("=" * 100)


for k, v in dense_metrics.items():

    print(
        f"  {k}: {v:.6f}"
    )


# ======================================================================================
# 19. FINAL APPLES-TO-APPLES MODEL COMPARISON
# ======================================================================================

model_comparison = pd.DataFrame(
    [
        {
            "Model": (
                "BSE entropy only + linear regression"
            ),
            "n_test_particles": len(comparison),
            **entropy_test_metrics,
        },

        {
            "Model": (
                "Global BSE intensity statistics + Ridge"
            ),
            "n_test_particles": len(comparison),
            **ridge_test_metrics,
        },

        {
            "Model": (
                "DenseNet121 (original test images only)"
            ),
            "n_test_particles": len(comparison),
            **dense_metrics,
        },
    ]
)


print("\n" + "=" * 100)
print("FINAL LOCKED-TEST COMPARISON")
print("=" * 100)


print(
    model_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


model_comparison_path = os.path.join(
    OUTPUT_DIR,
    "locked_test_model_comparison.csv"
)


model_comparison.to_csv(
    model_comparison_path,
    index=False,
    float_format="%.10f"
)


# ======================================================================================
# 20. PARTICLE-LEVEL ERROR TABLE
# ======================================================================================

comparison[
    "EntropyLinear_AbsError"
] = np.abs(
    comparison[TARGET]
    -
    comparison["EntropyLinear_Pred"]
)


comparison[
    "Ridge_AbsError"
] = np.abs(
    comparison[TARGET]
    -
    comparison["Ridge_Pred"]
)


comparison[
    "DenseNet_AbsError"
] = np.abs(
    comparison[TARGET]
    -
    comparison["DenseNet_Pred"]
)


prediction_path = os.path.join(
    OUTPUT_DIR,
    "locked_test_particle_predictions.csv"
)


comparison.to_csv(
    prediction_path,
    index=False,
    float_format="%.10f"
)


# ======================================================================================
# 21. PARITY FIGURE — IDENTICAL TEST PARTICLES
# ======================================================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15.5, 4.8),
    sharex=True,
    sharey=True
)


models_for_plot = [
    (
        "BSE entropy + linear regression",
        "EntropyLinear_Pred",
        entropy_test_metrics,
    ),

    (
        "Global BSE statistics + Ridge",
        "Ridge_Pred",
        ridge_test_metrics,
    ),

    (
        "DenseNet121",
        "DenseNet_Pred",
        dense_metrics,
    ),
]


all_true = (
    comparison[TARGET]
    .to_numpy(dtype=float)
)


all_preds = np.concatenate(
    [
        comparison[col]
        .to_numpy(dtype=float)

        for _, col, _
        in models_for_plot
    ]
)


plot_min = float(
    min(
        np.min(all_true),
        np.min(all_preds)
    )
)


plot_max = float(
    max(
        np.max(all_true),
        np.max(all_preds)
    )
)


pad = 0.03 * (
    plot_max - plot_min
)


plot_min -= pad
plot_max += pad


for ax, (
    title,
    pred_name,
    metrics
) in zip(
    axes,
    models_for_plot
):

    ax.scatter(
        comparison[TARGET],
        comparison[pred_name],
        s=28,
        alpha=0.75
    )


    ax.plot(
        [
            plot_min,
            plot_max
        ],
        [
            plot_min,
            plot_max
        ],
        linestyle="--",
        linewidth=1.2
    )


    ax.set_xlim(
        plot_min,
        plot_max
    )


    ax.set_ylim(
        plot_min,
        plot_max
    )


    ax.set_title(
        title
    )


    ax.set_xlabel(
        "True EDS-derived $H_{norm}$"
    )


    metric_text = (
        f"MAE = {metrics['MAE']:.3f}\n"
        f"RMSE = {metrics['RMSE']:.3f}\n"
        f"$R^2$ = {metrics['R2']:.3f}"
    )


    ax.text(
        0.05,
        0.95,
        metric_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="round",
            alpha=0.15
        )
    )


    ax.grid(
        alpha=0.20
    )


axes[0].set_ylabel(
    "Predicted $H_{norm}$"
)


fig.suptitle(
    "Held-out compositional-homogeneity prediction:\n"
    "global BSE intensity baselines versus DenseNet121",
    fontsize=14
)


fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.92
    ]
)


parity_path = os.path.join(
    OUTPUT_DIR,
    "locked_test_baseline_vs_densenet.png"
)


fig.savefig(
    parity_path,
    dpi=600,
    bbox_inches="tight"
)


plt.show()


# ======================================================================================
# 22. DIRECT BSE ENTROPY vs H_norm FIGURE
#
# Descriptive figure across all physical particles.
# ======================================================================================

fig, ax = plt.subplots(
    figsize=(6.5, 5.5)
)


ax.scatter(
    df["BSE_Entropy"],
    df[TARGET],
    s=25,
    alpha=0.65
)


visual_fit = LinearRegression()


visual_fit.fit(
    df[
        ["BSE_Entropy"]
    ],
    df[TARGET]
)


x_line = np.linspace(
    df["BSE_Entropy"].min(),
    df["BSE_Entropy"].max(),
    200
).reshape(
    -1,
    1
)


y_line = visual_fit.predict(
    x_line
)


ax.plot(
    x_line[:, 0],
    y_line,
    linewidth=1.5
)


ax.set_xlabel(
    "BSE histogram entropy"
)


ax.set_ylabel(
    "EDS-derived $H_{norm}$"
)


ax.set_title(
    "BSE intensity-distribution entropy vs\n"
    "EDS-derived compositional homogeneity"
)


annotation = (
    f"Pearson r = "
    f"{entropy_corr['Pearson_r']:.3f}\n"
    f"Spearman ρ = "
    f"{entropy_corr['Spearman_rho']:.3f}"
)


ax.text(
    0.05,
    0.95,
    annotation,
    transform=ax.transAxes,
    va="top",
    ha="left",
    bbox=dict(
        boxstyle="round",
        alpha=0.15
    )
)


ax.grid(
    alpha=0.20
)


fig.tight_layout()


entropy_scatter_path = os.path.join(
    OUTPUT_DIR,
    "bse_entropy_vs_hnorm.png"
)


fig.savefig(
    entropy_scatter_path,
    dpi=600,
    bbox_inches="tight"
)


plt.show()


# ======================================================================================
# 23. FINAL SUMMARY
# ======================================================================================

print("\n")
print("=" * 100)
print("CODE 2 v2 COMPLETE")
print("=" * 100)


print(
    f"\nDevelopment particles: "
    f"{len(dev_df):,}"
)


print(
    f"Locked DenseNet test particles: "
    f"{len(test_df):,}"
)


print(
    "\nFinal locked-test metrics:"
)


print(
    model_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print(
    "\nSaved outputs:"
)


outputs = [
    split_csv,
    corr_csv,
    corr_fig_path,
    ridge_search_path,
    model_comparison_path,
    prediction_path,
    parity_path,
    entropy_scatter_path,
]


for path in outputs:

    print(
        "  ",
        os.path.abspath(path)
    )


print("\nINTERPRETATION RULE:")

print(
    "The three predictive methods above were evaluated "
    "on exactly the same original physical test particles."
)

print(
    "DenseNet test membership came directly from the "
    "historical DenseNet prediction file and was never "
    "reconstructed from a new random split."
)
