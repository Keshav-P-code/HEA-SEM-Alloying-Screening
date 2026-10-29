# =============================================================================
# REVIEWER 1 + REVIEWER 2
# FINAL LOCKED-TEST BOOTSTRAP + MASK OCCUPANCY ANALYSIS
#
# INPUTS:
#   1. Existing 121-particle locked-test DenseNet121 predictions CSV
#   2. ORIGINAL masks used by DenseNet121 training/testing
#
# NOT USED:
#   - bse_baseline_features_originals.csv
#   - Original-resolution Area_px
#   - Raw BSE intensity descriptors
#
# EXACT TRAINING MASK PREPROCESSING:
#   cv2.imread(..., IMREAD_UNCHANGED)
#   ensure_binary()
#   cv2.resize(mask, (512,512), INTER_NEAREST)
#
# ANALYSES:
#   1. Locked-test integrity checks
#   2. Overall MAE, RMSE, R2 + bootstrap MAE CI
#   3. Three H_norm regimes + bootstrap MAE CIs
#   4. Seven H_norm bins + bootstrap MAE CIs
#   5. Low-vs-high MAE difference + bootstrap CI
#   6. Resized-mask occupancy vs absolute error
#   7. Resized-mask occupancy vs signed residual
#   8. Occupancy quartiles + bootstrap MAE CIs
#   9. Within-regime Spearman associations
#  10. Seven publication-quality figures
#
# NO MODEL TRAINING OR INFERENCE
# NO AUGMENTED PARTICLES
# =============================================================================


# =============================================================================
# 0. IMPORTS
# =============================================================================

from pathlib import Path
import re
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.metrics import r2_score


# =============================================================================
# 1. PATHS — CHECK THESE
# =============================================================================

BASE = Path(
    r"C:\Users\Keshav's\OneDrive - University of New Brunswick"
    r"\Testrun-3Sep9\Data-SEM-EDS\Images\Data-verse Ready"
    r"\bse_baseline_results"
)

PREDICTIONS_CSV = (
    BASE
    / "analysis_v2_locked_densenet_test"
    / "reviewer3_locked_test_particle_predictions.csv"
)

# ORIGINAL MASK DIRECTORY FROM YOUR DENSENET TRAINING SCRIPT.
# Update username/path if the masks are now elsewhere.
MASKS_DIR = Path(
    r"C:\Users\Keshav's\OneDrive - University of New Brunswick"
    r"\Testrun-3Sep9\Data-SEM-EDS\Images"
    r"\Augmented_output_masks\masks"
)

OUT = BASE / "reviewer_1_2_FINAL_NO_BSE_FEATURES"

OUT.mkdir(parents=True, exist_ok=True)

print("Predictions:", PREDICTIONS_CSV)
print("Masks:", MASKS_DIR)
print("Output:", OUT)


# =============================================================================
# 2. SETTINGS
# =============================================================================

IMG_SIZE = 512

N_BOOT = 10000
SEED = 42

RNG = np.random.default_rng(SEED)

EXPECTED_TEST_PARTICLES = 121

BROAD_LABELS = [
    "Low",
    "Medium",
    "High"
]

BROAD_EDGES = [
    -np.inf,
    0.80,
    0.95,
    np.inf
]

FINE_LABELS = [
    "<0.60",
    "0.60–0.70",
    "0.70–0.80",
    "0.80–0.90",
    "0.90–0.95",
    "0.95–0.975",
    "≥0.975"
]

FINE_EDGES = [
    -np.inf,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.975,
    np.inf
]

COLORS = {
    "Low": "#BA7139",
    "Medium": "#3B7FA8",
    "High": "#25846E"
}


# =============================================================================
# 3. FIGURE SETTINGS
# =============================================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600
})


def style_axes(ax):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(direction="in")

    ax.grid(
        axis="y",
        alpha=0.17,
        linewidth=0.6
    )

    ax.set_axisbelow(True)


def save_figure(fig, name):

    fig.tight_layout()

    for extension in ["pdf", "svg", "png"]:

        fig.savefig(
            OUT / f"{name}.{extension}",
            dpi=600,
            bbox_inches="tight",
            facecolor="white"
        )

    plt.show()
    plt.close(fig)

    print("Saved:", name)


# =============================================================================
# 4. EXACT MASK PREPROCESSING FROM ORIGINAL DENSENET SCRIPT
# =============================================================================

def ensure_binary(mask):

    if mask.ndim == 3:

        mask = cv2.cvtColor(
            mask,
            cv2.COLOR_BGR2GRAY
        )

    if mask.dtype != np.uint8:

        mask = cv2.normalize(
            mask,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    return (mask > 127).astype(np.uint8)


def load_densenet_mask(mask_path, img_size=IMG_SIZE):

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_UNCHANGED
    )

    if mask is None:

        raise FileNotFoundError(
            f"Cannot read mask:\n{mask_path}"
        )

    # EXACT original training function.
    mask = ensure_binary(mask)

    # EXACT original training resizing.
    mask = cv2.resize(
        mask,
        (img_size, img_size),
        interpolation=cv2.INTER_NEAREST
    )

    return mask


# =============================================================================
# 5. STRICT ORIGINAL MASK RESOLUTION
#
# The training script used find_by_stem() with several extensions
# and an optional wildcard fallback.
#
# For this analysis, we deliberately use ONLY exact filename stems,
# avoiding accidental selection of __aug_* masks.
#
# We will not silently substitute augmented masks.
# =============================================================================

EXTENSIONS = [
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp"
]


def original_stem_from_id(base_id):

    stem = Path(str(base_id)).stem

    # Reject augmented identifiers.
    if re.search(
        r"__aug_[A-Za-z0-9]+$",
        stem,
        flags=re.IGNORECASE
    ):

        raise RuntimeError(
            f"Augmented Base_ID encountered: {base_id}"
        )

    return stem


def find_original_mask(base_id):

    stem = original_stem_from_id(base_id)

    matches = []

    for path in MASKS_DIR.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() not in EXTENSIONS:
            continue

        if path.stem == stem:

            matches.append(path)

    if len(matches) == 0:

        raise FileNotFoundError(
            f"No exact original mask for {base_id} "
            f"in {MASKS_DIR}"
        )

    if len(matches) > 1:

        raise RuntimeError(
            f"Multiple exact original masks for {base_id}:\n"
            + "\n".join(str(x) for x in matches)
        )

    return matches[0]


# =============================================================================
# 6. BOOTSTRAP FUNCTIONS
# =============================================================================

def bootstrap_mae_ci(
    absolute_errors,
    n_boot=N_BOOT
):

    errors = np.asarray(
        absolute_errors,
        dtype=float
    )

    if len(errors) == 0:

        return np.nan, np.nan

    boot_means = np.empty(n_boot)

    for start in range(0, n_boot, 1000):

        stop = min(
            start + 1000,
            n_boot
        )

        idx = RNG.integers(
            0,
            len(errors),
            size=(stop - start, len(errors))
        )

        boot_means[start:stop] = (
            errors[idx].mean(axis=1)
        )

    low, high = np.percentile(
        boot_means,
        [2.5, 97.5]
    )

    return float(low), float(high)


def calculate_metrics(group):

    n = len(group)

    if n == 0:

        return {
            "n": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "Bias": np.nan,
            "MAE_CI_Low": np.nan,
            "MAE_CI_High": np.nan
        }

    residual = group[
        "Residual"
    ].to_numpy(dtype=float)

    abs_error = np.abs(residual)

    ci_low, ci_high = bootstrap_mae_ci(
        abs_error
    )

    return {
        "n": int(n),
        "MAE": float(abs_error.mean()),
        "RMSE": float(
            np.sqrt(
                np.mean(residual ** 2)
            )
        ),
        "Bias": float(residual.mean()),
        "MAE_CI_Low": ci_low,
        "MAE_CI_High": ci_high
    }


def safe_spearman(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    n = int(valid.sum())

    if n < 4:

        return np.nan, np.nan, n

    if (
        np.unique(x[valid]).size < 2
        or np.unique(y[valid]).size < 2
    ):

        return np.nan, np.nan, n

    rho, p = spearmanr(
        x[valid],
        y[valid]
    )

    return float(rho), float(p), n


# =============================================================================
# 7. LOAD LOCKED TEST PREDICTIONS
# =============================================================================

print("\n" + "=" * 85)
print("LOADING LOCKED TEST PREDICTIONS")
print("=" * 85)

if not PREDICTIONS_CSV.is_file():

    raise FileNotFoundError(
        PREDICTIONS_CSV
    )

if not MASKS_DIR.is_dir():

    raise NotADirectoryError(
        MASKS_DIR
    )

df = pd.read_csv(
    PREDICTIONS_CSV
)

required = [
    "Base_ID",
    "DenseNet_True",
    "DenseNet_Pred",
    "H_norm_atomic5"
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:

    raise ValueError(
        f"Missing prediction columns: {missing}"
    )

if len(df) != EXPECTED_TEST_PARTICLES:

    raise RuntimeError(
        f"Expected 121 locked-test particles, got {len(df)}"
    )

if df["Base_ID"].isna().any():

    raise RuntimeError(
        "Missing Base_ID values."
    )

if df["Base_ID"].duplicated().any():

    raise RuntimeError(
        "Duplicate physical particles in test CSV."
    )

if not np.allclose(
    df["DenseNet_True"].to_numpy(dtype=float),
    df["H_norm_atomic5"].to_numpy(dtype=float),
    atol=1e-6
):

    raise RuntimeError(
        "DenseNet_True disagrees with H_norm_atomic5."
    )

# Keep only the columns needed for this analysis.
# This removes all previously extracted raw BSE descriptors.
df = df[
    [
        "Base_ID",
        "DenseNet_True",
        "DenseNet_Pred"
    ]
].copy()

df["H_true"] = pd.to_numeric(
    df["DenseNet_True"],
    errors="raise"
)

df["H_pred"] = pd.to_numeric(
    df["DenseNet_Pred"],
    errors="raise"
)

if not np.isfinite(
    df[["H_true", "H_pred"]].to_numpy()
).all():

    raise RuntimeError(
        "Nonfinite target/prediction values."
    )

df["Residual"] = (
    df["H_pred"] - df["H_true"]
)

df["Absolute_Error"] = (
    df["Residual"].abs()
)

print(
    f"PASS: {len(df)} unique locked-test physical particles."
)


# =============================================================================
# 8. RECREATE DENSENET MASK INPUTS AND CALCULATE OCCUPANCY
# =============================================================================

print("\n" + "=" * 85)
print("RECREATING EXACT DENSENET MASK PREPROCESSING")
print("=" * 85)

mask_records = []

for i, row in df.iterrows():

    base_id = row["Base_ID"]

    mask_path = find_original_mask(
        base_id
    )

    resized_mask = load_densenet_mask(
        mask_path,
        img_size=IMG_SIZE
    )

    if resized_mask.shape != (
        IMG_SIZE,
        IMG_SIZE
    ):

        raise RuntimeError(
            f"Unexpected mask shape for {base_id}"
        )

    if not np.isin(
        np.unique(resized_mask),
        [0, 1]
    ).all():

        raise RuntimeError(
            f"Mask is not binary for {base_id}"
        )

    area_px = int(
        np.count_nonzero(resized_mask)
    )

    occupancy_pct = (
        100.0 * area_px
        / (IMG_SIZE * IMG_SIZE)
    )

    mask_records.append({
        "Base_ID": base_id,
        "Mask_Path": str(mask_path),
        "Resized_Mask_Height": IMG_SIZE,
        "Resized_Mask_Width": IMG_SIZE,
        "Resized_Mask_Area_px": area_px,
        "Mask_Occupancy_Percent": occupancy_pct
    })

mask_table = pd.DataFrame(
    mask_records
)

mask_table.to_csv(
    OUT / "resized_mask_area_verification.csv",
    index=False
)

df = df.merge(
    mask_table,
    on="Base_ID",
    how="left",
    validate="one_to_one"
)

if df[
    "Mask_Occupancy_Percent"
].isna().any():

    raise RuntimeError(
        "Missing mask occupancy values."
    )

print(
    "PASS: All 121 original masks processed "
    "using DenseNet's binary conversion and nearest-neighbor resize."
)

print("\nResized mask area (pixels):")

print(
    df["Resized_Mask_Area_px"].describe()
)

print("\nDenseNet mask occupancy (%):")

print(
    df["Mask_Occupancy_Percent"].describe()
)


# =============================================================================
# 9. DEFINE H_norm GROUPS
# =============================================================================

df["Broad_Regime"] = pd.cut(
    df["H_true"],
    bins=BROAD_EDGES,
    labels=BROAD_LABELS,
    right=False
)

df["Fine_Bin"] = pd.cut(
    df["H_true"],
    bins=FINE_EDGES,
    labels=FINE_LABELS,
    right=False
)


# =============================================================================
# 10. OVERALL BOOTSTRAP PERFORMANCE
# =============================================================================

overall_metrics = calculate_metrics(
    df
)

overall_r2 = r2_score(
    df["H_true"],
    df["H_pred"]
)

overall_table = pd.DataFrame([
    {
        "Group": "All locked-test particles",
        **overall_metrics,
        "R2": overall_r2
    }
])

overall_table.to_csv(
    OUT / "overall_bootstrap_metrics.csv",
    index=False
)

print("\n" + "=" * 85)
print("OVERALL DENSENET121 PERFORMANCE")
print("=" * 85)

print(
    overall_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# Verify previously confirmed model performance.
if not np.isclose(
    overall_metrics["MAE"],
    0.030887,
    atol=1e-5
):

    raise RuntimeError(
        "DenseNet MAE does not match verified locked-test result."
    )

if not np.isclose(
    overall_metrics["RMSE"],
    0.051410,
    atol=1e-5
):

    raise RuntimeError(
        "DenseNet RMSE does not match verified locked-test result."
    )


# =============================================================================
# 11. THREE BROAD H_norm REGIMES
# =============================================================================

broad_records = []

for regime in BROAD_LABELS:

    subset = df[
        df["Broad_Regime"] == regime
    ]

    broad_records.append({
        "Regime": regime,
        **calculate_metrics(subset),
        "Percent_of_Test_Set": (
            100.0 * len(subset) / len(df)
        )
    })

broad_table = pd.DataFrame(
    broad_records
)

broad_table.to_csv(
    OUT / "broad_regime_bootstrap_metrics.csv",
    index=False
)

print("\n" + "=" * 85)
print("THREE H_norm REGIMES")
print("=" * 85)

print(
    broad_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =============================================================================
# 12. SEVEN FINE H_norm BINS
# =============================================================================

fine_records = []

for label in FINE_LABELS:

    subset = df[
        df["Fine_Bin"] == label
    ]

    fine_records.append({
        "H_norm_Bin": label,
        **calculate_metrics(subset)
    })

fine_table = pd.DataFrame(
    fine_records
)

fine_table.to_csv(
    OUT / "fine_bin_bootstrap_metrics.csv",
    index=False
)

print("\n" + "=" * 85)
print("SEVEN H_norm BINS")
print("=" * 85)

print(
    fine_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =============================================================================
# 13. BOOTSTRAP DIFFERENCE: LOW MAE - HIGH MAE
# =============================================================================

low_errors = df.loc[
    df["Broad_Regime"] == "Low",
    "Absolute_Error"
].to_numpy()

high_errors = df.loc[
    df["Broad_Regime"] == "High",
    "Absolute_Error"
].to_numpy()

bootstrap_differences = np.empty(
    N_BOOT
)

for b in range(N_BOOT):

    low_sample = low_errors[
        RNG.integers(
            0,
            len(low_errors),
            size=len(low_errors)
        )
    ]

    high_sample = high_errors[
        RNG.integers(
            0,
            len(high_errors),
            size=len(high_errors)
        )
    ]

    bootstrap_differences[b] = (
        low_sample.mean()
        -
        high_sample.mean()
    )

difference_ci = np.percentile(
    bootstrap_differences,
    [2.5, 97.5]
)

difference_table = pd.DataFrame([
    {
        "Comparison": "Low MAE minus High MAE",
        "Observed_Difference": (
            low_errors.mean()
            -
            high_errors.mean()
        ),
        "CI_95_Low": difference_ci[0],
        "CI_95_High": difference_ci[1]
    }
])

difference_table.to_csv(
    OUT / "low_vs_high_MAE_difference.csv",
    index=False
)

print("\nLow-minus-high MAE difference:")

print(
    difference_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =============================================================================
# 14. MASK OCCUPANCY CORRELATIONS
# =============================================================================

association_records = []

groups = [
    ("Overall", df)
]

for regime in BROAD_LABELS:

    groups.append(
        (
            regime,
            df[df["Broad_Regime"] == regime]
        )
    )

for group_name, subset in groups:

    for outcome in [
        "Absolute_Error",
        "Residual"
    ]:

        rho, p_value, n = safe_spearman(
            subset["Mask_Occupancy_Percent"],
            subset[outcome]
        )

        association_records.append({
            "Group": group_name,
            "Outcome": outcome,
            "n": n,
            "Spearman_rho": rho,
            "p_value_exploratory": p_value
        })

association_table = pd.DataFrame(
    association_records
)

association_table.to_csv(
    OUT / "occupancy_error_correlations.csv",
    index=False
)

print("\n" + "=" * 85)
print("MASK OCCUPANCY VS PREDICTION ERROR")
print("=" * 85)

print(
    association_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =============================================================================
# 15. MASK OCCUPANCY QUARTILES
# =============================================================================

df["Occupancy_Quartile"] = pd.qcut(
    df["Mask_Occupancy_Percent"],
    q=4,
    labels=[
        "Q1",
        "Q2",
        "Q3",
        "Q4"
    ]
)

quartile_records = []

for quartile in [
    "Q1",
    "Q2",
    "Q3",
    "Q4"
]:

    subset = df[
        df["Occupancy_Quartile"] == quartile
    ]

    quartile_records.append({
        "Quartile": quartile,
        "Median_Occupancy_Percent": (
            subset["Mask_Occupancy_Percent"].median()
        ),
        "Min_Occupancy_Percent": (
            subset["Mask_Occupancy_Percent"].min()
        ),
        "Max_Occupancy_Percent": (
            subset["Mask_Occupancy_Percent"].max()
        ),
        **calculate_metrics(subset)
    })

quartile_table = pd.DataFrame(
    quartile_records
)

quartile_table.to_csv(
    OUT / "occupancy_quartile_bootstrap_metrics.csv",
    index=False
)

print("\n" + "=" * 85)
print("MASK OCCUPANCY QUARTILE PERFORMANCE")
print("=" * 85)

print(
    quartile_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =============================================================================
# 16. FIGURE 1 — PARITY BY H_norm REGIME
# =============================================================================

fig, ax = plt.subplots(
    figsize=(4.8, 4.5)
)

for regime in BROAD_LABELS:

    subset = df[
        df["Broad_Regime"] == regime
    ]

    ax.scatter(
        subset["H_true"],
        subset["H_pred"],
        s=29,
        alpha=0.82,
        color=COLORS[regime],
        edgecolors="white",
        linewidths=0.35,
        label=f"{regime} (n={len(subset)})"
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="#333333",
    linewidth=1,
    label="Ideal prediction"
)

ax.set_xlabel(
    r"EDS-derived $H_{\mathrm{norm}}$"
)

ax.set_ylabel(
    r"DenseNet121-predicted $H_{\mathrm{norm}}$"
)

ax.set_xlim(0.2, 1.02)
ax.set_ylim(0.2, 1.02)

ax.set_aspect(
    "equal",
    adjustable="box"
)

style_axes(ax)

ax.legend(
    frameon=False,
    loc="upper left"
)

save_figure(
    fig,
    "Fig01_parity_by_Hnorm"
)


# =============================================================================
# 17. FIGURE 2 — RESIDUAL VS TRUE H_norm
# =============================================================================

fig, ax = plt.subplots(
    figsize=(5.4, 3.6)
)

for regime in BROAD_LABELS:

    subset = df[
        df["Broad_Regime"] == regime
    ]

    ax.scatter(
        subset["H_true"],
        subset["Residual"],
        s=28,
        alpha=0.8,
        color=COLORS[regime],
        edgecolors="white",
        linewidths=0.3,
        label=f"{regime} (n={len(subset)})"
    )

ax.axhline(
    0,
    color="#333333",
    linestyle="--",
    linewidth=1
)

for boundary in [0.80, 0.95]:

    ax.axvline(
        boundary,
        color="#999999",
        linestyle=":",
        linewidth=0.8
    )

ax.set_xlabel(
    r"EDS-derived $H_{\mathrm{norm}}$"
)

ax.set_ylabel(
    r"Residual ($H_{\mathrm{pred}}-H_{\mathrm{true}}$)"
)

ax.set_xlim(0.2, 1.02)

style_axes(ax)

ax.legend(
    frameon=False,
    ncol=2
)

save_figure(
    fig,
    "Fig02_residual_vs_Hnorm"
)


# =============================================================================
# 18. FIGURE 3 — THREE-REGIME MAE WITH BOOTSTRAP CI
# =============================================================================

fig, ax = plt.subplots(
    figsize=(4.8, 3.6)
)

x = np.arange(
    len(broad_table)
)

y = broad_table[
    "MAE"
].to_numpy()

ci_low = broad_table[
    "MAE_CI_Low"
].to_numpy()

ci_high = broad_table[
    "MAE_CI_High"
].to_numpy()

ax.bar(
    x,
    y,
    width=0.6,
    color=[
        COLORS[r]
        for r in broad_table["Regime"]
    ]
)

ax.errorbar(
    x,
    y,
    yerr=[
        y - ci_low,
        ci_high - y
    ],
    fmt="none",
    ecolor="#222222",
    capsize=4,
    elinewidth=1.1
)

ax.set_xticks(x)

ax.set_xticklabels([
    f"{row['Regime']}\n(n={int(row['n'])})"
    for _, row in broad_table.iterrows()
])

ax.set_ylabel(
    r"Mean absolute error in $H_{\mathrm{norm}}$"
)

style_axes(ax)

save_figure(
    fig,
    "Fig03_broad_MAE_bootstrap_CI"
)


# =============================================================================
# 19. FIGURE 4 — SEVEN-BIN MAE WITH BOOTSTRAP CI
# =============================================================================

fig, ax = plt.subplots(
    figsize=(7.5, 3.8)
)

x = np.arange(
    len(fine_table)
)

y = fine_table[
    "MAE"
].to_numpy()

ci_low = fine_table[
    "MAE_CI_Low"
].to_numpy()

ci_high = fine_table[
    "MAE_CI_High"
].to_numpy()

ax.bar(
    x,
    y,
    width=0.67,
    color="#477F9B"
)

ax.errorbar(
    x,
    y,
    yerr=[
        y - ci_low,
        ci_high - y
    ],
    fmt="none",
    ecolor="#222222",
    capsize=3
)

ax.set_xticks(x)

ax.set_xticklabels(
    FINE_LABELS,
    rotation=25,
    ha="right"
)

ax.set_ylabel(
    r"Mean absolute error in $H_{\mathrm{norm}}$"
)

ax.set_ylim(
    0,
    fine_table["MAE_CI_High"].max() * 1.28
)

for i, row in fine_table.iterrows():

    ax.annotate(
        f"n={int(row['n'])}",
        (i, row["MAE_CI_High"]),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=8
    )

style_axes(ax)

save_figure(
    fig,
    "Fig04_fine_MAE_bootstrap_CI"
)


# =============================================================================
# 20. FIGURE 5 — ABSOLUTE ERROR VS RESIZED MASK OCCUPANCY
# =============================================================================

fig, ax = plt.subplots(
    figsize=(5.5, 3.7)
)

for regime in BROAD_LABELS:

    subset = df[
        df["Broad_Regime"] == regime
    ]

    ax.scatter(
        subset["Mask_Occupancy_Percent"],
        subset["Absolute_Error"],
        s=30,
        alpha=0.78,
        color=COLORS[regime],
        edgecolors="white",
        linewidths=0.3,
        label=f"{regime} (n={len(subset)})"
    )

ax.set_xlabel(
    "Particle mask occupancy in 512 × 512 input (%)"
)

ax.set_ylabel(
    r"Absolute prediction error in $H_{\mathrm{norm}}$"
)

ax.set_xlim(0, 100)

style_axes(ax)

ax.legend(frameon=False)

save_figure(
    fig,
    "Fig05_absolute_error_vs_mask_occupancy"
)


# =============================================================================
# 21. FIGURE 6 — SIGNED RESIDUAL VS RESIZED MASK OCCUPANCY
# =============================================================================

fig, ax = plt.subplots(
    figsize=(5.5, 3.7)
)

for regime in BROAD_LABELS:

    subset = df[
        df["Broad_Regime"] == regime
    ]

    ax.scatter(
        subset["Mask_Occupancy_Percent"],
        subset["Residual"],
        s=30,
        alpha=0.78,
        color=COLORS[regime],
        edgecolors="white",
        linewidths=0.3,
        label=f"{regime} (n={len(subset)})"
    )

ax.axhline(
    0,
    color="#333333",
    linestyle="--",
    linewidth=1
)

ax.set_xlabel(
    "Particle mask occupancy in 512 × 512 input (%)"
)

ax.set_ylabel(
    r"Residual ($H_{\mathrm{pred}}-H_{\mathrm{true}}$)"
)

ax.set_xlim(0, 100)

style_axes(ax)

ax.legend(frameon=False)

save_figure(
    fig,
    "Fig06_residual_vs_mask_occupancy"
)


# =============================================================================
# 22. FIGURE 7 — OCCUPANCY QUARTILE MAE WITH BOOTSTRAP CI
# =============================================================================

fig, ax = plt.subplots(
    figsize=(5.2, 3.7)
)

x = np.arange(
    len(quartile_table)
)

y = quartile_table[
    "MAE"
].to_numpy()

ci_low = quartile_table[
    "MAE_CI_Low"
].to_numpy()

ci_high = quartile_table[
    "MAE_CI_High"
].to_numpy()

ax.bar(
    x,
    y,
    width=0.62,
    color="#477F9B"
)

ax.errorbar(
    x,
    y,
    yerr=[
        y - ci_low,
        ci_high - y
    ],
    fmt="none",
    ecolor="#222222",
    capsize=4
)

ax.set_xticks(x)

ax.set_xticklabels([
    f"{row['Quartile']}\n(n={int(row['n'])})"
    for _, row in quartile_table.iterrows()
])

ax.set_xlabel(
    "Mask occupancy quartile (Q1 = lowest)"
)

ax.set_ylabel(
    r"Mean absolute error in $H_{\mathrm{norm}}$"
)

style_axes(ax)

save_figure(
    fig,
    "Fig07_occupancy_quartile_MAE_bootstrap_CI"
)


# =============================================================================
# 23. EXPORT COMPLETE PARTICLE-LEVEL DATA
# =============================================================================

df.to_csv(
    OUT / "locked_121_particle_full_analysis.csv",
    index=False,
    float_format="%.10f"
)


# =============================================================================
# 24. REPRODUCIBILITY MANIFEST
# =============================================================================

manifest = [
    "REVIEWER 1 + REVIEWER 2 — FINAL ANALYSIS",
    "",
    f"Predictions: {PREDICTIONS_CSV}",
    f"Original masks: {MASKS_DIR}",
    "",
    f"Original locked-test particles: {len(df)}",
    f"Unique Base_ID: {df['Base_ID'].nunique()}",
    "",
    f"Bootstrap resamples: {N_BOOT}",
    f"Random seed: {SEED}",
    "Bootstrap method: nonparametric percentile CI",
    "Bootstrap unit: original physical particle",
    "",
    "Mask preprocessing reproduced from DenseNet121 training:",
    "cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)",
    "ensure_binary(mask)",
    "cv2.resize(mask, (512,512), cv2.INTER_NEAREST)",
    "",
    "Mask occupancy =",
    "100 * foreground pixels in resized mask / 262144",
    "",
    "Original BSE features CSV: NOT USED",
    "Original Area_px: NOT USED",
    "Original BSE descriptors: NOT USED",
    "",
    "Mask occupancy is not calibrated physical particle size.",
    "H_norm is EDS-derived elemental compositional evenness.",
    "",
    "No model training.",
    "No model inference.",
    "No new data split.",
    "No augmented particles as independent observations.",
    "",
    "Spearman tests are exploratory and unadjusted.",
    "Fine H_norm bins with few particles have uncertain estimates."
]

(OUT / "analysis_manifest.txt").write_text(
    "\n".join(manifest),
    encoding="utf-8"
)


# =============================================================================
# 25. FINAL REPORT
# =============================================================================

print("\n" + "=" * 85)
print("FINAL ANALYSIS COMPLETED")
print("=" * 85)

print(
    "\nLocked-test physical particles:",
    len(df)
)

print(
    "Overall MAE:",
    f"{overall_metrics['MAE']:.6f}"
)

print(
    "Overall RMSE:",
    f"{overall_metrics['RMSE']:.6f}"
)

print(
    "Overall R2:",
    f"{overall_r2:.6f}"
)

print(
    "Overall MAE bootstrap 95% CI:",
    f"[{overall_metrics['MAE_CI_Low']:.6f}, "
    f"{overall_metrics['MAE_CI_High']:.6f}]"
)

print("\nFigures generated: 7")
print("Each figure saved as PDF, SVG, and 600-dpi PNG.")

print("\nAll outputs saved to:")
print(OUT)

print("\nCONFIRMED:")
print("No BSE features CSV was used.")
print("No original-resolution Area_px was used.")
print("Masks were resized exactly as in DenseNet121 training.")
print("Occupancy was calculated from the resized 512 x 512 masks.")
