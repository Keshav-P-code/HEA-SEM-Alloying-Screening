# =============================================================================

from pathlib import Path
import os
import re
import json
import hashlib
import warnings
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# =============================================================================
# 1. PATHS — CHANGE ONLY IF YOUR FOLDERS HAVE MOVED
# =============================================================================

ROOT = Path(r"PATH_TO_SEM_EDS_IMAGES_DIRECTORY"
)

IMAGES_DIR = ROOT / "augmented_images"
MASKS_DIR = ROOT / "Augmented_output_masks" / "masks"

LABEL_CSV = (
    ROOT / "Data-verse Ready" / "HEA10_30_labels-Aug - Copy.csv"
)

BASELINE_ROOT = ROOT / "Data-verse Ready" / "bse_baseline_results"

# Historical RAW baseline feature table.
RAW_FEATURE_CSV = (
    BASELINE_ROOT / "bse_baseline_features_originals.csv"
)

# Historical locked-test predictions:
# Contains raw entropy, raw Ridge and DenseNet121 predictions.
OLD_PREDICTIONS_CSV = (
    BASELINE_ROOT
    / "analysis_v2_locked_densenet_test"
    / "locked_test_particle_predictions.csv"
)

# Historical DenseNet predictions defining test membership.
DENSENET_PREDICTIONS_CSV = Path(
    r"PATH_TO_DENSENET_TEST_PREDICTIONS_CSV"
)

# ALL NEW RESULTS GO INTO THIS FOLDER.
OUTPUT_DIR = (
    BASELINE_ROOT
    / "CLAHE_RESIZED_BSE_BASELINES_LOCKED_TEST_121"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("NEW EXPERIMENT: CLAHE + RESIZED BSE BASELINES")
print("=" * 90)
print("\nALL NEW OUTPUTS WILL BE SAVED HERE:")
print(OUTPUT_DIR.resolve())
print("\nExisting baseline results will NOT be overwritten.")

# =============================================================================
# 2. FIXED EXPERIMENT SETTINGS
# =============================================================================

SEED = 42
TARGET = "H_norm_atomic5"

IMG_SIZE = 512
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

HIST_BINS = 256
HIST_RANGE = (0, 256)

CV_FOLDS = 5

RIDGE_ALPHAS = [
    1e-4, 1e-3, 1e-2, 1e-1,
    1.0, 10.0, 100.0, 1000.0
]

DROP_BASE_IDS = {"Particle_733", "Particle_795"}

FEATURES = [
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

EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# =============================================================================
# 3. GENERAL HELPERS
# =============================================================================

def split_stem_aug(pid):
    s = os.path.splitext(str(pid))[0]
    match = re.search(r"^(.+?)__aug_([A-Za-z0-9]+)$", s)

    if match:
        return match.group(1), "aug_" + match.group(2).lower()

    return s, "original"


def ensure_binary(mask):
    """Exact binary-mask logic from the original extraction script."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask.dtype != np.uint8:
        mask = cv2.normalize(
            mask, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

    return (mask > 127).astype(np.uint8)


def bse_histogram_entropy(pixel_values):
    """Same fixed 256-bin Shannon entropy as the original baseline."""
    counts, _ = np.histogram(
        pixel_values,
        bins=HIST_BINS,
        range=HIST_RANGE
    )

    counts = counts.astype(np.float64)
    total = counts.sum()

    if total <= 0:
        return np.nan

    probabilities = counts / total
    probabilities = probabilities[probabilities > 0]

    return float(
        -np.sum(probabilities * np.log(probabilities))
    )


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def check_finite(df, columns, name):
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"{name}: missing columns: {missing}"
        )

    values = df[columns].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"{name}: NaN or infinite numeric values found."
        )


def index_files(folder):
    """
    Index files by EXACT filename stem.
    No prefix matching that could accidentally select augmented images.
    """
    if not folder.is_dir():
        raise FileNotFoundError(folder)

    index = {}

    for path in folder.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in EXTENSIONS:
            continue

        index.setdefault(path.stem, []).append(path)

    return index


def get_exact_file(index, stem, kind):
    matches = index.get(stem, [])

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No {kind} found with exact stem '{stem}'"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous {kind} for '{stem}':\n"
            + "\n".join(map(str, matches))
        )

    return matches[0]


# =============================================================================
# 4. EXACT ORIGINAL FEATURE DEFINITIONS
# =============================================================================

def extract_descriptors(pixels):
    """
    Preserve original Code 1 definitions:
      std ddof=0
      CV = std / mean
      skew bias=False
      kurtosis fisher=True, bias=False
      fixed-bin Shannon entropy
    """
    pixels = np.asarray(pixels, dtype=np.float64)

    if pixels.size == 0:
        raise ValueError("Empty foreground pixels.")

    mean_value = float(np.mean(pixels))
    std_value = float(np.std(pixels, ddof=0))

    cv_value = (
        float(std_value / mean_value)
        if abs(mean_value) > 1e-12
        else np.nan
    )

    p10, p25, p75, p90 = np.percentile(
        pixels, [10, 25, 75, 90]
    )

    if std_value > 1e-12:
        skew_value = float(skew(pixels, bias=False))
        kurt_value = float(
            kurtosis(
                pixels,
                fisher=True,
                bias=False
            )
        )
    else:
        skew_value = 0.0
        kurt_value = 0.0

    return {
        "Area_px": int(pixels.size),
        "Mean": mean_value,
        "Std": std_value,
        "CV": cv_value,
        "Median": float(np.median(pixels)),
        "IQR": float(p75 - p25),
        "P10": float(p10),
        "P90": float(p90),
        "Skewness": skew_value,
        "Kurtosis": kurt_value,
        "BSE_Entropy": bse_histogram_entropy(pixels),
        "Min_Intensity": float(np.min(pixels)),
        "Max_Intensity": float(np.max(pixels)),
    }


# =============================================================================
# 5. PREPROCESS ONE ORIGINAL IMAGE + MASK
# =============================================================================

def preprocess_and_extract(image_path, mask_path):
    """
    Main sensitivity experiment:
      grayscale -> CLAHE -> INTER_AREA resize
      mask -> ensure_binary -> INTER_NEAREST resize
      extract foreground pixels -> original 10 descriptors

    No per-image z-normalization.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    if mask is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")

    mask = ensure_binary(mask)

    if image.shape != mask.shape:
        raise RuntimeError(
            f"Image/mask mismatch: {image_path.name} "
            f"image={image.shape}, mask={mask.shape}"
        )

    original_shape = image.shape

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID
    )

    image_clahe = clahe.apply(image)

    image_512 = cv2.resize(
        image_clahe,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    mask_512 = cv2.resize(
        mask,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    # Equivalent foreground values to gating by the binary mask.
    pixels = image_512[mask_512.astype(bool)]

    descriptors = extract_descriptors(pixels)

    descriptors["Original_Height"] = int(original_shape[0])
    descriptors["Original_Width"] = int(original_shape[1])
    descriptors["Mask_Occupancy_Percent"] = (
        100.0 * np.count_nonzero(mask_512)
        / float(IMG_SIZE * IMG_SIZE)
    )

    return descriptors


# =============================================================================
# 6. READ AND VERIFY ORIGINAL LABELS
# =============================================================================

for required_path in [
    LABEL_CSV,
    RAW_FEATURE_CSV,
    OLD_PREDICTIONS_CSV,
    DENSENET_PREDICTIONS_CSV,
]:
    if not required_path.is_file():
        raise FileNotFoundError(
            f"Required input file missing:\n{required_path}"
        )

labels = pd.read_csv(LABEL_CSV)
labels.columns = labels.columns.str.strip()

for col in ["Particle_ID", TARGET]:
    if col not in labels.columns:
        raise RuntimeError(
            f"Label CSV missing required column: {col}"
        )

parsed = labels["Particle_ID"].astype(str).map(split_stem_aug)

labels["Base_ID"] = [x[0] for x in parsed]
labels["Aug_Tag"] = [x[1] for x in parsed]

originals = labels.loc[
    (labels["Aug_Tag"] == "original")
    & (~labels["Base_ID"].isin(DROP_BASE_IDS))
].copy()

if originals["Base_ID"].duplicated().any():
    raise RuntimeError(
        "Duplicate original Base_ID in labels."
    )

originals = originals.reset_index(drop=True)

print("\nOriginal usable physical particles:", len(originals))

# =============================================================================
# 7. VERIFY AGAINST HISTORICAL RAW FEATURE TABLE
# =============================================================================

raw_features = pd.read_csv(RAW_FEATURE_CSV)
raw_features.columns = raw_features.columns.str.strip()

if raw_features["Base_ID"].duplicated().any():
    raise RuntimeError(
        "Duplicate Base_ID in original raw BSE feature table."
    )

label_ids = set(originals["Base_ID"])
raw_ids = set(raw_features["Base_ID"])

if label_ids != raw_ids:
    raise RuntimeError(
        "Original label IDs do not match the historical "
        "raw BSE feature table.\n"
        f"Only in labels: {sorted(label_ids - raw_ids)[:20]}\n"
        f"Only in raw features: {sorted(raw_ids - label_ids)[:20]}"
    )

raw_target_check = originals[
    ["Base_ID", TARGET]
].merge(
    raw_features[["Base_ID", TARGET]],
    on="Base_ID",
    suffixes=("_labels", "_raw"),
    validate="one_to_one"
)

max_raw_target_diff = np.max(np.abs(
    raw_target_check[f"{TARGET}_labels"].to_numpy(dtype=float)
    - raw_target_check[f"{TARGET}_raw"].to_numpy(dtype=float)
))

if max_raw_target_diff > 1e-6:
    raise RuntimeError(
        "EDS target mismatch between labels and raw features."
    )

print("PASS: Original labels match historical raw feature table.")

# =============================================================================
# 8. LOCK HISTORICAL DENSENET TEST PARTICLES
# =============================================================================

dense_raw = pd.read_csv(DENSENET_PREDICTIONS_CSV)
dense_raw.columns = dense_raw.columns.str.strip()

if "Particle_ID" not in dense_raw.columns:
    raise RuntimeError(
        "DenseNet predictions missing Particle_ID."
    )

parsed_dense = (
    dense_raw["Particle_ID"]
    .astype(str)
    .map(split_stem_aug)
)

dense_raw["Base_ID"] = [x[0] for x in parsed_dense]
dense_raw["Aug_Tag"] = [x[1] for x in parsed_dense]

dense_original = dense_raw.loc[
    dense_raw["Aug_Tag"] == "original"
].copy()

if dense_original["Base_ID"].duplicated().any():
    raise RuntimeError(
        "Duplicate original Base_ID in DenseNet predictions."
    )

true_candidates = [
    "True", "y_true", "True_H_norm", "H_norm_atomic5"
]
pred_candidates = [
    "Pred", "Prediction", "y_pred", "Pred_H_norm"
]

true_col = next(
    (c for c in true_candidates if c in dense_original.columns),
    None
)

pred_col = next(
    (c for c in pred_candidates if c in dense_original.columns),
    None
)

if true_col is None or pred_col is None:
    raise RuntimeError(
        "Could not identify DenseNet true/prediction columns.\n"
        f"Available: {list(dense_original.columns)}"
    )

dense_original = dense_original[
    ["Base_ID", true_col, pred_col]
].rename(
    columns={
        true_col: "DenseNet_True",
        pred_col: "DenseNet_Pred"
    }
)

test_ids = set(dense_original["Base_ID"])

if len(test_ids) != 121:
    raise RuntimeError(
        f"Expected exactly 121 locked test particles; "
        f"found {len(test_ids)}."
    )

if not test_ids.issubset(label_ids):
    raise RuntimeError(
        "Some historical DenseNet test particles "
        "are absent from original labels."
    )

originals["Split"] = np.where(
    originals["Base_ID"].isin(test_ids),
    "locked_test",
    "development"
)

print("PASS: Locked DenseNet test particles:", len(test_ids))
print("Development particles:", (originals["Split"] == "development").sum())

# =============================================================================
# 9. EXTRACT PREPROCESSED FEATURES — ALL ORIGINAL PARTICLES
# =============================================================================

print("\n" + "=" * 90)
print("EXTRACTING CLAHE + RESIZED FOREGROUND DESCRIPTORS")
print("=" * 90)

image_index = index_files(IMAGES_DIR)
mask_index = index_files(MASKS_DIR)

records = []
failures = []

for i, row in originals.iterrows():

    particle_id = str(row["Particle_ID"])
    base_id = str(row["Base_ID"])

    try:
        image_path = get_exact_file(
            image_index, base_id, "image"
        )

        mask_path = get_exact_file(
            mask_index, base_id, "mask"
        )

        features = preprocess_and_extract(
            image_path,
            mask_path
        )

        record = {
            "Particle_ID": particle_id,
            "Base_ID": base_id,
            TARGET: float(row[TARGET]),
            "Split": row["Split"],
            "Image_Path": str(image_path),
            "Mask_Path": str(mask_path),
        }

        for optional_col in ["Sample_ID", "Spectrum"]:
            if optional_col in originals.columns:
                record[optional_col] = row[optional_col]

        record.update(features)
        records.append(record)

    except Exception as exc:
        failures.append((base_id, str(exc)))

    if (i + 1) % 100 == 0 or (i + 1) == len(originals):
        print(f"Processed {i + 1:,}/{len(originals):,}")

if failures:
    print("\nEXTRACTION FAILURES:")
    for item in failures[:30]:
        print(item)

    raise RuntimeError(
        f"{len(failures)} particle(s) failed extraction. "
        "Stopping before regression."
    )

features_df = pd.DataFrame(records)

if len(features_df) != len(originals):
    raise RuntimeError(
        "Feature extraction lost original particles."
    )

if features_df["Base_ID"].duplicated().any():
    raise RuntimeError(
        "Duplicate Base_ID after extraction."
    )

check_finite(
    features_df,
    [TARGET, *FEATURES],
    "Preprocessed BSE feature table"
)

# Verify exact ID membership.
if set(features_df["Base_ID"]) != raw_ids:
    raise RuntimeError(
        "Preprocessed and raw feature sets differ in Base_ID."
    )

feature_path = (
    OUTPUT_DIR
    / "01_preprocessed_BSE_features_all_originals.csv"
)

features_df.to_csv(
    feature_path,
    index=False,
    float_format="%.10f"
)

print("\nPASS: Preprocessed features extracted:", len(features_df))
print("Saved:", feature_path.name)

# =============================================================================
# 10. SAVE LOCKED SPLIT ASSIGNMENTS
# =============================================================================

split_df = features_df[
    ["Particle_ID", "Base_ID", TARGET, "Split"]
].copy()

split_path = (
    OUTPUT_DIR / "02_locked_test_split_assignments.csv"
)

split_df.to_csv(split_path, index=False)

dev_df = features_df.loc[
    features_df["Split"] == "development"
].copy()

test_df = features_df.loc[
    features_df["Split"] == "locked_test"
].copy()

if len(test_df) != 121:
    raise RuntimeError(
        "Locked test must contain 121 particles."
    )

if set(dev_df["Base_ID"]) & set(test_df["Base_ID"]):
    raise RuntimeError(
        "Development/test Base_ID leakage detected."
    )

# =============================================================================
# 11. TRAIN ENTROPY-ONLY LINEAR REGRESSION
# =============================================================================

print("\n" + "=" * 90)
print("MODEL A: PREPROCESSED BSE ENTROPY -> LINEAR REGRESSION")
print("=" * 90)

y_dev = dev_df[TARGET].to_numpy(dtype=float)
y_test = test_df[TARGET].to_numpy(dtype=float)

X_dev_entropy = dev_df[
    ["BSE_Entropy"]
].to_numpy(dtype=float)

X_test_entropy = test_df[
    ["BSE_Entropy"]
].to_numpy(dtype=float)

entropy_model = LinearRegression()
entropy_model.fit(X_dev_entropy, y_dev)

entropy_pred = entropy_model.predict(X_test_entropy)
entropy_metrics = regression_metrics(y_test, entropy_pred)

print("Fitted entropy-only model:")
print(
    f"Predicted H_norm = {entropy_model.intercept_:.8f} "
    f"+ ({entropy_model.coef_[0]:.8f}) * BSE_Entropy"
)
print("Locked-test metrics:", entropy_metrics)

# =============================================================================
# 12. TRAIN RIDGE WITH DEVELOPMENT-ONLY CV
# =============================================================================

print("\n" + "=" * 90)
print("MODEL B: PREPROCESSED BSE DESCRIPTORS -> RIDGE")
print("=" * 90)

X_dev = dev_df[FEATURES].to_numpy(dtype=float)
X_test = test_df[FEATURES].to_numpy(dtype=float)

cv = KFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=SEED
)

cv_records = []

for alpha in RIDGE_ALPHAS:

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])

    fold_maes = -cross_val_score(
        pipeline,
        X_dev,
        y_dev,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    cv_records.append({
        "alpha": alpha,
        "CV_MAE_mean": float(np.mean(fold_maes)),
        "CV_MAE_std": float(np.std(fold_maes, ddof=1)),
    })

cv_results = pd.DataFrame(cv_records).sort_values(
    ["CV_MAE_mean", "CV_MAE_std"]
).reset_index(drop=True)

best_alpha = float(cv_results.iloc[0]["alpha"])

cv_path = OUTPUT_DIR / "03_ridge_development_CV_results.csv"
cv_results.to_csv(cv_path, index=False, float_format="%.10f")

ridge_model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=best_alpha)),
])

ridge_model.fit(X_dev, y_dev)

ridge_pred = ridge_model.predict(X_test)
ridge_metrics = regression_metrics(y_test, ridge_pred)

print("\nRidge CV results:")
print(cv_results.to_string(index=False))
print("\nSelected alpha:", best_alpha)
print("Locked-test metrics:", ridge_metrics)

# =============================================================================
# 13. ALIGN HISTORICAL RAW BASELINES + DENSENET
# =============================================================================

old_predictions = pd.read_csv(OLD_PREDICTIONS_CSV)
old_predictions.columns = old_predictions.columns.str.strip()

required_old_cols = [
    "Base_ID",
    TARGET,
    "EntropyLinear_Pred",
    "Ridge_Pred",
    "DenseNet_Pred",
]

missing_old = [
    c for c in required_old_cols
    if c not in old_predictions.columns
]

if missing_old:
    raise RuntimeError(
        f"Historical prediction CSV missing: {missing_old}"
    )

if old_predictions["Base_ID"].duplicated().any():
    raise RuntimeError(
        "Duplicate Base_ID in historical prediction CSV."
    )

if set(old_predictions["Base_ID"]) != test_ids:
    raise RuntimeError(
        "Historical prediction CSV does not contain "
        "exactly the 121 locked test Base_IDs."
    )

comparison = test_df[
    ["Particle_ID", "Base_ID", TARGET]
].copy()

comparison["CLAHE_EntropyLinear_Pred"] = entropy_pred
comparison["CLAHE_Ridge_Pred"] = ridge_pred

comparison = comparison.merge(
    old_predictions[
        [
            "Base_ID",
            TARGET,
            "EntropyLinear_Pred",
            "Ridge_Pred",
            "DenseNet_Pred",
        ]
    ],
    on="Base_ID",
    how="left",
    validate="one_to_one",
    suffixes=("", "_historical")
)

comparison = comparison.merge(
    dense_original,
    on="Base_ID",
    how="left",
    validate="one_to_one",
    suffixes=("", "_historical_dense")
)

target_checks = [
    f"{TARGET}_historical",
    "DenseNet_True",
]

for check_col in target_checks:
    difference = np.abs(
        comparison[TARGET].to_numpy(dtype=float)
        - comparison[check_col].to_numpy(dtype=float)
    )

    if not np.isfinite(difference).all() or np.max(difference) > 1e-6:
        raise RuntimeError(
            f"Target mismatch against {check_col}."
        )

# Historical comparison CSV DenseNet predictions and original
# DenseNet prediction file must also agree.
if "DenseNet_Pred_historical_dense" in comparison.columns:
    pred_difference = np.abs(
        comparison["DenseNet_Pred"].to_numpy(dtype=float)
        - comparison["DenseNet_Pred_historical_dense"].to_numpy(dtype=float)
    )

    if not np.isfinite(pred_difference).all() or np.max(pred_difference) > 1e-6:
        raise RuntimeError(
            "DenseNet predictions differ across historical files."
        )

prediction_columns = {
    "RAW_Entropy_OLS": "EntropyLinear_Pred",
    "RAW_Descriptors_Ridge": "Ridge_Pred",
    "CLAHE_Resized_Entropy_OLS": "CLAHE_EntropyLinear_Pred",
    "CLAHE_Resized_Descriptors_Ridge": "CLAHE_Ridge_Pred",
    "DenseNet121": "DenseNet_Pred",
}

for model_name, pred_col in prediction_columns.items():
    comparison[f"{model_name}_AbsError"] = np.abs(
        comparison[TARGET] - comparison[pred_col]
    )

prediction_path = (
    OUTPUT_DIR / "04_locked_test_particle_predictions.csv"
)

comparison.to_csv(
    prediction_path,
    index=False,
    float_format="%.10f"
)

# =============================================================================
# 14. FINAL MODEL COMPARISON — ALL FIVE METHODS
# =============================================================================

comparison_records = []

for model_name, pred_col in prediction_columns.items():
    metrics = regression_metrics(
        comparison[TARGET],
        comparison[pred_col]
    )

    comparison_records.append({
        "Model": model_name,
        "n_test_particles": len(comparison),
        **metrics,
    })

model_comparison = pd.DataFrame(comparison_records)

comparison_path = (
    OUTPUT_DIR / "05_final_model_comparison.csv"
)

model_comparison.to_csv(
    comparison_path,
    index=False,
    float_format="%.10f"
)

print("\n" + "=" * 90)
print("FINAL COMPARISON — SAME 121 ORIGINAL TEST PARTICLES")
print("=" * 90)
print(
    model_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# =============================================================================
# 15. PUBLICATION-READY MAE COMPARISON
# =============================================================================

plot_df = model_comparison.copy()

labels_plot = [
    "Raw BSE\nentropy + OLS",
    "Raw BSE\ndescriptors + Ridge",
    "CLAHE + resized\nentropy + OLS",
    "CLAHE + resized\ndescriptors + Ridge",
    "DenseNet121",
]

fig, ax = plt.subplots(figsize=(11.5, 5.8))

x = np.arange(len(plot_df))

bars = ax.bar(
    x,
    plot_df["MAE"].to_numpy(),
    width=0.68,
    edgecolor="black",
    linewidth=0.7
)

ax.set_xticks(x)
ax.set_xticklabels(labels_plot, fontsize=10)

ax.set_ylabel(
    "Mean absolute error in EDS-derived $H_{\\mathrm{norm}}$",
    fontsize=12
)

ax.set_title(
    "BSE descriptor baselines versus DenseNet121\n"
    "Same 121 held-out original particles",
    fontsize=13
)

ax.set_ylim(0, float(plot_df["MAE"].max()) * 1.25)
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

for bar, mae in zip(bars, plot_df["MAE"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + float(plot_df["MAE"].max()) * 0.025,
        f"{mae:.4f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

fig.tight_layout()

for extension in ["png", "pdf"]:
    fig.savefig(
        OUTPUT_DIR / f"06_model_comparison_MAE.{extension}",
        dpi=600,
        bbox_inches="tight"
    )

plt.show()

# =============================================================================
# 16. FIVE-PANEL PARITY FIGURE
# =============================================================================

fig, axes = plt.subplots(
    1, 5,
    figsize=(20, 4.3),
    sharex=True,
    sharey=True
)

all_predictions = np.concatenate([
    comparison[col].to_numpy(dtype=float)
    for col in prediction_columns.values()
])

all_true = comparison[TARGET].to_numpy(dtype=float)

plot_low = min(
    float(np.min(all_true)),
    float(np.min(all_predictions))
) - 0.035

plot_high = max(
    float(np.max(all_true)),
    float(np.max(all_predictions))
) + 0.035

titles = [
    "Raw entropy",
    "Raw descriptors",
    "CLAHE entropy",
    "CLAHE descriptors",
    "DenseNet121",
]

for ax, title, (model_name, pred_col) in zip(
    axes,
    titles,
    prediction_columns.items()
):
    metrics = regression_metrics(
        comparison[TARGET],
        comparison[pred_col]
    )

    ax.scatter(
        comparison[TARGET],
        comparison[pred_col],
        s=18,
        alpha=0.72
    )

    ax.plot(
        [plot_low, plot_high],
        [plot_low, plot_high],
        "--",
        color="black",
        linewidth=1
    )

    ax.set_xlim(plot_low, plot_high)
    ax.set_ylim(plot_low, plot_high)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("True $H_{\\mathrm{norm}}$")

    ax.text(
        0.05,
        0.95,
        f"MAE = {metrics['MAE']:.3f}\n"
        f"$R^2$ = {metrics['R2']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="none"
        )
    )

    ax.grid(alpha=0.2)

axes[0].set_ylabel("Predicted $H_{\\mathrm{norm}}$")

fig.suptitle(
    "Locked-test prediction comparison (n = 121)",
    fontsize=14
)

fig.tight_layout(rect=[0, 0, 1, 0.94])

for extension in ["png", "pdf"]:
    fig.savefig(
        OUTPUT_DIR / f"07_model_comparison_parity.{extension}",
        dpi=600,
        bbox_inches="tight"
    )

plt.show()

# =============================================================================
# 17. FEATURE-LEVEL PREPROCESSING COMPARISON
# =============================================================================

feature_compare = raw_features[
    ["Base_ID", *FEATURES]
].merge(
    features_df[
        ["Base_ID", *FEATURES]
    ],
    on="Base_ID",
    how="inner",
    suffixes=("_RAW", "_CLAHE_RESIZED"),
    validate="one_to_one"
)

feature_summary_records = []

for feature in FEATURES:
    raw_values = feature_compare[
        f"{feature}_RAW"
    ].to_numpy(dtype=float)

    processed_values = feature_compare[
        f"{feature}_CLAHE_RESIZED"
    ].to_numpy(dtype=float)

    feature_summary_records.append({
        "Feature": feature,
        "RAW_Mean": float(np.mean(raw_values)),
        "CLAHE_Resized_Mean": float(np.mean(processed_values)),
        "Mean_Difference": float(
            np.mean(processed_values - raw_values)
        ),
        "Mean_Absolute_Difference": float(
            np.mean(np.abs(processed_values - raw_values))
        ),
        "Pearson_r_Raw_vs_Preprocessed": (
            float(np.corrcoef(raw_values, processed_values)[0, 1])
            if np.std(raw_values) > 0 and np.std(processed_values) > 0
            else np.nan
        ),
    })

feature_summary = pd.DataFrame(feature_summary_records)

feature_summary_path = (
    OUTPUT_DIR / "08_feature_preprocessing_comparison.csv"
)

feature_summary.to_csv(
    feature_summary_path,
    index=False,
    float_format="%.10f"
)

# =============================================================================
# 18. REPRODUCIBILITY MANIFEST
# =============================================================================

manifest_path = OUTPUT_DIR / "09_analysis_manifest.txt"

manifest = f"""
EXPERIMENT
 — CLAHE + resized BSE intensity descriptor sensitivity analysis

CREATED
{datetime.now().isoformat(timespec="seconds")}

OUTPUT DIRECTORY
{OUTPUT_DIR.resolve()}

INPUT FILES
Labels: {LABEL_CSV}
Original images: {IMAGES_DIR}
Original masks: {MASKS_DIR}
Raw feature table: {RAW_FEATURE_CSV}
Historical baseline predictions: {OLD_PREDICTIONS_CSV}
Historical DenseNet predictions: {DENSENET_PREDICTIONS_CSV}

DATA
Total original physical particles: {len(features_df)}
Development particles: {len(dev_df)}
Locked test particles: {len(test_df)}
Excluded Base_IDs: {sorted(DROP_BASE_IDS)}

PREPROCESSING
Grayscale read: cv2.IMREAD_GRAYSCALE
CLAHE clipLimit: {CLAHE_CLIP_LIMIT}
CLAHE tileGridSize: {CLAHE_TILE_GRID}
Image resize: {IMG_SIZE}x{IMG_SIZE}, cv2.INTER_AREA
Mask resize: {IMG_SIZE}x{IMG_SIZE}, cv2.INTER_NEAREST
Foreground extraction: resized binary mask
Per-image z-normalization: NO
Mask area as regression predictor: NO

DESCRIPTORS
{FEATURES}
Histogram entropy: Shannon, natural logarithm,
fixed {HIST_BINS} bins in range {HIST_RANGE}
Std: ddof=0
Skewness: bias=False
Kurtosis: fisher=True, bias=False

MODEL SELECTION
Entropy baseline: ordinary least squares
Ridge baseline: StandardScaler + Ridge
CV: {CV_FOLDS}-fold shuffled KFold, seed {SEED}
Ridge alpha grid: {RIDGE_ALPHAS}
Selected alpha: {best_alpha}
Test labels excluded from fitting and model selection: YES

INTERPRETATION
This analysis matches CLAHE, resizing, and mask use with the
DenseNet input pipeline, but not the final per-image z-normalization.
It does not fully equalize model representation or capacity.
The target H_norm is EDS-derived elemental compositional evenness,
not a direct measure of spatial homogeneity or phase purity.

VERSIONS
Python/OpenCV: {cv2.__version__}
NumPy: {np.__version__}
Pandas: {pd.__version__}

FINAL RESULTS
{model_comparison.to_string(index=False)}
"""

manifest_path.write_text(
    manifest.strip() + "\n",
    encoding="utf-8"
)

# =============================================================================
# 19. FINAL VERIFICATION
# =============================================================================

expected_files = [
    "01_preprocessed_BSE_features_all_originals.csv",
    "02_locked_test_split_assignments.csv",
    "03_ridge_development_CV_results.csv",
    "04_locked_test_particle_predictions.csv",
    "05_final_model_comparison.csv",
    "06_model_comparison_MAE.png",
    "06_model_comparison_MAE.pdf",
    "07_model_comparison_parity.png",
    "07_model_comparison_parity.pdf",
    "08_feature_preprocessing_comparison.csv",
    "09_analysis_manifest.txt",
]

missing_outputs = [
    filename
    for filename in expected_files
    if not (OUTPUT_DIR / filename).is_file()
]

if missing_outputs:
    raise RuntimeError(
        f"Expected output files missing: {missing_outputs}"
    )

print("\n" + "=" * 90)
print("EXPERIMENT COMPLETE — ALL OUTPUTS VERIFIED")
print("=" * 90)

print("\nFINAL MODEL RESULTS:")
print(
    model_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nALL RESULTS SAVED TO:")
print(OUTPUT_DIR.resolve())

print("\nFILES:")
for filename in expected_files:
    print("  ", filename)

print("\nIMPORTANT:")
print(
    "This is a CLAHE + resizing matched sensitivity analysis, "
    "not an exactly identical preprocessing comparison."
)
