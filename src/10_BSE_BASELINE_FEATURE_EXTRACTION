import os
import re
import glob
import math
import numpy as np
import pandas as pd
import cv2

from scipy.stats import skew, kurtosis


# ======================================================================================
# 1. PATHS
#    These match the paths used in the DenseNet training script.
# ======================================================================================

IMAGES_DIR = r"PATH_TO_ORIGINAL_BSE_IMAGES"

MASKS_DIR = r"PATH_TO_EXISTING_PARTICLE_MASKS"

CSV_PATH = r"PATH_TO_PARTICLE_LABELS_CSV"

OUTPUT_DIR = r"PATH_TO_BASELINE_RESULTS"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "bse_baseline_features_originals.csv"
)


# ======================================================================================
# 2. STUDY SETTINGS
# ======================================================================================

ID_COL = "Particle_ID"
TARGET_COL = "H_norm_atomic5"

DROP_BASE_IDS = {
    "Particle_733",
    "Particle_795"
}

# Fixed histogram definition for BSE entropy.
#
# cv2.imread(..., IMREAD_GRAYSCALE) gives us an 8-bit grayscale image,
# therefore we use the SAME 256 bins [0, 256] for every particle.
#
# This is important:
# DO NOT construct a separate min-max histogram for each particle.
HIST_BINS = 256
HIST_RANGE = (0, 256)


# ======================================================================================
# 3. FILE HELPERS
# ======================================================================================

EXTS = [
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp",
    ".TIF", ".TIFF", ".PNG", ".JPG", ".JPEG", ".BMP"
]


def find_by_stem(folder: str, stem: str):
    """
    Match the same general file-resolution logic used in the DenseNet script.
    """

    # First try exact stem + extension
    for ext in EXTS:
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return p

    # Then allow a stem prefix match
    hits = []

    for ext in EXTS:
        hits += glob.glob(os.path.join(folder, stem + "*" + ext))

    return hits[0] if hits else None


def split_stem_aug(pid: str):
    """
    Same logic as DenseNet training.

    Examples
    --------
    Particle_001
        -> Base_ID = Particle_001
        -> Aug_Tag = original

    Particle_001__aug_r90
        -> Base_ID = Particle_001
        -> Aug_Tag = aug_r90
    """

    s = os.path.splitext(str(pid))[0]

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


def ensure_binary(mask):
    """
    Convert an existing mask to {0,1}.

    No threshold/mask regeneration is performed.
    """

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


# ======================================================================================
# 4. BSE HISTOGRAM ENTROPY
# ======================================================================================

def bse_histogram_entropy(pixel_values):
    """
    Shannon entropy of the grayscale BSE intensity distribution.

        H_BSE = -sum_k p_k ln(p_k)

    IMPORTANT:
    This is IMAGE histogram entropy.

    It is NOT the EDS-derived compositional H_norm target.
    """

    counts, _ = np.histogram(
        pixel_values,
        bins=HIST_BINS,
        range=HIST_RANGE
    )

    counts = counts.astype(np.float64)

    total = counts.sum()

    if total <= 0:
        return np.nan

    p = counts / total

    # Remove zero-probability bins before log
    p = p[p > 0]

    H = -np.sum(
        p * np.log(p)
    )

    return float(H)


# ======================================================================================
# 5. EXTRACT FEATURES FROM ONE PARTICLE
# ======================================================================================

def extract_particle_features(image_path, mask_path):
    """
    Read ORIGINAL grayscale BSE image + EXISTING binary mask.

    Statistics are calculated ONLY from pixels inside the mask.
    """

    # ------------------------------------------------------------------
    # Read image exactly as grayscale.
    #
    # This matches the effective image reading used in DenseNet training:
    # cv2.IMREAD_GRAYSCALE
    #
    # No CLAHE is applied here.
    # ------------------------------------------------------------------

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    # ------------------------------------------------------------------
    # Read existing mask
    # ------------------------------------------------------------------

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_UNCHANGED
    )

    if mask is None:
        raise ValueError(
            f"Could not read mask: {mask_path}"
        )

    mask = ensure_binary(mask)

    # ------------------------------------------------------------------
    # QC: dimensions must match
    # ------------------------------------------------------------------

    if image.shape != mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch: "
            f"image={image.shape}, mask={mask.shape}"
        )

    # ------------------------------------------------------------------
    # Extract ONLY particle pixels
    # ------------------------------------------------------------------

    pixels = image[
        mask.astype(bool)
    ].astype(np.float64)

    if pixels.size == 0:
        raise ValueError(
            "Mask contains zero foreground pixels."
        )

    # ------------------------------------------------------------------
    # Particle area
    # ------------------------------------------------------------------

    area_px = int(
        pixels.size
    )

    # ------------------------------------------------------------------
    # Basic intensity statistics
    # ------------------------------------------------------------------

    mean_intensity = float(
        np.mean(pixels)
    )

    std_intensity = float(
        np.std(pixels, ddof=0)
    )

    if abs(mean_intensity) > 1e-12:
        cv_intensity = (
            std_intensity /
            mean_intensity
        )
    else:
        cv_intensity = np.nan

    median_intensity = float(
        np.median(pixels)
    )

    p10 = float(
        np.percentile(pixels, 10)
    )

    p25 = float(
        np.percentile(pixels, 25)
    )

    p75 = float(
        np.percentile(pixels, 75)
    )

    p90 = float(
        np.percentile(pixels, 90)
    )

    iqr = (
        p75 - p25
    )

    # ------------------------------------------------------------------
    # Shape of intensity distribution
    #
    # bias=False gives bias-corrected estimates.
    # fisher=True means normal distribution has kurtosis = 0.
    # ------------------------------------------------------------------

    if std_intensity > 1e-12:

        skewness = float(
            skew(
                pixels,
                bias=False
            )
        )

        kurt = float(
            kurtosis(
                pixels,
                fisher=True,
                bias=False
            )
        )

    else:

        skewness = 0.0
        kurt = 0.0

    # ------------------------------------------------------------------
    # Histogram entropy
    # ------------------------------------------------------------------

    H_bse = bse_histogram_entropy(
        pixels
    )

    # ------------------------------------------------------------------
    # Additional QC information
    # ------------------------------------------------------------------

    min_intensity = float(
        np.min(pixels)
    )

    max_intensity = float(
        np.max(pixels)
    )

    return {
        "Area_px": area_px,

        "Mean": mean_intensity,
        "Std": std_intensity,
        "CV": cv_intensity,

        "Median": median_intensity,
        "IQR": float(iqr),

        "P10": p10,
        "P90": p90,

        "Skewness": skewness,
        "Kurtosis": kurt,

        "BSE_Entropy": H_bse,

        # QC-only columns
        "Min_Intensity": min_intensity,
        "Max_Intensity": max_intensity,
    }


# ======================================================================================
# 6. LOAD LABEL CSV
# ======================================================================================

print("=" * 80)
print("BSE BASELINE FEATURE EXTRACTION")
print("=" * 80)

print("\nLoading label CSV:")
print(CSV_PATH)

raw = pd.read_csv(
    CSV_PATH
)

raw.columns = [
    str(c).strip()
    for c in raw.columns
]

print(
    f"\nRows in label CSV: {len(raw):,}"
)

print(
    f"Columns: {list(raw.columns)}"
)


# ======================================================================================
# 7. CHECK REQUIRED COLUMNS
# ======================================================================================

required_columns = [
    ID_COL,
    TARGET_COL
]

missing_columns = [
    c
    for c in required_columns
    if c not in raw.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )


# ======================================================================================
# 8. IDENTIFY BASE_ID AND AUGMENTATION TAG
# ======================================================================================

split_results = raw[
    ID_COL
].astype(str).map(
    split_stem_aug
)

raw["Base_ID"] = [
    x[0]
    for x in split_results
]

raw["Aug_Tag"] = [
    x[1]
    for x in split_results
]


print("\n----------------------------------------")
print("INSTANCE STRUCTURE")
print("----------------------------------------")

print(
    f"Unique Base_IDs before exclusions: "
    f"{raw['Base_ID'].nunique():,}"
)

print("\nAugmentation tag counts:")

print(
    raw["Aug_Tag"]
    .value_counts()
    .sort_index()
)


# ======================================================================================
# 9. KEEP ORIGINAL IMAGES ONLY
# ======================================================================================

originals = raw[
    raw["Aug_Tag"] == "original"
].copy()

print(
    f"\nOriginal rows found: "
    f"{len(originals):,}"
)

print(
    f"Unique Base_IDs among originals: "
    f"{originals['Base_ID'].nunique():,}"
)


# ======================================================================================
# 10. DROP KNOWN FAULTY MASKS
# ======================================================================================

before_drop = len(originals)

originals = originals[
    ~originals["Base_ID"].isin(
        DROP_BASE_IDS
    )
].copy()

dropped_count = (
    before_drop -
    len(originals)
)

print(
    f"\nDropped known faulty Base_IDs: "
    f"{sorted(DROP_BASE_IDS)}"
)

print(
    f"Rows removed: {dropped_count}"
)

print(
    f"Usable original rows expected: "
    f"{len(originals):,}"
)


# ======================================================================================
# 11. CRITICAL QC: ONE ORIGINAL PER BASE_ID
# ======================================================================================

duplicate_bases = (
    originals["Base_ID"]
    .value_counts()
)

duplicate_bases = duplicate_bases[
    duplicate_bases > 1
]

if len(duplicate_bases) > 0:

    print(
        "\nWARNING: Some Base_IDs have "
        "multiple rows tagged as original:"
    )

    print(
        duplicate_bases
    )

    raise RuntimeError(
        "Expected exactly ONE original row "
        "per Base_ID."
    )


# ======================================================================================
# 12. RESOLVE IMAGE + MASK AND EXTRACT FEATURES
# ======================================================================================

records = []

missing_images = []
missing_masks = []
failed_particles = []

print("\n----------------------------------------")
print("EXTRACTING FEATURES")
print("----------------------------------------")


for i, row in originals.reset_index(drop=True).iterrows():

    particle_id = str(
        row[ID_COL]
    )

    base_id = str(
        row["Base_ID"]
    )

    h_norm = float(
        row[TARGET_COL]
    )

    stem = os.path.splitext(
        particle_id
    )[0]

    # --------------------------------------------------
    # Locate image
    # --------------------------------------------------

    image_path = find_by_stem(
        IMAGES_DIR,
        stem
    )

    if image_path is None:

        missing_images.append(
            particle_id
        )

        continue

    # --------------------------------------------------
    # Locate existing mask
    # --------------------------------------------------

    mask_path = find_by_stem(
        MASKS_DIR,
        stem
    )

    if mask_path is None:

        missing_masks.append(
            particle_id
        )

        continue

    # --------------------------------------------------
    # Extract features
    # --------------------------------------------------

    try:

        features = extract_particle_features(
            image_path,
            mask_path
        )

    except Exception as e:

        failed_particles.append(
            (
                particle_id,
                str(e)
            )
        )

        continue

    # --------------------------------------------------
    # Save useful metadata
    # --------------------------------------------------

    rec = {
        "Particle_ID": particle_id,
        "Base_ID": base_id,
        "H_norm_atomic5": h_norm,
        "Image_Path": image_path,
        "Mask_Path": mask_path,
    }

    # Preserve Sample_ID / Spectrum if available.
    if "Sample_ID" in originals.columns:
        rec["Sample_ID"] = row["Sample_ID"]

    if "Spectrum" in originals.columns:
        rec["Spectrum"] = row["Spectrum"]

    rec.update(
        features
    )

    records.append(
        rec
    )

    # Progress report every 100 particles
    if (
        (i + 1) % 100 == 0
        or (i + 1) == len(originals)
    ):

        print(
            f"Processed "
            f"{i + 1:,} / "
            f"{len(originals):,}"
        )


# ======================================================================================
# 13. BUILD FEATURE TABLE
# ======================================================================================

features_df = pd.DataFrame(
    records
)


# ======================================================================================
# 14. QC REPORT
# ======================================================================================

print("\n")
print("=" * 80)
print("QC SUMMARY")
print("=" * 80)

print(
    f"\nOriginal usable rows expected: "
    f"{len(originals):,}"
)

print(
    f"Successfully extracted: "
    f"{len(features_df):,}"
)

print(
    f"Unique Base_IDs extracted: "
    f"{features_df['Base_ID'].nunique() if len(features_df) else 0:,}"
)

print(
    f"Missing images: "
    f"{len(missing_images):,}"
)

print(
    f"Missing masks: "
    f"{len(missing_masks):,}"
)

print(
    f"Failed during extraction: "
    f"{len(failed_particles):,}"
)


if missing_images:

    print(
        "\nMissing image IDs:"
    )

    print(
        missing_images[:20]
    )


if missing_masks:

    print(
        "\nMissing mask IDs:"
    )

    print(
        missing_masks[:20]
    )


if failed_particles:

    print(
        "\nFailed particles:"
    )

    for item in failed_particles[:20]:

        print(
            item
        )


# ======================================================================================
# 15. FINAL DUPLICATE CHECK
# ======================================================================================

if len(features_df) > 0:

    duplicate_final = (
        features_df["Base_ID"]
        .duplicated()
        .sum()
    )

    print(
        f"\nDuplicate Base_ID rows in "
        f"final table: "
        f"{duplicate_final}"
    )

    if duplicate_final != 0:

        raise RuntimeError(
            "Final baseline table contains "
            "duplicate physical particles."
        )


# ======================================================================================
# 16. FEATURE SANITY CHECKS
# ======================================================================================

feature_cols = [
    "Area_px",
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
    "Min_Intensity",
    "Max_Intensity",
]


if len(features_df) > 0:

    print("\n----------------------------------------")
    print("TARGET SUMMARY")
    print("----------------------------------------")

    print(
        features_df[
            "H_norm_atomic5"
        ].describe()
    )

    print("\n----------------------------------------")
    print("FEATURE SUMMARY")
    print("----------------------------------------")

    print(
        features_df[
            feature_cols
        ].describe().T
    )


# ======================================================================================
# 17. CHECK NaN / INF
# ======================================================================================

if len(features_df) > 0:

    numeric_check_cols = [
        "H_norm_atomic5",
        *feature_cols
    ]

    nan_counts = (
        features_df[
            numeric_check_cols
        ]
        .isna()
        .sum()
    )

    inf_counts = pd.Series(
        {
            c: np.isinf(
                features_df[c]
                .to_numpy(dtype=float)
            ).sum()

            for c in numeric_check_cols
        }
    )

    print("\n----------------------------------------")
    print("NaN COUNTS")
    print("----------------------------------------")

    print(
        nan_counts[
            nan_counts > 0
        ]
        if (nan_counts > 0).any()
        else "No NaNs detected."
    )

    print("\n----------------------------------------")
    print("INF COUNTS")
    print("----------------------------------------")

    print(
        inf_counts[
            inf_counts > 0
        ]
        if (inf_counts > 0).any()
        else "No infinities detected."
    )


# ======================================================================================
# 18. SAVE
# ======================================================================================

features_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8",
    float_format="%.8f"
)

print("\n")
print("=" * 80)
print("DONE")
print("=" * 80)

print(
    f"\nSaved baseline feature table:\n"
    f"{os.path.abspath(OUTPUT_CSV)}"
)

print(
    f"\nFinal physical-particle count: "
    f"{len(features_df):,}"
)

print(
    "\nDO NOT proceed to regression yet."
)

print(
    "First inspect this QC output and "
    "the generated CSV."
)
