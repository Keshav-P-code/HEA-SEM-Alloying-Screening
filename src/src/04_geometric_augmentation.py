# This script performs offline geometric augmentations (rotations and flips)
# as described in the manuscript. Augmentations do not alter composition labels.
import os, glob, random, shutil
import pandas as pd
from PIL import Image, ImageOps

# ============================
# 1) EDIT THESE PATHS
# ============================
INPUT_CSV = r"RAW-DATA-COMP__atomic5_entropy.csv"   # set local path to entropy CSV
IMAGE_DIR = r"Particle_Images"                     # folder containing original images
OUTPUT_IMAGE_DIR = r"augmented_images"             # output folder for originals + augmentations

# ============================
# 2) AUGMENTATION SETTINGS
# ============================
INCLUDE_ROTATIONS = True    # adds 90°, 180°, 270°
INCLUDE_FLIP_H    = True    # horizontal flip
INCLUDE_FLIP_V    = True    # vertical flip
EXHAUSTIVE_COMBINATIONS = False   # True = all rotation×flip combos
MAX_AUG_PER_IMAGE = None          # None = no cap
RANDOM_SEED = 42

IMG_EXTS = (".png",".jpg",".jpeg",".tif",".tiff",".bmp",".PNG",".JPG",".JPEG",".TIF",".TIFF",".BMP")

# ============================
# 3) HELPERS
# ============================
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def find_by_stem(folder, stem):
    # exact matches first
    for ext in IMG_EXTS:
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p): return p
    # then any file starting with stem + dot
    hits = [h for h in glob.glob(os.path.join(folder, stem + ".*")) if os.path.splitext(h)[1] in IMG_EXTS]
    if hits:
        # prefer tif/tiff, else shortest path
        hits = sorted(hits, key=lambda x: (not x.lower().endswith((".tif",".tiff")), len(x)))
        return hits[0]
    return None

def load_image(path):
    im = Image.open(path)
    # keep grayscale as is; only convert paletted to RGB for safety
    if im.mode == "P":
        im = im.convert("RGB")
    return im

def apply_geometric(im, rot=0, flip_h=False, flip_v=False):
    """Apply 90/180/270 rotation and optional H/V flips (no interpolation artifacts)."""
    out = im
    if rot in (90, 180, 270):
        out = out.rotate(rot, expand=True)
    elif rot not in (0,):
        raise ValueError("Rotation must be one of {0,90,180,270}")
    if flip_h:
        out = ImageOps.mirror(out)
    if flip_v:
        out = ImageOps.flip(out)
    return out

def aug_name_suffix(rot, flip_h, flip_v):
    parts = []
    if rot: parts.append(f"r{rot}")
    if flip_h: parts.append("fh")
    if flip_v: parts.append("fv")
    return "__aug_" + ("_".join(parts) if parts else "id")

def enumerate_aug_combinations():
    rots = [0]
    if INCLUDE_ROTATIONS:
        rots += [90, 180, 270]
    combos = []
    if EXHAUSTIVE_COMBINATIONS:
        for r in rots:
            for fh in ([False, True] if INCLUDE_FLIP_H else [False]):
                for fv in ([False, True] if INCLUDE_FLIP_V else [False]):
                    if r == 0 and not fh and not fv:
                        continue  # skip identity; original already included
                    combos.append((r, fh, fv))
    else:
        # minimal unique set: each rotation once, plus separate flips
        for r in rots:
            if r != 0:
                combos.append((r, False, False))
        if INCLUDE_FLIP_H:
            combos.append((0, True,  False))
        if INCLUDE_FLIP_V:
            combos.append((0, False, True))
    # dedupe while preserving order
    seen, uniq = set(), []
    for c in combos:
        if c not in seen:
            uniq.append(c); seen.add(c)
    return uniq

# ============================
# 4) MAIN
# ============================
def main():
    random.seed(RANDOM_SEED)

    df = pd.read_csv(INPUT_CSV)
    csv_dir = os.path.dirname(INPUT_CSV)
    ensure_dir(OUTPUT_IMAGE_DIR)

    # pick image column
    img_col = IMAGE_COL_NAME
    if img_col is None:
        if "Image_Name" in df.columns:
            img_col = "Image_Name"
        elif "Particle_ID" in df.columns:
            img_col = "Particle_ID"
        else:
            raise ValueError("Could not find image column. Add 'Image_Name' (filename) or 'Particle_ID' (stem).")

    # build augmentation list
    aug_combos = enumerate_aug_combinations()
    if MAX_AUG_PER_IMAGE and MAX_AUG_PER_IMAGE < len(aug_combos):
        random.shuffle(aug_combos)
        aug_combos = aug_combos[:MAX_AUG_PER_IMAGE]

    new_rows = []

    for _, row in df.iterrows():
        key = str(row[img_col])

        # resolve image path and base/ext (robust: try multiple heuristics; skip row if not found)
        def _find_image_for_key(k):
            # direct absolute/relative path
            if os.path.isabs(k) and os.path.isfile(k):
                return k
            # join with IMAGE_DIR (common case when CSV holds filenames)
            p = os.path.join(IMAGE_DIR, k)
            if os.path.isfile(p):
                return p
            # try exact-stem matching helper (find_by_stem expects a stem)
            stem = os.path.splitext(os.path.basename(k))[0]
            p = find_by_stem(IMAGE_DIR, stem)
            if p:
                return p
            # fallback: case-insensitive substring match among image files
            lower_stem = stem.lower()
            candidates = [f for f in glob.glob(os.path.join(IMAGE_DIR, "*")) if os.path.splitext(f)[1] in IMG_EXTS]
            hits = [f for f in candidates if lower_stem in os.path.basename(f).lower()]
            if hits:
                # prefer tif/tiff then shortest path
                hits = sorted(hits, key=lambda x: (not x.lower().endswith((".tif",".tiff")), len(x)))
                return hits[0]
            return None

        if img_col == "Image_Name":
            src_path = _find_image_for_key(key)
            if src_path is None:
                print(f"  Image not found for Image_Name='{key}' in {IMAGE_DIR}. Skipping this row.")
                continue
            base, ext = os.path.splitext(os.path.basename(src_path))
        else:
            src_path = _find_image_for_key(key)
            if src_path is None:
                print(f"  No image found for Particle_ID='{key}' in {IMAGE_DIR}. Skipping this row.")
                continue
            base, ext = os.path.splitext(os.path.basename(src_path))

        # 1) copy original into OUTPUT_IMAGE_DIR (once)
        dst_orig = os.path.join(OUTPUT_IMAGE_DIR, base + ext)
        if os.path.abspath(src_path) != os.path.abspath(dst_orig):
            if not os.path.isfile(dst_orig):
                shutil.copy2(src_path, dst_orig)

        # append original row unchanged
        new_rows.append(row.copy())

        # 2) create augmented variants
        img = load_image(src_path)
        for (rot, fh, fv) in aug_combos:
            aug = apply_geometric(img, rot=rot, flip_h=fh, flip_v=fv)
            suffix = aug_name_suffix(rot, fh, fv)
            aug_name = f"{base}{suffix}{ext}"
            aug_path = os.path.join(OUTPUT_IMAGE_DIR, aug_name)
            aug.save(aug_path)

            aug_row = row.copy()
            aug_row[img_col] = aug_name
            new_rows.append(aug_row)

    # write output CSV (original + augs)
    out_df = pd.DataFrame(new_rows)
    out_csv = os.path.join(csv_dir, os.path.splitext(os.path.basename(INPUT_CSV))[0] + "__with_augs_geom.csv")
    out_df.to_csv(out_csv, index=False)

    print(" Done.")
    print(f"• Images saved to: {OUTPUT_IMAGE_DIR}")
    print(f"• Output CSV: {out_csv}")
    print(f"• Augmentations per image: {len(aug_combos)}")
    print(f"• Total rows: {len(out_df)} (orig {len(df)} + aug {len(out_df)-len(df)})")

if __name__ == "__main__":
    main()
