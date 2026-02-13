# === Batch masks for ALL particle images using ImageJ-style Otsu =================
# - Reads native depth (8/16-bit) and computes Otsu exactly like ImageJ:
#     * full-range histogram on the active intensity span
#     * a few histogram smoothing passes
# - Builds BOTH polarities (mask / inverted), keeps the one that scores better:
#     * prefers the largest connected component
#     * prefers the mask that includes the image center and has a sane area
# - Saves binary masks (PNG, 0/255), optional overlays, and an index CSV.
# - Mirrors the input folder structure under OUT_DIR.
# ===============================================================================

import os, re
import cv2
import numpy as np
import pandas as pd

# -------- EDIT THESE --------
IMAGE_DIR = r"augmented_images"         # input images (can contain subfolders)
OUT_DIR   = r"Augmented_output_masks"   # masks/overlays + index CSV will be written here
# ----------------------------

RECURSIVE           = True     # walk subfolders
SKIP_IF_EXISTS      = True     # don't recompute if a mask already exists
SAVE_OVERLAY_QC     = True     # also save red-on-gray overlays for quick inspection
OVERLAY_ALPHA       = 0.35
PNG_COMP            = [cv2.IMWRITE_PNG_COMPRESSION, 3]

SMOOTH_PASSES       = 3        # histogram smoothing passes (2–4 is typical)
AREA_MIN_FRAC       = 0.05     # keep if area ≥ 5% of image
AREA_MAX_FRAC       = 0.95     # keep if area ≤ 95% of image

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True); return path

def is_image(path):
    return os.path.splitext(path)[1].lower() in {".tif",".tiff",".png",".jpg",".jpeg",".bmp"}

def rel_noext(path, root):
    rel = os.path.relpath(path, root)
    return os.path.splitext(rel)[0]

def to_gray_native(path):
    """Read image preserving native bit depth; return single-channel grayscale."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: return None
    if img.ndim == 3:
        # Weighted grayscale while preserving range
        if img.dtype == np.uint16:
            gray = (0.114*img[...,0] + 0.587*img[...,1] + 0.299*img[...,2]).astype(np.uint16)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray
    return img

def smooth_histogram(h, passes=3):
    h = h.astype(np.float64)
    for _ in range(max(0, passes)):
        hp = h.copy()
        hp[1:-1] = (h[:-2] + 2*h[1:-1] + h[2:]) / 4.0
        hp[0]    = (3*h[0] + h[1]) / 4.0
        hp[-1]   = (h[-2] + 3*h[-1]) / 4.0
        h = hp
    return h

def otsu_from_histogram(hist):
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0: return 0
    bins = np.arange(len(hist))
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * bins)
    mu_t = mu[-1]
    denom = np.maximum(omega * (1.0 - omega), 1e-12)
    sigma_b2 = (mu_t*omega - mu)**2 / denom
    sigma_b2[~np.isfinite(sigma_b2)] = 0.0
    return int(np.argmax(sigma_b2))

def imagej_like_otsu(gray):
    """Return threshold (native scale), mask (>=thr), mask_inv (<thr)."""
    if gray.dtype == np.uint8:
        hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
        hist = smooth_histogram(hist, SMOOTH_PASSES)
        t = otsu_from_histogram(hist)
        mask     = (gray >= t).astype(np.uint8)*255
        mask_inv = (gray <  t).astype(np.uint8)*255
        return t, mask, mask_inv

    # uint16 (or wider) → use active range [gmin,gmax]
    if gray.dtype == np.uint16:
        gmin, gmax = int(gray.min()), int(gray.max())
        if gmax <= gmin:
            t = gmin
            mask     = (gray >= t).astype(np.uint8)*255
            mask_inv = (gray <  t).astype(np.uint8)*255
            return t, mask, mask_inv
        nbins = min(gmax - gmin + 1, 65536)
        hist = cv2.calcHist([gray],[0],None,[nbins],[gmin, gmax+1]).ravel()
        hist = smooth_histogram(hist, SMOOTH_PASSES)
        t_bin = otsu_from_histogram(hist)
        t_val = gmin + t_bin
        mask     = (gray >= t_val).astype(np.uint8)*255
        mask_inv = (gray <  t_val).astype(np.uint8)*255
        return t_val, mask, mask_inv

    # Fallback: normalize to 16-bit first
    gray16 = cv2.normalize(gray, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)
    return imagej_like_otsu(gray16)

def largest_cc(mask255):
    lab = cv2.connectedComponents((mask255>0).astype(np.uint8))[1]
    if lab.max() == 0:  # nothing
        return mask255
    areas = [(lab==i).sum() for i in range(1, lab.max()+1)]
    idx = 1 + int(np.argmax(areas))
    return ((lab==idx).astype(np.uint8))*255

def score_mask(mask255):
    """Score by (center inside) + area sanity; returns (score, area_frac)."""
    H, W = mask255.shape
    frac = float(mask255.sum()) / (255.0*H*W + 1e-6)
    cy, cx = H//2, W//2
    center = 1.0 if mask255[cy, cx] > 0 else 0.0
    # penalize far-out areas; mild bonus for larger masks
    band_penalty = 0.0
    if frac < AREA_MIN_FRAC:
        band_penalty = -(AREA_MIN_FRAC - frac)*5.0
    elif frac > AREA_MAX_FRAC:
        band_penalty = -(frac - AREA_MAX_FRAC)*5.0
    score = center + band_penalty + 0.1*frac
    return score, frac

def pick_best_polarity(mask_pos, mask_neg):
    """Choose between normal/inverted masks after largest-CC and sanity checks."""
    m1 = largest_cc(mask_pos.astype(np.uint8))
    m2 = largest_cc(mask_neg.astype(np.uint8))
    s1, f1 = score_mask(m1)
    s2, f2 = score_mask(m2)
    return (m1, f1) if s1 >= s2 else (m2, f2)

def overlay_for_preview(gray, mask255, alpha=0.35):
    g8 = gray if gray.dtype==np.uint8 else cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    base = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR).astype(np.float32)/255.0
    red = np.zeros_like(base); red[...,2] = 1.0
    m = (mask255>0)[...,None].astype(np.float32)
    over = np.clip(base*(1-alpha*m) + red*(alpha*m), 0, 1)
    return (over*255).astype(np.uint8)

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    IMAGE_DIR_abs = os.path.abspath(IMAGE_DIR)
    OUT_DIR_abs   = os.path.abspath(OUT_DIR)
    MASK_DIR      = ensure_dir(os.path.join(OUT_DIR_abs, "masks"))
    OVERLAY_DIR   = ensure_dir(os.path.join(OUT_DIR_abs, "overlays"))

    # gather images
    image_paths = []
    if RECURSIVE:
        for root, _, files in os.walk(IMAGE_DIR_abs):
            for fn in files:
                p = os.path.join(root, fn)
                if is_image(p): image_paths.append(p)
    else:
        for fn in os.listdir(IMAGE_DIR_abs):
            p = os.path.join(IMAGE_DIR_abs, fn)
            if os.path.isfile(p) and is_image(p): image_paths.append(p)

    if not image_paths:
        raise FileNotFoundError("No images found under IMAGE_DIR.")

    records = []
    made = skipped = failed = 0

    for ipath in image_paths:
        try:
            rel = rel_noext(ipath, IMAGE_DIR_abs)
            mask_path    = os.path.join(MASK_DIR,    rel + ".png")
            overlay_path = os.path.join(OVERLAY_DIR, rel + ".png")
            ensure_dir(os.path.dirname(mask_path))
            ensure_dir(os.path.dirname(overlay_path))

            if SKIP_IF_EXISTS and os.path.exists(mask_path):
                skipped += 1
                continue

            gray = to_gray_native(ipath)
            if gray is None:
                failed += 1
                continue

            # ImageJ-like Otsu (both polarities)
            thr, m_pos, m_neg = imagej_like_otsu(gray)

            # choose polarity; keep largest CC and sanity-check area
            best_mask, area = pick_best_polarity(m_pos, m_neg)

            # final tiny cleanup (optional)
            kernel = np.ones((3,3), np.uint8)
            best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            # write mask
            cv2.imwrite(mask_path, best_mask, PNG_COMP)

            # overlay (optional)
            if SAVE_OVERLAY_QC:
                ov = overlay_for_preview(gray, best_mask, alpha=OVERLAY_ALPHA)
                cv2.imwrite(overlay_path, ov, PNG_COMP)
            else:
                overlay_path = ""

            made += 1
            records.append({
                "Image_Path": ipath,
                "Mask_Path": mask_path,
                "Overlay_Path": overlay_path,
                "Thresh_Value": thr,
                "Area_Frac": round(area, 4)
            })

        except Exception as e:
            print(f"  {ipath}: {e}")
            failed += 1

    # index CSV
    index_csv = os.path.join(OUT_DIR_abs, "imagej_otsu_masks_index.csv")
    pd.DataFrame(records).to_csv(index_csv, index=False)

    print("\n Done.")
    print(f"  Images found:       {len(image_paths)}")
    print(f"  Masks created:      {made}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Failed:             {failed}")
    print(f"Index CSV: {index_csv}")
    print(f"Masks dir:    {os.path.join(OUT_DIR_abs,'masks')}")
    print(f"Overlays dir: {os.path.join(OUT_DIR_abs,'overlays')}")

if __name__ == "__main__":
    main()
