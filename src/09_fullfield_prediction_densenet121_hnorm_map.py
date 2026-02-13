# ================================================
# Full-Field H_norm Map with DenseNet121 Regression
# ================================================
# - Uses bridge-broken connected components to define particles
# - Crops from ORIGINAL SEM + ORIGINAL MASK (eroded mask used ONLY for CC)
# - Preprocessing matches training:
#     CLAHE -> resize(512, INTER_AREA) -> hard-gate -> mask-aware z-norm -> 3ch
# - DenseNet121 regression head predicts H_norm
# - T jittered crops per particle, mean H_norm aggregation

"""
NOTES:
- IMG_PATH: full-field SEM/BSE image (grayscale), any bit-depth.
- MASK_PATH: binary particle mask (same size as IMG_PATH), background=0, particles>0.
- CKPT_PATH: DenseNet121 regression checkpoint (.pt) trained on particle crops.
- Data is not expected to be on GitHub (can be hosted in your UNB repository).
"""

import os, math, random, csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

# --------------------- EDIT THESE ---------------------
IMG_PATH  = r"PATH/TO/FULLFIELD_SEM.tif"
MASK_PATH = r"PATH/TO/FULLFIELD_MASK.tif"
CKPT_PATH = r"PATH/TO/best_Densenet121_regression.pt"
OUT_DIR   = r"PATH/TO/OUTPUT_FOLDER"
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------- Inference knobs ---------------------
NET_SIZE        = 512      # model input (must match training IMG_SIZE)
USE_CLAHE       = True
CLAHE_CLIP      = 2.0
CLAHE_TILE      = (8, 8)

R_BRIDGE        = 45       # bridge-breaking radius (pixels) for CC
MIN_PART_AREA   = 25000     # ignore tiny CC (in pruned mask) for full-field prediction

JITTERS_PER_ID  = 8        # total views per particle (1 deterministic + 7 jittered)
JITTER_SHIFT    = 0.06     # ±6% bbox shift
JITTER_SCALE    = 0.08     # ±8% bbox expansion
CONTEXT_PAD     = 0.10     # include 10% bbox context around particle
BATCH_SIZE      = 32

# Overlay blending
ALPHA_SEM       = 0.6      # weight for SEM
ALPHA_HEAT      = 0.4      # weight for heatmap

# --------------------- Device & model ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Build DenseNet121 regression head (must match training architecture)
model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
model.classifier = nn.Linear(model.classifier.in_features, 1)

state = torch.load(CKPT_PATH, map_location=device)
# Training code saved state_dict directly, so no "model" key
if isinstance(state, dict) and "model" in state:
    model.load_state_dict(state["model"])
else:
    model.load_state_dict(state)

model.eval().to(device)

# --------------------- Label helper (from your numbering script) ---------------------
def put_label_with_bg(
    img, text, center_xy, *,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=12,
    text_color=(0, 0, 0),       # black text
    bg_color=(255, 255, 255),   # white background
    alpha=0.85,                 # background opacity
    pad=6,                      # padding around text
    thickness=12,
    radius=4
):
    """
    Draw a white rounded rectangle background, then the text centered at center_xy.
    """
    h, w = img.shape[:2]
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_w, box_h = tw + 2*pad, th + baseline + 2*pad

    # place the box so that its center is at center_xy
    x = int(round(center_xy[0] - box_w/2))
    y = int(round(center_xy[1] + box_h/2))  # bottom-left y for putText

    # clamp box inside image
    x = max(0, min(x, w - box_w))
    y = max(box_h, min(y, h))

    # rectangle coords
    tl = (x, y - box_h)
    br = (x + box_w, y)

    # draw filled rounded rectangle on overlay
    overlay = img.copy()
    cv2.rectangle(overlay, tl, br, bg_color, -1)
    # corners
    cv2.circle(overlay, (tl[0] + radius, tl[1] + radius), radius, bg_color, -1)
    cv2.circle(overlay, (br[0] - radius, tl[1] + radius), radius, bg_color, -1)
    cv2.circle(overlay, (tl[0] + radius, br[1] - radius), radius, bg_color, -1)
    cv2.circle(overlay, (br[0] - radius, br[1] - radius), radius, bg_color, -1)
    # alpha blend
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # subtle dark border
    cv2.rectangle(img, tl, br, (30, 30, 30), 1, cv2.LINE_AA)

    # text
    text_org = (x + pad, y - pad - baseline // 2)
    cv2.putText(img, text, text_org, font, font_scale, text_color, thickness, cv2.LINE_AA)

# --------------------- Image utils ---------------------
def ensure_gray(img):
    if img is None:
        raise ValueError("Image is None (check path).")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def ensure_binary(mask_u8):
    """
    Ensure {0,1} mask from input (which may be 0/255 or gray-ish).
    """
    mask_u8 = ensure_gray(mask_u8)
    if mask_u8.dtype != np.uint8:
        mask_u8 = cv2.normalize(mask_u8, None, 0, 255,
                                cv2.NORM_MINMAX).astype(np.uint8)
    _, m_bin = cv2.threshold(mask_u8, 0, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return (m_bin > 0).astype(np.uint8)

CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE) if USE_CLAHE else None

def clamp_int(a, lo, hi):
    return max(lo, min(int(a), hi))

def mask_aware_norm_np(x3: np.ndarray, m01: np.ndarray):
    """
    x3: (3,H,W) float32 in [0,1]
    m01: (H,W) uint8 {0,1} mask, where 1 == particle
    """
    m = m01.astype(bool)
    if m.sum() > 0:
        for c in range(3):
            vals = x3[c][m]
            mu = float(vals.mean())
            sd = float(vals.std() + 1e-6)
            x3[c] = (x3[c] - mu) / sd
    else:
        mu = float(x3.mean())
        sd = float(x3.std() + 1e-6)
        x3 = (x3 - mu) / sd
    return x3

def preprocess_crop(g_roi, m_roi, size=NET_SIZE):
    """
    Match training preprocessing for each ROI:
    1) CLAHE on grayscale ROI
    2) Resize to NET_SIZE (INTER_AREA for SEM, NEAREST for mask)
    3) Hard gate SEM by mask
    4) Convert to [0,1], 3ch, mask-aware z-norm
    Returns: torch.Tensor (3, size, size) float32
    """
    if CLAHE is not None:
        g_roi = CLAHE.apply(g_roi)

    g_resized = cv2.resize(g_roi, (size, size), interpolation=cv2.INTER_AREA)
    m_resized = cv2.resize(m_roi, (size, size), interpolation=cv2.INTER_NEAREST)
    m_resized = (m_resized > 0).astype(np.uint8)

    gated = (g_resized * m_resized).astype(np.uint8)
    x01 = gated.astype(np.float32) / 255.0
    x3 = np.stack([x01, x01, x01], axis=0)  # (3,H,W)
    x3 = mask_aware_norm_np(x3, m_resized)
    return torch.from_numpy(x3).float(), m_resized

# --------------------- Load ORIGINAL SEM + ORIGINAL MASK ---------------------
g_orig = cv2.imread(IMG_PATH, cv2.IMREAD_UNCHANGED)
m_orig = cv2.imread(MASK_PATH, cv2.IMREAD_UNCHANGED)

g_orig = ensure_gray(g_orig)
m_orig = ensure_binary(m_orig)

H, W = g_orig.shape
print(f"Image loaded: {W}x{H}")
print(f"Original mask fg px: {int(m_orig.sum())}")

# Ensure SEM is uint8 (for visualization / CLAHE)
if g_orig.dtype != np.uint8:
    g_u8 = cv2.normalize(g_orig, None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)
else:
    g_u8 = g_orig.copy()

# --------------------- Clean mask lightly BEFORE bridge breaking ---------------------
kernel_small = np.ones((2, 2), np.uint8)
m_clean = cv2.morphologyEx(m_orig, cv2.MORPH_OPEN, kernel_small)
cv2.imwrite(os.path.join(OUT_DIR, "mask_clean.png"),
            (m_clean * 255).astype(np.uint8))

# --------------------- Bridge breaking via morphological opening ---------------------
k_size = 2 * R_BRIDGE + 1
kernel_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

# Opening breaks thin necks/bridges
m_open = cv2.morphologyEx(m_clean, cv2.MORPH_OPEN, kernel_bridge)

# Optional: restore some area with a slightly smaller dilation
kernel_restore = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (2 * R_BRIDGE // 3 + 1, 2 * R_BRIDGE // 3 + 1)
)
m_pruned = cv2.dilate(m_open, kernel_restore, iterations=1)
m_pruned = (m_pruned > 0).astype(np.uint8)

cv2.imwrite(os.path.join(OUT_DIR, f"mask_pruned_R{R_BRIDGE}.png"),
            (m_pruned * 255).astype(np.uint8))

# --------------------- Connected Components AFTER bridge breaking + area filter ---------------------
num_cc, labels_cc = cv2.connectedComponents(m_pruned, connectivity=8)
print(f"Bridge-broken CC labels (including background): {num_cc}")

labels_filtered = labels_cc.copy()
for lab in range(1, num_cc):
    area = int((labels_cc == lab).sum())
    if area < MIN_PART_AREA:
        labels_filtered[labels_filtered == lab] = 0  # drop small particles

# Relabel after area filtering
num_final, labels_final = cv2.connectedComponents(
    (labels_filtered > 0).astype(np.uint8), connectivity=8
)
print(f"Final particles after MIN_PART_AREA={MIN_PART_AREA}: {num_final - 1}")

# --------------------- Build bboxes + centroids for each final CC label ---------------------
bboxes = {}       # lab -> (x0,y0,w,h,area_in_orig_mask)
centroids = {}    # lab -> (cx, cy) in full-image coords

for lab in range(1, num_final):
    ys, xs = np.where(labels_final == lab)
    if len(xs) == 0:
        continue
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w = x1 - x0 + 1
    h = y1 - y0 + 1

    area_orig = int(m_orig[ys, xs].sum())  # area in original mask (0/1)
    bboxes[lab] = (x0, y0, w, h, area_orig)

    # centroid for labeling (mean of coordinates)
    cx = float(xs.mean())
    cy = float(ys.mean())
    centroids[lab] = (cx, cy)

kept_ids = sorted(list(bboxes.keys()))
print(f"Particles kept: {len(kept_ids)}")

# Map CC label -> display ID (1..N) for overlay & CSV
display_id_map = {lab: i + 1 for i, lab in enumerate(kept_ids)}

# --------------------- Jittered crops, using ORIGINAL SEM + ORIGINAL MASK ---------------------
def jitter_bbox(x0, y0, w, h):
    sx = 1.0 + random.uniform(-JITTER_SCALE, JITTER_SCALE)
    sy = 1.0 + random.uniform(-JITTER_SCALE, JITTER_SCALE)
    nw = int(round(w * sx))
    nh = int(round(h * sy))

    cx = x0 + w // 2
    cy = y0 + h // 2

    dx = int(round(random.uniform(-JITTER_SHIFT, JITTER_SHIFT) * w))
    dy = int(round(random.uniform(-JITTER_SHIFT, JITTER_SHIFT) * h))

    x0n = clamp_int(cx - nw // 2 + dx, 0, W - 1)
    y0n = clamp_int(cy - nh // 2 + dy, 0, H - 1)
    x1n = clamp_int(x0n + nw, 0, W)
    y1n = clamp_int(y0n + nh, 0, H)

    nw = max(4, x1n - x0n)
    nh = max(4, y1n - y0n)
    return x0n, y0n, nw, nh

def crop_to_tensor_for_bbox(x0, y0, w, h):
    # add context around bbox (on ORIGINAL SEM + ORIGINAL MASK)
    pad_x = int(round(CONTEXT_PAD * w))
    pad_y = int(round(CONTEXT_PAD * h))

    xa = clamp_int(x0 - pad_x, 0, W - 1)
    ya = clamp_int(y0 - pad_y, 0, H - 1)
    xb = clamp_int(x0 + w + pad_x, 0, W)
    yb = clamp_int(y0 + h + pad_y, 0, H)

    roi_g = g_u8[ya:yb, xa:xb]
    roi_m = m_orig[ya:yb, xa:xb]  # ORIGINAL mask

    x3_t, _ = preprocess_crop(roi_g, roi_m, size=NET_SIZE)  # matches training
    return x3_t

# Assemble tensors + owner mapping
all_tensors = []
owners = []  # particle label for each tensor

for lab in kept_ids:
    x0, y0, w, h, _ = bboxes[lab]

    # Deterministic center crop
    all_tensors.append(crop_to_tensor_for_bbox(x0, y0, w, h))
    owners.append(lab)

    # Jittered crops
    for _ in range(JITTERS_PER_ID - 1):
        jx0, jy0, jw, jh = jitter_bbox(x0, y0, w, h)
        all_tensors.append(crop_to_tensor_for_bbox(jx0, jy0, jw, jh))
        owners.append(lab)

print(f"Prepared {len(all_tensors)} crops ({JITTERS_PER_ID} per particle). Running inference...")

# --------------------- Batched inference ---------------------
preds = {lab: [] for lab in kept_ids}

with torch.no_grad():
    for i in range(0, len(all_tensors), BATCH_SIZE):
        batch = all_tensors[i:i + BATCH_SIZE]
        xt = torch.stack(batch, dim=0).to(device)  # (B,3,H,W)
        out = model(xt)                             # (B,1)
        h_pred = out.view(-1).detach().cpu().numpy()

        for hp, lab in zip(h_pred, owners[i:i + BATCH_SIZE]):
            preds[lab].append(float(hp))

# Aggregate per particle
agg = {}
all_means = []
for lab in kept_ids:
    ps = np.array(preds[lab], dtype=np.float32)
    if len(ps) == 0:
        mean_h = 0.5
        std_h  = 0.0
        min_h  = 0.5
        max_h  = 0.5
    else:
        mean_h = float(ps.mean())
        std_h  = float(ps.std())
        min_h  = float(ps.min())
        max_h  = float(ps.max())
    agg[lab] = {
        "H_mean": mean_h,
        "H_std":  std_h,
        "H_min":  min_h,
        "H_max":  max_h,
        "n_views": len(ps)
    }
    all_means.append(mean_h)

all_means = np.array(all_means, dtype=np.float32)
global_min_H = float(all_means.min()) if len(all_means) else 0.0
global_max_H = float(all_means.max()) if len(all_means) else 1.0
print(f"H_mean range across particles: [{global_min_H:.3f}, {global_max_H:.3f}]")

# --------------------- Save per-particle CSV ---------------------
csv_path = os.path.join(OUT_DIR, "per_particle_Hnorm_predictions.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "ParticleLabel",         # CC label (labels_final)
        "DisplayID",             # 1..N, matches overlay numbers
        "x", "y", "w", "h",
        "area_px_originalMask",
        "H_mean", "H_std", "H_min", "H_max", "n_views"
    ])
    for lab in kept_ids:
        x0, y0, w0, h0, area = bboxes[lab]
        row = agg[lab]
        w.writerow([
            lab,
            display_id_map[lab],
            x0, y0, w0, h0, area,
            f"{row['H_mean']:.6f}",
            f"{row['H_std']:.6f}",
            f"{row['H_min']:.6f}",
            f"{row['H_max']:.6f}",
            row["n_views"],
        ])
print(f"CSV saved -> {csv_path}")

# --------------------- Build per-pixel H_map (continuous heat map) ---------------------
# Initialize with global_min_H everywhere
H_map = np.full((H, W), global_min_H, dtype=np.float32)

# Assign mean H to pixels belonging to each final CC label
for lab in kept_ids:
    H_map[labels_final == lab] = agg[lab]["H_mean"]

# Force epoxy/background (original mask == 0) to global_min_H as well
H_map[m_orig == 0] = global_min_H

# Normalize to [0,1] for colormap
den = (global_max_H - global_min_H + 1e-8)
H_norm01 = (H_map - global_min_H) / den
H_norm01 = np.clip(H_norm01, 0.0, 1.0)

H_u8 = (H_norm01 * 255.0).astype(np.uint8)

# --------------------- Make heatmap overlays (plasma + inferno) ---------------------
# Convert SEM to BGR
sem_bgr = cv2.cvtColor(g_u8, cv2.COLOR_GRAY2BGR)

# OpenCV colormap expects single-channel 8-bit
heat_plasma  = cv2.applyColorMap(H_u8, cv2.COLORMAP_PLASMA)
heat_inferno = cv2.applyColorMap(H_u8, cv2.COLORMAP_INFERNO)

# Blend overlays
overlay_plasma  = cv2.addWeighted(sem_bgr, ALPHA_SEM, heat_plasma,  ALPHA_HEAT, 0)
overlay_inferno = cv2.addWeighted(sem_bgr, ALPHA_SEM, heat_inferno, ALPHA_HEAT, 0)

# --------------------- Draw numbered labels on both overlays ---------------------
# Font scale scales with image width similarly to your earlier script
label_font_scale = max(1.2, W / 4000.0 * 1.2)
label_thickness  = 2

for lab in kept_ids:
    cx, cy = centroids[lab]
    disp_id = display_id_map[lab]
    center_xy = (int(round(cx)), int(round(cy)))
    put_label_with_bg(
        overlay_plasma,
        str(disp_id),
        center_xy,
        font_scale=label_font_scale,
        thickness=label_thickness,
        alpha=0.9,
        pad=6,
        radius=5
    )
    put_label_with_bg(
        overlay_inferno,
        str(disp_id),
        center_xy,
        font_scale=label_font_scale,
        thickness=label_thickness,
        alpha=0.9,
        pad=6,
        radius=5
    )

# --------------------- Save overlays ---------------------
out_plasma  = os.path.join(OUT_DIR, "Hnorm_fullfield_plasma.png")
out_inferno = os.path.join(OUT_DIR, "Hnorm_fullfield_inferno.png")

cv2.imwrite(out_plasma,  overlay_plasma)
cv2.imwrite(out_inferno, overlay_inferno)

print(f"Saved heatmap overlays ->")
print(f" - {out_plasma}")
print(f" - {out_inferno}")
print("Done.")
# --------------------- Viridis overlay WITH continuous colorbar ---------------------
# This uses H_norm01 directly, overlays on SEM, and adds a 0–1 scale bar.

fig, ax = plt.subplots(figsize=(8, 6))

# Show SEM in the background
ax.imshow(g_u8, cmap="gray")

# Overlay normalized H_map with viridis (purple -> green)
im = ax.imshow(
    H_norm01,
    cmap="viridis",        # purple to yellow-green, like your pentagon plot
    alpha=ALPHA_HEAT,
    vmin=0.0,
    vmax=1.0,
)

ax.set_axis_off()

# Add continuous colorbar for H_norm
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Predicted $H_{norm}$", fontsize=10)
cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"])

fig.tight_layout()
out_viridis = os.path.join(OUT_DIR, "Hnorm_fullfield_viridis_with_scale.png")
fig.savefig(out_viridis, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved viridis overlay with scale -> {out_viridis}")
