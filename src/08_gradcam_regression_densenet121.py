# ==================== Grad-CAM for DenseNet121 Regression (masked Pred-CAM only) ====================
# - Reads: best_Densenet121_regression.pt  +  test_predictions_regression.csv
# - Selects: (A) AbsError > 0.10, (B) Top-10 least AbsError
# - Panels: [Gated BSE] [Pred-CAM heatmap (masked)] [Pred-CAM overlay (masked + contour)]
# - Uses forward-hook + autograd.grad (no backward hooks) to avoid inplace-view errors
# ====================================================================================================

import os, glob, re, warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

warnings.filterwarnings("ignore")

# --------------------- PATHS (EDIT IF NEEDED) ---------------------
IMAGES_DIR = r"augmented_images"
MASKS_DIR  = r"Augmented_output_masks/masks"
MODEL_DIR  = r"outputs_regression_baseid_only_dense121"

CKPT_PATH  = os.path.join(MODEL_DIR, "best_Densenet121_regression.pt")
PRED_CSV   = os.path.join(MODEL_DIR, "test_predictions_regression.csv")

OUT_DIR    = os.path.join(MODEL_DIR, "gradcam_regression")
OUT_HIGH   = os.path.join(OUT_DIR, "high_error")
OUT_LOW    = os.path.join(OUT_DIR, "low_error")
os.makedirs(OUT_HIGH, exist_ok=True)
os.makedirs(OUT_LOW, exist_ok=True)

# --------------------- SETTINGS ---------------------
IMG_SIZE     = 512
USE_CLAHE    = True
THRESHOLD_AE = 0.10
TOPK_LOW_ERR = 10     
SHOW_FIGS    = False

SEED = 42
def set_seed(s):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --------------------- FILE HELPERS ---------------------
EXTS = [".tif",".tiff",".png",".jpg",".jpeg",".bmp",".TIF",".TIFF",".PNG",".JPG",".JPEG",".BMP"]

def find_by_stem(folder: str, stem: str):
    for ext in EXTS:
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p): return p
    hits = []
    for ext in EXTS:
        hits += glob.glob(os.path.join(folder, stem + "*" + ext))
    return hits[0] if hits else None

# --------------------- IMAGE UTILS (match training) ---------------------
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if USE_CLAHE else None

def ensure_binary(mask):
    if mask.ndim == 3: mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return (mask > 127).astype(np.uint8)

def load_gated(ipath, mpath, img_size=IMG_SIZE):
    g = cv2.imread(ipath, cv2.IMREAD_GRAYSCALE); assert g is not None, f"Missing image: {ipath}"
    m = cv2.imread(mpath, cv2.IMREAD_UNCHANGED);  assert m is not None, f"Missing mask: {mpath}"
    m = ensure_binary(m)
    if CLAHE is not None:
        g = CLAHE.apply(g)
    g = cv2.resize(g, (img_size, img_size), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    gated = (g * m).astype(np.uint8)
    return g, m, gated

def to_tensor_maskaware(gated_u8, mask01):
    x01 = gated_u8.astype(np.float32) / 255.0
    x3  = np.stack([x01, x01, x01], axis=0)  # (3,H,W)
    m = mask01.astype(bool)
    if m.sum() > 0:
        for c in range(3):
            vals = x3[c][m]
            mu, sd = float(vals.mean()), float(vals.std() + 1e-6)
            x3[c] = (x3[c] - mu) / sd
    else:
        mu, sd = float(x3.mean()), float(x3.std() + 1e-6)
        x3 = (x3 - mu) / sd
    return torch.from_numpy(x3).float()

def mask_contour_rgb(mask01):
    mask_u8 = (mask01 * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros((mask_u8.shape[0], mask_u8.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(canvas, contours, -1, (255,255,255), 1)
    return canvas

def mask_and_normalize_cam(cam01, mask01):
    cam = cam01 * mask01
    if mask01.any():
        inside = cam[mask01 == 1]
        vmin, vmax = float(inside.min()), float(inside.max())
        if vmax > vmin:
            cam = (cam - vmin) / (vmax - vmin + 1e-12)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)
    else:
        cam = np.zeros_like(cam, dtype=np.float32)
    return cam

def overlay_heatmap_on_gray(gray_u8, heat01, alpha=0.45):
    heat = (np.clip(heat01,0,1)*255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[:, :, ::-1]
    base = gray_u8 if gray_u8.ndim == 3 else np.stack([gray_u8]*3, axis=-1)
    overlay = (alpha*heat_color + (1-alpha)*base).astype(np.uint8)
    return overlay

# --------------------- MODEL: DenseNet121 → 1D regression ---------------------
model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
model.classifier = nn.Linear(model.classifier.in_features, 1)

state = torch.load(CKPT_PATH, map_location="cpu")
if isinstance(state, dict) and "model" in state:
    model.load_state_dict(state["model"])
else:
    model.load_state_dict(state)
model = model.to(device).eval()

# --------------------- Grad-CAM capture (DenseNet-friendly) ---------------------
# Avoid backward hooks to prevent "view + inplace" autograd errors.
target_layer = model.features  # last conv feature stack before pooling
_feats_tensor = {"t": None}

def fwd_hook(module, inp, out):
    _feats_tensor["t"] = out  # keep computation graph (no .detach())

# Remove previous hook if re-running in notebook
try:
    _fwd_handle.remove()
except Exception:
    pass
_fwd_handle = target_layer.register_forward_hook(fwd_hook)

def grad_cam_regression(x_tensor):
    # x_tensor: [1, 3, H, W]
    assert x_tensor.ndim == 4 and x_tensor.size(0) == 1
    _feats_tensor["t"] = None

    x_tensor = x_tensor.requires_grad_(True)
    out = model(x_tensor)                 # [1,1]
    score = out[0, 0]

    feats = _feats_tensor["t"]            # [1,C,h,w]
    assert feats is not None, "Forward hook did not capture features."

    grads = torch.autograd.grad(
        outputs=score,
        inputs=feats,
        grad_outputs=torch.ones_like(score),
        retain_graph=False,
        create_graph=False,
        allow_unused=False
    )[0]  # [1,C,h,w]

    weights = grads.mean(dim=(2, 3), keepdim=True)     # [1,C,1,1]
    cam = (weights * feats).sum(dim=1, keepdim=True)   # [1,1,h,w]
    cam = torch.relu(cam)[0, 0].detach().cpu().numpy()
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    return cam

# --------------------- CSV: read + force safe column names ---------------------
def _clean_header(s: str) -> str:
    s = str(s).replace("\ufeff","").replace("\u00A0"," ").strip()
    for _ in range(2):
        if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
            s = s[1:-1].strip()
    return s

def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

pred_tbl = pd.read_csv(PRED_CSV)
pred_tbl.columns = [_clean_header(c) for c in pred_tbl.columns]

# Build case/space-insensitive lookup
idx = { _norm_key(c): c for c in pred_tbl.columns }
def _pick(*cands):
    for c in cands:
        k = _norm_key(c)
        if k in idx: return idx[k]
    raise KeyError(f"None of {cands} found in columns: {list(pred_tbl.columns)}")

particle_col = _pick("Particle_ID","particle id","particleid","id","stem","name","image_id")
true_col     = _pick("TRUE","True","y_true","ytrue","target","gt","y")
pred_col     = _pick("Pred","Prediction","y_pred","ypred","output")
ae_col       = _pick("AbsError","abs error","ae","abserror","|pred-true|")

# Force safe names (avoid reserved words like 'True')
pred_tbl = pred_tbl.rename(columns={
    particle_col: "particle_id",
    true_col:     "y_true",
    pred_col:     "y_pred",
    ae_col:       "abs_error",
})

# Strong types
for c in ("y_true","y_pred","abs_error"):
    pred_tbl[c] = pd.to_numeric(pred_tbl[c], errors="coerce")

pred_tbl = pred_tbl.dropna(subset=["y_true","y_pred","abs_error"]).reset_index(drop=True)

# Selections
high_err_df = pred_tbl[pred_tbl["abs_error"] > THRESHOLD_AE].copy()
low_err_df  = pred_tbl.nsmallest(TOPK_LOW_ERR, "abs_error").copy()
print(f"Selected: high_error={len(high_err_df)} (AE>{THRESHOLD_AE}), low_error={len(low_err_df)} (TOP {TOPK_LOW_ERR})")

# --------------------- 3-panel composite ---------------------
def make_composite(g, m, gated, pred_cam_masked, title):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    ax = axes

    ax[0].imshow(gated, cmap='gray', vmin=0, vmax=255); ax[0].set_title("Gated BSE", fontsize=11); ax[0].axis('off')
    ax[1].imshow(pred_cam_masked, cmap='jet', vmin=0, vmax=1); ax[1].set_title("Pred-CAM (masked)", fontsize=11); ax[1].axis('off')

    overlay = overlay_heatmap_on_gray(gated, pred_cam_masked, alpha=0.45)
    contour = mask_contour_rgb(m)
    out = cv2.addWeighted(overlay, 1.0, contour, 1.0, 0)
    ax[2].imshow(out); ax[2].set_title("Overlay (masked + contour)", fontsize=11); ax[2].axis('off')

    fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout()
    return fig

# --------------------- Runner (dict records, no attribute access) ---------------------
def run_bucket(df, out_dir, label):
    os.makedirs(out_dir, exist_ok=True)
    saved = 0

    for rec in df.to_dict(orient="records"):
        pid    = str(rec["particle_id"])
        y_true = float(rec["y_true"])
        y_pred = float(rec["y_pred"])
        ae     = float(rec["abs_error"])

        stem = os.path.splitext(pid)[0]
        ip = find_by_stem(IMAGES_DIR, stem)
        mp = find_by_stem(MASKS_DIR,  stem)
        if ip is None or mp is None:
            print(f" Skipping (missing file) → PID={pid} | img={ip} | msk={mp}")
            continue

        g, m, gated = load_gated(ip, mp, IMG_SIZE)
        x3 = to_tensor_maskaware(gated, m)[None, ...].to(device)

        cam_raw  = grad_cam_regression(x3)
        cam_mask = mask_and_normalize_cam(cam_raw, m)

        title = f"PID: {pid} | True: {y_true:.3f} | Pred: {y_pred:.3f} | AE: {ae:.3f}"
        fig = make_composite(g, m, gated, cam_mask, title)

        safe_pid = "".join(ch for ch in pid if ch.isalnum() or ch in ("-","_"))
        out_name = f"{label}__PID_{safe_pid}__true-{y_true:.3f}__pred-{y_pred:.3f}__AE-{ae:.3f}.png"
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, dpi=170)
        if SHOW_FIGS: plt.show()
        plt.close(fig)
        saved += 1

    print(f" Saved {saved} composites → {out_dir}")

# Run both buckets
run_bucket(high_err_df, OUT_HIGH, label="high_error")
run_bucket(low_err_df,  OUT_LOW,  label="low_error")
print("\nDone. Output root:", OUT_DIR)
