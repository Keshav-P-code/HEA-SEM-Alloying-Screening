# NOTE:
# This script trains DenseNet121 for the regression of H_norm_atomic5 using masked SEM images.
# Data files (CSV, images, masks) are hosted separately (UNB repository) and must be downloaded locally.
# ==================================== DenseNet121 → Regression (H_norm / Entropy) ====================================
# - Grouped split by Base_ID ONLY
# - Keeps ALL offline augmentations (original + every __aug_* present)
# - Preprocess: CLAHE -> mask gate -> resize(512, INTER_AREA) -> re-gate -> mask-aware z-norm -> 3ch
# - Loss: MSE; Early stopping on Val MAE; Metrics: MAE / RMSE / R^2
# - Outputs: best ckpt, test_predictions_regression.csv, truth_vs_pred.png, residuals_hist.png
# ==================================================================================================================

import os, re, glob, math, time, warnings
from collections import Counter
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import DenseNet121_Weights   # CHANGED: import DenseNet weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# --------------------- YOUR PATHS ---------------------
IMAGES_DIR = r"augmented_images"                 # local folder containing images
MASKS_DIR  = r"Augmented_output_masks/masks"     # local folder containing masks
CSV_PATH   = r"HEA10_30_labels-Aug.csv"          # local labels CSV
OUTPUT_DIR = r"outputs_regression_baseid_only_dense121"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column with particle ID (with/without __aug_* suffix)
ID_COL = "Particle_ID"

# Target column (set explicitly if autodetect fails)
ENTROPY_COL = "H_norm_atomic5"  # e.g., "H_norm" or "Entropy"

# Entire bases to drop (originals and all augmentations)
DROP_BASE_IDS = {"Particle_733", "Particle_795"}

# --------------------- TRAINING CONFIG ---------------------
IMG_SIZE     = 512
USE_CLAHE    = True
BATCH_SIZE   = 8
EPOCHS       = 40
PATIENCE     = 10
LR           = 1e-4
WEIGHT_DECAY = 1e-4
SEED         = 42
NUM_WORKERS  = 0

# Optional: keep False per your request (no stratification).
# If you later want similar H_norm distributions across splits, set to True to stratify by H_norm quantile bins.
STRATIFY_ON_TARGET = True
N_BINS_FOR_TARGET  = 10

# --------------------- Repro & Device ---------------------
def set_seed(s):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = (device.type == "cuda")
print("Device:", device)

# --------------------- Helpers ---------------------
EXTS = [".tif",".tiff",".png",".jpg",".jpeg",".bmp",".TIF",".TIFF",".PNG",".JPG",".JPEG",".BMP"]

def find_by_stem(folder: str, stem: str):
    for ext in EXTS:
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p): return p
    hits = []
    for ext in EXTS:
        hits += glob.glob(os.path.join(folder, stem + "*" + ext))
    return hits[0] if hits else None

def ensure_binary(mask):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return (mask > 127).astype(np.uint8)

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if USE_CLAHE else None

def load_gated(ipath, mpath, img_size=IMG_SIZE):
    g = cv2.imread(ipath, cv2.IMREAD_GRAYSCALE); assert g is not None, ipath
    m = cv2.imread(mpath, cv2.IMREAD_UNCHANGED);  assert m is not None, mpath
    m = ensure_binary(m)
    if CLAHE is not None:
        g = CLAHE.apply(g)
    g = cv2.resize(g, (img_size, img_size), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    gated = (g * m).astype(np.uint8)
    return gated, m

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

def split_stem_aug(pid: str):
    """Return (base_id, aug_tag) where base_id has no __aug_* suffix; tag is 'original' or 'aug_xxx'."""
    s = os.path.splitext(str(pid))[0]
    m = re.search(r"^(.+?)__aug_([A-Za-z0-9]+)$", s)
    if m:
        return m.group(1), f"aug_{m.group(2).lower()}"
    return s, "original"

def auto_pick_entropy_column(df):
    if ENTROPY_COL is not None:
        assert ENTROPY_COL in df.columns, f"ENTROPY_COL='{ENTROPY_COL}' not in CSV. Available: {list(df.columns)}"
        return ENTROPY_COL
    candidates = [c for c in df.columns if c.lower() in ["h","h_norm","hnorm","entropy"] or "entropy" in c.lower()]
    if len(candidates) == 0:
        raise AssertionError(
            "Could not auto-detect the entropy column. "
            "Set ENTROPY_COL (e.g. 'H_norm', 'Entropy'). "
            f"Available columns: {list(df.columns)}"
        )
    if len(candidates) > 1:
        print(f"Detected multiple target-like columns {candidates}. Using '{candidates[0]}'.")
    return candidates[0]

# --------------------- Load CSV & keep ALL augs ---------------------
raw = pd.read_csv(CSV_PATH)
assert ID_COL in raw.columns, f"CSV must include '{ID_COL}'. Got: {list(raw.columns)}"
TARGET_COL = auto_pick_entropy_column(raw)

raw["Base_ID"], raw["Aug_Tag"] = zip(*raw[ID_COL].astype(str).map(split_stem_aug))

# Drop bases 733 & 795
before = len(raw)
raw = raw[ ~raw["Base_ID"].isin(DROP_BASE_IDS) ].reset_index(drop=True)
print(f"Dropped {before - len(raw)} rows for bases {sorted(list(DROP_BASE_IDS))}")

# Resolve image/mask paths and collect target
items = []
missing = 0
for r in raw.itertuples(index=False):
    pid   = getattr(r, ID_COL)
    base  = getattr(r, "Base_ID")
    y_reg = float(getattr(r, TARGET_COL))
    stem = os.path.splitext(str(pid))[0]
    ip = find_by_stem(IMAGES_DIR, stem)
    mp = find_by_stem(MASKS_DIR,  stem)
    if ip is None or mp is None:
        missing += 1
        continue
    items.append((pid, base, ip, mp, y_reg))
if missing:
    print(f"⚠️ Skipped {missing} rows due to missing image/mask.")

df = pd.DataFrame(items, columns=["Particle_ID","Base_ID","Image_Path","Mask_Path","y_reg"])
print("Resolved items:", len(df))

# --------------------- Grouped split by Base_ID (no class stratification) ---------------------
# Optional: stratify on binned target to keep distribution similar (OFF by default)
base_target = df.groupby("Base_ID")["y_reg"].mean()  # one representative value per base

if STRATIFY_ON_TARGET:
    # build quantile bins for target stratification at the base level
    bins = pd.qcut(base_target.rank(method="first"), q=N_BINS_FOR_TARGET, labels=False)
    stratify_array = bins.to_numpy()
else:
    stratify_array = None

base_ids = base_target.index.to_list()

train_bases, temp_bases = train_test_split(
    base_ids, test_size=0.30, random_state=SEED, stratify=stratify_array
)

# For second split, create stratify labels for temp if enabled
if STRATIFY_ON_TARGET:
    temp_mask = base_target.index.isin(temp_bases)
    temp_bins = pd.qcut(base_target[temp_mask].rank(method="first"), q=min(N_BINS_FOR_TARGET, len(temp_bases)), labels=False)
    stratify_temp = temp_bins.to_numpy()
else:
    stratify_temp = None

val_bases, test_bases = train_test_split(
    temp_bases, test_size=0.50, random_state=SEED, stratify=stratify_temp
)

train_df = df[df["Base_ID"].isin(train_bases)].reset_index(drop=True)
val_df   = df[df["Base_ID"].isin(val_bases)].reset_index(drop=True)
test_df  = df[df["Base_ID"].isin(test_bases)].reset_index(drop=True)

def _brief(name, subset):
    print(f"{name:5s}: n={len(subset):4d} | Bases={subset['Base_ID'].nunique()}")

print("\n=== Split Summary (ALL augs kept; grouped by Base_ID) ===")
_brief("Train", train_df); _brief("Val", val_df); _brief("Test", test_df)

# --------------------- Dataset ---------------------
class BSERegressionDS(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        gated, m01 = load_gated(r["Image_Path"], r["Mask_Path"], IMG_SIZE)
        x3 = to_tensor_maskaware(gated, m01)    # (3,H,W)
        y  = float(r["y_reg"])
        return x3, torch.tensor([y], dtype=torch.float32), r["Particle_ID"]

train_loader = DataLoader(BSERegressionDS(train_df), batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=USE_CUDA)
val_loader   = DataLoader(BSERegressionDS(val_df),   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=USE_CUDA)
test_loader  = DataLoader(BSERegressionDS(test_df),  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=USE_CUDA)

print(f"\nLoaders -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# --------------------- Model: DenseNet121 → 1D regression ---------------------
weights = DenseNet121_Weights.IMAGENET1K_V1            # CHANGED: use DenseNet121 weights
model = models.densenet121(weights=weights)            # CHANGED: build DenseNet121
model.classifier = nn.Linear(model.classifier.in_features, 1)  # CHANGED: reg head
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler('cuda', enabled=USE_CUDA)

def epoch_pass(loader, train=False):
    model.train(train)
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []
    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)  # (B,1)
        if train: optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=USE_CUDA):
            out = model(xb)         # (B,1)
            loss = criterion(out, yb)
        if train:
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer); scaler.update()
        total_loss += loss.item() * xb.size(0); n += xb.size(0)
        y_true.append(yb.detach().cpu().numpy())
        y_pred.append(out.detach().cpu().numpy())
    y_true = np.concatenate(y_true, axis=0).ravel()
    y_pred = np.concatenate(y_pred, axis=0).ravel()
    mse = ((y_true - y_pred)**2).mean()
    mae = np.abs(y_true - y_pred).mean()
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    return total_loss / max(1,n), mae, rmse, r2, y_true, y_pred

# --------------------- Train with early stopping on Val MAE ---------------------
hist = {"epoch": [], "train_mae": [], "val_mae": [], "train_rmse": [], "val_rmse": [], "train_r2": [], "val_r2": []}
best_val_mae = float("inf")
SAVE_PATH = os.path.join(OUTPUT_DIR, "best_resnet18_regression.pt")  # kept EXACTLY the same on purpose
no_improve = 0

for ep in range(1, EPOCHS+1):
    t0 = time.perf_counter()
    tr_loss, tr_mae, tr_rmse, tr_r2, _, _ = epoch_pass(train_loader, train=True)
    va_loss, va_mae, va_rmse, va_r2, _, _ = epoch_pass(val_loader, train=False)
    scheduler.step()

    hist["epoch"].append(ep)
    hist["train_mae"].append(tr_mae); hist["val_mae"].append(va_mae)
    hist["train_rmse"].append(tr_rmse); hist["val_rmse"].append(va_rmse)
    hist["train_r2"].append(tr_r2); hist["val_r2"].append(va_r2)

    saved = ""
    if va_mae + 1e-9 < best_val_mae:
        best_val_mae = va_mae
        torch.save(model.state_dict(), SAVE_PATH)
        no_improve = 0; saved = " [*saved*]"
    else:
        no_improve += 1

    dt = (time.perf_counter() - t0) / 60.0
    print(f"Epoch {ep:02d} | Train MAE={tr_mae:.4f} RMSE={tr_rmse:.4f} R2={tr_r2:.3f} | "
          f"Val MAE={va_mae:.4f} RMSE={va_rmse:.4f} R2={va_r2:.3f}{saved} | {dt:.1f} min")

    if no_improve >= PATIENCE:
        print(f"Early stopping (patience={PATIENCE}). Best Val MAE={best_val_mae:.4f}")
        break

print(f"\n✅ Training complete. Best model -> {SAVE_PATH}")

# --------------------- Evaluate on TEST + Plots ---------------------
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
model.eval()

_, te_mae, te_rmse, te_r2, y_true, y_pred = epoch_pass(test_loader, train=False)
print(f"\n📊 TEST: MAE={te_mae:.4f} | RMSE={te_rmse:.4f} | R²={te_r2:.3f}")

# Save per-sample predictions
test_ids = []
with torch.no_grad():
    for _, _, pid in DataLoader(BSERegressionDS(test_df), batch_size=64, shuffle=False):
        test_ids.extend(list(pid))
pred_tbl = pd.DataFrame({"Particle_ID": test_ids, "True": y_true, "Pred": y_pred, "AbsError": np.abs(y_true - y_pred)})
pred_csv = os.path.join(OUTPUT_DIR, "test_predictions_regression.csv")
pred_tbl.to_csv(pred_csv, index=False)
print(f"Saved: {pred_csv}")

# Truth vs Prediction scatter (+ y=x)
plt.figure(figsize=(5.8,5.6))
plt.scatter(y_true, y_pred, s=14, alpha=0.7)
mn = float(np.min([y_true.min(), y_pred.min()]))
mx = float(np.max([y_true.max(), y_pred.max()]))
plt.plot([mn, mx], [mn, mx], '--', linewidth=1)
plt.xlabel("True H_norm"); plt.ylabel("Predicted H_norm")
plt.title(f"Test: MAE={te_mae:.3f}, RMSE={te_rmse:.3f}, R²={te_r2:.3f}")
plt.tight_layout()
out_scatter = os.path.join(OUTPUT_DIR, "truth_vs_pred.png")
plt.savefig(out_scatter, dpi=180); plt.close()
print(f"Saved: {out_scatter}")

# Residual histogram
resid = y_pred - y_true
plt.figure(figsize=(6.0,3.8))
plt.hist(resid, bins=30, edgecolor='k')
plt.xlabel("Residual (Pred - True)"); plt.ylabel("Count")
plt.title("Residuals (TEST)")
plt.tight_layout()
out_hist = os.path.join(OUTPUT_DIR, "residuals_hist.png")
plt.savefig(out_hist, dpi=180); plt.close()
print(f"Saved: {out_hist}")

# Save training history
pd.DataFrame(hist).to_csv(os.path.join(OUTPUT_DIR, "training_history_regression.csv"), index=False)
