# NOTE:
# This script extracts penultimate-layer embeddings from a trained DenseNet121 model
# and performs PCA analysis for visualization.
# It assumes the same preprocessing, dataset splits, and DataLoader definitions
# as used in the training script (06_densenet121_regression_training.py).
# Data and checkpoints must be available locally.
# ================= Embedding dump + PCA for DenseNet-121 =================

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---- Paths / config ----
CKPT_PATH = r"best_densenet121_regression.pt" 
EMB_DIR = r"embeddings"
os.makedirs(EMB_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = (device.type == "cuda")

# ---- Build the SAME DenseNet-121 architecture and load weights ----
from torchvision import models
from torchvision.models import DenseNet121_Weights
model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
# Replace final layer to match regression head shape (will be overwritten by state_dict anyway)
model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
state = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval().to(device)

# ---- Embedding function (penultimate 1024-D vector) ----
@torch.no_grad()
def extract_embeddings(x3: torch.Tensor):
    """
    x3: (B,3,512,512) preprocessed as in training.
    Returns: emb (B,1024) and pred (B,1)
    """
    feats = model.features(x3)                      # (B,C,H,W)
    feats = F.relu(feats, inplace=False)
    pooled = F.adaptive_avg_pool2d(feats, (1,1))   # (B,C,1,1)
    emb = torch.flatten(pooled, 1)                 # (B,1024)
    pred = model.classifier(emb)                   # (B,1)
    return emb, pred

# ---- Helper to run one loader and collect embeddings/targets/ids ----
def run_loader_for_embeddings(loader, subset_name: str):
    embs, ys, ids = [], [], []
    with torch.no_grad():
        for xb, yb, pid in loader:
            xb = xb.to(device, non_blocking=USE_CUDA)
            emb, pred = extract_embeddings(xb)
            embs.append(emb.cpu().numpy())
            ys.append(yb.cpu().numpy().ravel())
            ids.extend(list(pid))
    E = np.concatenate(embs, axis=0) if len(embs) else np.zeros((0,1024), dtype=np.float32)
    Y = np.concatenate(ys, axis=0)   if len(ys)   else np.zeros((0,), dtype=np.float32)
    df = pd.DataFrame(E, columns=[f"f{i:04d}" for i in range(E.shape[1])])
    df.insert(0, "Particle_ID", ids)
    df.insert(1, "H_true", Y)
    out_csv = os.path.join(EMB_DIR, f"embeddings_{subset_name}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved embeddings: {out_csv} | shape={E.shape}")
    return df

# ---- Collect embeddings for train/val/test ----
emb_train = run_loader_for_embeddings(train_loader, "train")
emb_val   = run_loader_for_embeddings(val_loader,   "val")
emb_test  = run_loader_for_embeddings(test_loader,  "test")

# ---- Combine for PCA (use only train if you want zero leakage; here we combine for visualization) ----
emb_all = pd.concat([
    emb_train.assign(Split="train"),
    emb_val.assign(Split="val"),
    emb_test.assign(Split="test"),
], axis=0, ignore_index=True)

feat_cols = [c for c in emb_all.columns if c.startswith("f")]
X = emb_all[feat_cols].to_numpy(dtype=np.float32)
y = emb_all["H_true"].to_numpy(dtype=np.float32)

# ---- PCA to 2D (for plotting) and 50D (for analysis) ----
pca2  = PCA(n_components=2, random_state=42)
pca50 = PCA(n_components=50, random_state=42)
Z2    = pca2.fit_transform(X)
Z50   = pca50.fit_transform(X)

# Save PCA results
emb_all["PC1"] = Z2[:,0]
emb_all["PC2"] = Z2[:,1]
pca_csv = os.path.join(EMB_DIR, "embeddings_with_PCA.csv")
emb_all.to_csv(pca_csv, index=False)
print(f"Saved: {pca_csv}")

# Save PCA components / explained variance
meta = pd.DataFrame({
    "component": [f"PC{i+1}" for i in range(pca50.n_components_)],
    "explained_variance_ratio": pca50.explained_variance_ratio_
})
meta_csv = os.path.join(EMB_DIR, "pca50_explained_variance.csv")
meta.to_csv(meta_csv, index=False)
print(f"Saved: {meta_csv} (sum EVR={meta['explained_variance_ratio'].sum():.3f})")

# ---- 2D scatter colored by H_true, faceted by split ----
plt.figure(figsize=(7.0, 5.8))
sc = plt.scatter(emb_all["PC1"], emb_all["PC2"], c=emb_all["H_true"], s=10, alpha=0.8)
plt.xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("DenseNet-121 penultimate embeddings → PCA (colored by H_true)")
cbar = plt.colorbar(sc); cbar.set_label("H_true")
plt.tight_layout()
plot_path = os.path.join(EMB_DIR, "pca2_scatter_by_H.png")
plt.savefig(plot_path, dpi=180)
plt.close()
print(f"Saved: {plot_path}")

# Optional: separate plots by split
for split in ["train","val","test"]:
    sub = emb_all[emb_all["Split"]==split]
    if len(sub) == 0: continue
    plt.figure(figsize=(6.4, 5.2))
    sc = plt.scatter(sub["PC1"], sub["PC2"], c=sub["H_true"], s=10, alpha=0.85)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(f"PCA (penultimate) — {split}")
    cbar = plt.colorbar(sc); cbar.set_label("H_true")
    plt.tight_layout()
    pth = os.path.join(EMB_DIR, f"pca2_scatter_{split}.png")
    plt.savefig(pth, dpi=180); plt.close()
    print(f"Saved: {pth}")
