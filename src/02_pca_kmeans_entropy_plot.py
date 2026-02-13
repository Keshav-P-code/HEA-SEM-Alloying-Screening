import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==========================
# 1) EDIT THIS: input file
# ==========================
INPUT_CSV = r"RAW-DATA-COMP__atomic5_entropy.csv"  # <-- set your local file path

# Columns to use (exact names per your file)
ID_COL = "Particle_ID"
FEAT_COLS = ["Al_at%5","Co_at%5","Cr_at%5","Fe_at%5","Ni_at%5"]
ENTROPY_COL = "H_norm_atomic5"

# K-means settings
K = 3
RANDOM_STATE = 42
POINT_SIZE = 48

# ==========================
# 2) Load & sanitize
# ==========================
df = pd.read_csv(INPUT_CSV)
# Strip accidental spaces in column names
df.columns = [c.strip() for c in df.columns]

needed = [ID_COL, *FEAT_COLS, ENTROPY_COL]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

X = df[FEAT_COLS].astype(float).to_numpy()  # atomic % (within 5 metals), rows should sum≈100
# Convert to fractions for PCA (optional; scaler will normalize anyway)
X_frac = X / 100.0

# ==========================
# 3) Standardize → PCA(2)
# ==========================
scaler = StandardScaler()
Xz = scaler.fit_transform(X_frac)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
PC = pca.fit_transform(Xz)  # shape (n,2)
expl = pca.explained_variance_ratio_ * 100  # %

# ==========================
# 4) KMeans on standardized 5D
# ==========================
kmeans = KMeans(n_clusters=K, n_init=20, random_state=RANDOM_STATE)
clusters = kmeans.fit_predict(Xz)  # labels 0..K-1

# Project centroids to PCA plane for plotting
centroids_5d = kmeans.cluster_centers_
centroids_2d = pca.transform(centroids_5d)

# Silhouette (quick quality gauge)
sil = silhouette_score(Xz, clusters) if len(np.unique(clusters)) > 1 else np.nan
print(f"K={K}  |  Silhouette (std+PCA space): {sil:.3f}")
print(f"PCA explained variance: PC1={expl[0]:.1f}%  PC2={expl[1]:.1f}%")

# ==========================
# 5) Plot: PC1 vs PC2 colored by H_norm_atomic5
# ==========================
Hn = df[ENTROPY_COL].astype(float).to_numpy()
plt.figure(figsize=(8.5, 6.5))

sc = plt.scatter(
    PC[:,0], PC[:,1],
    c=Hn, s=POINT_SIZE, cmap="viridis",
    vmin=0.0, vmax=1.0, alpha=0.9,
    edgecolors="none"
)

# overlay centroids
plt.scatter(
    centroids_2d[:,0], centroids_2d[:,1],
    s=220, marker="X", c="white",
    edgecolors="black", linewidths=1.5, zorder=3
)
for i, (cx, cy) in enumerate(centroids_2d):
    plt.text(cx, cy, f"K{i+1}", ha="center", va="center",
             fontsize=9, weight="bold", color="black", zorder=4)

plt.xlabel(f"PC1 ({expl[0]:.1f}% var)")
plt.ylabel(f"PC2 ({expl[1]:.1f}% var)")
plt.title(f"PCA of 5-metal atomic fractions  •  KMeans K={K}  •  Silhouette={sil:.3f}")
cbar = plt.colorbar(sc)
cbar.set_label("H_norm_atomic5 (0=segregated  →  1=HEA-like)", rotation=90)

plt.tight_layout()

# ==========================
# 6) Save outputs
# ==========================
out_dir = os.path.dirname(INPUT_CSV)
base = os.path.splitext(os.path.basename(INPUT_CSV))[0]

plot_path = os.path.join(out_dir, f"{base}__PCA_K{K}_colored_by_Hnorm.png")
plt.savefig(plot_path, dpi=220)
plt.show()
print("Saved plot:", plot_path)

out_df = pd.DataFrame({
    ID_COL: df[ID_COL].values,
    "PC1": PC[:,0],
    "PC2": PC[:,1],
    "Cluster": clusters,
    ENTROPY_COL: Hn
})
out_csv = os.path.join(out_dir, f"{base}__PCA_K{K}_scores.csv")
out_df.to_csv(out_csv, index=False)
print("Saved scores:", out_csv)

# (Optional) Quick cluster composition summary (atomic % medians)
summary = df[[ID_COL, *FEAT_COLS]].copy()
summary["Cluster"] = clusters
cluster_medians = summary.groupby("Cluster")[FEAT_COLS].median().rename_axis("Cluster")
print("\nCluster median atomic % (within 5 metals):\n", cluster_medians)
