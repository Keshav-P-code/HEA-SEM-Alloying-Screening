import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================== EDIT THESE ===================
CSV_PATH = r"RAW-DATA-COMP__atomic5_entropy__with_Edist.csv"  # <-- your E-distance CSV (local path)
OUT_DIR  = os.path.join(os.path.dirname(CSV_PATH), "pentagon_map")
USE_EDIST_RINGS = True
RING_LEVELS = [0.10, 0.20, 0.30, 0.40]
# ==================================================
os.makedirs(OUT_DIR, exist_ok=True)

cols = ["Al_at%5","Co_at%5","Cr_at%5","Fe_at%5","Ni_at%5"]
pretty = ["Al","Co","Cr","Fe","Ni"]  # labels for vertices
need = ["Particle_ID","H_norm_atomic5"] + cols
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

miss = [c for c in need if c not in df.columns]
if miss:
    raise ValueError(f"CSV must contain: {miss}")

# ---- 1) Prepare atomic fractions (sum ≈ 1)
X_pct = df[cols].astype(float).to_numpy()
X = np.clip(X_pct/100.0, 0.0, 1.0)
row_sum = X.sum(axis=1, keepdims=True)
row_sum[row_sum == 0] = 1.0
X = X / row_sum  # safety renormalization

# Ideal equiatomic (fraction)
ideal = np.full(5, 0.2, dtype=float)

# Signed deltas and distances
deltas = X - ideal  # n×5, positive => richer than equiatomic for that element
dist_frac = np.sqrt(((X - ideal)**2).sum(axis=1))  # Euclidean in fraction space
MAX_DIST_FRAC = np.sqrt(0.8)  # distance from [1,0,0,0,0] to [0.2]*5
dist_frac_norm = dist_frac / MAX_DIST_FRAC

# If your CSV already has distances, keep them and add checks (no overwrite)
if "E_dist_frac" not in df.columns:
    df["E_dist_frac"] = dist_frac
if "E_dist_frac_norm" not in df.columns:
    df["E_dist_frac_norm"] = dist_frac_norm

# ---- 2) Regular pentagon (unit circle), vertices in nice order for labels above
# Order vertices to match [Al, Co, Cr, Fe, Ni] around the circle (counterclockwise)
angles_deg = [90, 90-72, 90-2*72, 90-3*72, 90-4*72]  # 90°, 18°, -54°, -126°, -198°
angles = np.deg2rad(angles_deg)
V = np.column_stack([np.cos(angles), np.sin(angles)])  # (5,2)

# Barycentric projection: 2D position = sum_i p_i * v_i
XY = X @ V
cx, cy = (ideal @ V)  # 2D center of equiatomic
df["X2D"] = XY[:,0]
df["Y2D"] = XY[:,1]

# Dominant element (sector)
dom_idx = X.argmax(axis=1)
df["Dominant_Element"] = np.array(pretty)[dom_idx]

# ---- 3) Save augmented CSV
out_csv = os.path.join(OUT_DIR, "particles_with_pentagon_map.csv")
df_out = df[["Particle_ID", *cols, "H_norm_atomic5", "E_dist_frac", "E_dist_frac_norm", "Dominant_Element", "X2D", "Y2D"]]
df_out.to_csv(out_csv, index=False)
print(" Saved augmented CSV:", out_csv)

# ---- 4) Plot helpers
def draw_pentagon(ax):
    # polygon outline
    ax.plot(np.r_[V[:,0], V[0,0]], np.r_[V[:,1], V[0,1]], color="k", lw=1)
    # spokes and labels
    for i, lab in enumerate(pretty):
        ax.plot([cx, V[i,0]], [cy, V[i,1]], color="lightgray", lw=0.8, zorder=0)
        ax.text(V[i,0]*1.06, V[i,1]*1.06, lab, ha="center", va="center", fontsize=10)
    # center
    ax.scatter([cx], [cy], s=50, c="k", marker="o", zorder=3)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

def draw_rings(ax, levels):
    # Draw rings as circles around center in the 2D map scaled to match typical spread.
    # We map normalized fraction distance roughly to this 2D plane by matching mean radial scaling.
    # Compute a linear scale factor so that mean 2D radius ~ mean true normalized distance.
    r2d = np.sqrt((df["X2D"]-cx)**2 + (df["Y2D"]-cy)**2).to_numpy()
    valid = r2d > 0
    if valid.any():
        scale = (r2d[valid].mean() / dist_frac_norm[valid].mean()) if dist_frac_norm[valid].mean() > 0 else 1.0
    else:
        scale = 1.0
    theta = np.linspace(0, 2*np.pi, 512)
    for lv in levels:
        rr = lv * scale
        ax.plot(cx + rr*np.cos(theta), cy + rr*np.sin(theta),
                linestyle="--", color="gray", lw=0.9, alpha=0.8)

# ---- 5) Plot A: color by H_norm (continuous)
plt.figure(figsize=(7.2, 6.4))
ax = plt.gca()
draw_pentagon(ax)

sc = ax.scatter(df["X2D"], df["Y2D"],
                c=df["H_norm_atomic5"], cmap="viridis",
                s=26, alpha=0.95, edgecolors="none", zorder=2)
cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("H_norm_atomic5 (0 → 1)")

if USE_EDIST_RINGS and "E_dist_frac_norm" in df.columns:
    draw_rings(ax, RING_LEVELS)

plt.title("Pentagon composition map — colored by H_norm (ideal HEA at center)")
plt.tight_layout()
png1 = os.path.join(OUT_DIR, "pentagon_map_entropy.png")
plt.savefig(png1, dpi=220)
plt.show()
print(" Saved:", png1)

# ---- 6) Plot B: dominant element colors + (optional) distance rings
colors_elem = {"Al":"#1f77b4", "Cr":"#ff7f0e", "Fe":"#2ca02c", "Co":"#d62728", "Ni":"#9467bd"}

plt.figure(figsize=(7.2, 6.4))
ax = plt.gca()
draw_pentagon(ax)
if USE_EDIST_RINGS and "E_dist_frac_norm" in df.columns:
    draw_rings(ax, RING_LEVELS)

for el in pretty:
    m = (df["Dominant_Element"] == el)
    ax.scatter(df.loc[m,"X2D"], df.loc[m,"Y2D"], s=28, c=colors_elem[el], label=el,
               alpha=0.9, edgecolors="none")

plt.legend(title="Dominant element", loc="upper right", frameon=False)
plt.title("Pentagon composition map — dominant direction (spokes show triangle sectors)")
plt.tight_layout()
png2 = os.path.join(OUT_DIR, "pentagon_map_dominant.png")
plt.savefig(png2, dpi=220)
plt.show()
print(" Saved:", png2)
