import os
import math
import pandas as pd
import numpy as np

# ========= EDIT THIS =========
# Use a repo-relative path (do NOT commit personal C:\Users\... paths to GitHub)
INPUT_CSV = r"RAW-DATA-COMP.csv"   # <-- put your raw composition CSV here
# =============================

# We will compute *atomic percent* (at%) from wt% for these 5 metals only,
# by converting wt% -> "moles" = wt / atomic_weight, then renormalizing over the 5.
METALS = ["Al", "Co", "Cr", "Fe", "Ni"]  # output order
ATOMIC_WEIGHTS = {  # g/mol (IUPAC)
    "Al": 26.9815385,
    "Co": 58.933194,
    "Cr": 51.9961,
    "Fe": 55.845,
    "Ni": 58.6934,
}

def to_numeric_cols(df, cols):
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def wt_to_atomic_percent_5(df, metals, aw):
    """
    From wt% columns in `metals`, compute at% *within those 5 metals only*:
      moles_i = wt_i / A_i
      p_i = moles_i / sum_j moles_j
      at%_i = 100 * p_i
    Returns (at_pct_df, p_df) where p_df are the fractions p_i (sum to 1).
    """
    moles = pd.DataFrame({m: df[m] / aw[m] for m in metals}, index=df.index)
    denom = moles[metals].sum(axis=1).replace(0, np.nan)
    p = moles.div(denom, axis=0).fillna(0.0)              # fractions over the 5 metals
    at_pct = (p * 100.0).rename(columns={m: f"{m}_at%5" for m in metals})
    return at_pct, p

def entropy_from_fractions(p_df):
    """Shannon H and H_norm from fractions p_i (row-wise, sum=1). Natural log."""
    eps = 1e-12
    H = -(p_df * np.log(p_df + eps)).sum(axis=1)
    H_norm = H / math.log(p_df.shape[1])  # divide by ln(5)
    return H, H_norm

def main():
    # Read input
    df = pd.read_csv(INPUT_CSV)

    # Keep convenient ID columns if they exist
    id_cols = [c for c in ["Particle_ID", "Sample_ID", "Spectrum"] if c in df.columns]
    out = df[id_cols].copy() if id_cols else pd.DataFrame()

    # Ensure our 5 metals exist and are numeric wt%
    df = to_numeric_cols(df, METALS)

    # Convert wt% -> atomic% (5-metal subspace)
    at_pct5, p5 = wt_to_atomic_percent_5(df, METALS, ATOMIC_WEIGHTS)

    # Entropy (from atomic fractions over these 5)
    H, H_norm = entropy_from_fractions(p5)

    # Assemble output
    # Include original wt% for reference (optional, helpful)
    for m in METALS:
        out[f"{m}_wt%"] = df[m]
    out = pd.concat([out, at_pct5], axis=1)
    out["H_atomic"] = H
    out["H_norm_atomic5"] = H_norm

    # Write CSV next to input
    base = os.path.splitext(os.path.basename(INPUT_CSV))[0]
    out_dir = os.path.dirname(INPUT_CSV)
    out_path = os.path.join(out_dir, f"{base}__atomic5_entropy.csv")
    out.to_csv(out_path, index=False, encoding="utf-8", float_format="%.6f")

    print("Wrote:", out_path)
    print("Columns:", list(out.columns))

if __name__ == "__main__":
    main()
