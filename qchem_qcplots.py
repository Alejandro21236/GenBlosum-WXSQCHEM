#!/usr/bin/env python

import os
import json
import math
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: PCA. If sklearn isn't installed, PCA plots will be skipped.
try:
    from sklearn.decomposition import PCA
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


# =========================
# CONFIG
# =========================
CSV_PATH = "qchem_all_summary_indestructible.csv"
FIG_DIR = "qchem_figs"
os.makedirs(FIG_DIR, exist_ok=True)


# =========================
# HELPERS
# =========================

def parse_vector(cell):
    """Convert JSON/list-like string to Python list of floats."""
    if pd.isna(cell):
        return None
    if isinstance(cell, (list, np.ndarray)):
        return list(cell)
    s = str(cell).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        # handle JSON-like list
        return [float(x) for x in json.loads(s)]
    except Exception:
        try:
            return [float(x) for x in ast.literal_eval(s)]
        except Exception:
            return None


def stack_vectors(series):
    """
    Turn a Series of vector-like cells into a 2D array (n_samples x n_atoms).
    Truncates all vectors to the minimum length.
    Returns matrix and list of index labels.
    """
    vecs = []
    labels = []
    lengths = []

    for idx, cell in series.items():
        v = parse_vector(cell)
        if v is None:
            continue
        vecs.append(v)
        labels.append(idx)
        lengths.append(len(v))

    if not vecs:
        return None, []

    L = min(lengths)
    mat = np.array([v[:L] for v in vecs], dtype=float)
    return mat, labels


def find_wildtype_index(df):
    """
    Try to detect a WT row by job_id containing 'WT', 'wt', or 'wild'.
    Returns index label or None.
    """
    for idx, job in df["job_id"].items():
        s = str(job).lower()
        if "wt" in s or "wild" in s:
            return idx
    return None


def get_wildtype_vector(series, wt_idx):
    if wt_idx is None or wt_idx not in series.index:
        return None
    return parse_vector(series.loc[wt_idx])


def l1_delta_matrix(mat, ref_vec):
    """Compute L1 difference for each row vs ref_vec."""
    ref = np.array(ref_vec, dtype=float)
    L = min(mat.shape[1], ref.shape[0])
    return np.sum(np.abs(mat[:, :L] - ref[:L]), axis=1)


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")
print("Columns:", df.columns.tolist())

# Use index aligned with rows
df.index = np.arange(len(df))


# =========================
# 1. NPA CHARGES: VECTOR PLOTS & HEATMAPS
# =========================

if "NPA_charges" in df.columns:
    npa_mat, npa_idx = stack_vectors(df["NPA_charges"])
    if npa_mat is not None:
        mutants = df.loc[npa_idx, "job_id"].tolist()
        n_atoms = npa_mat.shape[1]

        # (A) line plots for first few mutants
        n_show = min(5, npa_mat.shape[0])
        plt.figure(figsize=(8, 5))
        x = np.arange(n_atoms)
        for i in range(n_show):
            plt.plot(x, npa_mat[i], label=mutants[i][:20])
        plt.xlabel("Atom index")
        plt.ylabel("NPA charge (e)")
        plt.title("NPA atomic charge distribution (first few mutants)")
        plt.legend(fontsize=7, loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "npa_lineplots_first_mutants.png"), dpi=300)
        plt.close()

        # (B) heatmap of NPA charges
        plt.figure(figsize=(8, 6))
        plt.imshow(npa_mat, aspect="auto", cmap="bwr", interpolation="nearest")
        plt.colorbar(label="NPA charge (e)")
        plt.xlabel("Atom index")
        plt.ylabel("Mutant index")
        plt.title("NPA atomic charge heatmap across mutants")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "npa_heatmap.png"), dpi=300)
        plt.close()

        # (C) WT-based delta heatmap if we can find WT
        wt_idx = find_wildtype_index(df)
        if wt_idx is not None:
            wt_vec = get_wildtype_vector(df["NPA_charges"], wt_idx)
            if wt_vec is not None:
                wt_mat = np.tile(wt_vec[:n_atoms], (npa_mat.shape[0], 1))
                delta = npa_mat - wt_mat
                plt.figure(figsize=(8, 6))
                plt.imshow(delta, aspect="auto", cmap="bwr", interpolation="nearest")
                plt.colorbar(label="ΔNPA charge vs WT (e)")
                plt.xlabel("Atom index")
                plt.ylabel("Mutant index")
                plt.title("ΔNPA charge heatmap vs WT")
                plt.tight_layout()
                plt.savefig(os.path.join(FIG_DIR, "npa_delta_vs_wt_heatmap.png"), dpi=300)
                plt.close()

                # Save L1 norm vs mutant for later correlation
                l1 = l1_delta_matrix(npa_mat, wt_vec)
                df.loc[npa_idx, "NPA_L1_delta_vs_WT"] = l1

        # PCA on NPA if sklearn available
        if HAVE_SKLEARN and npa_mat.shape[0] > 2:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(npa_mat)
            plt.figure(figsize=(5, 5))
            plt.scatter(coords[:, 0], coords[:, 1], s=15)
            for i, name in enumerate(mutants):
                if i < 30:  # don't annotate everything
                    plt.text(coords[i, 0], coords[i, 1], str(i), fontsize=6)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("PCA of NPA charge vectors")
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, "npa_pca.png"), dpi=300)
            plt.close()

else:
    print("No NPA_charges column found; skipping NPA vector plots.")


# =========================
# 2. ESP (CHELPG) CHARGES: VECTOR PLOTS & HEATMAPS
# =========================

if "CHELPG_charges" in df.columns:
    esp_mat, esp_idx = stack_vectors(df["CHELPG_charges"])
    if esp_mat is not None:
        mutants_esp = df.loc[esp_idx, "job_id"].tolist()
        n_atoms_e = esp_mat.shape[1]

        # Line plots for first few mutants
        n_show = min(5, esp_mat.shape[0])
        plt.figure(figsize=(8, 5))
        x = np.arange(n_atoms_e)
        for i in range(n_show):
            plt.plot(x, esp_mat[i], label=mutants_esp[i][:20])
        plt.xlabel("Atom index")
        plt.ylabel("ESP (ChElPG) charge (e)")
        plt.title("ESP atomic charge distribution (first few mutants)")
        plt.legend(fontsize=7, loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "esp_lineplots_first_mutants.png"), dpi=300)
        plt.close()

        # Heatmap across mutants
        plt.figure(figsize=(8, 6))
        plt.imshow(esp_mat, aspect="auto", cmap="bwr", interpolation="nearest")
        plt.colorbar(label="ESP (ChElPG) charge (e)")
        plt.xlabel("Atom index")
        plt.ylabel("Mutant index")
        plt.title("ESP (ChElPG) atomic charge heatmap across mutants")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "esp_heatmap.png"), dpi=300)
        plt.close()

        # WT-based delta heatmap if WT present
        wt_idx2 = find_wildtype_index(df)
        if wt_idx2 is not None:
            wt_vec2 = get_wildtype_vector(df["CHELPG_charges"], wt_idx2)
            if wt_vec2 is not None:
                wt_mat2 = np.tile(wt_vec2[:n_atoms_e], (esp_mat.shape[0], 1))
                delta2 = esp_mat - wt_mat2
                plt.figure(figsize=(8, 6))
                plt.imshow(delta2, aspect="auto", cmap="bwr", interpolation="nearest")
                plt.colorbar(label="ΔESP charge vs WT (e)")
                plt.xlabel("Atom index")
                plt.ylabel("Mutant index")
                plt.title("ΔESP charge heatmap vs WT")
                plt.tight_layout()
                plt.savefig(os.path.join(FIG_DIR, "esp_delta_vs_wt_heatmap.png"), dpi=300)
                plt.close()

                l1e = l1_delta_matrix(esp_mat, wt_vec2)
                df.loc[esp_idx, "ESP_L1_delta_vs_WT"] = l1e

        # PCA on ESP if sklearn available
        if HAVE_SKLEARN and esp_mat.shape[0] > 2:
            pca2 = PCA(n_components=2)
            coords2 = pca2.fit_transform(esp_mat)
            plt.figure(figsize=(5, 5))
            plt.scatter(coords2[:, 0], coords2[:, 1], s=15)
            for i, name in enumerate(mutants_esp):
                if i < 30:
                    plt.text(coords2[i, 0], coords2[i, 1], str(i), fontsize=6)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("PCA of ESP (ChElPG) charge vectors")
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, "esp_pca.png"), dpi=300)
            plt.close()

else:
    print("No CHELPG_charges column found; skipping ESP vector plots.")


# =========================
# 3. BOX/HIST PLOTS FOR TOTAL CHARGES
# =========================

if "NPA_sum" in df.columns:
    plt.figure(figsize=(5, 4))
    plt.hist(df["NPA_sum"].dropna(), bins=20)
    plt.xlabel("Total NPA charge (sum over atoms)")
    plt.ylabel("Number of mutants")
    plt.title("Distribution of total NPA charge")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "npa_sum_hist.png"), dpi=300)
    plt.close()

if "CHELPG_sum" in df.columns:
    plt.figure(figsize=(5, 4))
    plt.hist(df["CHELPG_sum"].dropna(), bins=20)
    plt.xlabel("Total ESP (ChElPG) charge (sum over atoms)")
    plt.ylabel("Number of mutants")
    plt.title("Distribution of total ESP (ChElPG) charge")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "esp_sum_hist.png"), dpi=300)
    plt.close()


# =========================
# 4. HOMO / LUMO / GAP PLOTS
# =========================

if "HOMO_eV" in df.columns and "LUMO_eV" in df.columns and "gap_eV" in df.columns:
    # Scatter of HOMO/LUMO vs mutant index
    idx = np.arange(len(df))
    plt.figure(figsize=(7, 4))
    plt.scatter(idx, df["HOMO_eV"], s=10, label="HOMO")
    plt.scatter(idx, df["LUMO_eV"], s=10, label="LUMO")
    plt.xlabel("Mutant index")
    plt.ylabel("Energy (eV)")
    plt.title("HOMO and LUMO energies across mutants")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "homo_lumo_scatter.png"), dpi=300)
    plt.close()

    # Histogram of gap
    plt.figure(figsize=(5, 4))
    plt.hist(df["gap_eV"].dropna(), bins=20)
    plt.xlabel("HOMO-LUMO gap (eV)")
    plt.ylabel("Number of mutants")
    plt.title("Distribution of HOMO-LUMO gaps")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "gap_hist.png"), dpi=300)
    plt.close()
else:
    print("HOMO/LUMO columns not found; skipping orbital energy plots.")


# =========================
# 5. CORRELATION: ΔCHARGE VS GAP
# =========================

if "gap_eV" in df.columns:
    # Prefer NPA_L1_delta_vs_WT if present; fall back to ESP
    corr_data = None
    label = None

    if "NPA_L1_delta_vs_WT" in df.columns:
        corr_data = df["NPA_L1_delta_vs_WT"]
        label = "NPA L1 Δcharge vs WT"
    elif "ESP_L1_delta_vs_WT" in df.columns:
        corr_data = df["ESP_L1_delta_vs_WT"]
        label = "ESP L1 Δcharge vs WT"

    if corr_data is not None:
        valid = (~df["gap_eV"].isna()) & (~corr_data.isna())
        x = df.loc[valid, "gap_eV"]
        y = corr_data.loc[valid]
        plt.figure(figsize=(5, 4))
        plt.scatter(x, y, s=15)
        plt.xlabel("HOMO-LUMO gap (eV)")
        plt.ylabel(label)
        plt.title("Correlation between electronic gap and Δcharge vs WT")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "gap_vs_delta_charge.png"), dpi=300)
        plt.close()


# =========================
# 6. SAVE AUGMENTED TABLE
# =========================

df.to_csv(os.path.join(FIG_DIR, "qchem_summary_with_deltas.csv"), index=False)
print(f"Plots written to: {FIG_DIR}")
print("Augmented summary written to qchem_summary_with_deltas.csv in that folder.")
