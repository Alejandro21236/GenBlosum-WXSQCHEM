#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Codon-aware neutral model + domain-level analysis for TP53 and PIK3CA in BRCA.

- For each gene (TP53, PIK3CA):
  * Load canonical CDS (coding DNA sequence) – pasted below as strings
  * Load wild-type AA sequence from FASTA
  * Extract simple missense mutations from mutant_sequences_by_id CSVs
  * Build codon-aware neutral AA-substitution model (single-nt changes only)
  * Weight base substitutions using a simple mutational signature model
  * Compare observed vs neutral BLOSUM62:
      - global distribution (hist)
      - mean BLOSUM (with Monte Carlo)
      - radical fraction (BLOSUM <= 0)
  * Domain-level analysis:
      - observed vs neutral domain mutation fractions
      - observed vs neutral mean BLOSUM per domain

Outputs:
  out/fig4/fig4A_{gene}_blosum_hist.png
  out/fig4/fig4B_{gene}_mean_blosum.png
  out/fig4/fig4C_{gene}_radical_fraction.png
  out/fig4/summary_stats_fig4_{gene}.csv
  out/fig4/domain_stats_{gene}.csv
"""

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# CONFIG
# ------------------------------

BASE_DIR = "/fs/scratch/PAS2942/mutations/BRCA"
MUT_CSV_DIR = os.path.join(BASE_DIR, "out", "mutant_sequences_by_id")
FIG_DIR = os.path.join(BASE_DIR, "out", "fig4")

N_SIMULATIONS = 5000
N_NEUTRAL_SCORES = 50000
RANDOM_SEED = 2025

# Paste canonical CDS for each gene (CDS only, no UTR, no whitespace)
TP53_CDS = "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAGACCTATGGAAACTACTTCCTGAAAACAACGTTCTGTCCCCCTTGCCGTCCCAAGCAATGGATGATTTGATGCTGTCCCCGGACGATATTGAACAATGGTTCACTGAAGACCCAGGTCCAGATGAAGCTCCCAGAATGCCAGAGGCTGCTCCCCCCGTGGCCCCTGCACCAGCAGCTCCTACACCGGCGGCCCCTGCACCAGCCCCCTCCTGGCCCCTGTCATCTTCTGTCCCTTCCCAGAAAACCTACCAGGGCAGCTACGGTTTCCGTCTGGGCTTCTTGCATTCTGGGACAGCCAAGTCTGTGACTTGCACGTACTCCCCTGCCCTCAACAAGATGTTTTGCCAACTGGCCAAGACCTGCCCTGTGCAGCTGTGGGTTGATTCCACACCCCCGCCCGGCACCCGCGTCCGCGCCATGGCCATCTACAAGCAGTCACAGCACATGACGGAGGTTGTGAGGCGCTGCCCCCACCATGAGCGCTGCTCAGATAGCGATGGTCTGGCCCCTCCTCAGCATCTTATCCGAGTGGAAGGAAATTTGCGTGTGGAGTATTTGGATGACAGAAACACTTTTCGACATAGTGTGGTGGTGCCCTATGAGCCGCCTGAGGTTGGCTCTGACTGTACCACCATCCACTACAACTACATGTGTAACAGTTCCTGCATGGGCGGCATGAACCGGAGGCCCATCCTCACCATCATCACACTGGAAGACTCCAGTGGTAATCTACTGGGACGGAACAGCTTTGAGGTGCGTGTTTGTGCCTGTCCTGGGAGAGACCGGCGCACAGAGGAAGAGAATCTCCGCAAGAAAGGGGAGCCTCACCACGAGCTGCCCCCAGGGAGCACTAAGCGAGCACTGCCCAACAACACCAGCTCCTCTCCCCAGCCAAAGAAGAAACCACTGGATGGAGAATATTTCACCCTTCAGATCCGTGGGCGTGAGCGCTTCGAGATGTTCCGAGAGCTGAATGAGGCCTTGGAACTCAAGGATGCCCAGGCTGGGAAGGAGCCAGGGGGGAGCAGGGCTCACTCCAGCCACCTGAAGTCCAAAAAGGGTCAGTCTACCTCCCGCCATAAAAAACTCATGTTCAAGACAGAAGGGCCTGACTCAGACTGA"
PIK3CA_CDS = "ATGCCTCCACGACCATCATCAGGTGAACTGTGGGGCATCCACTTGATGCCCCCAAGAATCCTAGTAGAATGTTTACTACCAAATGGAATGATAGTGACTTTAGAATGCCTCCGTGAGGCTACATTAATAACCATAAAGCATGAACTATTTAAAGAAGCAAGAAAATACCCCCTCCATCAACTTCTTCAAGATGAATCTTCTTACATTTTCGTAAGTGTTACTCAAGAAGCAGAAAGGGAAGAATTTTTTGATGAAACAAGACGACTTTGTGACCTTCGGCTTTTTCAACCCTTTTTAAAAGTAATTGAACCAGTAGGCAACCGTGAAGAAAAGATCCTCAATCGAGAAATTGGTTTTGCTATCGGCATGCCAGTGTGTGAATTTGATATGGTTAAAGATCCAGAAGTACAGGACTTCCGAAGAAATATTCTGAACGTTTGTAAAGAAGCTGTGGATCTTAGGGACCTCAATTCACCTCATAGTAGAGCAATGTATGTCTATCCTCCAAATGTAGAATCTTCACCAGAATTGCCAAAGCACATATATAATAAATTAGATAAAGGGCAAATAATAGTGGTGATCTGGGTAATAGTTTCTCCAAATAATGACAAGCAGAAGTATACTCTGAAAATCAACCATGACTGTGTACCAGAACAAGTAATTGCTGAAGCAATCAGGAAAAAAACTCGAAGTATGTTGCTATCCTCTGAACAACTAAAACTCTGTGTTTTAGAATATCAGGGCAAGTATATTTTAAAAGTGTGTGGATGTGATGAATACTTCCTAGAAAAATATCCTCTGAGTCAGTATAAGTATATAAGAAGCTGTATAATGCTTGGGAGGATGCCCAATTTGATGTTGATGGCTAAAGAAAGCCTTTATTCTCAACTGCCAATGGACTGTTTTACAATGCCATCTTATTCCAGACGCATTTCCACAGCTACACCATATATGAATGGAGAAACATCTACAAAATCCCTTTGGGTTATAAATAGTGCACTCAGAATAAAAATTCTTTGTGCAACCTACGTGAATGTAAATATTCGAGACATTGATAAGATCTATGTTCGAACAGGTATCTACCATGGAGGAGAACCCTTATGTGACAATGTGAACACTCAAAGAGTACCTTGTTCCAATCCCAGGTGGAATGAATGGCTGAATTATGATATATACATTCCTGATCTTCCTCGTGCTGCTCGACTTTGCCTTTCCATTTGCTCTGTTAAAGGCCGAAAGGGTGCTAAAGAGGAACACTGTCCATTGGCATGGGGAAATATAAACTTGTTTGATTACACAGACACTCTAGTATCTGGAAAAATGGCTTTGAATCTTTGGCCAGTACCTCATGGATTAGAAGATTTGCTGAACCCTATTGGTGTTACTGGATCAAATCCAAATAAAGAAACTCCATGCTTAGAGTTGGAGTTTGACTGGTTCAGCAGTGTGGTAAAGTTCCCAGATATGTCAGTGATTGAAGAGCATGCCAATTGGTCTGTATCCCGAGAAGCAGGATTTAGCTATTCCCACGCAGGACTGAGTAACAGACTAGCTAGAGACAATGAATTAAGGGAAAATGACAAAGAACAGCTCAAAGCAATTTCTACACGAGATCCTCTCTCTGAAATCACTGAGCAGGAGAAAGATTTTCTATGGAGTCACAGACACTATTGTGTAACTATCCCCGAAATTCTACCCAAATTGCTTCTGTCTGTTAAATGGAATTCTAGAGATGAAGTAGCCCAGATGTATTGCTTGGTAAAAGATTGGCCTCCAATCAAACCTGAACAGGCTATGGAACTTCTGGACTGTAATTACCCAGATCCTATGGTTCGAGGTTTTGCTGTTCGGTGCTTGGAAAAATATTTAACAGATGACAAACTTTCTCAGTATTTAATTCAGCTAGTACAGGTCCTAAAATATGAACAATATTTGGATAACTTGCTTGTGAGATTTTTACTGAAGAAAGCATTGACTAATCAAAGGATTGGGCACTTTTTCTTTTGGCATTTAAAATCTGAGATGCACAATAAAACAGTTAGCCAGAGGTTTGGCCTGCTTTTGGAGTCCTATTGTCGTGCATGTGGGATGTATTTGAAGCACCTGAATAGGCAAGTCGAGGCAATGGAAAAGCTCATTAACTTAACTGACATTCTCAAACAGGAGAAGAAGGATGAAACACAAAAGGTACAGATGAAGTTTTTAGTTGAGCAAATGAGGCGACCAGATTTCATGGATGCTCTACAGGGCTTTCTGTCTCCTCTAAACCCTGCTCATCAACTAGGAAACCTCAGGCTTGAAGAGTGTCGAATTATGTCCTCTGCAAAAAGGCCACTGTGGTTGAATTGGGAGAACCCAGACATCATGTCAGAGTTACTGTTTCAGAACAATGAGATCATCTTTAAAAATGGGGATGATTTACGGCAAGATATGCTAACACTTCAAATTATTCGTATTATGGAAAATATCTGGCAAAATCAAGGTCTTGATCTTCGAATGTTACCTTATGGTTGTCTGTCAATCGGTGACTGTGTGGGACTTATTGAGGTGGTGCGAAATTCTCACACTATTATGCAAATTCAGTGCAAAGGCGGCTTGAAAGGTGCACTGCAGTTCAACAGCCACACACTACATCAGTGGCTCAAAGACAAGAACAAAGGAGAAATATATGATGCAGCCATTGACCTGTTTACACGTTCATGTGCTGGATACTGTGTAGCTACCTTCATTTTGGGAATTGGAGATCGTCACAATAGTAACATCATGGTGAAAGACGATGGACAACTGTTTCATATAGATTTTGGACACTTTTTGGATCACAAGAAGAAAAAATTTGGTTATAAACGAGAACGTGTGCCATTTGTTTTGACACAGGATTTCTTAATAGTGATTAGTAAAGGAGCCCAAGAATGCACAAAGACAAGAGAATTTGAGAGGTTTCAGGAGATGTGTTACAAGGCTTATCTAGCTATTCGACAGCATGCCAATCTCTTCATAAATCTTTTCTCAATGATGCTTGGCTCTGGAATGCCAGAACTACAATCTTTTGATGACATTGCATACATTCGAAAGACCCTAGCCTTAGATAAAACTGAGCAAGAGGCTTTGGAGTATTTCATGAAACAAATGAATGATGCACATCATGGTGGCTGGACAACAAAAATGGATTGGATCTTCCACACAATTAAACAGCATGCATTGAACTGA"

# Paths to wild-type protein FASTAs
GENE_CONFIG = {
    "TP53": {
        "cds": TP53_CDS,
        "aa_fasta": os.path.join(BASE_DIR, "TP53.fasta"),
        # Domain annotations (1-based inclusive)
        # Based on canonical p53 structure :contentReference[oaicite:0]{index=0}
        "domains": [
            ("TAD", 1, 61),
            ("PRD", 64, 92),
            ("DBD", 102, 292),
            ("OD", 307, 355),
            ("CTD", 356, 393),
        ],
    },
    "PIK3CA": {
        "cds": PIK3CA_CDS,
        "aa_fasta": os.path.join(BASE_DIR, "PIK3CA.fasta"),
        # p110α domains: ABD, RBD, C2, helical, kinase :contentReference[oaicite:1]{index=1}
        "domains": [
            ("ABD", 16, 105),
            ("RBD", 187, 289),
            ("C2", 330, 487),
            ("Helical", 517, 694),
            ("Kinase", 696, 1068),
        ],
    },
}

# Optional: context-free mutational signature weights for base changes.
# Keys: "A>C", "A>G", ..., "T>G". Default = 1.0 (uniform) – edit if you want BRCA-specific spectrum.
BASES = ["A", "C", "G", "T"]
SIGNATURE_WEIGHTS = {
    "C>T": 9.0,
    "C>G": 5.0,
    "C>A": 3.0,
    "T>C": 2.0,
    "T>G": 1.5,
    "T>A": 1.0,
    "A>G": 1.0,
    "A>C": 1.0,
    "A>T": 1.0,
    "G>A": 1.0,
    "G>T": 1.0,
    "G>C": 1.0,
}


# ------------------------------
# BLOSUM62
# ------------------------------

AA_ORDER = [
    "A", "R", "N", "D", "C",
    "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P",
    "S", "T", "W", "Y", "V",
]

_BLOSUM62_ROWS = [
    # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
    [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],  # A
    [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
    [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],  # N
    [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],  # D
    [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
    [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],  # Q
    [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],  # E
    [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],  # G
    [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],  # H
    [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],  # I
    [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],  # L
    [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],  # K
    [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],  # M
    [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],  # F
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],  # P
    [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],  # S
    [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],  # T
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],  # W
    [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],  # Y
    [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4],  # V
]

BLOSUM62 = {
    aa1: {aa2: _BLOSUM62_ROWS[i][j] for j, aa2 in enumerate(AA_ORDER)}
    for i, aa1 in enumerate(AA_ORDER)
}


def blosum_score(a1: str, a2: str) -> float:
    a1 = a1.upper()
    a2 = a2.upper()
    return BLOSUM62[a1][a2]


# ------------------------------
# GENETIC CODE
# ------------------------------

GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",

    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",

    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",

    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

def translate_codon(c: str) -> str:
    return GENETIC_CODE.get(c.upper(), "X")


# ------------------------------
# UTILITIES
# ------------------------------

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def read_single_fasta(path: str) -> str:
    seq_lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_lines.append(line)
    return "".join(seq_lines).strip()

def load_mutation_table(csv_dir: str) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csv_files:
        raise RuntimeError(f"No CSVs in {csv_dir}")
    dfs = []
    for p in csv_files:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
    if not dfs:
        raise RuntimeError("No readable CSVs")
    return pd.concat(dfs, ignore_index=True)

def extract_simple_missense_for_gene(df: pd.DataFrame, gene: str) -> pd.DataFrame:
    gene = gene.upper()
    records = []
    n_total = len(df)
    n_other = 0
    n_len_mismatch = 0
    n_multi = 0

    for _, row in df.iterrows():
        g = str(row.get("gene", "")).strip().upper()
        if g != gene:
            n_other += 1
            continue
        wt_seq = str(row["wt_seq"]).strip()
        mut_seq = str(row["mut_seq"]).strip()
        wt_len = int(row["wt_len"])
        mut_len = int(row["mut_len"])

        if wt_len != mut_len or len(wt_seq) != len(mut_seq):
            n_len_mismatch += 1
            continue

        diffs = []
        for i, (w, m) in enumerate(zip(wt_seq, mut_seq)):
            if w != m:
                diffs.append((i, w, m))
        if len(diffs) != 1:
            n_multi += 1
            continue

        idx, wt_aa, mut_aa = diffs[0]
        if wt_aa not in AA_ORDER or mut_aa not in AA_ORDER:
            continue
        score = blosum_score(wt_aa, mut_aa)
        records.append(
            {
                "sample_id": row["sample_id"],
                "pos": idx + 1,
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "blosum": score,
            }
        )

    df_out = pd.DataFrame.from_records(records)
    print(f"[{gene}] Total rows: {n_total}")
    print(f"[{gene}] Non-{gene} skipped: {n_other}")
    print(f"[{gene}] Len-mismatch skipped: {n_len_mismatch}")
    print(f"[{gene}] Multi-diff skipped: {n_multi}")
    print(f"[{gene}] Simple missense kept: {len(df_out)}")
    if df_out.empty:
        raise RuntimeError(f"No simple missense mutations for {gene}")
    return df_out

def check_cds(name: str, cds: str):
    cds = cds.strip().upper()
    if not cds:
        raise RuntimeError(f"{name} CDS is empty – paste canonical CDS.")
    if any(b not in BASES for b in cds):
        raise RuntimeError(f"{name} CDS contains non-ACGT characters.")
    if len(cds) % 3 != 0:
        raise RuntimeError(f"{name} CDS length {len(cds)} not divisible by 3.")
    return cds

def split_codons(cds: str):
    return [cds[i:i+3] for i in range(0, len(cds), 3)]

def build_base_substitution_model():
    model = {}
    for b1 in BASES:
        others = [b2 for b2 in BASES if b2 != b1]
        for b2 in others:
            w = SIGNATURE_WEIGHTS.get(f"{b1}>{b2}", 1.0)
            model[(b1, b2)] = float(w)
    return model

def domain_label(domains, pos: int) -> str:
    for name, start, end in domains:
        if start <= pos <= end:
            return name
    return "Other"


# ------------------------------
# CODON-AWARE NEUTRAL MODEL
# ------------------------------

def build_codon_neutral_pairs(cds: str, domains, base_model):
    codons = split_codons(cds)
    pairs = []
    weights = []
    pair_domains = []

    for codon_idx, codon in enumerate(codons):
        wt_aa = translate_codon(codon)
        if wt_aa == "*" or wt_aa == "X":
            continue
        pos = codon_idx + 1
        dom = domain_label(domains, pos)

        for i in range(3):
            b = codon[i]
            for b_new in BASES:
                if b_new == b:
                    continue
                new_codon_list = list(codon)
                new_codon_list[i] = b_new
                new_codon = "".join(new_codon_list)
                mut_aa = translate_codon(new_codon)
                if mut_aa == wt_aa:
                    continue
                if mut_aa == "*" or mut_aa == "X":
                    continue
                w = base_model[(b, b_new)]
                pairs.append((wt_aa, mut_aa))
                weights.append(w)
                pair_domains.append(dom)

    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    pair_domains = np.array(pair_domains, dtype=object)
    return pairs, weights, pair_domains

def simulate_neutral_sets(pairs, probs, n_mutations: int, n_sim: int, rng):
    n_pairs = len(pairs)
    idx_all = np.arange(n_pairs)
    sim_means = np.empty(n_sim, dtype=float)
    sim_rad = np.empty(n_sim, dtype=float)

    for s in range(n_sim):
        idx = rng.choice(idx_all, size=n_mutations, p=probs, replace=True)
        scores = np.empty(n_mutations, dtype=float)
        for k, j in enumerate(idx):
            a1, a2 = pairs[j]
            scores[k] = blosum_score(a1, a2)
        sim_means[s] = scores.mean()
        sim_rad[s] = np.mean(scores <= 0.0)
    return sim_means, sim_rad

def sample_neutral_scores(pairs, probs, n_scores: int, rng):
    n_pairs = len(pairs)
    idx_all = np.arange(n_pairs)
    idx = rng.choice(idx_all, size=n_scores, p=probs, replace=True)
    scores = np.empty(n_scores, dtype=float)
    for k, j in enumerate(idx):
        a1, a2 = pairs[j]
        scores[k] = blosum_score(a1, a2)
    return scores


# ------------------------------
# PLOTTING
# ------------------------------

def plot_hist(gene, observed, neutral_scores, out_path):
    plt.figure(figsize=(7, 5))
    lo = min(observed.min(), neutral_scores.min()) - 1
    hi = max(observed.max(), neutral_scores.max()) + 1
    bins = np.arange(lo, hi + 1)
    plt.hist(neutral_scores, bins=bins, alpha=0.6, density=True, label="Neutral (codon-aware)")
    plt.hist(observed, bins=bins, alpha=0.6, density=True, label="Observed")
    plt.xlabel("BLOSUM62 score")
    plt.ylabel("Density")
    plt.title(f"{gene}: BLOSUM62 observed vs codon-aware neutral")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_mean_box(gene, obs_mean, sim_means, out_path):
    plt.figure(figsize=(5, 5))
    plt.boxplot([sim_means], positions=[0], widths=0.5, showfliers=False)
    plt.scatter([0], [obs_mean], s=60, zorder=3, label="Observed mean")
    plt.xticks([0], ["Neutral (codon model)"])
    plt.ylabel("Mean BLOSUM62")
    plt.title(f"{gene}: mean BLOSUM62 vs neutral")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_radical_bar(gene, obs_frac, sim_frac, out_path):
    plt.figure(figsize=(6, 5))
    mean_sim = sim_frac.mean()
    lower = np.percentile(sim_frac, 2.5)
    upper = np.percentile(sim_frac, 97.5)
    x = np.arange(2)
    heights = [mean_sim, obs_frac]
    labels = ["Neutral (mean)", "Observed"]
    plt.bar(x, heights, width=0.6)
    plt.errorbar(
        x[0],
        mean_sim,
        yerr=[[mean_sim - lower], [upper - mean_sim]],
        fmt="none",
        capsize=5,
        linewidth=1.2,
    )
    plt.xticks(x, labels)
    plt.ylabel("Fraction with BLOSUM ≤ 0")
    plt.title(f"{gene}: radical substitution fraction")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ------------------------------
# DOMAIN-LEVEL NEUTRAL SUMMARY
# ------------------------------

def compute_domain_neutral_stats(pairs, weights, pair_domains):
    # total neutral probability per domain + mean BLOSUM per domain
    df = pd.DataFrame({
        "wt_aa": [p[0] for p in pairs],
        "mut_aa": [p[1] for p in pairs],
        "domain": pair_domains,
        "w": weights,
    })
    df["blosum"] = df.apply(lambda r: blosum_score(r["wt_aa"], r["mut_aa"]), axis=1)

    domain_stats = []
    for dom, sub in df.groupby("domain"):
        w_tot = sub["w"].sum()
        frac = w_tot  # since weights sum to 1 overall
        if w_tot > 0:
            mean_b = (sub["w"] * sub["blosum"]).sum() / w_tot
        else:
            mean_b = np.nan
        domain_stats.append(
            {
                "domain": dom,
                "neutral_frac": frac,
                "neutral_mean_blosum": mean_b,
            }
        )
    return pd.DataFrame(domain_stats)


# ------------------------------
# MAIN PER-GENE ANALYSIS
# ------------------------------

def run_for_gene(gene: str, df_all: pd.DataFrame, rng):
    cfg = GENE_CONFIG[gene]
    cds = check_cds(gene, cfg["cds"])
    aa_seq = read_single_fasta(cfg["aa_fasta"])
    print(f"[{gene}] CDS length: {len(cds)} (codons: {len(cds)//3})")
    print(f"[{gene}] WT AA length from FASTA: {len(aa_seq)}")

    # Observed
    df_obs = extract_simple_missense_for_gene(df_all, gene)
    obs_scores = df_obs["blosum"].to_numpy()
    n_mut = len(obs_scores)
    obs_mean = obs_scores.mean()
    obs_rad = np.mean(obs_scores <= 0.0)
    print(f"[{gene}] Observed missense count: {n_mut}")
    print(f"[{gene}] Observed mean BLOSUM: {obs_mean:.3f}")
    print(f"[{gene}] Observed radical fraction (≤0): {obs_rad:.3f}")

    # Add domain labels to observed
    doms = cfg["domains"]
    df_obs["domain"] = df_obs["pos"].apply(lambda p: domain_label(doms, p))

    # Neutral
    base_model = build_base_substitution_model()
    pairs, probs, pair_domains = build_codon_neutral_pairs(cds, doms, base_model)
    print(f"[{gene}] Codon-aware neutral AA pairs: {len(pairs)}")

    sim_means, sim_rad = simulate_neutral_sets(pairs, probs, n_mut, N_SIMULATIONS, rng)
    neutral_scores = sample_neutral_scores(pairs, probs, N_NEUTRAL_SCORES, rng)

    mean_sim_mean = sim_means.mean()
    mean_sim_sd = sim_means.std(ddof=1)
    mean_rad_mean = sim_rad.mean()
    mean_rad_sd = sim_rad.std(ddof=1)

    p_mean = np.mean(sim_means <= obs_mean)
    p_rad = np.mean(sim_rad >= obs_rad)

    print(f"[{gene}] Neutral mean BLOSUM (sim): {mean_sim_mean:.3f} ± {mean_sim_sd:.3f}")
    print(f"[{gene}] Neutral radical frac (sim): {mean_rad_mean:.3f} ± {mean_rad_sd:.3f}")
    print(f"[{gene}] p_mean (one-sided, more radical): {p_mean:.4g}")
    print(f"[{gene}] p_rad (one-sided, more radical): {p_rad:.4g}")

    # Figures
    fig_hist = os.path.join(FIG_DIR, f"fig4A_{gene}_blosum_hist.png")
    fig_mean = os.path.join(FIG_DIR, f"fig4B_{gene}_mean_blosum.png")
    fig_rad = os.path.join(FIG_DIR, f"fig4C_{gene}_radical_fraction.png")
    plot_hist(gene, obs_scores, neutral_scores, fig_hist)
    plot_mean_box(gene, obs_mean, sim_means, fig_mean)
    plot_radical_bar(gene, obs_rad, sim_rad, fig_rad)

    # Global summary CSV
    summary_path = os.path.join(FIG_DIR, f"summary_stats_fig4_{gene}.csv")
    summary = {
        "gene": gene,
        "n_observed_missense": n_mut,
        "observed_mean_blosum": obs_mean,
        "neutral_mean_blosum_mean": mean_sim_mean,
        "neutral_mean_blosum_sd": mean_sim_sd,
        "p_value_mean_blosum_one_sided_more_radical": p_mean,
        "observed_radical_fraction": obs_rad,
        "neutral_radical_fraction_mean": mean_rad_mean,
        "neutral_radical_fraction_sd": mean_rad_sd,
        "p_value_radical_fraction_one_sided_more_radical": p_rad,
        "n_simulations": N_SIMULATIONS,
        "n_neutral_scores_hist": N_NEUTRAL_SCORES,
        "random_seed": RANDOM_SEED,
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"[{gene}] Wrote summary to {summary_path}")

    # Domain-level neutral expectations
    df_dom_neutral = compute_domain_neutral_stats(pairs, probs, pair_domains)

    # Domain-level observed stats
    dom_obs = []
    total_obs = len(df_obs)
    for dom, sub in df_obs.groupby("domain"):
        n = len(sub)
        frac = n / total_obs
        mean_b = sub["blosum"].mean()
        dom_obs.append({"domain": dom, "observed_n": n, "observed_frac": frac, "observed_mean_blosum": mean_b})
    df_dom_obs = pd.DataFrame(dom_obs)

    df_dom = pd.merge(df_dom_neutral, df_dom_obs, on="domain", how="outer").fillna(0.0)
    dom_path = os.path.join(FIG_DIR, f"domain_stats_{gene}.csv")
    df_dom.to_csv(dom_path, index=False)
    print(f"[{gene}] Wrote domain stats to {dom_path}")

    return summary_path, dom_path


# ------------------------------
# MAIN
# ------------------------------

def main():
    ensure_dir(FIG_DIR)
    rng = np.random.default_rng(RANDOM_SEED)
    df_all = load_mutation_table(MUT_CSV_DIR)

    for gene in ["TP53", "PIK3CA"]:
        print(f"\n===== {gene} =====")
        run_for_gene(gene, df_all, rng)

    print("[INFO] Multi-gene codon-aware analysis finished.")


if __name__ == "__main__":
    main()
