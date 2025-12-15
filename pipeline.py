#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import pandas as pd
from Bio import SeqIO
import subprocess
import re
from typing import Dict, Tuple, Optional

# ------------------------------
# Utilities
# ------------------------------

AA3_TO_1 = {
    "Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G",
    "His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S",
    "Thr":"T","Trp":"W","Tyr":"Y","Val":"V","Sec":"U","Pyl":"O"
}

def load_protein_fasta(protein_fasta: Path) -> Dict[str, str]:
    """
    Load canonical protein sequences and index them under several convenient keys:
    - GN=<SYMBOL> from the description (preferred; e.g., TP53, PIK3CA)
    - UniProt accession (e.g., P04637, P42336)
    - Entry name (e.g., P53_HUMAN, PK3CA_HUMAN)
    - Full record id
    """
    import re
    seqs: Dict[str, str] = {}
    for rec in SeqIO.parse(str(protein_fasta), "fasta"):
        seq = str(rec.seq).upper()
        desc = rec.description

        # 1) GN= symbol
        m = re.search(r"\bGN=([A-Za-z0-9\-]+)\b", desc)
        if m:
            seqs[m.group(1).upper()] = seq  # e.g., TP53, PIK3CA

        # 2) UniProt-style id parts: sp|P04637|P53_HUMAN
        parts = rec.id.split("|")
        if len(parts) >= 3:
            acc = parts[1]     # P04637
            entry = parts[2]   # P53_HUMAN / PK3CA_HUMAN
            seqs[acc] = seq
            seqs[entry] = seq
        else:
            seqs[rec.id] = seq

        # 3) Also index by the token before a space in the header, if looks like GENE|ACC
        # (harmless fallback; covers >TP53|P04637 style too)
        head_token = rec.id.split()[0]
        if "|" in head_token:
            seqs[head_token.split("|", 1)[0].upper()] = seq

        # 4) Helpful synonyms for common cases
        synonyms = {"P53_HUMAN": "TP53", "PK3CA_HUMAN": "PIK3CA"}
        if len(parts) >= 3 and parts[2] in synonyms:
            seqs[synonyms[parts[2]]] = seq

    return seqs


def normalize_hgvsp(hgvsp: str) -> Optional[str]:
    """
    Normalize common HGVSp short forms to p.X###Y with one-letter AAs.
    Accepts things like:
      p.R273H
      p.Arg273His
      R273H  (we'll add 'p.')
    Returns 'p.R273H' or None if cannot parse.
    """
    if pd.isna(hgvsp) or not str(hgvsp).strip():
        return None
    s = str(hgvsp).strip()
    if s.startswith("p."):
        s0 = s[2:]
    else:
        s0 = s
    # Try one-letter form e.g. R273H
    m = re.fullmatch(r"([A-Z*])(\d+)([A-Z*])", s0)
    if m:
        return "p." + m.group(1) + m.group(2) + m.group(3)
    # Try three-letter form e.g. Arg273His
    m = re.fullmatch(r"([A-Za-z]{3})(\d+)([A-Za-z]{3})", s0)
    if m:
        a1 = AA3_TO_1.get(m.group(1).capitalize())
        a2 = AA3_TO_1.get(m.group(3).capitalize())
        if a1 and a2:
            return "p." + a1 + m.group(2) + a2
    # Handle special cases (e.g., '=' means silent)
    if s0.endswith("="):
        # we can map to p.X###X later if needed
        pos = re.findall(r"(\d+)", s0)
        if pos:
            # unknown AA; mark as silent placeholder
            return f"p.?{pos[0]}?"
    return None

def hgvsp_from_maf_row(row: pd.Series) -> Optional[str]:
    """
    Prefer HGVSp_Short. Fallback to Amino_acids + Protein_position if present.
    Amino_acids is often like "R/H"; Protein_position like "273".
    """
    for col in ["HGVSp_Short", "HGVSp_Short_RefSeq", "HGVSp"]:
        if col in row and pd.notna(row[col]):
            norm = normalize_hgvsp(str(row[col]))
            if norm:
                return norm
    # Fallback
    if "Amino_acids" in row and "Protein_position" in row:
        aa = str(row["Amino_acids"]) if pd.notna(row["Amino_acids"]) else ""
        pos = str(row["Protein_position"]) if pd.notna(row["Protein_position"]) else ""
        # Amino_acids often "R/H"
        m = re.fullmatch(r"([A-Za-z*])\s*[/|]\s*([A-Za-z*])", aa.strip())
        if m and pos.isdigit():
            return f"p.{m.group(1).upper()}{pos}{m.group(2).upper()}"
    return None

def apply_missense(wt_seq: str, hgvsp: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Apply a single missense change p.X###Y to WT sequence.
    Returns (mut_seq, errmsg). If error, mut_seq=None and errmsg contains reason.
    """
    m = re.fullmatch(r"p\.([A-Z*])(\d+)([A-Z*])", hgvsp)
    if not m:
        return None, f"Unsupported/Unparsed HGVSp: {hgvsp}"
    ref, pos_str, alt = m.groups()
    pos = int(pos_str)
    if pos < 1 or pos > len(wt_seq):
        return None, f"Position {pos} out of range (len={len(wt_seq)})"
    wt_ref = wt_seq[pos-1]
    if wt_ref != ref:
        # Allow if mismatch but continue (annot differences between isoforms)
        # You can choose to error instead
        pass
    mut_seq = wt_seq[:pos-1] + alt + wt_seq[pos:]
    return mut_seq, None

def write_fasta(path: Path, header: str, seq: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f">{header}\n")
        # wrap to 60 chars
        for i in range(0, len(seq), 60):
            f.write(seq[i:i+60] + "\n")

def run_colabfold_batch(fasta_dir: Path, out_dir: Path, extra_args: str = "") -> None:
    """
    Invoke colabfold_batch on all FASTAs in fasta_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = f'colabfold_batch "{fasta_dir}" "{out_dir}" {extra_args}'.strip()
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def find_best_pdbs(colab_out: Path) -> Dict[str, Path]:
    """
    For each job subfolder in colabfold output, pick a PDB (usually ranked_0.pdb or model_1).
    Returns mapping {jobname: pdb_path}
    """
    mapping = {}
    for sub in colab_out.iterdir():
        if not sub.is_dir():
            continue
        # Job folder name equals fasta basename without extension
        job = sub.name
        # Try typical ColabFold outputs
        candidates = list(sub.glob("*.pdb"))
        # Prefer ranked_0 or best_model
        ranked = [p for p in candidates if "ranked_0" in p.name]
        chosen = ranked[0] if ranked else (candidates[0] if candidates else None)
        if chosen:
            mapping[job] = chosen
    return mapping

# ------------------------------
# Main pipeline
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build mutant protein sequences from BRCA MAF CSVs and (optionally) run AlphaFold via ColabFold.")
    ap.add_argument("--brca_dir", default="/fs/scratch/PAS2942/mutations/BRCA", help="Directory with TP53_mutations.csv and PIK3CA_mutations.csv")
    ap.add_argument("--protein_fasta", required=True, help="FASTA with canonical protein sequences for TP53 and PIK3CA (e.g., >TP53|P04637, >PIK3CA|P42336)")
    ap.add_argument("--outdir", default="/fs/scratch/PAS2942/mutations/BRCA/out", help="Target output directory")
    ap.add_argument("--genes", nargs="+", default=["TP53","PIK3CA"], help="Genes to process")
    ap.add_argument("--run_colabfold", action="store_true", help="If set, runs colabfold_batch on unique mutant FASTAs")
    ap.add_argument("--colabfold_args", default="--amber --num-recycle 3", help="Extra args passed to colabfold_batch")
    args = ap.parse_args()

    brca_dir = Path(args.brca_dir)
    outdir = Path(args.outdir)
    fasta_dir = outdir / "fastas"
    per_id_csv_dir = outdir / "mutant_sequences_by_id"
    pdb_dir = outdir / "pdbs"

    # Load canonical protein sequences
    prot = load_protein_fasta(Path(args.protein_fasta))

    # Collect unique mutations per gene: { (gene, hgvsp_norm) : mutant_seq }
    unique_mutants: Dict[Tuple[str,str], str] = {}
    # Map id->info rows for per-ID CSVs
    per_id_rows = []

    for gene in args.genes:
        csv_path = brca_dir / f"{gene}_mutations.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing {csv_path}, skipping {gene}", file=sys.stderr)
            continue

        df = pd.read_csv(csv_path, dtype=str)

        # Determine where sample id is stored
        sample_col = "sample_id" if "sample_id" in df.columns else (
                     "Tumor_Sample_Barcode" if "Tumor_Sample_Barcode" in df.columns else None)
        if not sample_col:
            print(f"[WARN] No sample_id/Tumor_Sample_Barcode in {csv_path}. Will synthesize IDs.", file=sys.stderr)
            df["_synth_id"] = [f"{gene}_row{i}" for i in range(len(df))]
            sample_col = "_synth_id"

        # Get canonical WT sequence
        wt_seq = prot.get(gene)
        if not wt_seq:
            # Try alternate keys
            keys = [k for k in prot.keys() if k.startswith(gene)]
            if keys:
                wt_seq = prot[keys[0]]
        if not wt_seq:
            print(f"[ERROR] No canonical protein sequence found for {gene} in {args.protein_fasta}", file=sys.stderr)
            sys.exit(1)

        for _, row in df.iterrows():
            sid = str(row[sample_col])
            # Normalize HGVSp
            hgvsp = hgvsp_from_maf_row(row)
            if not hgvsp:
                # Skip non-missense or unparsed
                continue

            # Apply mutation (only single-AA missense supported here)
            mut_seq, err = apply_missense(wt_seq, hgvsp)
            if mut_seq is None:
                # Skip frameshift/nonsense/complex or isoform mismatch; could be extended later
                continue

            # Record per-ID CSV row
            per_id_rows.append({
                "sample_id": sid,
                "gene": gene,
                "HGVSp": hgvsp,
                "wt_len": len(wt_seq),
                "mut_len": len(mut_seq),
                "wt_seq": wt_seq,
                "mut_seq": mut_seq
            })

            # Deduplicate by (gene, HGVSp)
            unique_mutants[(gene, hgvsp)] = mut_seq

    # Write one CSV per sample ID as requested
    per_id_df = pd.DataFrame(per_id_rows)
    if not per_id_df.empty:
        # Group by sample and write
        per_id_csv_dir.mkdir(parents=True, exist_ok=True)
        for sid, g in per_id_df.groupby("sample_id"):
            out_csv = per_id_csv_dir / f"{sid}.csv"
            g.to_csv(out_csv, index=False)
        print(f"[OK] Wrote {len(per_id_df['sample_id'].unique())} per-ID CSVs -> {per_id_csv_dir}")
    else:
        print("[WARN] No missense mutations parsed; no per-ID CSVs written.")

    # Write one FASTA per unique mutation
    for (gene, hgvsp), mseq in unique_mutants.items():
        safe = hgvsp.replace("p.", "").replace("*", "STOP")
        fasta_path = fasta_dir / gene / f"{gene}_{hgvsp}.fasta"
        header = f"{gene}|{hgvsp}"
        write_fasta(fasta_path, header, mseq)
    print(f"[OK] Prepared {len(unique_mutants)} unique mutant FASTAs in {fasta_dir}")

    # Optionally run ColabFold/AlphaFold
    if args.run_colabfold and unique_mutants:
        for gene in args.genes:
            g_fasta_dir = fasta_dir / gene
            if not g_fasta_dir.exists():
                continue
            g_out_dir = pdb_dir / gene
            run_colabfold_batch(g_fasta_dir, g_out_dir, args.colabfold_args)

        # Collect best PDBs per job and rename as requested
        for gene in args.genes:
            g_out_dir = pdb_dir / gene
            if not g_out_dir.exists():
                continue
            best = find_best_pdbs(g_out_dir)
            # Rename to <gene>_<HGVSp>.pdb in the same gene folder
            for jobname, pdb_path in best.items():
                # jobname equals fasta basename without extension
                # Our fasta name is "{gene}_{HGVSp}.fasta"
                out_pdb = g_out_dir / f"{jobname}.pdb"
                try:
                    out_pdb.write_bytes(pdb_path.read_bytes())
                    print(f"[OK] {jobname} -> {out_pdb}")
                except Exception as e:
                    print(f"[WARN] Could not copy {pdb_path} -> {out_pdb}: {e}", file=sys.stderr)

        print(f"[DONE] PDBs written under {pdb_dir}")

if __name__ == "__main__":
    main()
