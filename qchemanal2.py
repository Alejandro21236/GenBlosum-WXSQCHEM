#!/usr/bin/env python

import os
import glob
import re
import json
import pandas as pd

# =========================
# CONFIG: set your folders
# =========================
# Change these to your actual directories
ESP_DIR  = "/fs/scratch/PAS2942/mutations/BRCA/out/pdbs/TP53/fixed_pdbs/qchem_outputs"         # ESP/CHELPG jobs
NPA_DIR  = "/fs/scratch/PAS2942/mutations/BRCA/out/pdbs/TP53/fixed_pdbs/qchem_outputs_NPA"     # NPA jobs
HOMO_DIR = "/fs/scratch/PAS2942/mutations/BRCA/out/pdbs/TP53/fixed_pdbs/qchem_outputs_HOMO"    # HOMO/LUMO jobs

OUTPUT_CSV = "qchem_all_summary_indestructible.csv"

EV_PER_HARTREE = 27.211386245988


# =========================
# Regex helpers
# =========================

ENERGY_RE = re.compile(r"Total energy in the final basis set\s*=\s*([-\d\.]+)")

ORBITAL_BLOCK_RE = re.compile(
    r"Orbital Energies\s*\(a\.u\.\)(.*?)(?:\n\s*\n|\Z)", re.S | re.M
)

CHELPG_START_RE = re.compile(r"Charges from ESP fit", re.I)

# NPA header variants
NPA_START_RE = re.compile(
    r"NATURAL POPULATION ANALYSIS|Summary of Natural Population Analysis", re.I
)


def safe_float(tok):
    try:
        return float(tok)
    except Exception:
        return None


# =========================
# Core parsers
# =========================

def parse_energy(text, calc_type, rec):
    """Parse SCF energy in Hartree and eV."""
    m = ENERGY_RE.search(text)
    if not m:
        return rec
    e_ha = float(m.group(1))
    rec[f"energy_{calc_type}_Ha"] = e_ha
    rec[f"energy_{calc_type}_eV"] = e_ha * EV_PER_HARTREE
    return rec


def parse_orbitals(text, rec):
    """
    Parse orbital energies (a.u.) from an 'Orbital Energies (a.u.)' block
    and approximate HOMO/LUMO as last negative / first positive.
    """
    m = re.search(r"Orbital Energies\s*\(a\.u\.\)", text)
    if not m:
        return rec

    start = m.end()
    lines = text[start:].splitlines()

    energies = []
    started = False

    for line in lines:
        # stop at first full blank line after we've started reading numbers
        if not line.strip():
            if started:
                break
            else:
                continue

        # skip headings
        if ("Alpha MOs" in line or "Beta MOs" in line or
            "Occupied" in line or "Virtual" in line or
            set(line.strip()) == set("-")):
            continue

        # collect floats
        vals = [safe_float(tok) for tok in line.split()]
        vals = [v for v in vals if v is not None]
        if vals:
            started = True
            energies.extend(vals)
        elif started:
            # if we already started and find a non-numeric line, stop
            break

    if not energies:
        return rec

    # HOMO = last negative, LUMO = first positive after it
    neg_indices = [i for i, e in enumerate(energies) if e < 0.0]
    if not neg_indices:
        return rec

    homo_idx = max(neg_indices)
    if homo_idx + 1 >= len(energies):
        return rec

    homo = energies[homo_idx]
    lumo = energies[homo_idx + 1]
    gap = lumo - homo

    rec["HOMO_Ha"] = homo
    rec["LUMO_Ha"] = lumo
    rec["gap_Ha"] = gap
    rec["HOMO_eV"] = homo * EV_PER_HARTREE
    rec["LUMO_eV"] = lumo * EV_PER_HARTREE
    rec["gap_eV"] = gap * EV_PER_HARTREE

    return rec



def parse_chelpg(text, rec):
    """
    Parse CHELPG/ChElPG charges from a 'Ground-State ChElPG Net Atomic Charges'
    block like the one in your screenshot.
    """
    m = re.search(r"Ground-State\s+ChElPG\s+Net\s+Atomic\s+Charges", text, re.I)
    if not m:
        return rec

    start = m.end()
    lines = text[start:].splitlines()

    charges = []
    started = False

    for line in lines:
        stripped = line.strip()

        # stop when we hit a blank line after starting, or another header
        if not stripped:
            if started:
                break
            else:
                continue

        # skip header / dashed line
        if "Atom" in stripped and "Charge" in stripped:
            continue
        if set(stripped) <= set("-+|"):
            continue

        parts = stripped.split()
        # expecting something like: "1  N  -0.635128"
        if len(parts) < 2:
            if started:
                break
            else:
                continue

        val = safe_float(parts[-1])
        if val is None:
            if started:
                break
            else:
                continue

        started = True
        charges.append(val)

    if charges:
        rec["CHELPG_charges"] = json.dumps(charges)
        rec["CHELPG_sum"] = float(sum(charges))

    return rec



def parse_npa(text, rec):
    """
    Parse NPA atomic charges from NBO/NPA block.
    We look for the header and then scan the following lines
    for a table with atomic charges in one of the early columns.
    This will not be perfect but should work for standard Q-Chem NPA output.
    """
    m = NPA_START_RE.search(text)
    if not m:
        return rec

    start = m.end()
    lines = text[start:].splitlines()

    # Try to find the line with something like: "Atom  No    Charge ..."
    header_idx = None
    for i, line in enumerate(lines):
        if "Charge" in line and "Atom" in line:
            header_idx = i
            break

    if header_idx is None:
        return rec

    charges = []
    for line in lines[header_idx + 1:]:
        # Stop if we hit a full blank line or a line of dashes
        if not line.strip():
            if charges:
                break
            else:
                continue
        if set(line.strip()) <= set("-"):
            if charges:
                break
            else:
                continue

        parts = line.split()
        # Heuristic: there should be at least 3-4 tokens, and one of them (usually the second
        # or third) is the NPA charge, typically between -3 and +3.
        floats = [safe_float(t) for t in parts]
        floats = [f for f in floats if f is not None]

        if not floats:
            # not a data line
            if charges:
                break
            else:
                continue

        # Pick the float whose abs value is < 5 and closest to zero as the "charge"
        candidate = min(floats, key=lambda x: abs(x)) if floats else None
        if candidate is not None:
            charges.append(candidate)

    if charges:
        rec["NPA_charges"] = json.dumps(charges)
        rec["NPA_sum"] = float(sum(charges))

    return rec


# =========================
# High-level per-file parse
# =========================

def parse_qchem_out(path, calc_type):
    """
    Pure-text parser for a single Q-Chem .out file.

    calc_type: "ESP", "NPA", or "HOMO" (only affects which extras we try to parse).
    """
    with open(path, "r", errors="ignore") as f:
        text = f.read()

    base = os.path.basename(path)
    job_id = base.replace(".out", "")

    rec = {
        "file": base,
        "job_id": job_id,
        "calc_type": calc_type,
    }

    # Always try energy
    rec = parse_energy(text, calc_type, rec)

    # HOMO/LUMO for HOMO jobs
    if calc_type == "HOMO":
        rec = parse_orbitals(text, rec)

    # CHELPG for ESP jobs
    if calc_type == "ESP":
        rec = parse_chelpg(text, rec)

    # NPA for NPA jobs
    if calc_type == "NPA":
        rec = parse_npa(text, rec)

    return rec


def parse_folder(folder, calc_type):
    out_files = sorted(glob.glob(os.path.join(folder, "*.out")))
    rows = []

    for path in out_files:
        try:
            rec = parse_qchem_out(path, calc_type)
            rows.append(rec)
        except Exception as e:
            print(f"[{calc_type}] Skipping {path}: {e}")

    if not rows:
        print(f"[{calc_type}] No usable records in {folder}")
        return pd.DataFrame()

    return pd.DataFrame(rows)


def main():
    dfs = []

    if ESP_DIR and os.path.isdir(ESP_DIR):
        dfs.append(parse_folder(ESP_DIR, "ESP"))

    if NPA_DIR and os.path.isdir(NPA_DIR):
        dfs.append(parse_folder(NPA_DIR, "NPA"))

    if HOMO_DIR and os.path.isdir(HOMO_DIR):
        dfs.append(parse_folder(HOMO_DIR, "HOMO"))

    if not dfs:
        print("No data parsed from any folder. Check paths.")
        return

    # Merge everything on job_id (outer join)
    summary = None
    for df in dfs:
        if df.empty:
            continue
        df_small = df.set_index("job_id")
        if summary is None:
            summary = df_small
        else:
            # Suffix overlapping columns with calc_type name to avoid collisions
            for col in df_small.columns:
                if col in summary.columns:
                    df_small = df_small.rename(columns={col: f"{col}_{df_small['calc_type'].iloc[0]}"})
            summary = summary.join(df_small, how="outer")

    if summary is None:
        print("No combined data to write.")
        return

    summary.reset_index(inplace=True)
    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"Written: {OUTPUT_CSV}")
    print(summary.head())


if __name__ == "__main__":
    main()
