#!/usr/bin/env python3
import argparse
import os
import re

from Bio.PDB import PDBParser, Selection

Z_TABLE = {
    "H": 1, "C": 6, "N": 7, "O": 8, "S": 16,
    "P": 15, "F": 9, "CL": 17, "BR": 35, "I": 53,
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--mut_tag", required=True, help="e.g. p.D281V")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--charge", type=int, default=0)
    return ap.parse_args()

def extract_mut_index(mut_tag):
    m = re.search(r"p\.[A-Z]([0-9]+)[A-Z]", mut_tag)
    if not m:
        raise ValueError(f"Could not parse residue index from {mut_tag}")
    return int(m.group(1))

def load_structure(pdb_path):
    parser = PDBParser(QUIET=True)
    return parser.get_structure("struct", pdb_path)

def get_chain_for_resid(struct, resid_target):
    for chain in struct.get_chains():
        for res in chain.get_residues():
            if not Selection.unfold_entities([res], "A"):
                continue
            _, seqid, _ = res.get_id()
            if seqid == resid_target:
                return chain.id
    raise RuntimeError(f"Residue {resid_target} not found in any chain")

def fragment_atoms(struct, chain_id, center_idx, window=1):
    atoms = []
    for chain in struct.get_chains():
        if chain.id != chain_id:
            continue
        for res in chain.get_residues():
            _, seqid, _ = res.get_id()
            if center_idx - window <= seqid <= center_idx + window:
                for atom in res.get_atoms():
                    if not atom.element or atom.element.strip() == "":
                        # crude guess from atom name
                        atom.element = atom.get_id()[0]
                    atoms.append(atom)
    return atoms

def guess_multiplicity(atoms, charge):
    total_Z = 0
    for atom in atoms:
        elem = atom.element.upper().strip()
        # handle e.g. 'CL', 'CA' etc.
        if elem in Z_TABLE:
            total_Z += Z_TABLE[elem]
        else:
            e0 = elem[0]
            if e0 in Z_TABLE:
                total_Z += Z_TABLE[e0]
            else:
                raise ValueError(f"Unknown element {elem} in fragment")
    nelec = total_Z - charge
    # even electrons → singlet, odd → doublet
    if nelec % 2 == 0:
        return 1
    else:
        return 2

def write_qchem_input(atoms, charge, mult, outfile):
    with open(outfile, "w") as f:
        f.write("$molecule\n")
        f.write(f"{charge} {mult}\n")
        for atom in atoms:
            elem = atom.element
            x, y, z = atom.coord
            f.write(f"{elem:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")
        f.write("$end\n\n")
        f.write("$rem\n")
        f.write("   JOBTYPE        SP\n")        # single-point, not NMR
        f.write("   METHOD         B3LYP\n")
        f.write("   BASIS          6-31G*\n")
        f.write("   SCF_CONVERGENCE 7\n")
        f.write("   MAX_SCF_CYCLES 200\n")
        f.write("   SCF_ALGORITHM  DIIS_GDM\n")
        f.write("   SYM_IGNORE     1\n")
        f.write("   NO_REORIENT    1\n")
        f.write("   CHELPG         TRUE\n")       # <-- ESP-derived charges
        # optional: Lebedev CHELPG grids for speed on larger systems:
        # f.write("   CHELPG_H       50\n")
        # f.write("   CHELPG_HA      110\n")
        f.write("$end\n")

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    resid = extract_mut_index(args.mut_tag)
    struct = load_structure(args.pdb)
    chain_id = get_chain_for_resid(struct, resid)

    atoms = fragment_atoms(struct, chain_id, resid, window=1)
    if not atoms:
        raise RuntimeError("No atoms in fragment – check residue index / PDB")

    mult = guess_multiplicity(atoms, args.charge)
    base = os.path.splitext(os.path.basename(args.pdb))[0]
    outpath = os.path.join(args.out_dir, base + ".inp")
    write_qchem_input(atoms, args.charge, mult, outpath)
    print(f"Wrote {outpath} with charge={args.charge}, mult={mult}")

if __name__ == "__main__":
    main()
