#!/usr/bin/env python3
import argparse
import os
import glob
import subprocess
import shutil

def run_cmd(cmd, cwd=None):
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"[WARN] Command failed (code {result.returncode}): {' '.join(cmd)}")
        print(f"[WARN] stdout:\n{result.stdout}")
        print(f"[WARN] stderr:\n{result.stderr}")
    return result.returncode == 0

def fix_with_pdb4amber(in_pdb, out_pdb):
    # --reduce adds hydrogens using reduce
    cmd = ["pdb4amber", "-i", in_pdb, "-o", out_pdb, "--reduce"]
    return run_cmd(cmd)

def fix_with_reduce(in_pdb, out_pdb):
    # MolProbity "reduce" binary
    cmd = ["reduce", "-BUILD", in_pdb]
    ok = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if ok.returncode != 0:
        print(f"[WARN] reduce failed on {in_pdb}:\n{ok.stderr}")
        return False
    with open(out_pdb, "w") as f:
        f.write(ok.stdout)
    return True

def fix_with_obabel(in_pdb, out_pdb):
    # OpenBabel: add hydrogens
    cmd = ["obabel", in_pdb, "-h", "-O", out_pdb]
    return run_cmd(cmd)

def guess_tool():
    # Try to auto-detect a usable tool
    for exe in ["pdb4amber", "reduce", "obabel"]:
        if shutil.which(exe) is not None:
            return exe
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_dir",
        default="/fs/scratch/PAS2942/mutations/BRCA/out/pdbs/TP53",
        help="Directory containing ColabFold TP53 PDBs",
    )
    ap.add_argument(
        "--tool",
        choices=["pdb4amber", "reduce", "obabel", "auto"],
        default="auto",
        help="Which external tool to use to fix/add H",
    )
    ap.add_argument(
        "--wt_pdb",
        default=None,
        help="Optional WT PDB file (relative or absolute) to also fix",
    )
    args = ap.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    out_dir = os.path.join(base_dir, "fixed_pdbs")
    os.makedirs(out_dir, exist_ok=True)

    tool = args.tool
    if tool == "auto":
        tool = guess_tool()
        if tool is None:
            raise SystemExit(
                "No pdb4amber / reduce / obabel found in PATH. "
                "Load a module and/or specify --tool explicitly."
            )
        print(f"[INFO] Auto-selected tool: {tool}")

    if tool == "pdb4amber":
        fixer = fix_with_pdb4amber
    elif tool == "reduce":
        fixer = fix_with_reduce
    elif tool == "obabel":
        fixer = fix_with_obabel
    else:
        raise SystemExit(f"Unknown tool: {tool}")

    # 1) Mutant PDBs: first pdb of each set = rank_001
    pattern = os.path.join(base_dir, "TP53_p.*_unrelaxed_rank_001_*.pdb")
    pdb_files = sorted(glob.glob(pattern))

    if not pdb_files:
        print(f"[WARN] No PDBs found matching pattern {pattern}")
    else:
        print(f"[INFO] Found {len(pdb_files)} mutant PDBs")

    for pdb in pdb_files:
        base = os.path.basename(pdb)
        out_pdb = os.path.join(out_dir, base.replace(".pdb", "_H.pdb"))
        print(f"[INFO] Fixing mutant {base} -> {out_pdb}")
        ok = fixer(pdb, out_pdb)
        if not ok:
            print(f"[WARN] Failed to fix {pdb}")

    # 2) Optional WT
    if args.wt_pdb is not None:
        wt_path = args.wt_pdb
        if not os.path.isabs(wt_path):
            wt_path = os.path.join(base_dir, wt_path)
        if os.path.exists(wt_path):
            wt_base = os.path.basename(wt_path)
            wt_out = os.path.join(out_dir, wt_base.replace(".pdb", "_H.pdb"))
            print(f"[INFO] Fixing WT {wt_base} -> {wt_out}")
            ok = fixer(wt_path, wt_out)
            if not ok:
                print(f"[WARN] Failed to fix WT {wt_path}")
        else:
            print(f"[WARN] WT PDB {wt_path} does not exist")

    print("[INFO] Done. Fixed PDBs (with H) in:", out_dir)

if __name__ == "__main__":
    main()
