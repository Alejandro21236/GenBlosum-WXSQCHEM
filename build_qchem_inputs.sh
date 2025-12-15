#!/usr/bin/env bash
#SBATCH --account=
#SBATCH --job-name=NMRin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --output=statefarm.out
#SBATCH --error=statefarm.err

set -euo pipefail

BASE_DIR="/fs/scratch/PAS2942/mutations/BRCA/out/pdbs/TP53/fixed_pdbs"
OUT_DIR="${BASE_DIR}/qchem_inputs_NPA"

mkdir -p "${OUT_DIR}"

for pdb in "${BASE_DIR}"/TP53_p.*_unrelaxed_rank_001_*.pdb; do
    [ -e "$pdb" ] || continue

    fname=$(basename "$pdb")
    # Example: TP53_p.D281V_unrelaxed_rank_001_alphafold2_ptm_model_2_seed_000.pdb
    mut=${fname#TP53_}              # p.D281V_unrelaxed...
    mut=${mut%%_unrelaxed*}         # p.D281V

    echo "Building fragment + Q-Chem input for ${mut}"

    python input_NPA.py \
        --pdb "$pdb" \
        --mut_tag "$mut" \
        --out_dir "$OUT_DIR"
done

echo "Done. Inputs in ${OUT_DIR}"
