#!/usr/bin/env bash
#SBATCH --job-name=tp53_nmr
#SBATCH --account=
#SBATCH --partition=nextgen
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-28    # adjust to number of inputs-1
#SBATCH --output=logs/tp53_nmr_%A_%a.out
#SBATCH --error=logs/tp53_nmr_%A_%a.err

module load qchem/6.3.0   # or whatever OSC uses
module load miniconda3/24.1.2-py310
module load cuda/11.8.0
conda activate rnaseq
BASE_DIR="/fs/scratch/PAS2942/mutations/BRCA/out/pdbs/TP53/fixed_pdbs"
IN_DIR="${BASE_DIR}/qchem_inputs_NPA"
OUT_DIR="${BASE_DIR}/qchem_outputs_NPA"

mkdir -p "${OUT_DIR}" logs

# Make an index of all .inp files
mapfile -t inputs < <(ls "${IN_DIR}"/*.inp)
inp="${inputs[$SLURM_ARRAY_TASK_ID]}"
base=$(basename "$inp" .inp)

out="${OUT_DIR}/${base}.out"

echo "Running Q-Chem on ${inp}"
qchem -nt ${SLURM_CPUS_PER_TASK} "$inp" "$out"
