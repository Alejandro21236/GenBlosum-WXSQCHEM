# GenBlosum-WXSQCHEM

Pipeline to (1) process WXS variants into protein substitutions, (2) score variants under a BLOSUM-based neutral model (“GenBlosum”), and (3) generate/run Q-Chem jobs and summarize electronic-structure outputs.

## What’s in this repo
- `pipeline.py` — end-to-end WXS→FASTA/PDB/CSV orchestration (expects local/authorized inputs)
- `two_neutral.py` — GenBlosum neutral model scoring
- `fix_pdbs.py` — PDB cleanup for downstream fragmentation/Q-Chem #YOU MAY NEED TO USE OPENBABEL
- `pdb_to_qchem_fragment.py` — build Q-Chem fragments from PDBs
- `input_HOMO.py`, `input_NPA.py` — write Q-Chem input decks
- `build_qchem_inputs.sh` — batch-generate Q-Chem inputs
- `run_qchem_array.sh` — run Q-Chem across an input directory
- `qchemanal2.py` — parse outputs → CSV
- `qchem_qcplots.py` — plots from the parsed CSV

## Data policy
This repo does **not** distribute TCGA/GDC-derived datasets or any controlled-access files.
Provide your own authorized inputs (e.g., MAF/VCF, FASTA/PDB) and run the pipeline locally/HPC.

## Quickstart
1. Create env / install deps
2. Run `pipeline.py` on your variant table to generate per-variant structures
3. Run `two_neutral.py` to compute GenBlosum neutral expectations/scores
4. Generate Q-Chem inputs via `build_qchem_inputs.sh`
5. Run Q-Chem with `run_qchem_array.sh`
6. Summarize via `qchemanal2.py` and plot via `qchem_qcplots.py`

## Requirements
- Python >= 3.10
- Q-Chem (version: 6.3.0 )
- (optional) ColabFold/AlphaFold runner if generating structures
