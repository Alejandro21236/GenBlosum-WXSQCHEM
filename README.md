1. pipeline - take the WXS maf files from TCGA, run them through, get the colabfold, FASTA, and and CSV files. this takes a while.
2. two_neutral.py- GenBlosum Script. please change the coding region or porbabilities if you want to try a different gene. (GenBlosum)
3. fix_pdbs.py - make sure the pdbs generated on colabfold are not messed up.
4. build_qchem_inputs.sh - use to execute the input developing scripts input_HOMO, input_NPA, and pdb_to_qchem.
5. run_qchem_array.sh - does the qchem run across all input files in the directory
6. qchemanal2.py - analyzes all the outputs and converts the results into a CSV
7. qplots.py - plots from the csv created in the previous script.
