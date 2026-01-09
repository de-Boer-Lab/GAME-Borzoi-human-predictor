# parse_borzoi_target.py
import os
import pandas as pd

utils_dir = os.path.dirname(__file__)
borzoi_api_dir = os.path.dirname(utils_dir)
borzoi_examples_dir = f"{borzoi_api_dir}/borzoi/examples"

input_data = pd.read_csv(f'{borzoi_examples_dir}/targets_human.txt', sep='\t', header=0)
input_data = input_data[['description']]

# Parsing for CAGE
CAGE = input_data[input_data['description'].str.contains("CAGE")]
CAGE_split = CAGE['description'].str.split(':', n=1, expand=True)
CAGE = pd.DataFrame({
    'Assay': CAGE_split[0],
    'Cell Type': CAGE_split[1]
})

# Parsing for DNASE
DNASE = input_data[input_data['description'].str.startswith('DNASE')]
DNASE_split = DNASE['description'].str.split(':', n=1, expand=True)
DNASE = pd.DataFrame({
    'Assay': DNASE_split[0],
    'Cell Type': DNASE_split[1]
})

# Parsing for ATAC
ATAC = input_data[input_data['description'].str.startswith('ATAC')]
ATAC_split = ATAC['description'].str.split(':', n=1, expand=True)
ATAC = pd.DataFrame({
    'Assay': ATAC_split[0],
    'Cell Type': ATAC_split[1]
})

# Parsing for CHIP (different structure)
CHIP = input_data[input_data['description'].str.startswith('CHIP')]
CHIP = CHIP['description'].str.split(r'[;:]', expand=True)
CHIP = CHIP.iloc[:, :3]
CHIP.columns = ['Assay', 'Molecule','Cell Type']

# Parsing for RNA
RNA = input_data[input_data['description'].str.startswith('RNA')]
RNA_split = RNA['description'].str.split(':', n=1, expand=True)
RNA = pd.DataFrame({
    'Assay': RNA_split[0],
    'Cell Type': RNA_split[1]
})

# Merged data: Note that CHIP will have one extra columns
targets_borzoi = pd.concat([CAGE, DNASE, ATAC, CHIP, RNA], ignore_index=True)
targets_borzoi.to_csv(f'{utils_dir}/borzoi_human_targets_simplified.txt', sep='\t', index=False)
