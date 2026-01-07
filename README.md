# KASSPer

KASSPer predicts kinase active-site conformational states using protein and compound language models.
Given a kinase amino acid sequence and a ligand SMILES string, it performs ligand-specific state prediction prior to structure-based virtual screening (SBVS).

## Resources

- KinCoRe (Kinase structure annotation): https://github.com/DunbrackLab/Kincore-standalone2  
- DUD-E (Ligand benchmark library): https://dude.docking.org/subsets  
- Kinase multi-state protein structures: https://doi.org/10.5281/zenodo.8272608

## Create conda enviroment for KASSPer

```bash
conda env create -f ./environment.yml
conda activate KASSPer
```

## Input formats

- FASTA: `>id` header with sequence lines
- SMI: `SMILES ID` format (space/tab delimited); header/comment (`#`) lines are ignored

## Usage

### 1) 1:1 matching prediction (obo)

Embeddings and predictions are created only when FASTA and SMI share the **same IDs**.

```bash
python KASSPER/main.py \
  --method obo \
  --fasta KASSPER/sample/sample.fasta \
  --smi KASSPER/sample/sample.smi \
  --output KASSPER/sample/obo_run \
  --force-remake
```

### 2) All-pairs prediction (obn)

Embeddings are created independently for FASTA and SMI inputs, then **all pairs** are predicted.

```bash
python KASSPER/main.py \
  --method obn \
  --fasta KASSPER/sample/sample.fasta \
  --smi KASSPER/sample/sample.smi \
  --output KASSPER/sample/obn_run \
  --force-remake

```

## Sample data

- `KASSPER/sample/sample.fasta`
- `KASSPER/sample/sample.smi`

Running the scripts with their default paths will create outputs under `KASSPER/sample/`.

## License

See the repository for license details.
