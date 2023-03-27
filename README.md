# End-to-End n-ary Relation Extraction for Combination Drug Therapies
The corresponding code for our paper: End-to-End n-ary Relation Extraction for Combination Drug Therapies
## Get access to CombDrugExt dataset
```
git clone https://github.com/allenai/drug-combo-extraction.git
```
### Preprocess dataset
```
python preprocess.py
```

## Installation of Seq2rel
```
curl -sSL https://install.python-poetry.org | python3 -
git clone https://github.com/JohnGiorgi/seq2rel.git
cd seq2rel
poetry install
```
