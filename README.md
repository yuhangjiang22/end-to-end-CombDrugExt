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
## Installation for Seq2rel
```
curl -sSL https://install.python-poetry.org | python3 -
cd seq2rel
poetry install
```
## Training
Use the `allennlp train` command with the provided `.jsonnet` config file to train the model
```
train_data_path="n-ary/train.txt" \
valid_data_path="n-ary/valid.txt" \
dataset_size=1362 \
allennlp train "training-config-seq2rel/n-ary.jsonnet" \
    --serialization-dir "/content/output"\
    --include-package "seq2rel" 
```
