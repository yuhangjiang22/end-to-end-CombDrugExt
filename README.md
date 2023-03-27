# End-to-End n-ary Relation Extraction for Combination Drug Therapies
The corresponding code for our paper: End-to-End n-ary Relation Extraction for Combination Drug Therapies
## Get access to CombDrugExt dataset
```
git clone https://github.com/allenai/drug-combo-extraction.git
```
### Preprocess dataset to our expected format
```
python preprocess.py
```
## Installing Seq2rel using Poetry
```
curl -sSL https://install.python-poetry.org | python3 -
cd seq2rel
poetry install
cd ..
```
## Training
Use the `allennlp train` command with `.jsonnet` config file we provid to train the model
```
train_data_path="n-ary/train.txt" \
valid_data_path="n-ary/valid.txt" \
dataset_size=1362 \
allennlp train "training-config-seq2rel/n-ary.jsonnet" \
    --serialization-dir "output"\
    --include-package "seq2rel" 
```
## Evaluation
Example for evaluation on `n-ary/test.txt` file with fine-tuned `model.tar.gz` of `positive combination f1` metric
```
python evaluation.py\
          --model output/model.tar.gz\
          --test_file n-ary/test.txt\
          --metric positive combination f1\
