# KGBERT Project

Classification prediction model exercise built using the model bert_uncased and the general knowledge graph dataset WN18

## Require

Python >= 3.8

```bash
pip install pytorch_pretrained_bert
pip install transformers
```

## Data

(1) The benchmark knowledge graph datasets are in ./data.

(2) entity2text.txt or entity2textlong.txt in each dataset contains entity textual sequences.

(3) relation2text.txt in each dataset contains relation textual sequences.

## Get a trained model

Get File From [KGE](https://drive.google.com/drive/folders/12Wa7fY-oMAYWGlZJ3lCesOhZt97U02zE?usp=sharing)

## run project

#### Execute training file

```shell
python train.py
--data_dir WN18/data
--output_dir output
```
#### Execute predict file
```shell
python predict.py
--data_dir WN18/data
--output_dir output
```
