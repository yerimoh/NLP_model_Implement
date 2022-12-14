# Word2Vec with non-ns
Implementation of the first paper on word2vec    
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- if you want to learn about detail this model, visit this [link](https://yerimoh.github.io/DL14/)
---

## Word2Vec Overview

There 2 model architectures desctibed in the paper:

- Continuous Bag-of-Words Model (CBOW), that predicts word based on its context;
- Continuous Skip-gram Model (Skip-Gram), that predicts context for a word.

Difference with the original paper:

- Trained on [WikiText103](https://pytorch.org/text/stable/datasets.html#wikitext-2) inxtead of Google News corpus.

-----

## Project Structure


```
├── README.md
├── config.yaml
├── requirements.txt
├── train.py
├── semantic.py
├── syntactic.py
├── utils
│   ├── test.py
│   ├── constants.py
│   ├── dataloader.py
│   ├── helper.py
│   ├── model.py
│   └── trainer.py
└── test_data
    ├── semantic_train.txt
    └── syntactic_train.txt
```

- **utils/dataloader.py** - data loader for WikiText-2 and WikiText103 datasets
- **utils/model.py** - model architectures
- **utils/trainer.py** - class for model training and evaluation

- **train.py** - script for training    
- **config.yaml** - file with training parameters     
- **utils/constants.py** - you can hadle more detail patameters       
- **weights/** - folder where expriments artifacts are stored if you want to retrain you must delete it

----

## Usage


### train

```
python3 train.py --config config.yaml
```

Before running the command, change the training parameters in the config.yaml, most important:

- model_name ("skipgram", "cbow")
- dataset ( "WikiText103")
- model_dir (directory to store experiment artifacts, should start with "weights/")


### test
After you train and get weight,  
you can test this semantic & syntactic test

```
python3 semantic.py 
```
```
python3 syntactic.py 
```

