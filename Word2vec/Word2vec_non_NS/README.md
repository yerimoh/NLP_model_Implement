# Word2Vec in PyTorch

Implementation of the first paper on word2vec - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)


## Word2Vec Overview

There 2 model architectures desctibed in the paper:

- Continuous Bag-of-Words Model (CBOW), that predicts word based on its context;
- Continuous Skip-gram Model (Skip-Gram), that predicts context for a word.

Difference with the original paper:

- Trained on [WikiText103](https://pytorch.org/text/stable/datasets.html#wikitext-2) inxtead of Google News corpus.



### CBOW Model in Details



### Skip-Gram Model in Details



## Project Structure


```
.
├── README.md
├── config.yaml
├── notebooks
│   └── Inference.ipynb
├── requirements.txt
├── train.py
├── utils
│   ├── constants.py
│   ├── dataloader.py
│   ├── helper.py
│   ├── model.py
│   └── trainer.py
└── weights
```

- **utils/dataloader.py** - data loader for WikiText-2 and WikiText103 datasets
- **utils/model.py** - model architectures
- **utils/trainer.py** - class for model training and evaluation

- **train.py** - script for training
- **config.yaml** - file with training parameters
- **weights/** - folder where expriments artifacts are stored


## Usage


```
python3 train.py --config config.yaml
```

Before running the command, change the training parameters in the config.yaml, most important:

- model_name ("skipgram", "cbow")
- dataset ( "WikiText103")
- model_dir (directory to store experiment artifacts, should start with "weights/")


