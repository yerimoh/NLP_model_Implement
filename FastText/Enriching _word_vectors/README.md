
# FastText
FastText: Enriching Word Vectors with Subword Information - [FastText: Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606.pdf)

if you want to more learn about this paper read [this](https://yerimoh.github.io/LAN7/)





-----



## Project Structure


```
.
├── README.md
├── WS353.py
├── fin_data.py
├── semantic.py
├── syntactic.py
├── train.py
├── data
│   └── ws353simrel.tar.gz
│   └── (you must add) one_bilion_dataset
├── test_data
│   └── semantic_train.txt
│   └── syntactic_train.txt
└── utils
    ├── test.py
    └── word.py

```

----

## Usage


### train
you must process the data first
```
python3 fin_data.py
```
and train it!
```
python3 train_model.py 
```



-----


### test
After you train and get weight,  
you can test this semantic & syntactic test

```
python3 semantic.py 
```
```
python3 syntactic.py 
```
```
python3 WS353.py
```
