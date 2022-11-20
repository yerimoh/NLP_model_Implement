# Word2vec with NS
- [Efficient Estimation of Word Representations in
Vector Space](https://arxiv.org/pdf/1301.3781.pdf)     
- [Distributed Representations of Words and Phrases and their Compositionality](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)    


-----

## Getting Started

download it and put in ```data```

```
Download google one billion dataset and put this folder in data directory.
```

----

## Preprocess data
- set the ```hyper.json```
- At first, run preprocess.py
```
python preprocess.py
```


## Training the Model
- fix features for model    

```
CBOW_HS.py
```

```
Huffman_encoding.py
```

```
Sg_HS.py
```

```
Sg_NS.py
```

```
Sg_NS_b.py
```



## Evaluation
Before evluation, you must change this path adapt with each model   
(in line 11)

```
if __name__ == "__main__":
    with open('FIX_ME.pkl', 'rb') as f:
        params = pickle.load(f)
```

and evaluate!

```
python semantic.py
```
```
python semantic.py
```

