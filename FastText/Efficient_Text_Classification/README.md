# FastText
FastText: Bag of Tricks for Efficient Text Classification 
: [FastText: Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)

❤️ if you want to more learn about this paper read [this](https://yerimoh.github.io/LAN13/)





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

# Dataset
We use [preprocessed data (See Xiang Zhang's folder)](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)
* [AG's news](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html), Sogou, DBpedia, Yelp P., Yelp F., Yahoo A., Amazon F., Amazon P.

---

# Usage

## train

* Experiment
    ```
    # Download a spacy "en_core_web_lg" model
    $ python3 -m spacy download en_core_web_lg --user
    
    # Download datasets (select your os (mac or ubuntu))
    $ sh download_datasets_mac.sh
    ```



## test

* AG
```
# Create a pickle file: data/ag_news_csv/ag.pkl
$ python3 dataset.py --data_dir ./data/ag_news_csv --pickle_name ag.pkl --num_classes 4 --max_len 467

# Run
$ python3 main.py --data_path ./data/ag_news_csv/ag.pkl --batch_size 2048 --lr 0.5 --log_interval 20
```

* Sogou
```
# Create a pickle file: data/sogou_news_csv/sogou.pkl
$ python3 dataset.py --data_dir ./data/sogou_news_csv --pickle_name sogou.pkl --num_classes 5 --max_len 90064

# Run
$ python3 main.py --data_path ./data/sogou_news_csv/sogou.pkl --batch_size 1024 --lr 0.1 --log_interval 40
```

* DBpedia
```
# Create a pickle file: data/dbpedia_csv/dbp.pkl
$ python3 dataset.py --data_dir ./data/dbpedia_csv --pickle_name dbp.pkl --num_classes 14 --max_len 3013

# Run
$ python3 main.py --data_path ./data/dbpedia_csv/dbp.pkl --batch_size 2048 --lr 0.1 --log_interval 20
```

* Yelp P.
```
# Create a pickle file: data/yelp_review_polarity_csv/yelp_p.pkl
$ python3 dataset.py --data_dir ./data/yelp_review_polarity_csv --pickle_name yelp_p.pkl --num_classes 2 --max_len 2955

# Run
$ python3 main.py --data_path ./data/yelp_review_polarity_csv/yelp_p.pkl --batch_size 1024 --lr 0.1 --log_interval 40
```

* Yelp F.
```
# Create a pickle file: data/yelp_review_full_csv/yelp_f.pkl
$ python3 dataset.py --data_dir ./data/yelp_review_full_csv --pickle_name yelp_f.pkl --num_classes 5 --max_len 2955

# Run
$ python3 main.py --data_path ./data/yelp_review_full_csv/yelp_f.pkl --batch_size 1024 --lr 0.05 --log_interval 40
```

* Yahoo A.
```
# Create a pickle file: data/yahoo_answers_csv/yahoo_a.pkl
$ python3 dataset.py --data_dir ./data/yahoo_answers_csv --pickle_name yahoo_a.pkl --num_classes 10 --max_len 8024

# Run
$ python3 main.py --data_path ./data/yahoo_answers_csv/yahoo_a.pkl --batch_size 1024 --lr 0.05 --log_interval 40
```

* Amazon F.
```
# Create a pickle file: data/amazon_review_full_csv/amazon_f.pkl
$ python3 dataset.py --data_dir ./data/amazon_review_full_csv --pickle_name amazon_f.pkl --num_classes 5 --max_len 1214

# Run
$ python3 main.py --data_path ./data/amazon_review_full_csv/amazon_f.pkl --batch_size 4096 --lr 0.25 --log_interval 10
```

* Amazon P.
```
# Create a pickle file: data/amazon_review_polarity_csv/amazon_p.pkl
$ python3 dataset.py --data_dir ./data/amazon_review_polarity_csv --pickle_name amazon_p.pkl --num_classes 2 --max_len 1318

# Run
$ python3 main.py --data_path ./data/amazon_review_polarity_csv/yahoo_a.pkl --batch_size 4096 --lr 0.25 --log_interval 10
```

