import numpy as np
import sys, pickle
import heapq
from collections import Counter
from tqdm.auto import tqdm
import pandas as pd

def clean_str(string):
    string = string.strip().lower()
    string = string.replace(".", " ")
    string = string.replace("\'s", " ")
    string = string.replace("\'", " ")
    string = string.replace('\"', " ")
    string = string.replace(",", " ")
    string = string.replace("!", " ")
    string = string.replace("?", " ")
    string = string.replace("\n", " ")
    string = string.replace("<br>", " ")
    string = string.replace("<br />", " ")
    string = string.replace("\\", " ")
    string = string.replace("//", " ")
    string = string.replace("<", " ")
    string = string.replace(">", " ")
    string = string.replace("=", " ")
    string = string.replace("+", " ")
    string = string.replace("  n", "")
    string = string.replace("-", " ")
    string = string.replace(")", " ")
    string = string.replace("(", " ")
    string = string.replace("[", " ")
    string = string.replace("]", " ")
    string = string.replace("{", " ")
    string = string.replace("}", " ")
    string = string.replace(";", " ")
    string = string.replace(":", " ")
    string = string.replace("@", " ")
    string = string.replace("#", " ")
    string = string.replace("$", " ")
    string = string.replace("%", " ")
    string = string.replace("^", " ")
    string = string.replace("&", " ")
    string = string.replace("*", " ")
    string = string.replace("  ", " ")
    return string.strip()

def freq(text, threshold=5):
    counts = Counter(text)
    output = []
    for w in tqdm(text, desc="Discarding rare words"):
        if w.isalpha(): #sogou에선 숫자 필요
            if counts[w] >= threshold:
                output.append(w)

    return output

def preprocess(corpus):
    corpus_cnt = Counter(corpus)
    max_cnt = max(corpus_cnt.values())
    corpus_cnt["<pad>"] = max_cnt + 1
    cnts = sorted(corpus_cnt, key=lambda x: -corpus_cnt[x])

    output = []
    w2id, id2w = {}, {}

    for word in tqdm(cnts, desc="Making dict"):
        idx = len(w2id)
        w2id[word] = idx
        id2w[idx] = word
    
    return w2id, id2w, corpus_cnt

def create_contexts_target(data, label, content):
    targets = np.array(data[label])
    tmp_contexts, contexts = [], []

    for line in tqdm(data[content]):
        cs = []
        line = clean_str(line)
        for word in line.split():
            if word in w2id.keys():
                cs.append(w2id[word])
        
        tmp_contexts.append(cs)

    max_len = min(len(max(tmp_contexts, key=len)), 512)
    for con in tmp_contexts:
        if len(con) > max_len:
            con = con[:max_len]
        elif len(con) < max_len:
            con.extend([0]*(max_len - len(con)))
        contexts.append(con)

    return np.array(contexts), targets

if __name__ == "__main__":
    if sys.argv[1] == "ag":
        data = pd.read_csv('./datasets/ag/modified_train.csv', header=None)
        test = pd.read_csv('./datasets/ag/modified_test.csv', header=None)
        label, content = 0, 1
    elif sys.argv[1] == "amzf":
        data = pd.read_csv('./datasets/amzf/modified_train.csv', header=None)
        test = pd.read_csv('./datasets/amzf/modified_test.csv', header=None)
        data = data.dropna().reset_index()
        test = test.dropna().reset_index()
        label, content = 0, 1
    elif sys.argv[1] == "amzp":
        data = pd.read_csv('./datasets/amzp/modified_train.csv', header=None)
        test = pd.read_csv('./datasets/amzp/modified_test.csv', header=None)
        data = data.dropna().reset_index()
        test = test.dropna().reset_index()
        label, content = 0, 1
    elif sys.argv[1] == "dbp":
        data = pd.read_csv('./datasets/dbp/modified_train.csv', header=None)
        test = pd.read_csv('./datasets/dbp/modified_test.csv', header=None)
        label, content = 0, 1
    elif sys.argv[1] == "sogou":
        data = pd.read_csv('./datasets/sogou/modified_train.csv', header=None)
        test = pd.read_csv('./datasets/sogou/modified_test.csv', header=None)
        label, content = 0, 1
    elif sys.argv[1] == "yaha":
        tmp_data = pd.read_csv('./datasets/yaha/train.csv', header=None)
        tmp_test = pd.read_csv('./datasets/yaha/test.csv', header=None)
        data = pd.DataFrame()
        test = pd.DataFrame()
        data[0] = tmp_data[0]
        data[1] = tmp_data[1].fillna("") + tmp_data[2].fillna("") + tmp_data[3].fillna("")
        test[0] = tmp_test[0]
        test[1] = tmp_test[1].fillna("") + tmp_test[2].fillna("") + tmp_test[3].fillna("")
        # data[0] = tmp_data[0]
        # data[1] = tmp_data[3]
        # test[0] = tmp_test[0]
        # test[1] = tmp_test[3]
        # data = data.dropna().reset_index(drop=True)
        # test = test.dropna().reset_index(drop=True)
        label, content = 0, 1
    elif sys.argv[1] == "yelpf":
        data = pd.read_csv('./datasets/yelpf/train.csv', header=None)
        test = pd.read_csv('./datasets/yelpf/test.csv', header=None)
        label, content = 0, 1
    elif sys.argv[1] == "yelpp":
        data = pd.read_csv('./datasets/yelpp/train.csv', header=None)
        test = pd.read_csv('./datasets/yelpp/test.csv', header=None)
        label, content = 0, 1


    words = []
    for i in range(len(data[content])):
        text = clean_str(data[content][i])
        words.extend(text.split())

    corpus = freq(words)
    w2id, id2w, corpus_cnt = preprocess(corpus)
    contexts, targets = create_contexts_target(data, label, content)
    test_contexts, test_targets = create_contexts_target(test, label, content)

    if sys.argv[1] == "ag":
        with open("./preprocessed_data/ag.pkl", "wb") as f:
            pickle.dump([w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets], f, -1)
    elif sys.argv[1] == "amzf":
        with open("./preprocessed_data/amzf.pkl", "wb") as f:
            pickle.dump([w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets], f, -1)
    elif sys.argv[1] == "amzp":
        with open("./preprocessed_data/amzp.pkl", "wb") as f:
            pickle.dump([w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets], f, -1)
    elif sys.argv[1] == "dbp":
        with open("./preprocessed_data/dbp.pkl", "wb") as f:
            pickle.dump([w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets], f, -1)
    elif sys.argv[1] == "sogou":
        with open("./preprocessed_data/sogou.pkl", "wb") as f:
            pickle.dump([w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets], f, -1)
    elif sys.argv[1] == "yaha":
        with open("./preprocessed_data/yaha.pkl", "wb") as f:
            pickle.dump([w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets], f, -1)
    elif sys.argv[1] == "yelpf":
        with open("./preprocessed_data/yelpf.pkl", "wb") as f:
            pickle.dump([w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets], f, -1)
    elif sys.argv[1] == "yelpp":
        with open("./preprocessed_data/yelpp.pkl", "wb") as f:
            pickle.dump([w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets], f, -1)