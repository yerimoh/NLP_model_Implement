import numpy as np
import time, json, csv, os, sys, pickle
import random
import pandas as pd
import heapq
from collections import Counter
from tqdm.auto import tqdm

def eval(contexts, targets):
    acc = 0
    cnt = 0

    for i in tqdm(range(len(targets))):
        try:
            t, c_ = targets[i]-1, contexts[i]
            c = np.array([tmp for tmp in c_ if tmp != 0])
            h = np.mean(W_in[c], axis=0)
            score = np.dot(h, W_affine) + b_affine
            pred = np.argmax(score)
            if pred == t:
                acc += 1
            cnt += 1
        except IndexError:
            continue

    print(acc / cnt) 

if __name__ == "__main__":
    if sys.argv[2] == "uni":
        if sys.argv[1] == "ag":
            with open('./preprocessed_data/ag.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)
            with open('ag_params.pkl', 'rb') as f:
                W_in, W_affine, b_affine = pickle.load(f)
        elif sys.argv[1] == "amzf":
            with open('./preprocessed_data/amzf.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)
            with open('amzf_params.pkl', 'rb') as f:
                W_in, W_affine, b_affine = pickle.load(f)
        elif sys.argv[1] == "amzp":
            with open('./preprocessed_data/amzp.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)
            with open('amzp_params.pkl', 'rb') as f:
                W_in, W_affine, b_affine = pickle.load(f)
        elif sys.argv[1] == "dbp":
            with open('./preprocessed_data/dbp.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)
            with open('dbp_params.pkl', 'rb') as f:
                W_in, W_affine, b_affine = pickle.load(f)
        elif sys.argv[1] == "sogou":
            with open('./preprocessed_data/sogou.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)
            with open('sogou_params.pkl', 'rb') as f:
                W_in, W_affine, b_affine = pickle.load(f)
        elif sys.argv[1] == "yaha":
            with open('./preprocessed_data/yaha.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)
            with open('yaha_params.pkl', 'rb') as f:
                W_in, W_affine, b_affine = pickle.load(f)
        elif sys.argv[1] == "yelpf":
            with open('./preprocessed_data/yelpf.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)
            with open('yelpf_params.pkl', 'rb') as f:
                W_in, W_affine, b_affine = pickle.load(f)
        elif sys.argv[1] == "yelpp":
            with open('./preprocessed_data/yelpp.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)
            with open('yelpp_params.pkl', 'rb') as f:
                W_in, W_affine, b_affine = pickle.load(f)

    elif sys.argv[2] == "bi":
        if sys.argv[1] == "ag":
            with open('./preprocessed_data/ag_bigram.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)
            with open('ag_bigram_params.pkl', 'rb') as f:
                W_in, W_affine, b_affine = pickle.load(f)
        elif sys.argv[1] == "amzf":
            with open('./preprocessed_data/amzf_bigram.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)
            with open('amzf_bigram_params.pkl', 'rb') as f:
                W_in, W_affine, b_affine = pickle.load(f)
        elif sys.argv[1] == "amzp":
            with open('./preprocessed_data/amzp_bigram.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)
            with open('amzp_bigram_params.pkl', 'rb') as f:
                W_in, W_affine, b_affine = pickle.load(f)
        elif sys.argv[1] == "dbp":
            with open('./preprocessed_data/dbp_bigram.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)
            with open('dbp_bigram_params.pkl', 'rb') as f:
                W_in, W_affine, b_affine = pickle.load(f)
        elif sys.argv[1] == "sogou":
            with open('./preprocessed_data/sogou_bigram.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)
            with open('sogou_bigram_params.pkl', 'rb') as f:
                W_in, W_affine, b_affine = pickle.load(f)
        elif sys.argv[1] == "yaha":
            with open('./preprocessed_data/yaha_bigram.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)
            with open('yaha_bigram_params.pkl', 'rb') as f:
                W_in, W_affine, b_affine = pickle.load(f)
        elif sys.argv[1] == "yelpf":
            with open('./preprocessed_data/yelpf_bigram.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)
            with open('yelpf_bigram_params.pkl', 'rb') as f:
                W_in, W_affine, b_affine = pickle.load(f)
        elif sys.argv[1] == "yelpp":
            with open('./preprocessed_data/yelpp_bigram.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)
            with open('yelpp_bigram_params.pkl', 'rb') as f:
                W_in, W_affine, b_affine = pickle.load(f)

    eval(test_contexts, test_targets)