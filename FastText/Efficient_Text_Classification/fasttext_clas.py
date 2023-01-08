import numpy as np
import time, json, csv, os, sys, pickle
import random
import pandas as pd
import heapq
from collections import Counter
from tqdm.auto import tqdm

def train(contexts, targets, epoch=5):
    num_classes = len(set(targets))
    W_in = (np.random.randn(len(w2id), 10) * 0.01).astype("float128")
    W_affine = (np.random.randn(10, num_classes) * 0.01).astype("float128")
    b_affine = np.zeros(num_classes)
    shuffle = np.random.permutation(len(targets))
    contexts, targets = contexts[shuffle], targets[shuffle]

    loss = 0
    lr = 0.25
    decay = lr / (5*len(targets))

    for e in range(5):
        for i in tqdm(range(len(targets))):
            try:
                t, c_ = targets[i]-1, contexts[i]
                c = np.array([tmp for tmp in c_ if tmp != 0])
                h = np.mean(W_in[c], axis=0)
                score = np.dot(h, W_affine) + b_affine
                
                score -= np.max(score)
                y = np.exp(score) / np.sum(np.exp(score))
                y = y.reshape(1, -1)
                t = t.reshape(1, -1)
                loss += -np.sum(np.log(y[0, t] +1e-7))

                dout = y.copy()
                dout[0, t] -= 1
                db = np.sum(dout, axis=0)
                dW = np.dot(h.reshape(1, -1).T, dout)

                W_affine -= lr * dW
                b_affine -= lr * db

                dh = np.dot(dout, W_affine.T) / len(c)
                dh *= -lr
                np.add.at(W_in, c, dh)

                if (i % 10000) == 0:
                    print(loss)
                    loss = 0
                lr -= decay
            except IndexError:
                continue
    return W_in, W_affine, b_affine
    

if __name__ == "__main__":
    if sys.argv[2] == "uni":
        if sys.argv[1] == "ag":
            with open('./preprocessed_data/ag.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)

            W_in, W_affine, b_affine = train(contexts, targets)
            with open('ag_params.pkl', 'wb') as f:
                pickle.dump([W_in, W_affine, b_affine], f, -1)

        elif sys.argv[1] == "amzf":
            with open('./preprocessed_data/amzf.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)

            W_in, W_affine, b_affine = train(contexts, targets)
            with open('amzf_params.pkl', 'wb') as f:
                pickle.dump([W_in, W_affine, b_affine], f, -1)

        elif sys.argv[1] == "amzp":
            with open('./preprocessed_data/amzp.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)

            W_in, W_affine, b_affine = train(contexts, targets)
            with open('amzp_params.pkl', 'wb') as f:
                pickle.dump([W_in, W_affine, b_affine], f, -1)

        elif sys.argv[1] == "dbp":
            with open('./preprocessed_data/dbp.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)

            W_in, W_affine, b_affine = train(contexts, targets)
            with open('dbp_params.pkl', 'wb') as f:
                pickle.dump([W_in, W_affine, b_affine], f, -1)

        elif sys.argv[1] == "sogou":
            with open('./preprocessed_data/sogou.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)

            W_in, W_affine, b_affine = train(contexts, targets)
            with open('sogou_params.pkl', 'wb') as f:
                pickle.dump([W_in, W_affine, b_affine], f, -1)
        
        elif sys.argv[1] == "yaha":
            with open('./preprocessed_data/yaha.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)

            W_in, W_affine, b_affine = train(contexts, targets)
            with open('yaha_params.pkl', 'wb') as f:
                pickle.dump([W_in, W_affine, b_affine], f, -1)

        elif sys.argv[1] == "yelpf":
            with open('./preprocessed_data/yelpf.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)

            W_in, W_affine, b_affine = train(contexts, targets)
            with open('yelpf_params.pkl', 'wb') as f:
                pickle.dump([W_in, W_affine, b_affine], f, -1)

        elif sys.argv[1] == "yelpp":
            with open('./preprocessed_data/yelpp.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)

            W_in, W_affine, b_affine = train(contexts, targets)
            with open('yelpp_params.pkl', 'wb') as f:
                pickle.dump([W_in, W_affine, b_affine], f, -1)

    elif sys.argv[2] == "bi":
        if sys.argv[1] == "ag":
            with open('./preprocessed_data/ag_bigram.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)

            W_in, W_affine, b_affine = train(contexts, targets)
            with open('ag_bigram_params.pkl', 'wb') as f:
                pickle.dump([W_in, W_affine, b_affine], f, -1)

        elif sys.argv[1] == "amzf":
            with open('./preprocessed_data/amzf_bigram.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)

            W_in, W_affine, b_affine = train(contexts, targets)
            with open('amzf_bigram_params.pkl', 'wb') as f:
                pickle.dump([W_in, W_affine, b_affine], f, -1)

        elif sys.argv[1] == "amzp":
            with open('./preprocessed_data/amzp_bigram.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)

            W_in, W_affine, b_affine = train(contexts, targets)
            with open('amzp_bigram_params.pkl', 'wb') as f:
                pickle.dump([W_in, W_affine, b_affine], f, -1)

        elif sys.argv[1] == "dbp":
            with open('./preprocessed_data/dbp_bigram.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)

            W_in, W_affine, b_affine = train(contexts, targets)
            with open('dbp_bigram_params.pkl', 'wb') as f:
                pickle.dump([W_in, W_affine, b_affine], f, -1)

        elif sys.argv[1] == "sogou":
            with open('./preprocessed_data/sogou_bigram.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)

            W_in, W_affine, b_affine = train(contexts, targets)
            with open('sogou_bigram_params.pkl', 'wb') as f:
                pickle.dump([W_in, W_affine, b_affine], f, -1)
        
        elif sys.argv[1] == "yaha":
            with open('./preprocessed_data/yaha_bigram.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)

            W_in, W_affine, b_affine = train(contexts, targets)
            with open('yaha_bigram_params.pkl', 'wb') as f:
                pickle.dump([W_in, W_affine, b_affine], f, -1)

        elif sys.argv[1] == "yelpf":
            with open('./preprocessed_data/yelpf_bigram.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)

            W_in, W_affine, b_affine = train(contexts, targets)
            with open('yelpf_bigram_params.pkl', 'wb') as f:
                pickle.dump([W_in, W_affine, b_affine], f, -1)

        elif sys.argv[1] == "yelpp":
            with open('./preprocessed_data/yelpp_bigram.pkl', 'rb') as f:
                w2id, id2w, corpus_cnt, contexts, targets, test_contexts, test_targets = pickle.load(f)

            W_in, W_affine, b_affine = train(contexts, targets)
            with open('yelpp_bigram_params.pkl', 'wb') as f:
                pickle.dump([W_in, W_affine, b_affine], f, -1)