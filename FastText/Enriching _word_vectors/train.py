import os, sys, pickle, time
import random
from collections import Counter

import numpy as np

np.random.seed(0)

if __name__ == "__main__":
    with open('../../word2vec/preprocessed_data/preprocessed_corpus', 'rb') as f:
        corpus, w2id, id2w, counter = pickle.load(f)
    with open("./preprocessed_data/dicts", "rb") as f:
        sub_w2id, sub_id2w, char_dict = pickle.load(f)
    with open("./preprocessed_data/w2v/plist", "rb") as f:
        p_list = pickle.load(f)

    try:
        with open("temp_params", "rb") as f:
            para = pickle.load(f)
        W_in = para["word_vecs"]
        sub_W = para["sub_vecs"]
        epoch = para["epoch"]+1
    except:
        W_in = (np.random.randn(len(w2id), 300) * 0.01).astype(np.float128)
        sub_W = (np.random.randn(len(sub_w2id), 300) * 0.01).astype(np.float128)
        W_out = (np.random.randn(len(w2id), 300) * 0.01).astype(np.float128)
        epoch = 0
    
    print(f"loaded at epoch {epoch}")
    
    vocab_size = W_in.shape[0]
    
    pad = [0]
    lr = 0.5
    decay = lr / (100-epoch)

    lr -= decay*epoch

    start_time = time.time()

    for file in os.listdir("../../word2vec/data/")[epoch:]:
        loss = 0
        with open("./preprocessed_data/w2v/"+file+"_data", "rb") as f:
            contexts, targets = pickle.load(f)
        
        for i in range(len(targets)):
            t, c_ = targets[i], contexts[i]
            c = np.setdiff1d(c_, pad)

            h1 = W_in[t] 
            h2 = sub_W[char_dict[t]]

            h = np.sum(np.vstack((h1, h2)), axis=0)
            
            negative_sample = np.random.choice(vocab_size, size=5*len(c), replace=True, p=p_list)
            context = np.append(c, negative_sample)
            
            score = np.dot(h, W_out[context].T).flatten()
            
            label = np.append(np.ones(len(c)), np.zeros(len(negative_sample))).astype(np.int)
            y = 1 / (1 + np.exp(-score))

            tmp = np.c_[1-y, y]
            batch_size = y.shape[0]
            loss += -np.sum(np.log(tmp[np.arange(batch_size), label] + 1e-7)) / batch_size

            dout = (y - label) / batch_size
            
            dout = dout.reshape(1, -1)

            h = h.reshape(1, -1)

            dW_tmp = np.dot(h.T, dout).T

            dx = np.dot(dout, W_out[context])

            dx /= len(context)
            dx = dx.reshape(-1)

            np.add.at(W_out, context, -lr * dW_tmp)

            W_in[t] -= lr * dx
            np.add.at(sub_W, char_dict[t], -lr*dx)

            if i % 100000 == 0:
                print(f"Epoch: {epoch} || iter: {i} / {len(targets)}")
                print("Time Elapsed: " + str(int(time.time() - start_time)) + " sec")
                if i > 0:
                    print(round(loss / 100000, 5))
                    loss = 0
                    
        lr -= decay
        epoch += 1

        try:
            para = {}
            para["word_vecs"] = W_in
            para["sub_vecs"] = sub_W
            para["epoch"] = epoch
            with open("temp_params.pkl", "wb") as f:
                pickle.dump(para, f, -1)
        except:
            continue

    para = {}
    para["word_vecs"] = W_in
    para["sub_vecs"] = sub_W
    with open("final_params.pkl", "wb") as f:
        pickle.dump(para, f, -1)