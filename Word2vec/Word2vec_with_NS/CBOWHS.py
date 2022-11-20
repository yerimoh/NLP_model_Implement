import pickle, os
from tqdm.auto import tqdm
import numpy as np
from preprocess import *

def train(W_in, W_h, paths, bin_paths, epoch=0):
    lr = 0.5
    decay = lr / 100
    loss = 0

    for file in tqdm(os.listdir("./data/")[epoch:], desc="Reading Data"):
        with open("./data/"+file, 'r') as f:
            data = f.read()
        data = unicodeToAscii(data)
        data = clean_str(data).strip().split()
        output = [w2id[word] for word in data if word in w2id]

        targets = output[5:len(output)-5]
        contexts = []
        
        for idx in tqdm(range(5, len(output)-5), desc="createing contexts and targets"):
            contexts.append(output[idx-5:idx] + output[idx+1:idx+6])

        contexts, targets = np.array(contexts), np.array(targets)

        for i in tqdm(range(len(targets)), desc="training"):
            t, c = targets[i], contexts[i]
            h = np.mean(W_in[c], axis=0)
            
            score = np.dot(h, W_h[paths[t]].T).flatten()
            y = 1 / (1 + np.exp(-score))
            # y = np.zeros_like(score)
            # for k in range(len(score)):
            #     if score[k] >= 0:
            #         y[k] = 1.0 / (1.0 + np.exp(-score[k]))
            #     else:
            #         y[k] = np.exp(score[k]) / (np.exp(score[k]) + 1)
            tmp = np.c_[1-y, y]
            
            batch_size = y.shape[0]
            loss += -np.sum(np.log(tmp[np.arange(batch_size), bin_paths[t]] + 1e-7)) / batch_size

            dout = (y - bin_paths[t]) / batch_size
            dout = dout.reshape(1, -1)
            
            h = h.reshape(1, -1)
            dW_tmp = np.dot(h.T, dout).T
            dh = np.dot(dout, W_h[paths[t]]).reshape(-1)

            np.add.at(W_h, paths[t], -lr*dW_tmp)
            W_in[c] -= lr*dh

            if i % 100000 == 0:
                print(loss/100000)
                loss = 0

        lr -= decay
        para = {}
        para["word_vecs"] = W_in
        para["word_to_id"] = w2id
        para["id_to_word"] = id2w
        para["epoch"] = epoch
        para["huffman"] = W_h
        with open("./params/HS_cbow300_temp_params.pkl", "wb") as f:
            pickle.dump(para, f, -1)

        epoch += 1
    return W_in

if __name__ == "__main__":
    np.random.seed(0)
    with open('./preprocessed_data/hs_data', 'rb') as f:
        heap, paths, bin_paths = pickle.load(f)

    with open('./preprocessed_data/preprocessed_corpus', 'rb') as f:
        corpus, w2id, id2w, counter = pickle.load(f)
    
    print("Data Loaded!")

    try:    
        with open(f"./params/HS_cbow300_temp_params.pkl", "rb") as f:
            para = pickle.load(f)
        W_in = para["word_vecs"]
        W_h = para["huffman"]
        epoch = para["epoch"]+1
    except:
        hidden_size = 300
        W_in = (np.random.randn(len(w2id), hidden_size) * 0.01).astype('float128')
        W_h = (np.random.randn(len(w2id)-1, hidden_size) * 0.01).astype('float128')
        epoch = 0
        print("Params Initialized!")

    print(f"resume trainig at {epoch}!")

    W_in = train(W_in, W_h, paths, bin_paths, epoch)
    para = {}
    para["word_vecs"] = W_in
    para["word_to_id"] = w2id
    para["id_to_word"] = id2w
    with open("./params/HS_cbow300_final_params.pkl", "wb") as f:
        pickle.dump(para, f, -1)