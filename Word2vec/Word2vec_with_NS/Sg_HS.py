import pickle
# import warnings
from tqdm.auto import tqdm
import numpy as np
# warnings.filterwarnings("error")

def train(W_in, W_h, paths, bin_paths, targets, contexts):
    loss = 0
    lr = 0.05
    decay = lr / len(targets)
    pad = [0]
    for i in tqdm(range(len(targets))):
        t, c_ = targets[i], contexts[i]
        c = np.setdiff1d(c_, pad)
        h = W_in[t]
        dh = np.zeros_like(h)
        for j in range(len(c)):
            score = np.dot(h, W_h[paths[c[j]]].T).flatten()
            # score = np.clip(score, -10.0, 20.0)
            # y = np.zeros_like(score)
            # for k in range(len(score)):
                # if score[k] >= 0:
                    # y[k] = 1.0 / (1.0 + np.exp(-score[k]))
                # else:
                    # y[k] = np.exp(score[k]) / (np.exp(score[k]) + 1)
            y = 1 / (1 + np.exp(-score))
            
            tmp = np.c_[1-y, y]
            batch_size = y.shape[0]
            loss += -np.sum(np.log(tmp[np.arange(batch_size), bin_paths[c[j]]] + 1e-7)) / batch_size

            dout = (y - bin_paths[c[j]]) / batch_size
            dout = dout.reshape(1, -1)
            h = h.reshape(1, -1)
            dW_tmp = np.dot(h.T, dout).T
            dh += np.dot(dout, W_h[paths[c[j]]]).reshape(-1) / len(c)
            
            np.add.at(W_h, paths[c[j]], -lr * dW_tmp)
        
        W_in[t] -= lr * dh

        if i % 10000 == 0:
            print(loss/10000)
            loss = 0

        if i % 5000000 == 0:
            para = {}
            para["word_vecs"] = W_in#.astype(np.float16)
            para["word_to_id"] = w2id
            para["id_to_word"] = id2w
            with open("HS_skipgram_temp_params.pkl", "wb") as f:
                pickle.dump(para, f, -1)
        lr -= decay

    return W_in


if __name__ == "__main__":
    np.random.seed(0)
    with open('./preprocessed_data/hs_data', 'rb') as f:
        heap, paths, bin_paths = pickle.load(f)

    with open('./preprocessed_data/data_subsampled', 'rb') as f:
        contexts, targets, corpus, w2id, id2w = pickle.load(f)

    window_size = 5
    hidden_size = 300
    W_in = (np.random.randn(len(w2id), hidden_size) * 0.01).astype('float128')
    W_h = (np.random.randn(len(w2id)-1, hidden_size) * 0.01).astype('float128')
    
    W_in = train(W_in, W_h, paths, bin_paths, targets, contexts)
    para = {}
    para["word_vecs"] = W_in#.astype(np.float16)
    para["word_to_id"] = w2id
    para["id_to_word"] = id2w
    with open("HS_skipgram_final_params.pkl", "wb") as f:
        pickle.dump(para, f, -1)
