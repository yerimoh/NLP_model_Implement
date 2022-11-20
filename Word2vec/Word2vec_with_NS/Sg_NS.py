import pickle
from tqdm.auto import tqdm
from collections import Counter
import numpy as np
from optimizers import AdaDelta


def get_unigram(corpus):
    counter = Counter(corpus)
    counter[0] = 0

    vocab_size = len(counter)
    p_list = np.zeros(vocab_size)
    for i in range(vocab_size):
        p_list[i] = counter[i]

    p_list = np.power(p_list, 0.75)
    p_list /= np.sum(p_list)
    return p_list


def train(W_in, W_out, targets, contexts, p_list):
    vocab_size = W_out.shape[0]
    pad = [0]
    lr = 0.05
    decay = lr / len(targets)
    
    loss = 0
    for i in tqdm(range(len(targets))):
        t, c_ = targets[i], contexts[i]
        c = np.setdiff1d(c_, pad)
        h = W_in[t]
        negative_sample = np.random.choice(vocab_size, size=5*len(c), replace=True, p=p_list)
        context = np.append(c, negative_sample)
        score = np.dot(h, W_out[context].T).flatten()
        label = np.append(np.ones(len(c)), np.zeros(len(negative_sample))).astype(np.int)

        y = 1 / (1 + np.exp(-score))
        # y = np.zeros_like(score)
        # for k in range(len(score)):
        #     if score[k] >= 0:
        #         y[k] = 1.0 / (1.0 + np.exp(-score[k]))
        #     else:
        #         y[k] = np.exp(score[k]) / (np.exp(score[k]) + 1)

        tmp = np.c_[1-y, y]
        batch_size = y.shape[0]
        loss += -np.sum(np.log(tmp[np.arange(batch_size), label] + 1e-7)) / batch_size

        dout = (y - label) / batch_size
        dout = dout.reshape(1, -1)
        h = h.reshape(1, -1)
        
        dW_tmp = np.dot(h.T, dout).T
        dx = np.dot(dout, W_out[context])
        
        np.add.at(W_out, context, -lr * dW_tmp)
        W_in[t] -= lr * dx.reshape(-1)

        if i % 10000 == 0:
            print(loss/10000)
            loss = 0

        if i % 5000000 == 0:
            try:
                para = {}
                para["word_vecs"] = W_in#.astype(np.float16)
                para["word_to_id"] = w2id
                para["id_to_word"] = id2w
                with open("NS_skipgram_temp_params.pkl", "wb") as f:
                    pickle.dump(para, f, -1)
            except OSError:
                continue
        lr -= decay
    return W_in


if __name__ == "__main__":
    with open('./preprocessed_data/data_subsampled', 'rb') as f:
        contexts, targets, corpus, w2id, id2w = pickle.load(f)
    
    W_in = (np.random.randn(len(w2id), 300) * 0.01).astype(np.float128)
    W_out = (np.random.randn(len(w2id), 300) * 0.01).astype(np.float128)
    p_list = get_unigram(corpus)
    
    W_in = train(W_in, W_out, targets, contexts, p_list)
    para = {}
    para["word_vecs"] = W_in#.astype(np.float16)
    para["word_to_id"] = w2id
    para["id_to_word"] = id2w
    with open("NS_skipgram_final_params.pkl", "wb") as f:
        pickle.dump(para, f, -1)
