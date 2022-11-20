import pickle
from tqdm.auto import tqdm
from collections import Counter
import numpy as np
from optimizers import AdaDelta

class NegativeSampler:
    def __init__(self, corpus):
        counter = Counter(corpus)

        self.vocab_size = len(counter)
        self.p = np.zeros(self.vocab_size)
        for i in range(self.vocab_size):
            self.p[i] = counter[i]

        self.p = np.power(self.p, 0.75)
        self.p /= np.sum(self.p)

    def get_negative_sample(self, sample_size):
        negative_sample = np.random.choice(self.vocab_size, size=(100, sample_size), replace=True, p=self.p)
        return negative_sample

def train(W_in, W_out, targets, contexts, sampler, batch_size=100):
    sampler = NegativeSampler(corpus)
    max_iter = len(targets)//batch_size
    correct_label = np.ones(batch_size, dtype=np.int8)
    negative_label = np.zeros(batch_size, dtype=np.int8)
    
    lr = 0.025
    decay = lr / max_iter
    loss = 0
    
    for i in tqdm(range(max_iter)):
        lr -= decay
        t = targets[i*batch_size : (i+1)*batch_size]
        c = contexts[i*batch_size : (i+1)*batch_size]
        h = W_in[t]
        
        for window in range(2*5):
            ns = sampler.get_negative_sample(5)
            target_W = W_out[c[:, window]]
            score = np.sum(target_W*h, axis=1)
            y = 1 / (1 + np.exp(-score))
            tmp = np.c_[1-y, y]
            loss += -np.sum(np.log(tmp[np.arange(batch_size), correct_label] + 1e-7)) / batch_size

            #backward
            dout = (y - correct_label) / batch_size
            dout = dout.reshape(dout.shape[0], 1)
            dtarget_W = -lr * (dout * h)
            np.add.at(W_out, c[:, window], dtarget_W)
            dh = dout * target_W

            
            for neg in range(5):
                neg_target_W = W_out[ns[:, neg]]
                neg_score = np.sum(neg_target_W*h, axis=1)
                neg_y = 1 / (1 + np.exp(-neg_score))
                neg_tmp = np.c_[1-neg_y, neg_y]
                loss += -np.sum(np.log(neg_tmp[np.arange(batch_size), negative_label] + 1e-7)) / batch_size

                #backward
                neg_dout = (neg_y - negative_label) / batch_size
                neg_dout = neg_dout.reshape(neg_dout.shape[0], 1)
                neg_dtarget_W = -lr * (neg_dout * h)
                np.add.at(W_out, ns[:, neg], neg_dtarget_W)
                dh += neg_dout * neg_target_W

        np.add.at(W_in, t, -lr * dh)
            
        if (i % 100 == 0):
            print(loss)
            loss = 0

        if (i % 100000 == 0) and (i > 1):
            para = {}
            para["word_vecs"] = W_in.astype(np.float16)
            para["word_to_id"] = w2id
            para["id_to_word"] = id2w
            with open("skipgram_temp_params.pkl", "wb") as f:
                pickle.dump(para, f, -1)

    return W_in




if __name__ == "__main__":
    with open('./preprocessed_data/data_subsampled', 'rb') as f:
        contexts, targets, corpus, w2id, id2w = pickle.load(f)


    W_in = (np.random.randn(len(w2id), 300) * 0.01).astype('float16')
    W_out = (np.random.randn(len(w2id), 300) * 0.01).astype('float16')
    sampler = NegativeSampler(corpus)
    embedding_matrix = train(W_in, W_out, targets, contexts, sampler)
    para = {}
    para["word_vecs"] = embedding_matrix.astype(np.float16)
    para["word_to_id"] = w2id
    para["id_to_word"] = id2w
    with open("skipgram_final_params.pkl", "wb") as f:
        pickle.dump(para, f, -1)
