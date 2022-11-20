import pickle
import numpy as np
from optimizers import Adam

class HierarchicalSoftmax:
    def __init__(self, W, paths):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.paths = paths
        self.cache = None

    def forward(self, h, t):
        W, = self.params
        out = np.dot(h, W[self.paths[t]].T).flatten()
        self.cache = (h, out, t)
        return out

    def backward(self, dout):
        W, = self.params
        h, out, t = self.cache
        
        dout = dout.reshape(1, -1)
        h = h.reshape(1, -1)
        dW = np.zeros_like(W).astype('f')
        dW_tmp = np.dot(h.T, dout).T
        np.add.at(dW, self.paths[t], dW_tmp)
        self.grads[0][...] = dW

        dx = np.dot(dout, W[self.paths[t]])
        return dx


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        self.idx = idx
        W, = self.params
        return W[idx]

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        dW[self.idx] = dout.reshape(-1)
        # np.add.at(dW, self.idx, dout)
        return None
        

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, x, t):
        y = 1 / (1 + np.exp(-x))

        tmp = np.c_[1-y, y]

        if tmp.ndim == 1:
            tmp = tmp.reshape(1, -1)
            t = t.reshape(1, -1)

        # if tmp.size == t.size:
            # t = np.argmax(t, axis=1)

        batch_size = y.shape[0]
        loss = -np.sum(np.log(tmp[np.arange(batch_size), t] + 1e-7)) / batch_size
        self.cache = (y, t, batch_size)
        return loss

    def backward(self, dout=1):
        y, t, batch_size = self.cache
        dout = (y - t) / batch_size

        return dout

if __name__ == "__main__":

    np.random.seed(0)
    with open('./preprocessed_data/hs_data', 'rb') as f:
        heap, paths, bin_paths = pickle.load(f)

    h = (np.random.randn(1, 300)*0.01).astype('f')
    HS = (np.random.randn(1000000, 300)*0.01).astype('f')

    tmp = HierarchicalSoftmax(HS, paths)
    loss_layer = SigmoidWithLoss()
    
    optimizer = Adam()
    for _ in range(10000):
        out = tmp.forward(h, 1)
        loss = loss_layer.forward(out, np.array(bin_paths[1]))
        print(loss)
        dout = loss_layer.backward()
        dout = tmp.backward(dout)
        optimizer.update(tmp.params, tmp.grads)