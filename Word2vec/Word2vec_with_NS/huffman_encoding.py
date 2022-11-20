import pickle
import numpy as np
import heapq

with open('./preprocessed_data/preprocessed_corpus', 'rb') as f:
    corpus, w2id, id2w, counter = pickle.load(f)


def huffman_encoding(counter, w2id):
    heap = [[count, w2id[word]] for word, count in counter.items()]
    heapq.heapify(heap)

    idx = 0
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        tmp = [ a[0]+b[0], idx, a, b]
        idx += 1
        heapq.heappush(heap, tmp)
    return heap

def mover(heap, path, path_counter):
    """node path"""
    if len(heap) > 2:
        mover(heap[2], path + " " + str(heap[1]), (path_counter << 1))
        mover(heap[3], path + " " + str(heap[1]), (path_counter << 1) + 0b1)
    elif len(heap) <= 2:
        paths[heap[1]] = [int(i) for i in path.split()]
        tmp = bin(int(path_counter)).replace("0b1", "")
        bin_paths[heap[1]] = [int(i) for i in tmp]

if __name__ == "__main__":        
    paths = [0]*(len(w2id)+1)
    bin_paths = [0]*(len(w2id)+1)
    heap = huffman_encoding(counter, w2id)[0]
    mover(heap, "", 0b1)
    hs_paths = [heap, paths, bin_paths]
    pickle.dump(hs_paths, open("./preprocessed_data/hs_data", 'wb'), protocol=-1)