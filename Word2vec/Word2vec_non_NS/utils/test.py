import numpy as np
from collections import Counter

def cos_similarity(vec1, matrix):
    return np.dot(vec1, matrix) / (np.linalg.norm(vec1) * np.linalg.norm(matrix, axis=1))


def analogy(a, b, c, vocab,  word_matrix, matrix_norm):
    for word in (a, b, c):
        if word not in vocab:
            return None
    
    a_vec, b_vec, c_vec = word_matrix[vocab[a]], word_matrix[vocab[b]], word_matrix[vocab[c]]
    query_vec = b_vec - a_vec + c_vec

    similarity = np.dot(query_vec, word_matrix.T) #/ matrix_norm
    similarity = (-1 * similarity).argsort()
    
    for i in range(40):
        word = vocab.lookup_token(similarity[i])
        if word not in (a, b, c):
            return word

