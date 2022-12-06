import pickle
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def cos_similarity(a, b, matrix, sub_matrix, w2id, sub_w2id):
    # a, b = "<"+a+">", "<"+b+">"
    if a in w2id.keys():
        vec1 = matrix[w2id[a]]
    else:
        vec1 = np.zeros((300))
        for i in range(3, 7):
            for j in range(len(a)-i+1):
                char_n = a[j:j+i]
                if char_n in sub_w2id.keys():
                    vec1 += sub_matrix[sub_w2id[char_n]]

    if b in w2id.keys():
        vec2 = matrix[w2id[b]]
    else:
        vec2 = np.zeros((300))
        for i in range(3, 7):
            for j in range(len(b)-i+1):
                char_n = b[j:j+i]
                if char_n in sub_w2id.keys():
                    vec2 += sub_matrix[sub_w2id[char_n]]
    
    vec1 /= np.linalg.norm(vec1)
    vec2 /= np.linalg.norm(vec2)

    return round(np.dot(vec1, vec2), 2)

def w2v_cos_similarity(a, b, matrix, w2id):
    if (a not in w2id.keys()) or (b not in w2id.keys()):
        return 0

    else:
        vec1 = matrix[w2id[a]]
        vec2 = matrix[w2id[b]]
    return np.dot(vec1, vec2)


def most_similar(query, w2id, id2w, word_matrix, top=10):
    # query = "<"+query+">"
    query_vec = word_matrix[w2id[query]]

    similarity = np.dot(query_vec, word_matrix.T)
    similarity = (-1 * similarity).argsort()
    
    words = []
    for i in range(top):
        word = id2w[similarity[i]]
        words.append(word)
        
    return words

def analogy(a, b, c, word_to_id, id_to_word, word_matrix, sub_matrix, sub_w2id, top=4):

    if a not in word_to_id.keys():
        a_vec = np.zeros((300))
        for i in range(3, 7):
            for j in range(len(a)-i+1):
                char_n = a[j:j+i]
                if char_n in sub_w2id.keys():
                    a_vec += sub_matrix[sub_w2id[char_n]]
        a_vec /= np.linalg.norm(a_vec)
    else:
        a_vec = word_matrix[word_to_id[a]]
    
    if b not in word_to_id.keys():
        b_vec = np.zeros((300))
        for i in range(3, 7):
            for j in range(len(b)-i+1):
                char_n = b[j:j+i]
                if char_n in sub_w2id.keys():
                    b_vec += sub_matrix[sub_w2id[char_n]]
        b_vec /= np.linalg.norm(b_vec)
    else:
        b_vec = word_matrix[word_to_id[b]]

    if c not in word_to_id.keys():
        c_vec = np.zeros((300))
        for i in range(3, 7):
            for j in range(len(c)-i+1):
                char_n = c[j:j+i]
                if char_n in sub_w2id.keys():
                    c_vec += sub_matrix[sub_w2id[char_n]]
        c_vec /= np.linalg.norm(c_vec)
    else:
        c_vec = word_matrix[word_to_id[c]]

    # a_vec, b_vec, c_vec = word_matrix[word_to_id[a]], word_matrix[word_to_id[b]], word_matrix[word_to_id[c]]
    query_vec = b_vec - a_vec + c_vec

    similarity = np.dot(query_vec, word_matrix.T)
    similarity = (-1 * similarity).argsort()
    
    words = []
    for i in range(top):
        word = id_to_word[similarity[i]]
        words.append(word)
        
    return words
