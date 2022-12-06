import sys
import pickle
import numpy as np
from tqdm.auto import tqdm
import fasttext
from utils import *

if __name__ == "__main__":
    words = []
    human = []
    ai = []
    if sys.argv[2] == "rw":
        with open('./data/rw/rw.txt', "r") as f:
            for line in f:
                items = line.split()
                words.append(items[:3])
    elif sys.argv[2] == "ws353":
        with open("./data/ws353/wordsim_similarity_goldstandard.txt", "r") as f:
            for line in f:
                items = line.split()
                words.append(items[:3])
        
    if sys.argv[1] == "w2v":
        with open('/hdd1/user22/word2vec/NS_skipgram_final_params.pkl', "rb") as f:
            params = pickle.load(f)
        embed_matrix = params["word_vecs"]
        w2id = params["word_to_id"]
        id2w = params["id_to_word"]
        norm = np.linalg.norm(embed_matrix, axis=1)
        embed_matrix /= norm[:, np.newaxis]
        for a, b, c in words:
            similarity = round(w2v_cos_similarity(a, b, embed_matrix, w2id)*10, 2)
            ai.append(similarity)
            human.append(float(c))
            print(similarity, float(c))
            
        cor = np.corrcoef(ai, human)
        print(cor[0, 1])

    elif sys.argv[1] == "fasttext":
        with open('../../word2vec/preprocessed_data/preprocessed_corpus', 'rb') as f:
            corpus, w2id, id2w, counter = pickle.load(f)
        with open("./preprocessed_data/dicts", "rb") as f:
            sub_w2id, sub_id2w, char_dict = pickle.load(f)
        with open("temp_params.pkl", "rb") as f:
            params = pickle.load(f)

        embed_matrix = params["word_vecs"]
        subword_matrix = params["sub_vecs"]

        full_matrix = np.zeros_like(embed_matrix)
        for i in tqdm(range(1, len(w2id)), desc="merging"):
            sub = subword_matrix[char_dict[i]]
            h = embed_matrix[i]
            full_matrix[i] = np.sum(np.vstack((h, sub)), axis=0)

        full_norm = np.linalg.norm(full_matrix, axis=1)
        full_matrix /= full_norm[:, np.newaxis]

        for a, b, c in words:
            similarity = cos_similarity(a.lower(), b.lower(), full_matrix, subword_matrix, w2id, sub_w2id)
            ai.append(similarity)
            human.append(float(c))
            print(a, b, similarity)
            
        cor = np.corrcoef(ai, human)
        print(cor[0, 1])
        
    elif sys.argv[1] == "package":
        model = fasttext.load_model("package_fil9.bin")

        for a, b, c in words:
            a_vec, b_vec = model.get_word_vector(a), model.get_word_vector(b)
            a_vec /= np.linalg.norm(a_vec)
            b_vec /= np.linalg.norm(b_vec)
            sim = round(np.dot(a_vec, b_vec), 2)
            ai.append(sim)
            human.append(float(c))
            print(sim, float(c))

        cor = np.corrcoef(ai, human)
        print(cor[0, 1])
