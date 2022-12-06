import pickle, sys
import numpy as np
from tqdm.auto import tqdm
import fasttext
from utils import *



if __name__ == "__main__":
    queries = {}
    cat = None
    q = []

    with open("/hdd1/user22/word2vec/analogies/questions-words.txt", 'r') as f:
        while True:
            line = f.readline()
            if not line:
                queries[cat] = q
                break
            line = str(line).lower()
            if cat == None:
                cat = line.split()[1]
            if line[0] == ":":
                queries[cat] = q
                cat = line.split()[1]
                q = []
                continue
            q.append(line.split())
    perf1 = []
    perf2 = []

    if sys.argv[1] == "fasttext":
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
            sub = np.sum(sub, axis=0)
            sub /= (np.linalg.norm(sub)+1e-7)
            h = embed_matrix[i]
            h /= (np.linalg.norm(h) + 1e-7)
            full_matrix[i] = h + sub

        full_norm = np.linalg.norm(full_matrix, axis=1)
        full_matrix /= (full_norm[:, np.newaxis] + 1e-7)
        sub_norm = np.linalg.norm(subword_matrix, axis=1)
        subword_matrix /= (sub_norm[:, np.newaxis] + 1e-7)

        for type in list(queries.keys())[:5]:
            acc, cnt = 0, 0
            for q in queries[type]:
                print('\n[analogy] ' + q[0] + ':' + q[1] + ' = ' + q[2] + ':?')
                cnt += 1
                predicted = analogy(q[0], q[1], q[2], w2id, id2w, full_matrix, subword_matrix, sub_w2id, 4)
                print(predicted)
                if predicted == None:
                    continue
                if q[-1] in predicted:
                    acc += 1
                    print("CORRECT!!")
                print(acc, " / ", cnt)
                
            perf1.append((acc, cnt))
            
        full_matrix = np.zeros_like(embed_matrix)
        for i in tqdm(range(1, len(w2id)), desc="merging"):
            sub = subword_matrix[char_dict[i]]
            h = embed_matrix[i]
            full_matrix[i] = np.sum(np.vstack((h, sub)), axis=0)

        full_norm = np.linalg.norm(full_matrix, axis=1)
        full_matrix /= (full_norm[:, np.newaxis] + 1e-7)

        for type in list(queries.keys())[5:]:
            acc, cnt = 0, 0
            for q in queries[type]:
                print('\n[analogy] ' + q[0] + ':' + q[1] + ' = ' + q[2] + ':?')
                cnt += 1
                predicted = analogy(q[0], q[1], q[2], w2id, id2w, full_matrix, subword_matrix, sub_w2id, 4)
                print(predicted)
                if predicted == None:
                    continue
                if q[-1] in predicted:
                    acc += 1
                    print("CORRECT!!")
                print(acc, " / ", cnt)
                
            perf2.append((acc, cnt))

        print(perf1)
        print(perf2)

    elif sys.argv[1] == "package":
        model = fasttext.load_model("package_fil9.bin")

        for type in list(queries.keys()):
            acc, cnt = 0, 0
            for q in queries[type]:
                print('\n[analogy] ' + q[0] + ':' + q[1] + ' = ' + q[-1] + ':?')
                cnt += 1
                predicted = model.get_analogies(q[0], q[1], q[-1])[:4]#[0][1]
                tmp = [word for score, word in predicted]
                print(tmp)
                if q[2] in tmp:
                    acc += 1
                    print("CORRECT!!")
                print(acc, " / ", cnt)
                
            perf.append((acc, cnt))
            
        print(perf)