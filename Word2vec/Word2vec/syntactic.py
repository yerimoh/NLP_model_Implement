import sys
import pickle
from tqdm.auto import tqdm
import numpy as np
from src.utill import *       


if __name__ == "__main__":
    #model = sys.argv[1]
    #if model == "HS":
    with open('cbow_params.pkl', 'rb') as f:
        params = pickle.load(f)
    embed_matrix = params["word_vecs"]
    w2id = params["word_to_id"]
    id2w = params["id_to_word"]
    norm = np.linalg.norm(embed_matrix, axis=1)
    embed_matrix /= norm[:, np.newaxis]
    matrix_norm = np.linalg.norm(embed_matrix)


    with open("./test_data/syntactic_train.txt", "r") as f1:
        answer = 0
        cnt = 0
        lines = f1.readlines()

        for i in lines:
            i = i.lower()
            a, b, c, d = i.split()
            #print(a, b, c, d)
            if analogy(a, b, c, w2id, id2w,  embed_matrix,matrix_norm):
                if d in analogy(a, b, c, w2id, id2w,  embed_matrix,matrix_norm):
                    answer += 1
                    print(a, b, c, d)
            cnt += 1
        print(cnt, answer)
        print("semantic accuracy:" + str(answer / cnt))

    '''   
    matrix_norm = np.linalg.norm(embed_matrix)
    queries = {}
    cat = None
    q = []

    with open("./test_data/semantic_train.txt", 'r') as f:
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

    perf = []
    for type in list(queries.keys()):
        acc, cnt = 0, 0
        for q in queries[type]:
            print('\n[analogy] ' + q[0] + ':' + q[1] + ' = ' + q[2] + ':?')
            cnt += 1
            predicted = analogy(q[0], q[1], q[2], w2id, id2w, embed_matrix, matrix_norm)
            print(predicted)
            if predicted == None:
                # cnt -= 1
                continue
            if q[-1] in predicted:
                acc += 1
                print("CORRECT!!")
        
            # print("answer: ", q[3], " predicted : ", predicted)
            print(acc, " / ", cnt)
            
        perf.append((acc, cnt))

        
    print("semantic accuracy:", perf)
    '''
