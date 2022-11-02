import sys
import pickle
from tqdm.auto import tqdm
import numpy as np
from utils.test import *       
import torch

def norm_mat(embeddings):
    # normalization
    norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
    norms = np.reshape(norms, (len(norms), 1))
    embeddings_norm = embeddings / norms
    embeddings_norm.shape



if __name__ == "__main__":
    #model = sys.argv[1]
    #if model == "HS":
    folder = "weights/cbow_WikiText2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(f"{folder}/model.pt", map_location=device)
    vocab = torch.load(f"{folder}/vocab.pt")
    
    embeddings = list(model.parameters())[0]
    embeddings = embeddings.cpu().detach().numpy()



    embed_matrix = embeddings
    w2id = vocab
    # id2w = vocab.lookup_token()
    norm = np.linalg.norm(embed_matrix, axis=1)
    #norm = norm_mat(embeddings)
    embed_matrix /= norm[:, np.newaxis]
   

    
    matrix_norm = np.linalg.norm(embed_matrix)
    queries = {}
    cat = None
    q = []

    with open("../test_data/syntactic_train.txt", 'r') as f:
        answer = 0
        cnt = 0
        lines = f.readlines()

        for i in lines:
            i = i.lower()
            a, b, c, d = i.split()
            #print(a, b, c, d)
            if analogy(a, b, c, vocab,  embed_matrix,matrix_norm):
                if d in analogy(a, b, c, vocab,  embed_matrix,matrix_norm):
                    answer += 1
                    print(a, b, c, d)
                cnt += 1
        print(cnt, answer)
        print("semantic accuracy:" + str(answer / cnt))
