import os, pickle, sys, string, unicodedata, random
from collections import Counter
import numpy as np
from tqdm.auto import tqdm

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def clean_str(string):
    string = string.strip().lower()
    string = string.replace(".", " ")
    string = string.replace("\'s", "")
    string = string.replace("\'", " ")
    string = string.replace("\"", " ")
    string = string.replace(",", " ")
    string = string.replace("!", " ")
    string = string.replace("?", " ")
    string = string.replace("\n", " ")
    string = string.replace("\\", " ")
    string = string.replace("//", " ")
    string = string.replace("/", " ")
    string = string.replace("<", " ")
    string = string.replace(">", " ")
    string = string.replace("=", " ")
    string = string.replace("_", " ")
    string = string.replace("+", " ")
    string = string.replace("  n", " ")
    string = string.replace("-", " ")
    string = string.replace(")", " ")
    string = string.replace("(", " ")
    string = string.replace("[", " ")
    string = string.replace("]", " ")
    string = string.replace("{", " ")
    string = string.replace("}", " ")
    string = string.replace(";", " ")
    string = string.replace(":", " ")
    string = string.replace("@", " ")
    string = string.replace("#", " ")
    string = string.replace("$", " ")
    string = string.replace("%", " ")
    string = string.replace("^", " ")
    string = string.replace("&", " ")
    string = string.replace("*", " ")
    string = string.replace("  ", " ")
    return string.strip()


def preprocess(w2id, sub_w2id, sub_id2w, char_dict, n=6):
    words = list(w2id.keys())

    for word in tqdm(words, desc="building sub dicts"):
        subword_list = []
        full = "<" + word + ">"
        
        if len(full) < 3:
            continue

        for i in range(3, n+1):
            for j in range(len(full)-i+1):
                char_n = full[j:j+i]
                if (char_n != full): 
                    if (char_n not in sub_w2id):
                        subidx = len(sub_w2id)
                        sub_w2id[char_n] = subidx
                        sub_id2w[subidx] = char_n
                    subword_list.append(sub_w2id[char_n])

        char_dict[w2id[word]] = subword_list

    return sub_w2id, sub_id2w, char_dict

def subsampling(corpus, threshold=1e-5):
    counter = Counter(corpus)
    subsample_scores = {}
    total_cnt = sum(counter.values())
    for word, cnt in tqdm(counter.items(), desc="Calculating subsampling scores"):
        if word == 0:
            continue
        score = 1 - np.sqrt(threshold / (cnt/total_cnt))
        subsample_scores[word] = score
    subsample_scores[0] = 10.

    return subsample_scores

def create_contexts_targets(corpus, window_size=5):
    subsample_scores = subsampling(corpus)
    targets, contexts = [], []

    for idx in tqdm(range(window_size, len(corpus)-window_size), desc="Creating contexts and targets"):
        draw = np.random.uniform()
        if draw > subsample_scores[corpus[idx]]:
            targets.append(corpus[idx])
            dynamic_window = random.randint(1, window_size)
            contexts.append([0]*(5-dynamic_window) + corpus[idx-dynamic_window : idx] + corpus[idx+1 : idx+dynamic_window+1] + [0]*(5-dynamic_window))

    return np.array(contexts), np.array(targets)

def get_unigram(counter, w2id):
    counter["pad"] = 0

    vocab_size = len(w2id)
    p_list = np.zeros(vocab_size)

    for word in w2id.keys():
        p_list[w2id[word]] = counter[word]

    p_list = np.power(p_list, 0.75)
    p_list /= np.sum(p_list)

    return p_list

if __name__ == "__main__":
    with open('../../word2vec/preprocessed_data/preprocessed_corpus', 'rb') as f:
        corpus, w2id, id2w, counter = pickle.load(f)
    
    print("Dicts Loaded!")

    sub_w2id, sub_id2w, char_dict = {}, {}, {}
    corpus = []

    for file in tqdm(os.listdir("../../word2vec/data/"), desc="reading data"):
        with open("../../word2vec/data/"+file, 'r') as f:
            data = f.read()

        data = unicodeToAscii(data)
        data = clean_str(data).strip().split()
        output = [w2id[word] for word in data if word in w2id]
        
        sub_w2id, sub_id2w, char_dict = preprocess(w2id, sub_w2id, sub_id2w, char_dict, 6)
        pickle.dump([sub_w2id, sub_id2w, char_dict], open("./preprocessed_data/dicts", "wb"), protocol=-1)
        contexts, targets = create_contexts_targets(output)

        pickle.dump([contexts, targets], open("./preprocessed_data/w2v/"+file+"_data", "wb"), protocol=-1)
        
    p_list = get_unigram(counter, w2id)
    pickle.dump(p_list, open("./preprocessed_data/w2v/plist", "wb"), protocol=-1)