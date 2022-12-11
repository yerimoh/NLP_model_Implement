import os, pickle, sys
import numpy as np
from tqdm.auto import tqdm
from collections import Counter
import random

def clean_str(string):
    string = string.strip().lower()
    string = string.replace(".", "")
    string = string.replace("\'s", "")
    string = string.replace("\'", "")
    string = string.replace(",", "")
    string = string.replace("!", "")
    string = string.replace("?", "")
    string = string.replace("\n", " ")
    string = string.replace("\\", " ")
    string = string.replace("//", " ")
    string = string.replace("<", " ")
    string = string.replace(">", " ")
    string = string.replace("=", " ")
    string = string.replace("+", " ")
    string = string.replace("  n", "")
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

def freq(text, threshold=10):
    text = clean_str(text)
    tmp = text.split()

    counts = Counter(tmp)
    output = []
    for w in tqdm(tmp, desc="Discarding rare words"):
        if w.isalpha():
            if counts[w] >= threshold:
                output.append(w)

    return output

def phrase_score(corpus, delta=10):
    word_cnt = Counter(corpus)
    phrase_counts = {}
    for i in tqdm(range(len(corpus)-1), desc="Making Phrases"):
        if corpus[i].isalpha() and corpus[i+1].isalpha():
            phrase = corpus[i] + " " + corpus[i+1]
            if phrase in phrase_counts.keys():
                phrase_counts[phrase] += 1
            else:
                phrase_counts[phrase] = 1

    phrase_scores = {}
    for phrase, count in tqdm(phrase_counts.items(), desc="Calculating Phrase Scores"):
        score = (count - delta) / (word_cnt[phrase.split()[0]] * word_cnt[phrase.split()[1]] + 1e-7)
        phrase_scores[phrase] = score

    return phrase_scores

def make_phrase(corp, times=3):
    for _ in range(times):
        phrase_scores = phrase_score(corp)
        corpus = []
        i = 0
        while i < (len(corp) - 1):
            if corp[i].isalpha() and corp[i+1].isalpha():
                if phrase_scores[corp[i] + " " + corp[i+1]] > 1e-2:
                    corpus.append(corp[i] + " " + corp[i+1])
                    # print(corp[i] + " " + corp[i+1])
                    i += 2
                else:
                    corpus.append(corp[i])
                    i += 1
            else:
                i += 1
        corp = corpus

    return corpus


def preprocess(corpus):
    # corpus = make_phrase(corpus)
    
    corpus_cnt = Counter(corpus)
    max_cnt = max(corpus_cnt.values())
    corpus_cnt["<pad>"] = max_cnt + 1
    cnts = sorted(corpus_cnt, key = lambda x: -corpus_cnt[x])
    
    output = []
    w2id, id2w = {}, {}

    for word in tqdm(cnts, desc="Making Dictionary"):
        idx = len(w2id)
        w2id[word] = idx
        id2w[idx] = word

    for word in tqdm(corpus, desc="Mapping words with indices"):
        output.append(w2id[word])

    return np.array(output), w2id, id2w, corpus_cnt

def subsampling(corpus, counter, threshold):
    subsample_scores = {}
    total_cnt = sum(counter.values()) - counter["<pad>"]
    for word, cnt in tqdm(counter.items(), desc="Calculating subsampling scores"):
        score = 1 - np.sqrt(threshold / (cnt/total_cnt))
        subsample_scores[word] = score
    subsample_scores["<pad>"] = 10.

    return subsample_scores


def create_contexts_target(corpus, window_size, counter, threshold=1e-5):
    subsample_scores = subsampling(corpus, counter, threshold)
    targets, contexts = [], []

    for idx in tqdm(range(window_size, len(corpus)-window_size), desc="Creating contexts and targets"):
        draw = np.random.uniform()
        if draw > subsample_scores[id2w[corpus[idx]]]:
            targets.append(corpus[idx])
            cs = []
            dynamic_window = random.randint(1, window_size)
            for t in range(-window_size, window_size+1):
                if (t < -dynamic_window) or (t > dynamic_window):
                    cs.append(0)
                else:
                    if t != 0:
                        cs.append(corpus[idx+t])

            contexts.append(cs)
    
    return np.array(contexts), np.array(targets)
