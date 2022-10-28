import torch
from random import shuffle
from collections import Counter
import argparse
import random
import os 
import pickle
import numpy as np

def getRandomContext(corpus, C=10):
    wordID = random.randint(0, len(corpus) - 1)
    
    context = corpus[max(0, wordID - C):wordID]
    if wordID+1 < len(corpus):
        context += corpus[wordID+1:min(len(corpus), wordID + C + 1)]

    centerword = corpus[wordID]
    context = [w for w in context if w != centerword]

    if len(context) > 0:
        return centerword, context
    else:
        return getRandomContext(corpus, C)


def Skipgram(centerWord, contextWord, inputMatrix, outputMatrix):
    ################################  Input  ################################
    # centerWord : Index of a centerword (type:int)                         #
    # contextWord : Index of a contextword (type:int)                       #
    # inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))         #
    # outputMatrix : Weight matrix of output (type:torch.tesnor(V,D))       #
    #########################################################################
    
    #get hidden layer
    center_word_vector = inputMatrix[centerWord,:].view(1,-1) #1,D
    #print(center_word_vector.size())
    #score
    score_vector = torch.matmul(center_word_vector, torch.t(outputMatrix)) # (1,D) * (D,V) = (1,V)
    
    e = torch.exp(score_vector) 
    softmax = e / (torch.sum(e, dim=1, keepdim=True)) #1,V
    #print(softmax.size())
    #print(softmax)
    
    loss = -torch.log(softmax[:,contextWord])
    
    #get grad
    softmax_grad = softmax
    softmax_grad[:,contextWord] -= 1.0
    
    grad_out = torch.matmul(torch.t(softmax_grad), center_word_vector) #(V,1) * (1,D) = (V,D)
    grad_emb = torch.matmul(softmax_grad, outputMatrix) #(1,V) * (V,D) = (1,D)
    
    ###############################  Output  ################################
    # loss : Loss value (type:torch.tensor(1))                              #
    # grad_emb : Gradient of word vector (type:torch.tensor(1,D))            #
    # grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))          #
    #########################################################################

    return loss, grad_emb, grad_out

def CBOW(centerWord, contextWords, inputMatrix, outputMatrix):
    ################################  Input  ################################
    # centerWord : Index of a centerword (type:int)                         #
    # contextWords : Indices of contextwords (type:list(int))               #
    # inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))         #
    # outputMatrix : Weight matrix of output (type:torch.tesnor(V,D))       #
    #########################################################################
    
    #get hidden layer
    sum_of_context_words_vector = torch.sum(inputMatrix[contextWords, :],dim=0,keepdim=True) #1,D
    
    #result_vector = result_vector / len(contextWords) #1,D
    #print(result_vector.size())
    #score
    score_vector = torch.matmul(sum_of_context_words_vector, torch.t(outputMatrix)) # (1,D) * (D,V) = (1,V)
    
    e = torch.exp(score_vector) 
    softmax = e / (torch.sum(e, dim=1, keepdim=True)) #1,V
    #print(softmax.size())
    #print(softmax)
    
    loss = -torch.log(softmax[:,centerWord])
    
    #get grad
    softmax_grad = softmax
    softmax_grad[:,centerWord] -= 1.0
    #softmax_grad = softmax_grad.view(1,-1)
    
    #grad_out = torch.matmul(torch.t(softmax_grad), result_vector) #V,D
    grad_out = torch.matmul(torch.t(softmax_grad), sum_of_context_words_vector) #(1,V) * (1,D) = (V,D)
    grad_emb = torch.matmul(softmax_grad, outputMatrix) #(1,V) * (V,D) = (1,D)

    ###############################  Output  ################################
    # loss : Loss value (type:torch.tensor(1))                              #
    # grad_emb : Gradient of word embedding (type:torch.tensor(1,D))        #
    # grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))          #
    #########################################################################

    return loss, grad_emb, grad_out


def word2vec_trainer(corpus, word2ind, mode="CBOW", dimension=50, learning_rate=0.025, iteration=100000):
# Xavier initialization of weight matrices
    W_emb = torch.randn(len(word2ind), dimension) / (dimension**0.5) 
    W_out = torch.randn(len(word2ind), dimension) / (dimension**0.5)  
    window_size = 5

    
    losses=[]
    for i in range(iteration):
        #Training word2vec using SGD
        centerword, context = getRandomContext(corpus, window_size)
        centerInd =  word2ind[centerword]
        contextInds = [word2ind[i] for i in context]
        
        if mode=="CBOW":
            L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out)
            W_emb[contextInds] -= learning_rate*G_emb
            W_out -= learning_rate*G_out
            losses.append(L.item())

        elif mode=="SG":
            for contextInd in contextInds:
                L, G_emb, G_out = Skipgram(centerInd, contextInd, W_emb, W_out)
                W_emb[centerInd] -= learning_rate*G_emb.squeeze()
                W_out -= learning_rate*G_out
                losses.append(L.item())

        else:
            print("Unkwnown mode : "+mode)
            exit()

        if i%10000==0:
        	avg_loss=sum(losses)/len(losses)
        	print("Loss : %f" %(avg_loss,))
        	print("Acc : " , 100-int(avg_loss))
        	losses=[]
    
    
    return W_emb, W_out


def sim(testword, word2ind, ind2word, matrix):
    length = (matrix*matrix).sum(1)**0.5
    wi = word2ind[testword]
    inputVector = matrix[wi].reshape(1,-1)/length[wi]
    sim = (inputVector@matrix.t())[0]/length
    values, indices = sim.squeeze().topk(5)
    
    print()
    print("===============================================")
    print("The most similar words to \"" + testword + "\"")
    for ind, val in zip(indices,values):
        print(ind2word[ind.item()]+":%.3f"%(val,))
    print("===============================================")
    print()







def main():
    # Arugment를 받기위해서는 우선ArgumentParser 를 이용해 Instance를 만든다
    parser = argparse.ArgumentParser(description='Word2vec')
    # Instance 생서 시 Option으로 description = '내용'을 이용해 Argument의 설명을 적을 수 있다. (Help Mode 시 표시됨)
    parser.add_argument('mode', metavar='mode', type=str,
                        help='"SG" for skipgram, "CBOW" for CBOW')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    parser.add_argument('trainingWord', metavar='Number of word ', type=int,
                        help='The number of training words you want')
    # 입력 Argument들을 args에 할당
    ## 이제 아래 코드들로 받은 입력인자들을 args에 넣어 원하는함수(args)로 사용가능
    args = parser.parse_args()
    mode = args.mode
    part = args.part
    trainingWord = args.trainingWord

	#Load and tokenize corpus
    print("loading...")
    if part=="part":
        text = open('text8',mode='r').readlines()[0][:1000000] #Load a part of corpus for debugging
    elif part=="full":
        # data_ 폴더 안에있는 데이터들 목록 가져오기
        file_names = os.listdir('data_')
        # 목록들 하나하나 열어서 한줄한줄 읽기
        f = open("fin_data",'a')
        count = 0
        for i in file_names:
            data = open('data_/'+i,mode='r').readlines() #Load full corpus for submissionl
            count += 1
            for i in data:
                f.write(i)
            if count == 15:
                break
        f.close()
        text = open('fin_data',mode='r').read() #Load full corpus for submissionl
    else:
        print("Unknown argument : " + part)
        exit()

    
    print("tokenizing...")
    corpus = text.split()
    frequency = Counter(corpus)
    #print(frequency)
    processed = [] # processed 여기에
    
    # rare word 지우고 TWM이면 멈추기
    ## processed에 빈도수가 4개 초과인 단어 TW million개 담기
    TW = trainingWord
    for word in corpus:
        if frequency[word]>=4:
            processed.append(word)
        #if len(processed) == TW*1000000:
        #    break
    print(len(processed))

    #fin_processed = []
    # 상위단어 빈도의 단어 30k 뽑기 (30000)
    #sorted_processed = sorted(processed.items(), key=lambda x: x[1], reverse=True) #정렬
    #for i in range(0,30000):
    #    fin_processed.append(sorted_processed[i][0]) # ex) sorted_processed = [('a', 222), ('b', 221), ('c', 220)]
    vocabulary = set(processed)
 

    #Assign an index number to a word
    word2ind = {}
    word2ind[" "]=0 # {' ': 0} 0번째 인덱스는 공백으로 채워준다
    i = 1
    for word in vocabulary:
        word2ind[word] = i # 각 단어마다 인덱스를 부여한다
        i+=1
    ind2word = {} # 이번엔 인덱스가 key 값이 value이다.
    for k,v in word2ind.items():
        ind2word[v]=k

    print("Vocabulary size")
    print(len(word2ind))
    print()

    #Training section
    W_emb, W_out = word2vec_trainer(processed, word2ind, mode=mode, dimension=50, learning_rate=0.025, iteration=300000)
    
    # word_vecs W_emb
    word_vecs = W_emb
    params = {}
   # params['word_vecs'] = word_vecs.astype(np.float16)
    params['word_vecs'] = word_vecs
    params['word_to_id'] = word2ind
    params['id_to_word'] = ind2word
    pkl_file = 'cbow_params.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump(params, f, -1)
    # word_vecs W_emb
    
    #Print similar words
    #testwords = ["one", "are", "he", "have", "many", "first", "all", "world", "people", "after"]
    #for tw in testwords:
    #	sim(tw,word2ind,ind2word,W_emb)

main()
