from math import log2
import re 
import pandas as pd
import math

# calculate the kl divergence
def kl_divergence(p, q):
	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

def kl_divergence_list(p, q):
	return [p[i] * log2(p[i]/q[i]) for i in range(len(p))]

def delta_kl_divergence_list(p, q):
	kl1 = kl_divergence_list(p, q)
	kl2 = kl_divergence_list(q, p)
	return [kl1[i] - kl2[i] for i in range(len(p))]

def print_summary_kl(key:str, document1:dict, document2:dict):
    print(f'# of word in document1: {sum(document1.values())} in document2: {sum(document2.values())}')
    print(f'# of word "{key}" in document1: {document1[key]} in document2: {document2[key]}')
    p = document1[key]/sum(document1.values())
    q = document2[key]/sum(document2.values())
    print(f'P of word "{key}" in document1: {p} in document2: {q}')
    kl1 = p * log2(p/q)
    kl2 = q * log2(q/p)
    print('---------------------------------')
    print('KL-divergence = P * log2(P/Q)')
    print('---------------------------------')
    print(f'KL-divergence for document1: {kl1}')
    print(f'KL-divergence for document2: {kl2}')
    print(f'delta KL-divergence for document1 and document2: {kl1 - kl2}')

def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

def computeIDF(uniqueWords:set, df:pd.DataFrame, text_column:str = 'Text'):
    idfDict = dict.fromkeys(uniqueWords, 0)
    for document in df[text_column]:
        word_list = re.findall(r'\b[A-Za-z][a-z]{2,9}\b',  document)
        for word in uniqueWords:
            if word in word_list:
                idfDict[word] += 1

    drop = []           
    for word, val in idfDict.items():
        if val == 0:
            drop.append(word)
        else:
            idfDict[word] = math.log(len(df) / float(val))
    
    for i in drop: 
        idfDict.pop(i)

    return idfDict

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        if word in idfs.keys():
            tfidf[word] = val * idfs[word]
    return tfidf

def print_summary_tf_idf(key:str, tfA:dict, tfB:dict, idfs:dict):
    print(f'# of word in document1: {sum(tfA.values())} in document2: {sum(tfB.values())}')
    print(f'tf of word "{key}" in document1: {tfA[key]} in document2: {tfB[key]}')
    print(f'idfs of "{key}" = {idfs[key]}')
    print(f'tf-idfs of "{key}" in document1: {tfA[key] * idfs[key]} in document2: {tfB[key] * idfs[key]}')

def print_summary(key:str , document1:dict, document2:dict, tfA:dict, tfB:dict, idfs:dict):
    print("kl divergence approach")
    print_summary_kl(key, document1, document2)
    print("\n")
    print("tf-idf approach")
    print_summary_tf_idf(key, tfA, tfB, idfs)