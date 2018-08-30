
import pandas as pd
import re
import collections
#from konlpy.tag import Twitter, Kkma
import pickle
import collections
import random
import numpy as np
import os
from gensim.models.word2vec import Word2Vec


def _save_pickle(path, data):
    # save pkl
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()


# data 불러옴
print("Load real data ...")
a = open('./data/pk_real_data.pkl', 'rb')
sentences = pickle.load(a)

# pk_idx2pos.pkl
a = open('./data/pk_idx2pos.pkl', 'rb')
idx2pos = pickle.load(a)

# pk_pos2idx.pkl
a = open('./data/pk_pos2idx.pkl', 'rb')
pos2idx = pickle.load(a)

sentence_idx = []
for sentence in sentences:
    words = []
    for word in sentence:
        words.append(pos2idx[word[0]])
    sentence_idx.append(words)

sentences_words = []
for sen in sentence_idx:
    sentence = []
    for pos_idx in sen:
        sentence.append(idx2pos[pos_idx])
        sentences_words.append(sentence)

# word2vec 학습
print("Training word2vec ...")
model = Word2Vec(sentences_words, size=30, window=5,min_count=0, workers=4, iter=10, sg=1)

# word2vec 테스트
print("Test word2vec ...")
print(model.most_similar("불"))

# word2vec에 <start>, UNK 등 추가 후 numpy로 저장
key = list(pos2idx.keys())
w2v = []
for k in key:
    if k == '<start>' or k == 'UNK' or k == '후다':
        print(k)
        w2v.append(np.random.randn(30)*0.1)
    else:
        w2v.append(model.wv[k])

w2v=np.array(w2v)

_save_pickle('./data/pk_embedding_vec.pkl', w2v)

print("Save word2vec !")

# pk_embedding_vec.pkl

a = open('./data/pk_embedding_vec.pkl', 'rb')
w2v_load = pickle.load(a)

print(np.shape(w2v_load))
print(w2v_load)