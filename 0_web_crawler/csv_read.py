import pandas as pd
import re
import collections
from konlpy.tag import Twitter
import random
import numpy as np

def build_dataset(train_text, min_count=0, sampling_rate=0):
    words = list()
    for line in train_text:
        sentence = re.sub(r"[^ㄱ-힣a-zA-Z0-9]+", ' ', line).strip().split()
        if sentence:
            words.append(sentence)

    word_counter = [['UNK', -1]]
    word_counter.extend(collections.Counter([word for sentence in words for word in sentence]).most_common())
    word_counter = [item for item in word_counter if item[1] >= min_count or item[0] == 'UNK']

    word_list = list()
    word_dict = dict()
    for word, count in word_counter:
        word_list.append(word) # 학습에 사용된 word를 저장한다. (visualize를 위해)
        word_dict[word] = len(word_dict)
    word_reverse_dict = dict(zip(word_dict.values(), word_dict.keys()))

    word_to_pos_li = dict()
    pos_list = list()
    twitter = Twitter()
    for w in word_dict:
        w_pos_li = list()
        for pos in twitter.pos(w, norm=True):
            w_pos_li.append(pos)

        word_to_pos_li[word_dict[w]] = w_pos_li
        pos_list += w_pos_li

    pos_counter = collections.Counter(pos_list).most_common()

    pos_dict = dict()
    for pos, _ in pos_counter:
        pos_dict[pos] = len(pos_dict)

    pos_reverse_dict = dict(zip(pos_dict.values(), pos_dict.keys()))

    word_to_pos_dict = dict()

    for word_id, pos_li in word_to_pos_li.items():
        pos_id_li = list()
        for pos in pos_li:
            pos_id_li.append(pos_dict[pos])
        word_to_pos_dict[word_id] = pos_id_li

    data = list()
    unk_count = 0
    for sentence in words:
        s = list()
        for word in sentence:
            if word in word_dict:
                index = word_dict[word]
            else:
                index = word_dict['UNK']
                unk_count += 1
            s.append(index)
        data.append(s)
    word_counter[0][1] = max(1, unk_count)

    # data = sub_sampling(data, word_counter, word_dict, sampling_rate)

    return data, word_dict, word_reverse_dict, pos_dict, pos_reverse_dict, word_to_pos_dict, word_list

DATA_PATH = "./data/"

pk_data = []
pk_desc = []
for i in range(1,8):
    data = pd.read_csv(DATA_PATH + 'pk_data_g{}.csv'.format(i))
    pk_data.append(data)
    pk_desc.append(data['desc'])

words = []
sentences = []
input = []
gen = 1
count = 0
for desc in pk_desc:
    print("{} 세대".format(gen))
    gen += 1
    print(desc)
    for d in desc:
        if type(d) == float:
            continue
        count += 1
        for sent in d.split("."):
            input.append(sent)
            sentence = re.sub(r"[^ㄱ-힣a-zA-Z0-9]+", ' ', sent).strip().split()
            if sentence:
                sentences.append(sentence)
                for word in sentence:
                    words.append(word)

data, word_dict, word_reverse_dict, pos_dict, pos_reverse_dict, word_to_pos_dict, word_list \
        = build_dataset(input)

print(len(sentences))
print(len(words))
print(words)

vocabulary_size = len(word_dict)
pos_size = len(pos_dict)
num_sentences = len(data)

print("number of sentences :", num_sentences)
print("vocabulary size :", vocabulary_size)
print("pos size :", pos_size)
print("poke num :", count)