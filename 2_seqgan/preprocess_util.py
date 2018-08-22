import pandas as pd
import re
import collections
from konlpy.tag import Twitter, Kkma
import pickle
import collections
import random
import numpy as np
import os

if not os.path.isdir('./data'):
    os.mkdir('./data')

def _save_pickle(path, data):
    # save pkl
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()


def sentence2pos(train_text, tag):
    if tag == "kkma":
        analyzer = Kkma()
    elif tag == "twitter":
        analyzer = Twitter()

    sentences = list()
    for line in train_text:
        sentence = re.sub(r"[^ㄱ-힣a-zA-Z0-9]+", ' ', line)
        if sentence:
            sentence = analyzer.pos(sentence)
            sentences.append(sentence)

    pos_counter = [['UNK', -1]]  # 빈도수 문제로 word_dict에 없는 word를 처리하기 위함. unknown
    pos_counter.extend(collections.Counter([word[0] for words in sentences for word in words]).most_common())
    print(pos_counter)

    pos_list = list()
    for pos, _ in pos_counter:
        pos_list.append(pos)

    return sentences, pos_list


def load_sentence(path):
    sentences = list()
    with open(path, 'r') as fout:
        for line in fout:
            pos_line = list()
            for pos in line.split():
                pos_line.append(pos)
            sentences.append(pos_line)
    return sentences


def get_sentence_pos(input):
    # Separate sentences into pos(morphemes)
    # 문장을 형태소로 분리
    # tag : "kkma" or "twitter"
    print("Sentence to pos in progressing...")
    data, pos_list = sentence2pos(input, tag="kkma")
    print("poke_count: ", count)
    print(len(data))
    print(data[0])
    print("pos_list: ", pos_list)

    # save pkl
    _save_pickle('./data/pk_real_data.pkl', data)
    _save_pickle('./data/pk_pos_list.pkl', pos_list)

    # save txt
    f = open('./data/pk_real_data.txt', 'w')
    for token in data:
        for word in token:
            word = str(word) + ' '
            f.write(word)
        f.write('\n')
    f.close()


def get_preprocess_data(embedpath, data_pos_list):
    """
    """
    # embed text 파일 읽기
    print("Loading embedding vector...")
    i = 0
    with open(embedpath, 'r') as fout:
        embed_pos_list = list()
        embedding_list = list()
        for line in fout:
            line = line.strip()
            if i == 0:
                pos_size = int(line.split(" ")[0])
                embedding_size = int(line.split(" ")[1])
                i += 1
                continue
            vector_list = list()
            line_sp = line.split(" ")
            for j in range(len(line_sp)):
                if j == 0:
                    continue
                elif j == 1:
                    # print(line_sp[j])
                    embed_pos_list.append(line_sp[j])
                else:
                    # print(line_sp[j])
                    vector_list.append(line_sp[j])
            embedding_list.append(vector_list)

    # embed vector의  pos2idx, idx2pos, embedding_vec 만듬
    pos2idx = dict()
    for pos in embed_pos_list:
        pos2idx[pos] = len(pos2idx)
    idx2pos = dict(zip(pos2idx.values(), pos2idx.keys()))
    print(pos2idx)
    print(idx2pos)

    embedding_vec = np.array(embedding_list, dtype=np.float32)
    print("before embed: ", np.shape(embedding_vec))

    print("Create new embedding vector...")
    # 현재 데이터에 해당되는 embedding vector만 추출.
    exist_idx = list()
    nonexist_pos = list()
    for data in data_pos_list:
        if data in list(pos2idx.keys()):
            exist_idx.append(pos2idx[data])
        else:
            nonexist_pos.append(data)

    embedding_vec = embedding_vec[sorted(exist_idx)]
    print(sorted(exist_idx))
    print("after embed: ", np.shape(embedding_vec))

    # 현재 데이터에는 있지만 embedding vector에 없는 데이터는 무작위 vector로 embedding vector에 추가
    start_embed = np.random.randn(1, embedding_size)
    add_embed = np.random.randn(len(nonexist_pos), embedding_size)
    embedding_vec = np.concatenate([start_embed, embedding_vec, add_embed], axis=0)
    print(len(nonexist_pos))
    print("after embed2: ", np.shape(embedding_vec))

    # 현재 데이터와 새로 만들어진 embedding vector에 맞는 pos2idx, idx2pos 만듬
    pos2idx = dict()
    pos2idx["<start>"] = len(pos2idx)
    for idx in sorted(exist_idx):
        pos = idx2pos[idx]
        pos2idx[pos] = len(pos2idx)
    for pos in nonexist_pos:
        pos2idx[pos] = len(pos2idx)
    idx2pos = dict(zip(pos2idx.values(), pos2idx.keys()))

    pos_size = len(pos2idx)

    # save pkl
    _save_pickle('./data/pk_pos2idx.pkl', pos2idx)
    _save_pickle('./data/pk_idx2pos.pkl', idx2pos)
    _save_pickle('./data/pk_embedding_vec.pkl', embedding_vec)
    print("Save all data as pkl !!")

    return pos_size, embedding_size


def _pkl_loading_test():
    # load sentences separated by pos (pkl)
    a = open('./data/pk_real_data.pkl', 'rb')
    sents = pickle.load(a)

    # load pos_list (pkl)
    a = open('./data/pk_pos_list.pkl', 'rb')
    pos_list = pickle.load(a)

    # load pos2idx (pkl)
    a = open('./data/pk_pos2idx.pkl', 'rb')
    pos2idx = pickle.load(a)

    # load idx2pos (pkl)
    a = open('./data/pk_idx2pos.pkl', 'rb')
    idx2pos = pickle.load(a)

    # load embedding_vec (pkl)
    a = open('./data/pk_embedding_vec.pkl', 'rb')
    embedding_vec = pickle.load(a)

    print(sents)
    print(len(pos_list))
    print(pos2idx)
    print(idx2pos)
    print(np.shape(embedding_vec))


if __name__ == "__main__":

    DATA_PATH = "./data/"
    embed_path = "./embed/vec.txt"

    pk_data = []
    pk_desc = []
    for i in range(1,8):
        data = pd.read_csv(DATA_PATH + 'pk_data_g{}.csv'.format(i))
        pk_data.append(data)
        pk_desc.append(data['desc'])

    input = []
    gen = 1
    count = 0
    for desc in pk_desc:
        print("{} 세대".format(gen))
        print(desc)
        gen += 1
        for d in desc:
            if type(d) == float:
                continue
            count += 1
            for sent in d.split("."):
                input.append(sent)

    # sentence와 pos_list pkl 만듬
    # 이미 pkl 만들었으면 주석 처리, 처음 사용시 주석 해제
    print("Data Loading and indexing...")
    # get_sentence_pos(input)

    # load sentences separated by pos (pkl)
    a = open('./data/pk_real_data.pkl', 'rb')
    sents = pickle.load(a)

    # load pos_list (pkl)
    a = open('./data/pk_pos_list.pkl', 'rb')
    pos_list = pickle.load(a)
    print(pos_list)

    # make embedding vector and etc
    print("Data preprocessing in progress..")
    pos_size, embedding_size = get_preprocess_data(embed_path, pos_list)
    print("pos_size: ", pos_size)
    print("embedding_size: ", embedding_size)

    print("#### test ####")
    _pkl_loading_test()


