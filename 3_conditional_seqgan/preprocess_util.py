import pickle
import collections
import numpy as np
import os

if not os.path.isdir('./data'):
    os.mkdir('./data')

def _save_pickle(path, data):
    # save pkl
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()


def get_pos_list(sentences):
    pos_counter = [['UNK', -1]]  # 빈도수 문제로 word_dict에 없는 word를 처리하기 위함. unknown
    pos_counter.extend(collections.Counter([word[0] for words in sentences for word in words]).most_common())
    print(pos_counter)

    pos_list = list()
    for pos, _ in pos_counter:
        pos_list.append(pos)

    return sentences, pos_list


def get_sentence_pos(type_dict):
    """
    type_dict를 받아 문장별로 분리하고 pos_list를 만든다.
    """
    # dict 의 모든 문장을 list에 저장
    type2idx = dict()
    sentence = list()
    for pk_type in type_dict.keys():
        type2idx[pk_type] = len(type2idx)
        for sent in type_dict[pk_type]:
            sentence.append(sent)
    idx2type = dict(zip(type2idx.values(), type2idx.keys()))

    print("Sentence to pos in progressing...")
    data, pos_list = get_pos_list(sentence)
    print(len(data))
    print(data[0])
    print("pos_list: ", pos_list)

    # save pkl
    _save_pickle('./data/1_pk_real_data.pkl', data)
    _save_pickle('./data/pk_pos_list.pkl', pos_list)
    _save_pickle('./data/pk_type2idx.pkl', type2idx)
    _save_pickle('./data/pk_idx2type.pkl', idx2type)

    # save txt
    f = open('./data/1_pk_real_data.txt', 'w')
    for token in data:
        for word in token:
            word = str(word) + ' '
            f.write(word)
        f.write('\n')
    f.close()


def _pkl_loading_test():
    # load sentences separated by pos (pkl)
    a = open('./data/1_pk_real_data.pkl', 'rb')
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
    a = open('./data/pretrain_embedding_vec.pkl', 'rb')
    embedding_vec = pickle.load(a)

    # load type2idx (pkl)
    a = open('./data/pk_type2idx.pkl', 'rb')
    type2idx = pickle.load(a)

    # load idx2type (pkl)
    a = open('./data/pk_idx2type.pkl', 'rb')
    idx2type = pickle.load(a)

    print(len(sents))
    print(len(pos_list))
    print(pos2idx)
    print(idx2pos)
    print(type2idx)
    print(idx2type)
    print(np.shape(embedding_vec))


if __name__ == "__main__":

    DATA_PATH = "./data/"
    embed_path = "./embed/vec.txt"

    print("Data Loading and indexing...")

    # load dictionary that changes type to sentences (pkl)
    a = open('./data/type_dict_khkim.pickle', 'rb')
    type_dict = pickle.load(a)

    # 이미 pkl 만들었으면 주석 처리, 처음 사용시 주석 해제
    get_sentence_pos(type_dict)

    # load sequence list (pkl)
    a = open('./data/1_pk_real_data.pkl', 'rb')
    real_data = pickle.load(a)
    print(real_data)

    # load pos_list (pkl)
    a = open('./data/pk_pos_list.pkl', 'rb')
    pos_list = pickle.load(a)
    print(pos_list)

    # load pos2idx (pkl)
    a = open('./data/pk_pos2idx.pkl', 'rb')
    pos2idx = pickle.load(a)

    # load idx2pos (pkl)
    a = open('./data/pk_idx2pos.pkl', 'rb')
    idx2pos = pickle.load(a)

    # data의 모든 단어가 pos2idx에 있는지 검증
    for sent in real_data:
        for word in sent:
            try:
                pos2idx[word[0]]
            except:
                print("{} is not in pos2idx".format(word[0]))

    print("#### test ####")
    _pkl_loading_test()
    # 9805
    # 6447