
import pickle


def _save_pickle(path, data):
    # save pkl
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()


def create_sequence(seq_length, type_dict):
    type_stories = list()
    for pk_type in type_dict.keys():
        sentences = list()
        for sent in type_dict[pk_type]:
            sentences.append(sent)
        type_stories.append(sentences)

    data = list()
    type_data = dict()
    for sent in type_stories:
        type_story = list()
        # 문장 개수만큼 for 문
        for i in range(len(sent)):
            seq_data = list()
            # seq_data 개수가 seq_length가 될때 까지
            while True:
                flag = 0
                for word in sent[i]:
                    if seq_length <= len(seq_data):
                        flag = 1
                        break
                    seq_data.append(word)
                if flag == 1:
                    break
                i += 1
                if i >= len(sent):
                    i -= 1
                    while seq_length > len(seq_data):
                        seq_data.append(('UNK', ''))
            data.append(seq_data)
            type_story.append(seq_data)
        type_data[len(type_data)] = type_story

    _save_pickle("./data/2_pk_preprocessed_data.pkl", data)
    _save_pickle("./data/2_pk_pre_type_data.pkl", type_data)

    f = open('./data/2_pk_preprocessed_data.txt', 'w')
    for tokens in data:
        for word in tokens:
            word = str(word) + ' '
            f.write(word)
        f.write('\n')
    f.close()


def data_to_index(datadict, pos2idx):
    idx_dict = dict()
    for key in datadict.keys():
        idx_dataset = list()
        for sent in datadict[key]:
            idx_sentence = list()
            for word in sent:
                if word[0] not in list(pos2idx.keys()):
                    print(word[0])
                    idx_sentence.append(pos2idx['UNK'])
                    continue
                idx_sentence.append(pos2idx[word[0]])
            idx_dataset.append(idx_sentence)
        idx_dict[len(idx_dict)] = idx_dataset

    _save_pickle("./data/3_pk_type_data_index.pkl", idx_dict)

    # save pk_data_index.txt
    f = open('./data/3_pk_type_data_index.txt', 'w')
    for key in idx_dict.keys():
        f.write("[{}]".format(str(key)))
        f.write('\n')
        for idx_sent in idx_dict[key]:
            for word in idx_sent:
                word = str(word) + ' '
                f.write(word)
            f.write('\n')
    f.close()


if __name__ == "__main__":
    DATA_PATH = "./data/"

    seq_length = 30  # max 52

    # load dictionary that changes type to sentences (pkl)
    a = open('./data/type_dict_khkim.pickle', 'rb')
    type_dict = pickle.load(a)
    print(type_dict)

    print("Create Sequence in a length of seq_length...")
    create_sequence(seq_length, type_dict)

    print("Complete Creating sequence !!")

    # load after dataset
    a = open("./data/2_pk_pre_type_data.pkl", 'rb')
    datadict = pickle.load(a)

    # load pos to index
    a = open("./data/pk_pos2idx.pkl", 'rb')
    pos2idx = pickle.load(a)

    print("Replace Sequence to Index...")
    data_to_index(datadict, pos2idx)

    print("Complete Creating sequence to index !!")
