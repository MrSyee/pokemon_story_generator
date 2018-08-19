import numpy as np


def load_pratrained_vector(embed_path, load_path):
    with open(embed_path, 'r') as inp, open(load_path, 'w') as outp:
        pos_count = '30185'    # line count of the tsv file (as string)
        dimensions = '200'    # vector size (as string)
        outp.write(' '.join([pos_count, dimensions]) + '\n')
        for line in inp:
            words = line.strip().split()
            outp.write(' ')
            if "]" in [w for word in words for w in word]:
                line = line.strip().replace("]", "")
                words = line.strip().split()
                outp.write(' '.join(words) + '\n')
            elif "[" in [w for word in words for w in word]:
                line = line.strip().replace("[", "")
                words = line.strip().split()
                outp.write(' '.join(words))
            else:
                outp.write(' '.join(words))

def load_vec_file(filepath):
    """
    Load .vec file. Get pos_dict, pos_embedding_vector
    :param filepath: String, path of .vec file
    :return:
        pos_size, embedding_size, pos2idx, idx2pos, embedding_vec
    """
    i = 0
    with open(filepath, 'r') as fout:
        pos_list = list()
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
                    pos_list.append(line_sp[j])
                else:
                    # print(line_sp[j])
                    vector_list.append(line_sp[j])
            embedding_list.append(vector_list)

    pos2idx = dict()
    for pos in pos_list:
        pos2idx[pos] = len(pos2idx)
    idx2pos = dict(zip(pos2idx.values(), pos2idx.keys()))
    embedding_vec = np.array(embedding_list, dtype=np.float32)

    return pos_size, embedding_size, pos2idx, idx2pos, embedding_vec


if __name__ == "__main__":
    embed_path = "./embed/ko.tsv"
    load_path = "./embed/vec.txt"

    print("Loading {}...".format(embed_path))
    load_pratrained_vector(embed_path, load_path)
    print("Saved Complete {} !! ".format(load_path))

    # pos_size, embedding_size, pos2idx, idx2pos, embedding_vec = load_vec_file(load_path)
    # print(pos_size)
    # print(embedding_size)
    # print(pos2idx)
    # print(idx2pos)
    # print(np.shape(embedding_vec))




