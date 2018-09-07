import numpy as np

class Gen_Data_loader():
    def __init__(self, batch_size, sen_length):
        self.batch_size = batch_size
        self.token_stream = []
        self.sen_length = sen_length

    def create_batches(self, data_file):
        self.token_type = []
        self.token_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line[0] == '[':
                    self.flag = line[1]
                    continue
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.sen_length:
                    self.token_stream.append(parse_line)
                    self.token_type.append(self.flag)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        # seqeunce
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        # type
        self.token_type = self.token_type[:self.num_batch * self.batch_size]
        self.type_batch = np.split(np.array(self.token_type), self.num_batch, 0)

        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        ret_type = self.type_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret, ret_type

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader():
    def __init__(self, batch_size, sen_length):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.types = np.array([])
        self.sen_length = sen_length

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        positive_type = []
        negative_examples = []
        negative_type = []
        with open(positive_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line[0] == '[':
                    self.flag = line[1]
                    continue
                line = line.split()
                parse_line = [int(x) for x in line]
                positive_examples.append(parse_line)
                positive_type.append(self.flag)

        with open(negative_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                negative_type.append(parse_line[0])
                negative_examples.append(parse_line[1:])

        # print("positive_len: ", np.shape(positive_examples))
        # print("positive_len: ", np.shape(positive_type))
        # print("negative_len: ", np.shape(negative_examples))
        # print("negative_len: ", np.shape(negative_type))
        self.types = np.array(positive_type + negative_type)
        self.sentences = np.array(positive_examples + negative_examples)
        # print("setences_len: ", np.shape(self.sentences))
        # print("types_len: ", np.shape(self.types))

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.types = self.types[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)

        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.types = self.types[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]

        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.types_batches = np.split(self.types, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.types_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

