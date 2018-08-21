import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
import pickle
import time

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 200 # embedding dimension
HIDDEN_DIM = 300 # hidden state dimension of lstm cell
SEQ_LENGTH = 30 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120  # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = EMB_DIM
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 200
generated_num = 100
sample_num = 10

# original seqgan parameter
# HIDDEN_DIM = 32
# PRE_EPOCH_NUM = 120
# TOTAL_BATCH = 200
# generated_num = 10000

positive_file = './data/pk_data_index.txt'
negative_file = 'save/negative_sample.txt'
eval_file = 'save/eval_file.txt'

a = open('./data/pk_data_index.pkl', 'rb')
real_data = pickle.load(a)

a = open('./data/pk_pos2idx.pkl', 'rb')
vocab_to_int = pickle.load(a)

a = open('./data/pk_idx2pos.pkl', 'rb')
int_to_vocab = pickle.load(a)
print(int_to_vocab)

a = open('./data/pk_embedding_vec.pkl', 'rb')
word_embedding_matrix = pickle.load(a)
word_embedding_matrix = word_embedding_matrix.astype(np.float32)

# a = open('./data/word_dict.pickle', 'rb')
# word_dict = pickle.load(a)

real_data_vocab = [[int_to_vocab[i] for i in sample if int_to_vocab[i] != '<PAD>'] for sample in real_data]
real_data_vocab = [' '.join(sample) for sample in real_data_vocab]
print(len(real_data_vocab))


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file, word_embedding_matrix):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess, word_embedding_matrix))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def pre_train_epoch(sess, trainable_model, data_loader, word_embedding_matrix):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch, word_embedding_matrix)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def make_sample(eval_file, int_to_vocab, sample_num):
    samples = []
    with open(eval_file, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            samples.append(parse_line)

    sample_int = samples[:sample_num]
    sample_vocab = [[int_to_vocab[i] for i in sample] for sample in sample_int]
    sample_vocab = [' '.join(sample) for sample in sample_vocab]

    return sample_vocab

################################## main() #########################################

# 시간측정
start_time = time.time()

tf.reset_default_graph()

random.seed(SEED)
np.random.seed(SEED)

gen_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH)
vocab_size = len(vocab_to_int)  # 6447
print(vocab_size)
dis_data_loader = Dis_dataloader(BATCH_SIZE, SEQ_LENGTH)

generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim,
                              filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

# First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
#  pre-train generator
gen_data_loader.create_batches(positive_file)
gen_sample = open('save/pretrain_sample.txt', 'w')
print('Start pre-training...')
gen_sample.write('pre-training...\n')
for epoch in range(PRE_EPOCH_NUM):
    loss = pre_train_epoch(sess, generator, gen_data_loader, word_embedding_matrix)
    if epoch % 5 == 0:
        generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file, word_embedding_matrix)
        sample_vocab = make_sample(eval_file, int_to_vocab, sample_num)

        print('pre-train epoch ', epoch)

        buffer = 'epoch:\t' + str(epoch) + '\n'
        gen_sample.write(buffer)
        for sample in sample_vocab:
            print(sample)
            buffer = sample + '\n'
            gen_sample.write(buffer)

#  pre-train discriminator
print('Start pre-training discriminator...')
# Train 3 epoch on the generated data and do this for 50 times
for _ in range(25):
    generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file, word_embedding_matrix)
    dis_data_loader.load_train_data(positive_file, negative_file)
    for _ in range(3):
        dis_data_loader.reset_pointer()
        for it in range(dis_data_loader.num_batch):
            x_batch, y_batch = dis_data_loader.next_batch()
            feed = {
                discriminator.input_x: x_batch,
                discriminator.input_y: y_batch,
                discriminator.dropout_keep_prob: dis_dropout_keep_prob
            }
            _ = sess.run(discriminator.train_op, feed)

rollout = ROLLOUT(generator, 0.8, word_embedding_matrix)

print('#########################################################################')
print('Start Adversarial Training...')
gen_sample.write('adversarial training...\n')
for total_batch in range(TOTAL_BATCH):
    # Train the generator for one step
    for it in range(1):
        samples = generator.generate(sess, word_embedding_matrix)
        rewards = rollout.get_reward(sess, samples, 16, discriminator)
        feed = {generator.x: samples, generator.rewards: rewards, generator.word_embedding_matrix: word_embedding_matrix}
        _ = sess.run(generator.g_updates, feed_dict=feed)

    # Test
    if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
        generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file, word_embedding_matrix)
        sample_vocab = make_sample(eval_file, int_to_vocab, sample_num)

        print('total_batch: ', total_batch)

        buffer = 'epoch:\t' + str(total_batch) + '\n'
        gen_sample.write(buffer)
        for sample in sample_vocab:
            print(sample)
            buffer = sample + '\n'
            gen_sample.write(buffer)

    # Update roll-out parameters
    rollout.update_params()

    # Train the discriminator
    for _ in range(5):
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file, word_embedding_matrix)
        dis_data_loader.load_train_data(positive_file, negative_file)

        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _ = sess.run(discriminator.train_op, feed)

    if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
        saver.save(sess, './checkpoint/seqGAN_ours')

gen_sample.close()

# 걸린 시간 출력
time_check = "--- total {} seconds ---".\
    format(time.time() - start_time)
print(time_check)

generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file, word_embedding_matrix)

samples = make_sample(eval_file, int_to_vocab, generated_num)
samples = [[word for word in sample.split() if word != 'UNK'] for sample in samples]
samples = [' '.join(sample) for sample in samples]

f = open('./save/final_output_vocab.txt', 'w')
for token in samples:
    token = token + '\n'
    f.write(token)
f.close()

# write the training time
f = open('./save/_parameters.txt', 'w')
f.write("Training time : {}\n".format(time_check))
f.write("add <start> signal as zero in word2vec lookup table\n")
f.close()