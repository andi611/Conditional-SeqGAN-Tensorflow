import os
from random import shuffle
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import numpy as np

import data_utils

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

class ValueNet:
    def __init__(self, size, num_layers, vocab_size, buckets):
        self.__name__ = 'ValueNet'
        self.cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = core_rnn_cell.EmbeddingWrapper(
            cell=self.enc_cell,
            embedding_classes=vocab_size,
            embedding_size=size)
        self.embedding = variable_scope.get_variable('embedding', [vocab_size, size])
        self.W = tf.Variable(xavier_init([size*num_layers, 1]))
        self.b = tf.Variable(tf.zeros(shape=[1]))
        self.real_data = [tf.placeholder(tf.int32, shape=[None], name='realdata{0}'.format(i)) for i in range(buckets[-1][1])]

    def discriminator(self, inp, inp_lens, ans, batch_size, dtype=tf.float32):
        # notice reversed parts
        with variable_scope.variable_scope('valuenet') as scope:
            _, inp_state = tf.nn.static_rnn(self.enc_cell, inp, sequence_length=inp_lens, dtype=dtype)
            prob, logit = self.decode(inp_state, ans)
            return prob, logit

    def decode(self, init_state, decoder_inputs):
        logits = []
        probs = []
        state = init_state
        emb_inputs = (embedding_ops.embedding_lookup(self.embedding, i)
                      for i in decoder_inputs)
        for i, emb_inp in enumerate(emb_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            state_vec = tf.concat(state, 1)
            logits.append(tf.matmul(state_vec, self.W)+self.b)
            probs.append(tf.nn.sigmoid(logits[-1]))
            # notice : the order is different from GAN
            output, state = self.cell(emb_inp, state)
        return probs, logits

"""
class SeqGAN:
    def __init__(self, size, num_layers, vocab_size, buckets):
        self.__name__ = 'SeqGAN'
        self.cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = core_rnn_cell.EmbeddingWrapper(
            cell=self.enc_cell,
            embedding_classes=vocab_size,
            embedding_size=size,
            reuse=None)
        self.cell = core_rnn_cell.EmbeddingWrapper(
            cell=self.cell,
            embedding_classes=vocab_size,
            embedding_size=size,
            reuse=True)
        self.D_W = [tf.Variable(xavier_init([size*num_layers, 1]))]
        self.D_b = [tf.Variable(tf.zeros(shape=[1]))]
        self.real_data = [tf.placeholder(tf.int32, shape=[None], name='realdata{0}'.format(i)) for i in range(buckets[-1][1])]

    def discriminator(self, inp, inp_lens, ans, batch_size, dtype=tf.float32):
        # notice reversed parts
        with variable_scope.variable_scope('critic') as scope:
            # get ans length
            ans_lens = [tf.Variable(0, trainable=False, dtype=tf.int32) for _ in range(batch_size)]
            for i in range(batch_size):
                tmp = tf.where([tf.equal(ans_step[i], data_utils.EOS_ID) for ans_step in ans])
                ans_lens[i] = ans_lens[i].assign(tf.cond(tf.not_equal(tf.shape(tmp)[0],0),
                                                         lambda: tf.cast(tmp[0][0], tf.int32),
                                                         lambda: tf.constant(-1)))
            # seq2seq
            _, inp_state = tf.nn.static_rnn(self.enc_cell, inp, sequence_length=inp_lens, dtype=dtype)
            with variable_scope.variable_scope(
                    variable_scope.get_variable_scope(), reuse=True):
                _, final_state = tf.nn.static_rnn(self.cell, ans, initial_state=inp_state, sequence_length=ans_lens, dtype=dtype)
            # project history vector into 1-dim value
            final_state = tf.concat(final_state, 1)
            D_logit = tf.matmul(final_state, self.D_W[-1])+self.D_b[-1]
            D_prob = tf.nn.sigmoid(D_logit)
            return D_prob, D_logit
"""
class Direct_WGAN_GP:
    def __init__(self, size, num_layers, vocab_size, buckets):
        self.__name__ = 'Direct_WGAN_GP'
        self.vocab_size = vocab_size
        self.cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.D_W = [tf.Variable(xavier_init([size*num_layers, 1]))]
        self.D_b = [tf.Variable(tf.zeros(shape=[1]))]
        self.real_data = [tf.placeholder(tf.int32, shape=[None], name='realdata{0}'.format(i)) for i in range(buckets[-1][1])]

    def discriminator(self, inp, inp_lens, ans, ans_samples, batch_size, dtype=tf.float32):
        # notice reversed parts
        with variable_scope.variable_scope('critic') as scope:
            # get ans length
            if ans_samples is not None:
                ans_lens = [tf.Variable(0, trainable=False, dtype=tf.int32) for _ in range(batch_size)]
                for i in range(batch_size):
                    tmp = tf.where([tf.equal(ans_step[i], data_utils.EOS_ID) for ans_step in ans_samples])
                    ans_lens[i] = ans_lens[i].assign(tf.cond(tf.not_equal(tf.shape(tmp)[0],0),
                                                             lambda: tf.cast(tmp[0][0], tf.int32),
                                                             lambda: tf.constant(-1)))
            # pack real data to one-hot
            inp_one_hot = tf.one_hot(tf.stack(inp), self.vocab_size)
            inp_inputs = [tf.squeeze(inp_step, squeeze_dims=[0]) for inp_step in tf.split(inp_one_hot, len(inp), 0)]
            # seq2seq
            _, inp_state = tf.nn.static_rnn(self.enc_cell, inp_inputs, sequence_length=inp_lens, dtype=dtype)
            with variable_scope.variable_scope(
                    variable_scope.get_variable_scope(), reuse=True):
                if ans_samples is not None:
                    _, final_state = tf.nn.static_rnn(self.cell, ans, initial_state=inp_state, sequence_length=ans_lens, dtype=dtype)
                else:
                    _, final_state = tf.nn.static_rnn(self.cell, ans, initial_state=inp_state, dtype=dtype)
            # project history vector into 1-dim value
            final_state = tf.concat(final_state, 1)
            D_logit = tf.matmul(final_state, self.D_W[-1])+self.D_b[-1]
            D_prob = tf.nn.sigmoid(D_logit)
            return D_prob, D_logit

class ESGAN:
    def __init__(self, size, num_layers, vocab_size, buckets):
        self.__name__ = 'ESGAN'
        self.cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = core_rnn_cell.EmbeddingWrapper(
            cell=self.enc_cell,
            embedding_classes=vocab_size,
            embedding_size=size)
        self.embedding = variable_scope.get_variable('embedding', [vocab_size, size])
        self.D_W = tf.Variable(xavier_init([size*num_layers, 1]))
        self.D_b = tf.Variable(tf.zeros(shape=[1]))
        self.real_data = [tf.placeholder(tf.int32, shape=[None], name='realdata{0}'.format(i)) for i in range(buckets[-1][1])]

    def discriminator(self, inp, inp_lens, ans, batch_size, dtype=tf.float32):
        # notice reversed parts
        with variable_scope.variable_scope('critic') as scope:
            _, inp_state = tf.nn.static_rnn(self.enc_cell, inp, sequence_length=inp_lens, dtype=dtype)
            D_prob, D_logit = self.decode(inp_state, ans)
            return D_prob, D_logit

    def decode(self, init_state, decoder_inputs):
        logits = []
        probs = []
        state = init_state
        emb_inputs = (embedding_ops.embedding_lookup(self.embedding, i)
                      for i in decoder_inputs)
        for i, emb_inp in enumerate(emb_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            output, state = self.cell(emb_inp, state)
            state_vec = tf.concat(state, 1)
            logits.append(tf.matmul(state_vec, self.D_W)+self.D_b)
            probs.append(tf.nn.sigmoid(logits[-1]))
        return probs, logits

class EBESGAN:
    def __init__(self, size, num_layers, vocab_size, buckets):
        self.__name__ = 'EBESGAN'
        self.vocab_size = vocab_size
        if vocab_size > 1000:
            self.num_sampled = 512
        elif vocab_size < 20:
            self.num_sampled = 5
        self.cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = core_rnn_cell.EmbeddingWrapper(
            cell=self.enc_cell,
            embedding_classes=vocab_size,
            embedding_size=size)
        self.embedding = variable_scope.get_variable('embedding', [vocab_size, size])
        self.w = tf.get_variable('proj_w', [size, vocab_size])
        self.w_t = tf.transpose(self.w)
        self.b = tf.get_variable('proj_b', [vocab_size])

        self.real_data = [tf.placeholder(tf.int32, shape=[None], name='realdata{0}'.format(i)) for i in range(buckets[-1][1])]

    def discriminator(self, inp, inp_lens, ans, batch_size, dtype=tf.float32):
        with variable_scope.variable_scope('critic') as scope:
            _, inp_state = tf.nn.static_rnn(self.enc_cell, inp, sequence_length=inp_lens, dtype=dtype)
            each_outs = self.decode(inp_state, ans)
            return each_outs

    def decode(self, init_state, decoder_inputs):
        outputs = []
        state = init_state
        emb_inputs = (embedding_ops.embedding_lookup(self.embedding, i)
                      for i in decoder_inputs)
        for i, emb_inp in enumerate(emb_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            output, state = self.cell(emb_inp, state)
            outputs.append(output)
        return outputs
    
    def softmax_loss_function(self, labels, inputs):
        labels = tf.reshape(labels, [-1, 1])
        local_w_t = tf.cast(self.w_t, tf.float32)
        local_b = tf.cast(self.b, tf.float32)
        local_inputs = tf.cast(inputs, tf.float32)
        return tf.cast(tf.nn.sampled_softmax_loss(
            weights = local_w_t,
            biases = local_b,
            inputs = local_inputs,
            labels = labels,
            num_sampled = self.num_sampled,
            num_classes = self.vocab_size),
            dtype = tf.float32)

################################
# Generate Scores for REINFORCE
################################
class Sequence_Task:
    def __init__(self):
        self.VOCAB_SIZE = 1000
        self.SEQ_LEN = 20

        vocab_path = 'data/sequence/vocab{}'.format(self.VOCAB_SIZE + 4)
        if os.path.exists(vocab_path):
            vocab, self.rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        def gen_data(f, num_data, test=False):
            for _ in range(num_data):
                inp_init = np.random.randint(self.VOCAB_SIZE)           
                inp_len = np.random.randint(1, high=self.SEQ_LEN+1)
                inp, out_init = self.compute(inp_init, inp_len)
                buf = ' '.join(str(i) for i in inp)
                buf += '\n'
                f.write(buf)
                if not test:
                    out_len = np.random.randint(1, high=self.SEQ_LEN+1)
                    out, _ = self.compute(out_init, out_len)
                    buf = ' '.join(str(i) for i in out)
                    buf += '\n'
                    f.write(buf)

        if not os.path.exists('data/sequence/train_sequence.txt'):
            with open('data/sequence/train_sequence.txt', 'w') as f:
                gen_data(f, 100000)
            with open('data/sequence/dev_sequence.txt', 'w') as f:
                gen_data(f, 10000)
            with open('data/sequence/test_sequence.txt', 'w') as f:
                gen_data(f, 10000, test=True)

    def compute(self, init, length):
        if self.VOCAB_SIZE > init + length:
            return range(init, length + init), init + length
        elif self.VOCAB_SIZE == init + length:
            return range(init, length + init), 0
        else:
            tmp1 = list(range(init, self.VOCAB_SIZE))
            tmp2 = list(range(0, length - self.VOCAB_SIZE + init))
            return tmp1 + tmp2, length - self.VOCAB_SIZE + init

    def possible_ans_num(self, inp):
        return self.SEQ_LEN

    def check_ans(self, inp, ans):
        tmp_ans = ans.split()
        buf = ' '.join(str(i) for i in range(int(inp[-1])+1,
                                             len(tmp_ans)+int(inp[-1])+1))
        return (ans == buf) and len(tmp_ans) > 0

class Counting_Task:
    def __init__(self):
        self.UPBOUND = 9
        self.SEQ_LEN = 20
        vocab_path = 'data/counting/vocab{}'.format(self.UPBOUND + 1 + 4)
        vocab, self.rev_vocab = data_utils.initialize_vocabulary(vocab_path)
        self.number_rev_vocab = tf.string_to_number(tf.constant(self.rev_vocab[4:]), tf.int32)

        def gen_data(f, num_data, test=False):
            for _ in range(num_data):
                inp_len = np.random.randint(1, high=self.SEQ_LEN)
                inp = np.random.randint(self.UPBOUND, size=inp_len)           
                buf = ' '.join(str(i) for i in inp)
                buf += '\n'
                f.write(buf)
                if not test:
                    #out_flags_num = np.random.randint(inp_len + 1)
                    out_flag = np.random.randint(inp_len)
                    out = [out_flag, inp[out_flag], len(inp) - out_flag - 1]
                    buf = ' '.join(str(i) for i in out)
                    buf += '\n'
                    f.write(buf)

        if not os.path.exists('data/counting/train_counting.txt'):
            with open('data/counting/train_counting.txt', 'w') as f:
                gen_data(f, 100000)
            with open('data/counting/dev_counting.txt', 'w') as f:
                gen_data(f, 10000)
            with open('data/counting/test_counting.txt', 'w') as f:
                gen_data(f, 10000, test=True)

    def possible_ans_num(self, inp):
        return len(inp)

    def check_ans(self, inp, ans):
        ans = ans.split()
        try:
            if len(ans) == 3:
                if ans[0] == "_UNK" and ans[2] != "_UNK":
                    if int(ans[2]) < len(inp) and len(inp) - int(ans[2]) > 9:
                        if inp[-int(ans[2])-1] == ans[1]:
                            #print(inp)
                            #print(ans)
                            return True
                elif ans[0] != "_UNK" and ans[2] == "_UNK":
                    if int(ans[0]) < len(inp) and len(inp) - int(ans[0]) > 9:
                        if inp[int(ans[0])] == ans[1]:
                            #print(inp)
                            #print(ans)
                            return True
                elif int(ans[0]) + int(ans[2]) + 1 == len(inp) and int(ans[0]) >= 0:
                    if ans[1] == inp[int(ans[0])]:
                        #print(inp)
                        #print(ans)
                        return True
        except ValueError:
            return False
        return False

class Addition_Task:
    def __init__(self):
        self.SEQ_LEN = 10

        vocab_path = 'data/addition/vocab{}'.format(10 + 4)
        if os.path.exists(vocab_path):
            _, self.rev_vocab = data_utils.initialize_vocabulary(vocab_path)
       
        def gen_data(f, num_data, test=False):
            for _ in range(num_data):
                seq_len = np.random.randint(2, high=self.SEQ_LEN+1)
                seq = np.random.randint(10, size=seq_len)
                inp_seq = ' '.join(str(i) for i in seq)
                inp_seq += '\n'
                f.write(inp_seq)
                if not test:
                    flag = np.random.randint(1, high=seq_len)
                    num1 = ''.join(str(i) for i in seq[:flag])
                    num1 = int(num1)
                    num2 = ''.join(str(i) for i in seq[flag:])
                    num2 = int(num2)
                    out = str(num1 + num2)
                    out_seq = ' '.join(i for i in out)
                    out_seq += '\n'
                    f.write(out_seq)

        if not os.path.exists('data/addition/train_addition.txt'):
            with open('data/addition/train_addition.txt', 'w') as f:
                gen_data(f, 100000)
            with open('data/addition/dev_addition.txt', 'w') as f:
                gen_data(f, 10000)
            with open('data/addition/test_addition.txt', 'w') as f:
                gen_data(f, 10000, test=True)

    def possible_ans_num(self, inp):
        buf = []
        for flag in range(1, len(inp)):
            num1 = ''.join(str(i) for i in inp[:flag])
            num2 = ''.join(str(i) for i in inp[flag:])
            buf.append(' '.join(i for i in str(int(num1)+int(num2))))
        return len(set(buf))

    def check_ans(self, inp, ans):
        buf = []
        for flag in range(1, len(inp)):
            num1 = ''.join(str(i) for i in inp[:flag])
            num2 = ''.join(str(i) for i in inp[flag:])
            buf.append(' '.join(i for i in str(int(num1)+int(num2))))
        if ans in buf:
            return True
        else:
            return False
