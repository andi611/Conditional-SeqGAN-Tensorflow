import tensorflow as tf
import numpy as np
import copy

import data_utils
from units import *
from critic import *

def load_critic(name, size=None, num_layers=None, vocab_size=None, buckets=None):
    with variable_scope.variable_scope('critic') as scope:
        if name == 'Counting_Task':
            return Counting_Task()
        elif name == 'Sequence_Task':
            return Sequence_Task()
        elif name == 'Addition_Task':
            return Addition_Task()
        elif name == 'SeqGAN' or name == 'MaliGAN':
            return ESGAN(size, num_layers, vocab_size, buckets)
        elif name == 'Direct_WGAN_GP':
            return Direct_WGAN_GP(size, num_layers, vocab_size, buckets)
        elif name == 'REGS':
            return ESGAN(size, num_layers, vocab_size, buckets)
        elif 'ESGAN' in name:
            return ESGAN(size, num_layers, vocab_size, buckets)

class Seq2Seq:
    def __init__(
            self,
            mode,
            size,
            num_layers,
            vocab_size,
            buckets,
            learning_rate=0.5,
            learning_rate_decay_factor=0.99,
            max_gradient_norm=5.0,
            critic=None,
            critic_size=None,
            critic_num_layers=None,
            other_option=None,
            use_attn=False,
            output_sample=False,
            input_embed=True,
            feed_prev=False,
            batch_size=32,
            D_lr=1e-4,
            D_lr_decay_factor=0.5,
            v_lr=1e-4,
            v_lr_decay_factor=0.5,
            dtype=tf.float32):

        # self-config
        self.size = size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        if vocab_size > 1000:
            num_sampled = 512
        elif vocab_size < 20:
            num_sampled = 5
        self.buckets = buckets
        self.critic = load_critic(critic, critic_size, critic_num_layers, vocab_size, buckets)
        self.critic_name = critic
        self.other_option = other_option
        self.use_attn = use_attn
        self.output_sample = output_sample
        self.input_embed = input_embed
        self.feed_prev = feed_prev
        self.batch_size = batch_size
        # general vars
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.D_lr = tf.Variable(float(D_lr), trainable=False, dtype=dtype)
        self.v_lr = tf.Variable(float(v_lr), trainable=False, dtype=dtype)
        self.op_lr_decay = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.op_D_lr_decay = self.D_lr.assign(self.D_lr * D_lr_decay_factor)
        self.op_v_lr_decay = self.v_lr.assign(self.v_lr * v_lr_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.global_D_step = tf.Variable(0, trainable=False)
        self.global_V_step = tf.Variable(0, trainable=False)
        # value network
        if critic is not None and critic is not 'None':
            with variable_scope.variable_scope('valuenet') as scope:
                self.value_net = ValueNet(critic_size, critic_num_layers, vocab_size, buckets)
        
        # core cells, encoder and decoder are separated
        self.cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = core_rnn_cell.EmbeddingWrapper(
            cell=self.enc_cell,
            embedding_classes=vocab_size,
            embedding_size=size)

        # output projection
        w = tf.get_variable('proj_w', [size, vocab_size])
        w_t = tf.transpose(w)
        b = tf.get_variable('proj_b', [vocab_size])
        self.output_projection = (w, b)
        # input embedding
        self.embedding = variable_scope.get_variable('embedding', [vocab_size, size])

        # seq2seq-specific functions
        def loop_function(prev):
            prev = nn_ops.xw_plus_b(prev, self.output_projection[0], self.output_projection[1])
            prev_symbol = math_ops.argmax(prev, axis=1)
            emb_prev = embedding_ops.embedding_lookup(self.embedding, prev_symbol)
            return emb_prev
        
        def sample_loop_function(prev):
            prev = nn_ops.xw_plus_b(prev, self.output_projection[0], self.output_projection[1])
            prev_index = tf.multinomial(tf.log(tf.nn.softmax(2*prev)), 1)
            prev_symbol = tf.reshape(prev_index, [-1])
            emb_prev = embedding_ops.embedding_lookup(self.embedding, prev_symbol)
            return [emb_prev, prev_symbol]

        def test_sample_loop_function(prev):
            prev = nn_ops.xw_plus_b(prev, self.output_projection[0], self.output_projection[1])
            prev_index = tf.multinomial(tf.log(tf.nn.softmax(prev)), 1)
            prev_symbol = tf.reshape(prev_index, [-1])
            emb_prev = embedding_ops.embedding_lookup(self.embedding, prev_symbol)
            return [emb_prev, prev_symbol]

        def softmax_loss_function(labels, inputs):
            labels = tf.reshape(labels, [-1, 1])
            local_w_t = tf.cast(w_t, tf.float32)
            local_b = tf.cast(b, tf.float32)
            local_inputs = tf.cast(inputs, tf.float32)
            return tf.cast(tf.nn.sampled_softmax_loss(
                weights = local_w_t,
                biases = local_b,
                inputs = local_inputs,
                labels = labels,
                num_sampled = num_sampled,
                num_classes = vocab_size),
                dtype = tf.float32)

        def compute_loss(logits, targets, weights):
            with ops.name_scope("sequence_loss", logits + targets + weights):
                log_perp_list = []
                for logit, target, weight in zip(logits, targets, weights):
                    crossent = softmax_loss_function(target, logit)
                    log_perp_list.append(crossent * weight)
                log_perps = math_ops.add_n(log_perp_list)
                total_size = math_ops.add_n(weights)
                total_size += 1e-12
                log_perps /= total_size
                cost = math_ops.reduce_sum(log_perps)
                batch_size = array_ops.shape(targets[0])[0]
                return cost / math_ops.cast(batch_size, cost.dtype)

        def get_eos_value(rewards, uniform_weights):
            eos = [tf.cast(tf.equal(math_ops.add_n(uniform_weights[:i+1]), math_ops.add_n(uniform_weights)), tf.float32)*uniform_weights[i] for i in range(len(rewards))]
            outs = []
            for r, w in zip(rewards, eos):
                outs.append(tf.reshape(r, [-1]) * w)
            eos_value = math_ops.add_n(outs)
            return eos_value

        def uniform_weights(targets):
            tmp = [tf.cast(tf.equal(target, data_utils.EOS_ID), tf.float32) for target in targets]
            tmp[-1] = tf.cast(tf.equal(math_ops.add_n(tmp), 0.0), tf.float32)
            #off = tf.cast(tf.not_equal(math_ops.add_n(tmp), 0.0), tf.float32)
            uniform_weights = [tf.cast(tf.equal(math_ops.add_n(tmp[i:]), math_ops.add_n(tmp)), tf.float32) \
                               for i in range(len(tmp))]
            return uniform_weights

        def weighted_rewards(rewards, targets, uniform_weights, method='uniform'):
            #tmp = [tf.cast(tf.equal(target, data_utils.EOS_ID), tf.float32) for target in targets]
            #tmp[-1] = tf.cast(tf.equal(math_ops.add_n(tmp), 0.0), tf.float32)
            #uniform_weights = [math_ops.add_n(tmp[i:]) for i in range(len(tmp))]
            if method == 'uniform':
                weights = uniform_weights
            elif method == 'random':
                rand = tf.random_uniform([1],maxval=tf.cast(len(uniform_weights),tf.int32),dtype=tf.int32)
                weights = []
                for i in range(len(uniform_weights)):
                    weights.append(tf.cond(tf.equal(i,tf.reshape(rand,[])),
                                           lambda: tf.ones(tf.shape(targets[0])),
                                           lambda: tf.zeros(tf.shape(targets[0]))))
            elif method == 'decrease':
                weights = [math_ops.add_n(uniform_weights[i:]) for i in range(len(uniform_weights))]                
            elif method == 'increase':
                weights = [math_ops.add_n(uniform_weights[:(i+1)]) * uniform_weights[i] for i in range(len(uniform_weights))]
            outs = []
            for r, w in zip(rewards, weights):
                outs.append(tf.reshape(r, [-1]) * w)
            return outs, math_ops.add_n(weights), math_ops.add_n(uniform_weights)

        def seq_log_prob(logits, targets, rewards=None):
            if rewards is None:
                rewards = [tf.ones(tf.shape(target), tf.float32) for target in targets]
            with ops.name_scope("sequence_log_prob", logits + targets + rewards):
                log_perp_list = []
                tmp = [tf.cast(tf.equal(target, data_utils.EOS_ID), tf.float32) for target in targets]
                tmp[-1] = tf.cast(tf.equal(math_ops.add_n(tmp), 0.0), tf.float32)
                weights = [math_ops.add_n(tmp[i:]) for i in range(len(tmp))]
                for logit, target, weight, reward in zip(logits, targets, weights, rewards):
                    crossent = softmax_loss_function(target, logit)
                    log_perp_list.append(crossent * weight * reward)
                log_perps = math_ops.add_n(log_perp_list)
                total_size = math_ops.add_n(weights)
                total_size += 1e-12
                log_perps /= total_size
                return log_perps

        def critic_seq_log_prob(logits, targets, rewards=None, w=False):
            if rewards is None:
                rewards = [tf.ones(tf.shape(target), tf.float32) for target in targets]
            with ops.name_scope("critic_sequence_log_prob", logits + targets + rewards):
                log_perp_list = []
                tmp = [tf.cast(tf.equal(target, data_utils.EOS_ID), tf.float32) for target in targets]
                tmp[-1] = tf.cast(tf.equal(math_ops.add_n(tmp), 0.0), tf.float32)
                uniform_weights = [math_ops.add_n(tmp[i:]) for i in range(len(tmp))]
                if w == True:
                    weights = uniform_weights
                    # decrease
                    #weights = [math_ops.add_n(uniform_weights[i:]) for i in range(len(uniform_weights))]                
                    # increase
                    #weights = [math_ops.add_n(uniform_weights[:(i+1)]) * uniform_weights[i] for i in range(len(uniform_weights))]
                else:
                    weights = uniform_weights
                for logit, target, weight, reward in zip(logits, targets, weights, rewards):
                    crossent = self.critic.softmax_loss_function(target, logit)
                    log_perp_list.append(crossent * weight * reward)
                log_perps = math_ops.add_n(log_perp_list)
                total_size = math_ops.add_n(weights)
                total_size += 1e-12
                log_perps /= total_size
                return log_perps
                #cost = math_ops.reduce_sum(log_perps)
                #return cost

        def each_perp(logits, targets, weights):
            with ops.name_scope("each_perp", logits + targets):
                log_perp_list = []
                #tmp = [tf.cast(tf.equal(target, data_utils.EOS_ID), tf.float32) for target in targets]
                #tmp[-1] = tf.cast(tf.equal(math_ops.add_n(tmp), 0.0), tf.float32)
                #weights = [math_ops.add_n(tmp[i:]) for i in range(len(tmp))]
                for logit, target, weight in zip(logits, targets, weights):
                    #crossent = self.critic.softmax_loss_function(target, logit)
                    crossent = softmax_loss_function(target, logit)
                    log_perp_list.append(crossent * weight)
                return log_perp_list 

        # encoder's placeholder
        self.encoder_inputs = []
        for bid in range(buckets[-1][0]):
            self.encoder_inputs.append(
                tf.placeholder(tf.int32, shape = [None],
                               name = 'encoder{0}'.format(bid)))
        self.seq_len = tf.placeholder(
            tf.int32, shape = [None],
            name = 'enc_seq_len')

        # decoder's placeholder
        self.decoder_inputs = []
        self.target_weights = []
        if not feed_prev and mode == 'TRAIN':
            for bid in range(buckets[-1][1] + 1):
                self.decoder_inputs.append(
                    tf.placeholder(tf.int32, shape = [None],
                                   name = 'decoder{0}'.format(bid)))
                self.target_weights.append(
                    tf.placeholder(tf.float32, shape = [None],
                                   name = 'weight{0}'.format(bid)))
            targets = [self.decoder_inputs[i+1] for i in range(len(self.decoder_inputs)-1)]
        elif mode == 'TEST' or mode == 'D_TEST':
            for bid in range(buckets[-1][1] + 1):
                self.decoder_inputs.append(
                    tf.placeholder(tf.int32, shape = [None],
                                   name = 'decoder{0}'.format(bid)))
        else:
            self.decoder_inputs = [tf.placeholder(tf.int32, shape = [None], name = 'decoder0')]
        # other placeholders
        if critic is not None and not hasattr(critic, 'reward_eval_in_graph'):
            self.fed_samples = [ tf.placeholder(tf.int32, shape = [None], name = 'fed_sample{0}'.format(i)) for i in range(buckets[-1][1]) ]
            self.fed_rewards = tf.placeholder(tf.float32, shape = [None], name = 'fed_rewards')

        if mode == 'TRAIN':
            #Maximum-Likelihood Estimation (MLE)
            if critic is None:
                self.enc_state = []
                self.outputs = []
                self.losses = []
                for j, bucket in enumerate(buckets):
                    with variable_scope.variable_scope(
                            variable_scope.get_variable_scope(), reuse=True if j > 0 else None):
                        enc_outputs, enc_state = \
                            encode(self.enc_cell, self.encoder_inputs[:bucket[0]], self.seq_len)
                        outputs, _, _ = \
                            decode(self.cell, enc_state, self.embedding, \
                                   self.decoder_inputs[:bucket[1]], \
                                   bucket[1]+1, feed_prev=False)
                        loss = compute_loss(outputs, targets[:bucket[1]], self.target_weights[:bucket[1]])
                        self.enc_state.append(enc_state)
                        self.outputs.append(outputs)
                        self.losses.append(loss)
                self.print_outputs = []
                self.tmp_outputs = []
                for j, outs in enumerate(self.outputs):
                    self.print_outputs.append([])
                    self.tmp_outputs.append([])
                    for i in range(len(outs)):
                        self.print_outputs[j].append(nn_ops.xw_plus_b(outs[i], self.output_projection[0], self.output_projection[1]))
                        self.tmp_outputs[j].append(math_ops.argmax(self.print_outputs[j][i], axis=1))
                # update methods
                self.op_update = []
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                params = tf.trainable_variables()
                print(params)
                for j in range(len(self.buckets)):
                    gradients = tf.gradients(self.losses[j], params)
                    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
                    self.op_update.append(optimizer.apply_gradients(zip(clipped_gradients, params),
                                                                    global_step=self.global_step))
            # REINFORCE based methods,
            # include REINFORCE, SeqGAN, MaliGAN, REGS, ESGAN
            else:
                # generate process
                self.enc_state = []
                self.outputs = []
                self.samples_dists = []
                self.each_probs = []
                self.perp = []
                # only for REINFORCE
                self.out_dists = []

                # score process
                self.each_rewards = []
                self.rewards = []
                self.for_G_rewards = []

                # training process
                self.losses = []
                self.value_losses = []
                self.D_losses = []
                self.D_real = []
                self.D_fake = []

                for j, bucket in enumerate(buckets):
                    with variable_scope.variable_scope(
                            variable_scope.get_variable_scope(), reuse=True if j > 0 else None):
                        enc_outputs, enc_state = \
                            encode(self.enc_cell, self.encoder_inputs[:bucket[0]], self.seq_len)
                        samples_dists, samples, hiddens = \
                            decode(self.cell, enc_state, self.embedding, \
                                   [self.decoder_inputs[0]], bucket[1], \
                                   feed_prev=True, loop_function=sample_loop_function)
                        prob = - seq_log_prob(samples_dists, samples)
                        
                        if critic == 'SeqGAN' or critic == 'MaliGAN':
                            def Monte_Carlo():
                                N = 5
                                rewards = []
                                for step in range(bucket[1]):
                                    each_step_reward = []
                                    for _ in range(N):
                                        with variable_scope.variable_scope(
                                            variable_scope.get_variable_scope(), reuse=True):
                                            _, MC_sample, _ = \
                                                decode(self.cell, hiddens[step], self.embedding, \
                                                       [samples[step]], bucket[1]-step-1, \
                                                       feed_prev=True, loop_function=sample_loop_function)
                                            r, _ = self.critic.discriminator(self.encoder_inputs[:bucket[0]], self.seq_len, samples[:step]+MC_sample, batch_size)
                                        each_step_reward.append(tf.reshape(r,[-1]))
                                    rewards.append(math_ops.add_n(each_step_reward) / N)
                                return rewards

                            # architecture
                            each_prob_fake, each_logit_fake = self.critic.discriminator(self.encoder_inputs[:bucket[0]], self.seq_len, samples, batch_size)
                            each_prob_fake_value, each_logit_fake_value = self.value_net.discriminator(self.encoder_inputs[:bucket[0]], self.seq_len, samples, batch_size)
                            with variable_scope.variable_scope(
                                    variable_scope.get_variable_scope(), reuse=True):
                                each_prob_real, each_logit_real = self.critic.discriminator(self.encoder_inputs[:bucket[0]], self.seq_len, self.critic.real_data[:bucket[1]], batch_size)
                                each_prob_real_value, each_logit_real_value = self.value_net.discriminator(self.encoder_inputs[:bucket[0]], self.seq_len, self.critic.real_data[:bucket[1]], batch_size)
                            # uniform weights
                            fake_uniW = uniform_weights(samples)
                            real_uniW = uniform_weights(self.critic.real_data[:bucket[1]])
                            # end of sentence
                            D_prob_fake = get_eos_value(each_prob_fake, fake_uniW)
                            D_prob_real = get_eos_value(each_prob_real, real_uniW)

                            # train D
                            D_loss = -tf.reduce_mean(tf.log(D_prob_real) + tf.log(1.-D_prob_fake))
                            # Monte Carlo
                            #rewards = Monte_Carlo()

                            # print reward
                            reward = tf.reshape(D_prob_fake, [-1])
                            if critic == 'SeqGAN':
                                returns = [D_prob_fake for i in range(bucket[1])]
                                real_returns = [D_prob_real for i in range(bucket[1])]
                            else:
                                returns = [D_prob_fake/tf.reduce_sum(D_prob_fake) for i in range(bucket[1])]
                                real_returns = [D_prob_real/tf.reduce_sum(D_prob_real) for i in range(bucket[1])]
                            value_loss_update = tf.reduce_sum([tf.square(returns[i] - tf.reshape(each_prob_fake_value[i],[-1]))*fake_uniW[i] for i in range(bucket[1])]) / (tf.reduce_sum(fake_uniW)+1e-12)
                            value_loss_update += tf.reduce_sum([tf.square(real_returns[i] - tf.reshape(each_prob_real_value[i],[-1]))*real_uniW[i] for i in range(bucket[1])]) / (tf.reduce_sum(real_uniW)+1e-12)
                            minus_baseline = [returns[i] - tf.reshape(each_prob_fake_value[i],[-1]) for i in range(bucket[1])]    

                            for_G_prob_fake, for_G_fake_credits, _ = \
                                weighted_rewards(minus_baseline, samples, fake_uniW, 'uniform')
                            #for_G_rewards = [r - tf.reduce_mean(r) for r in for_G_prob_fake]
                            for_G_rewards = for_G_prob_fake
                            loss_update = each_perp(samples_dists, samples, fake_uniW)

                        elif critic == 'REGS':
                            # architecture
                            each_prob_fake, each_logit_fake = self.critic.discriminator(self.encoder_inputs[:bucket[0]], self.seq_len, samples, batch_size)
                            each_prob_fake_value, each_logit_fake_value = self.value_net.discriminator(self.encoder_inputs[:bucket[0]], self.seq_len, samples, batch_size)
                            with variable_scope.variable_scope(
                                    variable_scope.get_variable_scope(), reuse=True):
                                each_prob_real, each_logit_real = self.critic.discriminator(self.encoder_inputs[:bucket[0]], self.seq_len, self.critic.real_data[:bucket[1]], batch_size)
                                each_prob_real_value, each_logit_real_value = self.value_net.discriminator(self.encoder_inputs[:bucket[0]], self.seq_len, self.critic.real_data[:bucket[1]], batch_size)

                            # uniform weights
                            fake_uniW = uniform_weights(samples)
                            real_uniW = uniform_weights(self.critic.real_data[:bucket[1]])

                            # weights (alpha D)
                            for_D_each_prob_fake, for_D_fake_credits, _ = \
                                weighted_rewards(each_prob_fake, samples, fake_uniW, 'random')
                            for_D_each_prob_real, for_D_real_credits, _ = \
                                weighted_rewards(each_prob_real, self.critic.real_data[:bucket[1]], \
                                                 real_uniW, 'random')
                            for_D_score_fake = math_ops.add_n(for_D_each_prob_fake)
                            for_D_score_real = math_ops.add_n(for_D_each_prob_real)
                            # update D
                            D_loss = -tf.reduce_mean(tf.log(for_D_score_real) + tf.log(1.-for_D_score_fake))

                            # print score (weighted D)
                            reward = tf.reduce_mean(for_D_score_fake)
                            D_prob_fake = [[r[b] for r in for_D_each_prob_fake] for b in range(batch_size)]
                            D_prob_real = [[r[b] for r in for_D_each_prob_real] for b in range(batch_size)]
                            
                            # get actual return, in REGS, the returns are the outputs of D
                            # weights (alpha G)
                            for_G_rewards = [tf.reshape((each_prob_fake[i] - each_prob_fake_value[i]),[-1])*fake_uniW[i] for i in range(bucket[1])]

                            # normal method
                            loss_update = each_perp(samples_dists, samples, fake_uniW)
                            value_loss_update = tf.reduce_mean([tf.square(each_prob_fake[i] - each_prob_fake_value[i])*tf.reshape(fake_uniW[i],[-1,1]) for i in range(bucket[1])])
                            value_loss_update += tf.reduce_mean([tf.square(each_prob_real[i] - each_prob_real_value[i])*tf.reshape(real_uniW[i],[-1,1]) for i in range(bucket[1])])

                        elif 'ESGAN' in critic:
                            # architecture
                            each_prob_fake, each_logit_fake = self.critic.discriminator(self.encoder_inputs[:bucket[0]], self.seq_len, samples, batch_size)
                            each_prob_fake_value, each_logit_fake_value = self.value_net.discriminator(self.encoder_inputs[:bucket[0]], self.seq_len, samples, batch_size)
                            with variable_scope.variable_scope(
                                    variable_scope.get_variable_scope(), reuse=True):
                                each_prob_real, each_logit_real = self.critic.discriminator(self.encoder_inputs[:bucket[0]], self.seq_len, self.critic.real_data[:bucket[1]], batch_size)
                                each_prob_real_value, each_logit_real_value = self.value_net.discriminator(self.encoder_inputs[:bucket[0]], self.seq_len, self.critic.real_data[:bucket[1]], batch_size)

                            # uniform weights
                            fake_uniW = uniform_weights(samples)
                            real_uniW = uniform_weights(self.critic.real_data[:bucket[1]])

                            # weights (alpha D)
                            for_D_each_prob_fake, for_D_fake_credits, _ = \
                                weighted_rewards(each_prob_fake, samples, fake_uniW, 'uniform')
                            for_D_each_prob_real, for_D_real_credits, _ = \
                                weighted_rewards(each_prob_real, self.critic.real_data[:bucket[1]], \
                                                 real_uniW, 'uniform')
                            for_D_score_fake = math_ops.add_n(for_D_each_prob_fake) / (for_D_fake_credits + 1e-12)
                            for_D_score_real = math_ops.add_n(for_D_each_prob_real) / (for_D_real_credits + 1e-12)
                            # update D
                            D_loss = -tf.reduce_mean(tf.log(for_D_score_real) + tf.log(1.-for_D_score_fake))

                            # print score (weighted D)
                            reward = tf.reduce_mean(for_D_score_fake)
                            D_prob_fake = [[r[b] for r in for_D_each_prob_fake] for b in range(batch_size)]
                            D_prob_real = [[r[b] for r in for_D_each_prob_real] for b in range(batch_size)]
                            
                            # get actual return
                            uni_each_prob_fake = for_D_each_prob_fake
                            uni_each_prob_real = for_D_each_prob_real
                            if 'seq' in critic:
                                returns = [math_ops.add_n(uni_each_prob_fake[i:]) for i in range(bucket[1])]
                                real_returns = [math_ops.add_n(uni_each_prob_real[i:]) for i in range(bucket[1])]
                            else:
                                returns = uni_each_prob_fake
                                real_returns = uni_each_prob_real
                            value_loss_update = tf.reduce_sum([tf.square(returns[i] - tf.reshape(each_prob_fake_value[i], [-1]))*fake_uniW[i] for i in range(bucket[1])]) / (tf.reduce_sum(fake_uniW)+1e-12)
                            value_loss_update += tf.reduce_sum([tf.square(real_returns[i] - tf.reshape(each_prob_real_value[i], [-1]))*real_uniW[i] for i in range(bucket[1])]) / (tf.reduce_sum(real_uniW)+1e-12)

                            minus_baseline = [returns[i] - tf.reshape(each_prob_fake_value[i],[-1]) for i in range(bucket[1])]
                            # weights (alpha G)
                            for_G_prob_fake, for_G_fake_credits, _ = \
                                weighted_rewards(minus_baseline, samples, fake_uniW, 'decrease')
                            for_G_rewards = for_G_prob_fake

                            # normal method
                            loss_update = each_perp(samples_dists, samples, fake_uniW)
                            
                        #REINFORCE
                        else:
                            with variable_scope.variable_scope(
                                variable_scope.get_variable_scope(), reuse=True):
                                out_dist, hiddens, _ = \
                                    decode(self.cell, enc_state, self.embedding, \
                                           self.decoder_inputs[:bucket[1]], bucket[1], \
                                           feed_prev=False)
                            self.out_dists.append(out_dist)
                            reward = self.fed_rewards
                            loss = seq_log_prob(out_dist, self.fed_samples[:bucket[1]], \
                                                [reward - tf.reduce_mean(reward) for _ in range(bucket[1])])
                            loss_update = tf.reduce_sum(loss) / batch_size
                        
                        # generate process
                        self.enc_state.append(enc_state)
                        self.outputs.append(samples)
                        self.samples_dists.append(samples_dists)
                        self.each_probs.append(prob)
                        self.perp.append(tf.reduce_sum(prob) / batch_size)
                        
                        # score process, REINFORCE's reward is got from environment
                        self.each_rewards.append(reward)
                        self.for_G_rewards.append(for_G_rewards)

                        # trainning process
                        self.losses.append(loss_update)
                        self.value_losses.append(value_loss_update)
                        if 'GAN' in critic or critic == 'REGS':
                            self.D_losses.append(D_loss)
                            self.D_real.append(D_prob_real)
                            self.D_fake.append(D_prob_fake)

                # parameter collection
                params = tf.trainable_variables()
                critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
                value_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='valuenet')
                s2s_params = [ x for x in params if x not in critic_params and x not in value_params ]
                critic_params.append(self.global_D_step)
                value_params.append(self.global_V_step)
                print(s2s_params)
                print(critic_params)

                # optimizer
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                D_optimizer = tf.train.GradientDescentOptimizer(self.D_lr)

                # update method
                self.op_update = []
                self.D_solver = []
                self.v_solver = []
                for j in range(len(self.buckets)):
                    # update generator
                    gradients = tf.gradients(self.losses[j], s2s_params, self.for_G_rewards[j])
                    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
                    self.op_update.append(optimizer.apply_gradients(
                        zip(clipped_gradients, s2s_params),
                        global_step=self.global_step))
                    # update discriminator
                    if 'GAN' in critic or critic == 'REGS':
                        D_grads = tf.gradients(self.D_losses[j], critic_params)
                        clipped_D_grads, _ = tf.clip_by_global_norm(D_grads, max_gradient_norm)
                        self.D_solver.append(D_optimizer.apply_gradients(
                            zip(clipped_D_grads, critic_params),
                            global_step=self.global_D_step))
                    # update value net
                    v_grads = tf.gradients(self.value_losses[j], value_params)
                    clipped_v_grads, _ = tf.clip_by_global_norm(v_grads, max_gradient_norm)
                    self.v_solver.append(D_optimizer.apply_gradients(
                        zip(clipped_v_grads, value_params),
                        global_step=self.global_V_step))

                self.pre_D_saver = tf.train.Saver(var_list=critic_params, max_to_keep=None, sharded=True)
                self.pre_value_saver = tf.train.Saver(var_list=value_params, max_to_keep=None, sharded=True)
                self.pre_saver = tf.train.Saver(var_list=s2s_params, sharded=True)

        elif mode == 'TEST':
            self.enc_state = []
            self.outputs = []
            enc_outputs, enc_state = \
                encode(self.enc_cell, self.encoder_inputs, self.seq_len)
            
            # for beam search
            outputs, _, _ = \
                decode(self.cell, enc_state, self.embedding, \
                       self.decoder_inputs, buckets[-1][1], \
                       feed_prev=False)
            self.outs = []
            for l in range(len(outputs)):
                outs = nn_ops.xw_plus_b(outputs[l], self.output_projection[0], self.output_projection[1])
                self.outs.append(tf.nn.softmax(outs))
            
            # for MMI
            local_batch_size = array_ops.shape(self.decoder_inputs[0])[0]
            lm_outputs, _, _ = \
                decode(self.cell, self.cell.zero_state(local_batch_size, tf.float32), self.embedding, \
                       self.decoder_inputs, buckets[-1][1], \
                       feed_prev=False)
            self.lm_outs = []
            for l in range(len(lm_outputs)):
                lm_outs = nn_ops.xw_plus_b(lm_outputs[l], self.output_projection[0], self.output_projection[1])
                self.lm_outs.append(tf.nn.softmax(lm_outs))
            
            self.enc_state.append(enc_state)

            # for argmax test
            with variable_scope.variable_scope(
                    variable_scope.get_variable_scope(), reuse=None):
                outputs, _, _ = \
                    decode(self.cell, enc_state, self.embedding, \
                           self.decoder_inputs, buckets[-1][1], \
                           feed_prev=True, loop_function=loop_function)
            self.outputs.append(outputs)
            self.print_outputs = []
            self.tmp_outputs = []
            for j, outs in enumerate(self.outputs):
                self.print_outputs.append([])
                self.tmp_outputs.append([])
                for i in range(len(outs)):
                    self.print_outputs[j].append(nn_ops.xw_plus_b(outs[i], self.output_projection[0], self.output_projection[1]))
                    self.tmp_outputs[j].append(math_ops.argmax(self.print_outputs[j][i], axis=1))
            self.max_log_prob = - seq_log_prob(outputs, self.tmp_outputs[0])

            # for sample test
            with variable_scope.variable_scope(
                    variable_scope.get_variable_scope(), reuse=True):
                tmp, self.samples, _ = \
                    decode(self.cell, enc_state, self.embedding, \
                           self.decoder_inputs, buckets[-1][1], \
                           feed_prev=True, loop_function=test_sample_loop_function)
            self.log_prob = - seq_log_prob(tmp, self.samples)
            params = tf.trainable_variables()
            critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
            s2s_params = [ x for x in params if x not in critic_params ]
            general_s2s_params = {}
            other_ver = False

            self.pre_saver = tf.train.Saver(var_list=s2s_params, sharded=True)

        elif mode == 'D_TEST':
            print(self.critic_name)
            print('D_TEST')
            each_prob, each_logit = self.critic.discriminator(self.encoder_inputs, self.seq_len, self.critic.real_data, batch_size)
            each_value_prob, each_value_logit = self.value_net.discriminator(self.encoder_inputs, self.seq_len, self.critic.real_data, batch_size)
            real_uniW = uniform_weights(self.critic.real_data)
            if self.critic_name == 'SeqGAN':
                print('SeqGAN') 
                def Monte_Carlo():
                    enc_outputs, enc_state = \
                        encode(self.enc_cell, self.encoder_inputs, self.seq_len)
                    _, hiddens, _ = \
                        decode(self.cell, enc_state, self.embedding, \
                               self.decoder_inputs, self.buckets[-1][1], \
                               feed_prev=False)
                    N = 10
                    rewards = []
                    for step in range(self.buckets[-1][1]):
                        each_step_reward = []
                        for _ in range(N):
                            with variable_scope.variable_scope(
                                variable_scope.get_variable_scope(), reuse=True):
                                _, MC_sample, _ = \
                                    decode(self.cell, hiddens[step], self.embedding, \
                                           [self.decoder_inputs[step+1]], self.buckets[-1][1]-step-1, \
                                           feed_prev=True, loop_function=sample_loop_function)
                                all_r, _ = self.critic.discriminator(self.encoder_inputs, self.seq_len, self.decoder_inputs[1:(step+2)]+MC_sample, batch_size)
                            uniW = uniform_weights(self.decoder_inputs[1:(step+2)]+MC_sample)
                            r = get_eos_value(all_r, uniW)
                            each_step_reward.append(tf.reshape(r,[-1]))
                        rewards.append(math_ops.add_n(each_step_reward) / N)
                        #rewards.append(list(each_step_reward))
                    return rewards
                for_D_score = get_eos_value(each_prob, real_uniW)
                for_D_each_prob = Monte_Carlo()
            else:
                # weights (alpha D)
                for_D_each_prob, for_D_credits, _ = \
                    weighted_rewards(each_prob, self.critic.real_data, real_uniW, 'uniform')
                for_D_score = math_ops.add_n(for_D_each_prob) / (for_D_credits + 1e-12)
                
            if 'seq' in self.critic_name:
                uni_each_value, _, _ = \
                    weighted_rewards(each_value_logit, self.critic.real_data, real_uniW, 'uniform')
            else:
                uni_each_value, _, _ = \
                    weighted_rewards(each_value_prob, self.critic.real_data, real_uniW, 'uniform')

            # print score (weighted D)
            self.reward = tf.reduce_mean(for_D_score)
            #self.D_probs = [[r[b] for r in for_D_each_prob] for b in range(1)]
            self.D_probs = for_D_each_prob
            self.uniW = uni_each_value#[[w[b] for w in uni_each_value] for b in range(1)]
            critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
            value_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='valuenet')
            self.pre_saver = tf.train.Saver(var_list=critic_params, sharded=True)
            self.pre_V_saver = tf.train.Saver(var_list=value_params, sharded=True)
            params = tf.trainable_variables()
            s2s_params = [ x for x in params if x not in critic_params and x not in value_params ]
            self.pre_s2s_saver = tf.train.Saver(var_list=s2s_params, sharded=True)

        # whole seq2seq saver
        self.saver = tf.train.Saver(max_to_keep=None, sharded=True)

    def train_step(
            self,
            sess,
            encoder_inputs,
            decoder_inputs,
            target_weights,
            bucket_id,
            encoder_lens=None,
            forward=False,
            decoder_outputs=None,#for REINFORCE, MIXER
            rewards=None,#for REINFORCE, MIXER
            GAN_mode=None#for GAN
    ):
    
        #MLE
        if self.critic is None:
            batch_size = encoder_inputs[0].shape[0]
            encoder_size, decoder_size = self.buckets[bucket_id]
            input_feed = {}
            for l in range(encoder_size):
                input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.seq_len] = encoder_lens
            for l in range(decoder_size):
                input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
                input_feed[self.target_weights[l].name] = target_weights[l]
            last_target = self.decoder_inputs[decoder_size].name
            input_feed[last_target] = np.zeros([batch_size], dtype = np.int32)

            if forward:
                output_feed = [self.losses[bucket_id], self.tmp_outputs[bucket_id]]
                outputs = sess.run(output_feed, input_feed)
                return outputs[0], outputs[1]
            else:
                output_feed = [self.losses[bucket_id], self.op_update[bucket_id]]
                outputs = sess.run(output_feed, input_feed)
                return outputs[0], outputs[1]

        #SeqGAN, MaliGAN, REGS, ESGAN, EBESGAN
        elif GAN_mode:
            batch_size = encoder_inputs[0].shape[0]
            encoder_size, decoder_size = self.buckets[bucket_id]
            input_feed = {}
            for l in range(encoder_size):
                input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.seq_len] = encoder_lens
            for l in range(decoder_size):
                input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
                input_feed[self.target_weights[l].name] = target_weights[l]
            last_target = self.decoder_inputs[decoder_size].name
            input_feed[last_target] = np.zeros([batch_size], dtype = np.int32)

            if GAN_mode == 'D':
                for l in range(decoder_size-1):
                    input_feed[self.critic.real_data[l].name] = decoder_inputs[l+1]
                input_feed[self.critic.real_data[decoder_size-1].name] = \
                    np.zeros([batch_size], dtype = np.int32)
                if forward:
                    output_feed = [self.D_losses[bucket_id],
                                   self.losses[bucket_id],
                                   self.D_real[bucket_id],
                                   self.outputs[bucket_id],
                                   self.D_fake[bucket_id]]
                    outputs = sess.run(output_feed, input_feed)
                    return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]
                else:
                    output_feed = [self.D_losses[bucket_id],
                                   self.D_solver[bucket_id]]
                    outputs = sess.run(output_feed, input_feed)
                    return outputs[0], outputs[1]

            elif GAN_mode == 'V':
                for l in range(decoder_size-1):
                    input_feed[self.critic.real_data[l].name] = decoder_inputs[l+1]
                input_feed[self.critic.real_data[decoder_size-1].name] = \
                    np.zeros([batch_size], dtype = np.int32)
                output_feed = [self.value_losses[bucket_id],
                               self.v_solver[bucket_id]]
                outputs = sess.run(output_feed, input_feed)
                return outputs[0], outputs[1]

            else:
                if forward:
                    output_feed = [self.losses[bucket_id],
                                   self.outputs[bucket_id],
                                   self.D_fake[bucket_id],
                                   self.each_probs[bucket_id]]
                    outputs = sess.run(output_feed, input_feed)
                    return outputs[0], outputs[1], outputs[2], outputs[3]
                else:
                    output_feed = [self.losses[bucket_id],
                                   self.perp[bucket_id],
                                   self.D_fake[bucket_id],
                                   self.op_update[bucket_id]]
                    outputs = sess.run(output_feed, input_feed)
                    return outputs[0], outputs[1], outputs[2]
            
        #REINFORCE
        else:
            batch_size = encoder_inputs[0].shape[0]
            encoder_size, decoder_size = self.buckets[bucket_id]
            input_feed = {}
            for l in range(encoder_size):
                input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.seq_len] = encoder_lens
            for l in range(decoder_size):
                input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
                input_feed[self.target_weights[l].name] = target_weights[l]
            last_target = self.decoder_inputs[decoder_size].name
            input_feed[last_target] = np.zeros([batch_size], dtype = np.int32)
            if forward:
                output_feed = [self.outputs[bucket_id], self.tmps[bucket_id]]
                outputs = sess.run(output_feed, input_feed)
                return outputs[0], outputs[1]
            else:
                for l in range(decoder_size-1):
                    input_feed[self.decoder_inputs[l+1].name] = decoder_outputs[l]
                for l in range(decoder_size):
                    input_feed[self.fed_samples[l].name] = decoder_outputs[l]
                input_feed[self.fed_rewards.name] = rewards
                output_feed = [self.losses[bucket_id],
                               self.perp[bucket_id], 
                               self.out_dists[bucket_id],
                               self.op_update[bucket_id]]
                outputs = sess.run(output_feed, input_feed)
                return outputs[0], outputs[1], outputs[2]

    def dynamic_decode(self, sess, encoder_inputs, encoder_lens, decoder_inputs, mode='argmax'):
        encoder_size = self.buckets[-1][0]
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        input_feed[self.seq_len] = encoder_lens
        input_feed[self.decoder_inputs[0].name] = decoder_inputs[0]
        if mode == 'argmax':
            output_feed = [self.tmp_outputs[0], self.max_log_prob]
        elif mode == 'sample':
            output_feed = [self.samples, self.log_prob]
            #output_feed = [self.samples, self.log_prob, self.reward]
        return sess.run(output_feed, input_feed)

    def test_discriminator(self, sess, encoder_inputs, encoder_lens, decoder_inputs):
        encoder_size, decoder_size = self.buckets[-1]
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        input_feed[self.seq_len] = encoder_lens
        input_feed[self.decoder_inputs[0].name] = [data_utils.GO_ID]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l+1].name] = decoder_inputs[l]
            input_feed[self.critic.real_data[l].name] = decoder_inputs[l]
        output_feed = [self.reward, self.D_probs, self.uniW]
        return sess.run(output_feed, input_feed)

    def stepwise_test_beam(self, sess, encoder_inputs, encoder_lens, decoder_inputs):
        encoder_size, decoder_size = self.buckets[-1]
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        input_feed[self.seq_len] = encoder_lens
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        output_feed = [self.outs]
        return sess.run(output_feed, input_feed)

    def lm_prob(self, sess, decoder_inputs):
        _, decoder_size = self.buckets[-1]
        input_feed = {}
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        output_feed = [self.lm_outs]
        return sess.run(output_feed, input_feed)
