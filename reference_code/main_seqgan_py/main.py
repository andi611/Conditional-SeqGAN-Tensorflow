import tensorflow as tf
import numpy as np
import pickle
import random
import re
import os
import sys
import time
import math
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import modified_precision

from seq2seq_model_comp import *
import data_utils
import args
#import gen_seq as mydata
from critic import *

def parse_buckets(str_buck):
    _pair = re.compile(r"(\d+,\d+)")
    _num = re.compile(r"\d+")
    buck_list = _pair.findall(str_buck)
    if len(buck_list) < 1:
        raise ValueError("There should be at least 1 specific bucket.\nPlease set buckets in configuration.")
    buckets = []
    for buck in buck_list:
        tmp = _num.findall(buck)
        d_tmp = (int(tmp[0]), int(tmp[1]))
        buckets.append(d_tmp)
    return buckets

FLAGS = args.parse()
FLAGS.data_dir, _ = FLAGS.data_path.rsplit('/',1)
_buckets = parse_buckets(FLAGS.buckets)

def train_s2s():
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    def build_summaries(): 
        #TODO: combine the vars into one plot
        train_loss = tf.Variable(0.)
        tf.summary.scalar("train_loss", train_loss)
        eval_losses = []
        for ids, _ in enumerate(_buckets):
            eval_losses.append(tf.Variable(0.))
            tf.summary.scalar("eval_loss_{}".format(ids), eval_losses[ids])
        summary_vars = [train_loss] + eval_losses
        summary_ops = tf.summary.merge_all()
        return summary_ops, summary_vars

    # parse data and build vocab if there do not exist one.
    train, dev, _ = data_utils.prepare_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
    vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
    
    with tf.Session() as sess:
        # build the model
        model = Seq2Seq(
            'TRAIN',
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.vocab_size,
            _buckets,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            critic=None,
            use_attn=FLAGS.use_attn,
            output_sample=True,
            input_embed=True,
            feed_prev=False,
            batch_size=FLAGS.batch_size,
            dtype=tf.float32)
        # build summary and intialize
        summary_ops, summary_vars = build_summaries()
        sess.run(tf.variables_initializer(tf.global_variables()))
        log_dir = os.path.join(FLAGS.model_dir, 'log')
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(ckpt.model_checkpoint_path))
            model.saver.restore(sess, ckpt.model_checkpoint_path)

        # load in train and dev(valid) data with buckets
        train_set = read_data_with_buckets(train, FLAGS.max_train_data_size)
        dev_set = read_data_with_buckets(dev)
        train_buckets_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_buckets_sizes))
        print ('each buckets has: {d}'.format(d=train_buckets_sizes))
        train_buckets_scale = [sum(train_buckets_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_buckets_sizes))]

        # main process
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # get batch from a random selected bucket
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            encoder_inputs, decoder_inputs, weights, seq_lens, _ = \
                get_batch_with_buckets(train_set, FLAGS.batch_size, bucket_id)

            # each training step
            start_time = time.time()
            step_loss, _ = model.train_step(sess, encoder_inputs, \
                                            decoder_inputs, weights, \
                                            bucket_id, seq_lens)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1
            # log, save and eval
            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                print ("global step %d; learning rate %.4f; step-time %.2f; perplexity "
                       "%.2f; loss %.2f"
                       % (model.global_step.eval(), model.learning_rate.eval(),
                          step_time, perplexity, loss))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.op_lr_decay)
                previous_losses.append(loss)
                # eval
                eval_losses = []
                for bucket_id in range(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        eval_losses.append(0.)
                        continue
                    encoder_inputs, decoder_inputs, weights, seq_lens, _ = \
                        get_batch_with_buckets(dev_set, FLAGS.batch_size, bucket_id)
                    eval_loss, outputs = model.train_step(sess, encoder_inputs, \
                                                    decoder_inputs, weights, \
                                                    bucket_id, seq_lens, forward=True)
                    outputs = [output_ids[0] for output_ids in outputs]
                    #if data_utils.EOS_ID in outputs:
                    #    outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                    print(outputs)
                    print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))
                    eval_losses.append(eval_loss)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                    print("  eval: bucket %d perplexity %.2f; loss %.2f" % (bucket_id, eval_ppx, eval_loss))
                # write summary
                feed_dict = {}
                for ids, key in enumerate(summary_vars[1:]):
                    feed_dict[key] = eval_losses[ids]
                feed_dict[summary_vars[0]] = loss
                summary_str = sess.run(summary_ops,
                                       feed_dict=feed_dict)
                writer.add_summary(summary_str, model.global_step.eval())
                writer.flush()
                # Save checkpoint and zero timer and loss.
                ckpt_path = os.path.join(FLAGS.model_dir, "ckpt")
                model.saver.save(sess, ckpt_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                sys.stdout.flush()

def train_gan():
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    def build_summaries():
        loss = tf.Variable(0.)
        tf.summary.scalar("loss", loss)
        perp = tf.Variable(0.)
        tf.summary.scalar("perp", perp)
        reward = tf.Variable(0.)
        tf.summary.scalar("reward", reward)
        if 'GAN' in FLAGS.gan_type or FLAGS.gan_type == 'REGS':
            D_loss = tf.Variable(0.)
            tf.summary.scalar("D_loss", D_loss)
            summary_vars = [loss, perp, reward, D_loss]
        else:
            summary_vars = [loss, perp, reward]
        summary_ops = tf.summary.merge_all()
        return summary_ops, summary_vars

    # parse data and build vocab if there do not exist one.
    train, _, _ = data_utils.prepare_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
    reward_out_path = os.path.join(FLAGS.model_dir, 'rewards_trajectory.txt')
    
    with tf.Session() as sess, open(reward_out_path,'w') as f_reward_out:
        # build the model
        model = Seq2Seq(
            'TRAIN',
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.vocab_size,
            _buckets,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            critic=FLAGS.gan_type,
            critic_size=FLAGS.gan_size,
            critic_num_layers=FLAGS.gan_num_layers,
            other_option=FLAGS.option,
            use_attn=FLAGS.use_attn,
            output_sample=True,
            input_embed=True,
            feed_prev=False,
            batch_size=FLAGS.batch_size,
            D_lr=FLAGS.D_lr,
            D_lr_decay_factor=FLAGS.D_lr_decay_factor,
            v_lr=FLAGS.v_lr,
            v_lr_decay_factor=FLAGS.v_lr_decay_factor,
            dtype=tf.float32)
        # build summary and initialize
        summary_ops, summary_vars = build_summaries()
        sess.run(tf.variables_initializer(tf.global_variables()))
        value_path = os.path.join(FLAGS.pre_D_model_dir, '..', 'value')
        log_dir = os.path.join(FLAGS.model_dir, 'log')
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(ckpt.model_checkpoint_path))
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        elif os.path.exists(FLAGS.pre_model_dir):
            pre_ckpt = tf.train.get_checkpoint_state(FLAGS.pre_model_dir)
            if pre_ckpt and tf.train.checkpoint_exists(pre_ckpt.model_checkpoint_path):
                print ('read in model from {}'.format(pre_ckpt.model_checkpoint_path))
                model.pre_saver.restore(sess, pre_ckpt.model_checkpoint_path)
            else:
                print ('no previous model, create a new one')
            if os.path.exists(FLAGS.pre_D_model_dir):
                pre_D_ckpt = tf.train.get_checkpoint_state(FLAGS.pre_D_model_dir)
                if pre_D_ckpt and tf.train.checkpoint_exists(pre_D_ckpt.model_checkpoint_path):
                    print ('read in model from {}'.format(pre_D_ckpt.model_checkpoint_path))
                    model.pre_D_saver.restore(sess, pre_D_ckpt.model_checkpoint_path)
                else:
                    print ('no previous critic, create a new one')
            if os.path.exists(value_path):
                pre_V_ckpt = tf.train.get_checkpoint_state(value_path)
                if pre_V_ckpt and tf.train.checkpoint_exists(pre_V_ckpt.model_checkpoint_path):
                    print ('read in model from {}'.format(pre_V_ckpt.model_checkpoint_path))
                    model.pre_value_saver.restore(sess, pre_V_ckpt.model_checkpoint_path)
                else:
                    print ('no previous critic, create a new one')

        # load in train and dev(valid) data with buckets
        train_set = read_data_with_buckets(train, FLAGS.max_train_data_size)
        if FLAGS.option == 'MIXER':# in MIXER, using the longest bucket
            train_buckets_sizes = [len(train_set[-1])]
        # in REINFORCE, SeqGAN, using buckets, (or can set a longest bucket)
        else:
            train_buckets_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_buckets_sizes))
        print ('each buckets has: {d}'.format(d=train_buckets_sizes))
        train_buckets_scale = [sum(train_buckets_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_buckets_sizes))]

        # check the operation of discriminator
        """
        if hasattr(model.critic, 'discriminator'):
            sys.stdout.write('Discriminator\'s testing stage.')
            sys.stdout.write('Please click Enter to test,')
            sys.stdout.write('or type exit() to continue training.\n')
            sys.stdout.flush()
            vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
            vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
            sys.stdout.write('> ')
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            while sentence:
                if sentence.strip() == 'exit()':
                    break
                # get batch from a random selected bucket
                random_number_01 = np.random.random_sample()
                bucket_id = min([i for i in range(len(train_buckets_scale))
                                 if train_buckets_scale[i] > random_number_01])
                encoder_inputs, decoder_inputs, weights, seq_lens, inputs_token = \
                    get_batch_with_buckets(train_set, FLAGS.batch_size, bucket_id)

                start_time = time.time()
                _, samples, rewards, logs = \
                    model.train_step(sess, encoder_inputs, \
                                     decoder_inputs, weights, \
                                     bucket_id, seq_lens, forward=True, GAN_mode='G')
                inputs_seq = []
                samples_seq = []
                for ids, input_token in enumerate(inputs_token):
                    if data_utils.PAD_ID in input_token:
                        input_token = input_token[:input_token.index(data_utils.PAD_ID)]
                    inputs_seq.append(" ".join([tf.compat.as_str(rev_vocab[output]) for output in reversed(input_token)]))
                    sample = [sample_ids[ids] for sample_ids in samples]
                    if data_utils.EOS_ID in sample:
                        sample = sample[:sample.index(data_utils.EOS_ID)]
                    samples_seq.append(" ".join([tf.compat.as_str(rev_vocab[output]) for output in sample]))

                for inp, out, log, r in zip(inputs_seq, samples_seq, logs, rewards):
                    print('Q: '+inp)
                    print('A: '+out)
                    print('reward:{}; log:{}'
                          .format(r, log))
                    print('time spent:{}'.format(time.time() - start_time))

                _, _, D_real, _, _ = \
                    model.train_step(sess, encoder_inputs, \
                                     decoder_inputs, weights, \
                                     bucket_id, seq_lens, forward=True, GAN_mode='D')
                print('Q: '+inputs_seq[0])
                sample = [inp[0] for inp in decoder_inputs]
                if data_utils.EOS_ID in sample:
                    sample = sample[:sample.index(data_utils.EOS_ID)]
                print_dec_inp = " ".join([tf.compat.as_str(rev_vocab[output]) for output in sample])
                print('A: '+print_dec_inp)
                print('reward:{}'.format(D_real[0]))

                sys.stdout.write('> ')
                sys.stdout.flush()
                sentence = sys.stdin.readline()

            sys.stdout.write('Type \'exit()\' to skip training stage > ')
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            if sentence.strip() == 'exit()':
                return
        """
        
        D_step = FLAGS.D_step
        G_step = FLAGS.G_step
        # main process
        step_time, loss, reward = 0.0, 0.0, 0.0
        perp, D_loss = 0.0, 0.0
        bucket_times = [0 for _ in range(len(_buckets))]
        #value_loss = 0.0
        s = _buckets[-1][1]
        current_step = 0
        previous_rewards = []
        np.random.seed(1234)
        while True:
            if FLAGS.option == 'MIXER':
                bucket_id = -1
            else:
                # get batch from a random selected bucket
                random_number_01 = np.random.random_sample()
                bucket_id = min([i for i in range(len(train_buckets_scale))
                                 if train_buckets_scale[i] > random_number_01])
            #TODO: extra mis-matching inputs, intend-generated wrong inputs
            encoder_inputs, decoder_inputs, weights, seq_lens, _ = \
                get_batch_with_buckets(train_set, FLAGS.batch_size, bucket_id)

            # each training step
            start_time = time.time()
            if hasattr(model.critic, 'discriminator'):
                for _ in range(D_step):# D steps
                    step_D_loss, _ = model.train_step(sess, encoder_inputs, \
                                                      decoder_inputs, weights, \
                                                      bucket_id, seq_lens, GAN_mode='D')
                    # each time get another batch
                    random_number_01 = np.random.random_sample()
                    bucket_id = min([i for i in range(len(train_buckets_scale))
                                     if train_buckets_scale[i] > random_number_01])
                    encoder_inputs, decoder_inputs, weights, seq_lens, _ = \
                        get_batch_with_buckets(train_set, FLAGS.batch_size, bucket_id)
                    D_loss += step_D_loss / FLAGS.steps_per_checkpoint / D_step
                # value net
                _, _ = model.train_step(sess, encoder_inputs, \
                                        decoder_inputs, weights, \
                                        bucket_id, seq_lens, GAN_mode='V')
                #TODO: notice the inputs batch now
                for _ in range(G_step):# G steps
                    step_loss, step_perp, step_reward = \
                        model.train_step(sess, encoder_inputs, \
                                         decoder_inputs, weights, \
                                         bucket_id, seq_lens, \
                                         GAN_mode='G')
                    if 'ESGAN' in FLAGS.gan_type or FLAGS.gan_type == 'REGS':
                        step_reward = np.concatenate((step_reward, \
                                        np.zeros((FLAGS.batch_size, _buckets[-1][1] - _buckets[bucket_id][1]))),
                                        axis=1)
                    bucket_times[bucket_id] += 1
                    # each time get another batch
                    random_number_01 = np.random.random_sample()
                    bucket_id = min([i for i in range(len(train_buckets_scale))
                                     if train_buckets_scale[i] > random_number_01])
                    encoder_inputs, decoder_inputs, weights, seq_lens, _ = \
                        get_batch_with_buckets(train_set, FLAGS.batch_size, bucket_id)
                    #TODO notice the relationship between steps_per_checkpoint and G steps
                    reward += np.sum(step_reward, axis=0) / FLAGS.batch_size / G_step

            #REINFORCE
            else:
                step_samples, _ = model.train_step(sess, encoder_inputs, \
                                                   decoder_inputs, weights, \
                                                   bucket_id, seq_lens, forward=True)
                step_reward = check_batch_ans(model.critic, encoder_inputs, seq_lens, step_samples)
                step_loss, step_perp, _ = \
                    model.train_step(sess, encoder_inputs, \
                                     decoder_inputs, weights, \
                                     bucket_id, seq_lens, \
                                     decoder_outputs=step_samples, \
                                     rewards=step_reward)
                reward += sum(step_reward) / FLAGS.batch_size / FLAGS.steps_per_checkpoint
                
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += sum(sum(step_loss)) / FLAGS.batch_size / FLAGS.steps_per_checkpoint
            perp += step_perp / FLAGS.steps_per_checkpoint
            # log, save and eval
            current_step += 1
            if current_step % FLAGS.steps_per_checkpoint == 0:
                #print(value)
                print ("global step %d; learning rate %.8f; D lr %.8f; step-time %.2f;"
                       % (model.global_step.eval(),
                          model.learning_rate.eval(),
                          model.D_lr.eval(),
                          step_time))
                print("perp %.4f" % perp)
                #print("loss %.8f" % loss)
                print(loss)
                if hasattr(model.critic, 'discriminator'):
                    print("D-loss %.4f" % D_loss)
                    #print("value-loss %.4f" % value_loss)
                if 'ESGAN' in FLAGS.gan_type or FLAGS.gan_type == 'REGS':
                    len_times = []
                    for k, bucket in enumerate(_buckets):
                        len_times += [sum(bucket_times[k:])+1e-12] * (bucket[1]-len(len_times))
                    reward = np.true_divide(reward, len_times)
                print("reward(D_fake_value) {}".format(reward))
                f_reward_out.write("{}\n".format(reward))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if 'EB' in FLAGS.gan_type:
                    if len(previous_rewards) > 2 and np.sum(reward) > max(previous_rewards[-3:]):
                        sess.run(model.op_lr_decay)
                    elif len(previous_rewards) > 2 and np.sum(reward) < min(previous_rewards[-3:]):
                        sess.run(model.op_D_lr_decay)
                else:
                    if len(previous_rewards) > 2 and np.sum(reward) < min(previous_rewards[-3:]):
                        sess.run(model.op_lr_decay)
                    elif len(previous_rewards) > 2 and np.sum(reward) > max(previous_rewards[-3:]):
                        sess.run(model.op_D_lr_decay)
                previous_rewards.append(np.sum(reward))
                # write summary
                feed_dict = {}
                feed_dict[summary_vars[0]] = loss
                feed_dict[summary_vars[1]] = perp
                feed_dict[summary_vars[2]] = np.sum(reward) / _buckets[-1][1]
                if hasattr(model.critic, 'discriminator'):
                    feed_dict[summary_vars[3]] = D_loss
                summary_str = sess.run(summary_ops,
                                       feed_dict=feed_dict)
                writer.add_summary(summary_str, model.global_step.eval())
                writer.flush()
                # Save checkpoint and zero timer and loss.
                ckpt_path = os.path.join(FLAGS.model_dir, "ckpt")
                model.saver.save(sess, ckpt_path, global_step=model.global_step)
                if model.global_step.eval() >= 12600:
                    return
                step_time, loss, reward = 0.0, 0.0, 0.0
                perp, D_loss = 0.0, 0.0
                bucket_times = [0 for _ in range(len(_buckets))]

                sys.stdout.flush()

def train_value():
    value_path = os.path.join(FLAGS.pre_D_model_dir, '..', 'value')
    if not os.path.exists(value_path):
        os.makedirs(value_path)
    def build_summaries():
        loss = tf.Variable(0.)
        tf.summary.scalar("V-loss", loss)
        summary_vars = [loss]
        summary_ops = tf.summary.merge_all()
        return summary_ops, summary_vars
    # parse data and build vocab if there do not exist one.
    train, _, _ = data_utils.prepare_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
    with tf.Session() as sess:
        # build the model
        model = Seq2Seq(
            'TRAIN',
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.vocab_size,
            _buckets,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            critic=FLAGS.gan_type,
            critic_size=FLAGS.gan_size,
            critic_num_layers=FLAGS.gan_num_layers,
            other_option=FLAGS.option,
            use_attn=FLAGS.use_attn,
            output_sample=True,
            input_embed=True,
            feed_prev=False,
            batch_size=FLAGS.batch_size,
            D_lr=FLAGS.D_lr,
            D_lr_decay_factor=FLAGS.D_lr_decay_factor,
            dtype=tf.float32)
        # build summary and initialize
        summary_ops, summary_vars = build_summaries()
        sess.run(tf.variables_initializer(tf.global_variables()))
        log_dir = os.path.join(value_path, 'log')
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        pre_V_ckpt = tf.train.get_checkpoint_state(value_path)
        if pre_V_ckpt and tf.train.checkpoint_exists(pre_V_ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(pre_V_ckpt.model_checkpoint_path))
            model.pre_value_saver.restore(sess, pre_V_ckpt.model_checkpoint_path)
        else:
            print ('no previous model, create a new one')
        pre_ckpt = tf.train.get_checkpoint_state(FLAGS.pre_model_dir)
        if pre_ckpt and tf.train.checkpoint_exists(pre_ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(pre_ckpt.model_checkpoint_path))
            model.pre_saver.restore(sess, pre_ckpt.model_checkpoint_path)
        pre_D_ckpt = tf.train.get_checkpoint_state(FLAGS.pre_D_model_dir)
        if pre_D_ckpt and tf.train.checkpoint_exists(pre_D_ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(pre_D_ckpt.model_checkpoint_path))
            model.pre_D_saver.restore(sess, pre_D_ckpt.model_checkpoint_path)

        # load in train and dev(valid) data with buckets
        train_set = read_data_with_buckets(train, FLAGS.max_train_data_size)
        train_buckets_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_buckets_sizes))
        print ('each buckets has: {d}'.format(d=train_buckets_sizes))
        train_buckets_scale = [sum(train_buckets_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_buckets_sizes))]

        # main process
        step_time, loss = 0.0, 0.0
        current_step = 0
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            encoder_inputs, decoder_inputs, weights, seq_lens, _ = \
                get_batch_with_buckets(train_set, FLAGS.batch_size, bucket_id)

            start_time = time.time()
            step_loss, _ = \
                model.train_step(sess, encoder_inputs, \
                                 decoder_inputs, weights, \
                                 bucket_id, seq_lens, GAN_mode='V')
            loss += step_loss / FLAGS.steps_per_checkpoint
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            
            current_step += 1
            if current_step % FLAGS.steps_per_checkpoint == 0:
                print("global step %d; step-time %.2f; V-loss %.4f"
                      % (model.global_V_step.eval(),
                         step_time, loss))
                feed_dict = {}
                feed_dict[summary_vars[0]] = loss
                summary_str = sess.run(summary_ops,
                                       feed_dict=feed_dict)
                writer.add_summary(summary_str, model.global_V_step.eval())
                writer.flush()

                ckpt_path = os.path.join(value_path, "ckpt")
                model.pre_value_saver.save(sess, ckpt_path, global_step=model.global_V_step)
                step_time, loss = 0.0, 0.0

                sys.stdout.flush()

def train_critic():
    if not os.path.exists(FLAGS.pre_D_model_dir):
        os.makedirs(FLAGS.pre_D_model_dir)
    def build_summaries():
        D_loss = tf.Variable(0.)
        tf.summary.scalar("D_loss", D_loss)
        summary_vars = [D_loss]
        summary_ops = tf.summary.merge_all()
        return summary_ops, summary_vars
    # parse data and build vocab if there do not exist one.
    train, _, _ = data_utils.prepare_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
    with tf.Session() as sess:
        # build the model
        model = Seq2Seq(
            'TRAIN',
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.vocab_size,
            _buckets,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            critic=FLAGS.gan_type,
            critic_size=FLAGS.gan_size,
            critic_num_layers=FLAGS.gan_num_layers,
            other_option=FLAGS.option,
            use_attn=FLAGS.use_attn,
            output_sample=True,
            input_embed=True,
            feed_prev=False,
            batch_size=FLAGS.batch_size,
            D_lr=FLAGS.D_lr,
            D_lr_decay_factor=FLAGS.D_lr_decay_factor,
            dtype=tf.float32)
        # build summary and initialize
        summary_ops, summary_vars = build_summaries()
        sess.run(tf.variables_initializer(tf.global_variables()))
        log_dir = os.path.join(FLAGS.pre_D_model_dir, 'log')
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        pre_D_ckpt = tf.train.get_checkpoint_state(FLAGS.pre_D_model_dir)
        if pre_D_ckpt and tf.train.checkpoint_exists(pre_D_ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(pre_D_ckpt.model_checkpoint_path))
            model.pre_D_saver.restore(sess, pre_D_ckpt.model_checkpoint_path)
        else:
            print ('no previous model, create a new one')
        
        pre_ckpt = tf.train.get_checkpoint_state(FLAGS.pre_model_dir)
        if pre_ckpt and tf.train.checkpoint_exists(pre_ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(pre_ckpt.model_checkpoint_path))
            model.pre_saver.restore(sess, pre_ckpt.model_checkpoint_path)

        # load in train and dev(valid) data with buckets
        train_set = read_data_with_buckets(train, FLAGS.max_train_data_size)
        train_buckets_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_buckets_sizes))
        print ('each buckets has: {d}'.format(d=train_buckets_sizes))
        train_buckets_scale = [sum(train_buckets_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_buckets_sizes))]

        # main process
        step_time, D_loss = 0.0, 0.0
        current_step = 0
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            encoder_inputs, decoder_inputs, weights, seq_lens, _ = \
                get_batch_with_buckets(train_set, FLAGS.batch_size, bucket_id)

            start_time = time.time()
            step_D_loss, _ = \
                model.train_step(sess, encoder_inputs, \
                                 decoder_inputs, weights, \
                                 bucket_id, seq_lens, GAN_mode='D')
            D_loss += step_D_loss / FLAGS.steps_per_checkpoint
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            
            current_step += 1
            if current_step % FLAGS.steps_per_checkpoint == 0:
                print("global step D %d; step-time %.2f; D-loss %.4f"
                      % (model.global_D_step.eval(),
                         step_time, D_loss))
                feed_dict = {}
                feed_dict[summary_vars[0]] = D_loss
                summary_str = sess.run(summary_ops,
                                       feed_dict=feed_dict)
                writer.add_summary(summary_str, model.global_D_step.eval())
                writer.flush()

                D_ckpt_path = os.path.join(FLAGS.pre_D_model_dir, "ckpt")
                model.pre_D_saver.save(sess, D_ckpt_path, global_step=model.global_D_step)
                if model.global_D_step.eval() >= 12600:
                    return
                step_time, D_loss = 0.0, 0.0
                sys.stdout.flush()

def test():
    
    with tf.Session() as sess:
        # build the model
        model = Seq2Seq(
            'TEST',
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.vocab_size,
            _buckets,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            critic=FLAGS.gan_type,
            critic_size=FLAGS.gan_size,
            critic_num_layers=FLAGS.gan_num_layers,
            use_attn=FLAGS.use_attn,
            output_sample=True,
            input_embed=True,
            feed_prev=True,
            batch_size=FLAGS.batch_size,
            dtype=tf.float32)

        sess.run(tf.variables_initializer(tf.global_variables()))
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        model.pre_saver.restore(sess, ckpt.model_checkpoint_path)
        print ('read in model from {}'.format(ckpt.model_checkpoint_path))
        
        vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        def beam_search(encoder_inputs, encoder_len):
            beam_size = 10
            beam_len = 1
            beam_encoder_inputs = []
            for l in range(_buckets[-1][0]):
                beam_encoder_inputs.append(
                    np.array([encoder_inputs[l][0] for _ in range(beam_size)], np.int32))
            beam_encoder_len = [encoder_len[0] for _ in range(beam_size)]
            decoder_inputs = [[data_utils.GO_ID]] + [[data_utils.PAD_ID] for _ in range(_buckets[-1][1]-1)]
            outs = model.stepwise_test_beam(sess, encoder_inputs, encoder_len, decoder_inputs)
            outs = outs[0][0]
            top_k_idx = np.argsort(outs[0])[-beam_size:]
            path = [[data_utils.GO_ID for _ in range(beam_size)], top_k_idx] + \
                   [[data_utils.PAD_ID for _ in range(beam_size)] for _ in range(_buckets[-1][1]-beam_len-1)]
            end_path = []
            probs = np.zeros(beam_size)
            for k in range(beam_size):
                probs[k] += np.log(outs[0][top_k_idx[k]])
                end_path.append(data_utils.EOS_ID == path[-1][k])

            #for _ in range(2-1):
            for _ in range(_buckets[-1][1]-1):
                beam_len += 1
                outs = model.stepwise_test_beam(sess, beam_encoder_inputs, beam_encoder_len, path)
                outs = outs[0][beam_len-1]
                top_k_idxes = [np.argsort(outs[p])[-beam_size:] for p in range(beam_size)]
                edges = []
                tmp_probs = []
                for p in range(beam_size):
                    if end_path[p]:
                        tmp_probs.extend([probs[p]])
                        edges.append(1)
                    else:
                        tmp_probs.extend([probs[p]+np.log(outs[p][k]) for k in top_k_idxes[p]])
                        edges.append(beam_size)
                top_k_path_idx = np.argsort(tmp_probs)[-beam_size:]
                edges_scale = [sum(edges[:i+1]) for i in range(beam_size)]
                tmp_path = []
                for l in range(beam_len):
                    step = []
                    for k in top_k_path_idx:
                        path_id = min([i for i in range(beam_size) if edges_scale[i] > k])
                        step.append(path[l][path_id])
                    tmp_path.append(step)
                path = tmp_path
                step = []
                expand_edges_scale = [0] + edges_scale
                for k in top_k_path_idx:
                    path_id = min([i for i in range(beam_size) if edges_scale[i] > k])
                    step_id = k - expand_edges_scale[path_id]
                    step.append(top_k_idxes[path_id][step_id])
                path.append(step)
                for i, k in enumerate(top_k_path_idx):
                    probs[i] = tmp_probs[k]
                    end_path[i] = False
                    for l in range(beam_len+1):
                        if path[l][i] == data_utils.EOS_ID:
                            end_path[i] = True
                path += [[data_utils.PAD_ID for _ in range(beam_size)] for _ in range(_buckets[-1][1]-beam_len-1)]
            return path, probs

        def MMI(decoder_inputs):
            lm_probs = model.lm_prob(sess, decoder_inputs)
            lm_probs = lm_probs[0]
            lm_prob = []
            lens = []
            for p in range(len(decoder_inputs[0])):
                tmp_prob = 0.0
                for l in range(len(decoder_inputs)-1):
                    if l < 3:
                        tmp_prob += np.log(lm_probs[l][p][decoder_inputs[l+1][p]])
                    if decoder_inputs[l+1][p] == data_utils.EOS_ID:
                        lens.append(l)
                        break
                lm_prob.append(tmp_prob)
                if len(lens) < len(lm_prob):
                    lens.append(len(decoder_inputs)-1)
            return lm_prob, lens

        # main testing process.
        if FLAGS.test_type == 'BLEU':
            count = 0
            bleu_1, bleu_2, bleu_3, bleu_4 = 0.0, 0.0, 0.0, 0.0
            data_dir, file_name = FLAGS.data_path.rsplit('/',1)
            test_data = os.path.join(data_dir, 'test_' + file_name)
            #test_fout = os.path.join(FLAGS.model_dir, 'test_argmax_outs.txt')
            #with open(test_data,'r') as f, open(test_fout,'w') as fout:
            with open(test_data,'r') as f:
                all_f = f.readlines()
                all_ref = [all_f[2*i+1] for i in range(int(len(all_f)/2))]
                all_f = [all_f[2*i] for i in range(int(len(all_f)/2))]
                tmp_encoder_inputs = []
                tmp_encoder_lens = []
                for _, sentence in enumerate(all_f[:1000]):
                    token_ids = data_utils.sentence_to_token_ids(
                        tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
                    encoder_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
                    tmp_encoder_lens.append(len(token_ids))
                    tmp_encoder_inputs.append(list(reversed(token_ids)) + encoder_pad)
                batch_encoder_inputs = []
                batch_encoder_lens = []
                batch_sentences = []
                batch_flag = 0
                for _ in range(int(len(tmp_encoder_inputs) / FLAGS.batch_size)):
                    encoder_inputs = []
                    for idx in range(_buckets[-1][0]):
                        encoder_inputs.append(
                            np.array([tmp_encoder_inputs[batch_flag+batch_idx][idx]
                                      for batch_idx in range(FLAGS.batch_size)],
                                     dtype = np.int32))
                    batch_encoder_inputs.append(encoder_inputs)
                    batch_encoder_lens.append(tmp_encoder_lens[batch_flag:batch_flag+FLAGS.batch_size])
                    batch_sentences.append(all_f[batch_flag:batch_flag+FLAGS.batch_size])
                    batch_flag += FLAGS.batch_size
                decoder_inputs = [[data_utils.GO_ID for _ in range(FLAGS.batch_size)]]

                for batch_idx, encoder_inputs in enumerate(batch_encoder_inputs):
                    outputs, _ = model.dynamic_decode(sess, encoder_inputs, batch_encoder_lens[batch_idx], \
                                                      decoder_inputs)
                    for idx, sentence in enumerate(all_f[batch_idx*FLAGS.batch_size:(batch_idx+1)*FLAGS.batch_size]):
                        an_output = [output_ids[idx] for output_ids in outputs]
                        if data_utils.EOS_ID in an_output:
                            an_output = an_output[:an_output.index(data_utils.EOS_ID)]
                        ans = " ".join([tf.compat.as_str(rev_vocab[output]) for output in an_output])
                        #fout.write(sentence)
                        #fout.write(ans)
                        #fout.write('\n')
                        #bleu_2 += sentence_bleu(all_ref[batch_idx*FLAGS.batch_size+idx].split(), ans.split(), weights=[0.5,0.5,0,0])
                        #bleu_3 += sentence_bleu(all_ref[batch_idx*FLAGS.batch_size+idx].split(), ans.split(), weights=[0.33,0.33,0.33,0])
                        #bleu_4 += sentence_bleu(all_ref[batch_idx*FLAGS.batch_size+idx].split(), ans.split(), weights=[0.25,0.25,0.25,0.25])
                        tmp_bleu_1 = modified_precision(all_ref[batch_idx*FLAGS.batch_size+idx].split(), ans.split(), 1)
                        tmp_bleu_2 = modified_precision(all_ref[batch_idx*FLAGS.batch_size+idx].split(), ans.split(), 2)
                        #print(tmp_bleu_1)
                        #print(tmp_bleu_2)
                        bleu_1 += tmp_bleu_1#*len(ans.split())
                        bleu_2 += tmp_bleu_2#*len(ans.split())
                        #count += len(ans.split())
                #if bleu_2/count == 0:
                #    BLEU_2 = np.exp(0.5*np.log(bleu_1/count).5*np.log(bleu_2/count))
                #else:
                #    BLEU_2 = np.exp(0.5*np.log(bleu_1/count)+0.5*np.log(bleu_2/count))
                #print('BLEU-2:{}'.format(BLEU_2))
                print('UNI:{}'.format(bleu_1))
                print('BI:{}'.format(bleu_2))

        elif FLAGS.test_type == 'per_print':
            beam_bleu_1, mmi_bleu_1 = 0.0, 0.0
            beam_bleu_2, mmi_bleu_2 = 0.0, 0.0
            data_dir, file_name = FLAGS.data_path.rsplit('/',1)
            #test_data = os.path.join(data_dir, 'test_' + file_name)
            test_data = os.path.join('other_test_' + file_name)
            bs_output_path = os.path.join(FLAGS.model_dir, 'other_test_per_BS.txt')
            mmi_output_path = os.path.join(FLAGS.model_dir, 'other_test_per_MMI.txt')
            with open(test_data,'r') as f, open(bs_output_path,'w') as bs_fout, open(mmi_output_path,'w') as mmi_fout:
                all_f = f.readlines()
                all_ref = [all_f[2*i+1] for i in range(int(len(all_f)/2))]
                all_f = [all_f[2*i] for i in range(int(len(all_f)/2))]
                for idx, sentence in enumerate(all_f):
                    token_ids = data_utils.sentence_to_token_ids(
                        tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
                    encoder_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
                    encoder_lens = [len(token_ids)]
                    token_ids = list(reversed(token_ids)) + encoder_pad
                    encoder_inputs = []
                    for idx in token_ids:
                        encoder_inputs.append([idx])
                    decoder_inputs = [[data_utils.GO_ID]]
                    bs_fout.write(sentence)
                    mmi_fout.write(sentence)
                    """
                    # greedy
                    outs, log_prob = model.dynamic_decode(sess, encoder_inputs, encoder_lens, \
                                                             decoder_inputs)
                    outputs = [output_ids[0] for output_ids in outs]
                    if data_utils.EOS_ID in outputs:
                        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                    greedy = " ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs])
                    greedy_bleu_2 += \
                        sentence_bleu(all_ref[idx].split(), \
                                      greedy.split(), weights=[0.5,0.5,0,0])
                    fout.write(greedy+",")
                    """
                    # beam_search
                    outs, log_prob = beam_search(encoder_inputs, encoder_lens)
                    outputs = [output_ids[-1] for output_ids in outs]
                    if data_utils.EOS_ID in outputs:
                        outputs = outputs[1:outputs.index(data_utils.EOS_ID)]
                    beam = " ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs])
                    bs_fout.write(beam+"\n")
                    #beam_bleu_1 += \
                    #        modified_precision(all_ref[idx].split(), beam.split(), 1)
                    #beam_bleu_2 += \
                    #        modified_precision(all_ref[idx].split(), beam.split(), 2)
                    # MMI
                    lm_log_prob, lens = MMI(outs)
                    MMI_scores = np.array(log_prob) - 0.5 * np.array(lm_log_prob) + 0.5 * np.array(lens)
                    MMI_idx = np.argsort(MMI_scores)
                    outputs = [output_ids[MMI_idx[-1]] for output_ids in outs]
                    if data_utils.EOS_ID in outputs:
                        outputs = outputs[1:outputs.index(data_utils.EOS_ID)]
                    mmi = " ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs])
                    mmi_fout.write(mmi+"\n")
                    #mmi_bleu_1 += \
                    #        modified_precision(all_ref[idx].split(), mmi.split(), 1)
                    #mmi_bleu_2 += \
                    #        modified_precision(all_ref[idx].split(), mmi.split(), 2)
                #print('beam search, BLEU-1:{}\n'.format(beam_bleu_1))
                #print('beam search, BLEU-2:{}\n'.format(beam_bleu_2))
                #print('MMI, BLEU-1:{}'.format(mmi_bleu_1))
                #print('MMI, BLEU-2:{}'.format(mmi_bleu_2))

        elif FLAGS.test_type == 'accuracy':
            grammar_critic = load_critic(FLAGS.test_critic)
            argmax_correct, argmax_count = 0, 0
            sample_correct, sample_count = 0, 0
            data_dir, file_name = FLAGS.data_path.rsplit('/',1)
            test_data = os.path.join(data_dir, 'test_' + file_name)
            with open(test_data,'r') as f:
                all_f = f.readlines()
                all_ref = [all_f[2*i+1] for i in range(int(len(all_f)/2))]
                all_f = [all_f[2*i] for i in range(int(len(all_f)/2))]
                tmp_encoder_inputs = []
                tmp_encoder_lens = []
                for _, sentence in enumerate(all_f):
                    # step
                    token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
                    encoder_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
                    tmp_encoder_lens.append(len(token_ids))
                    tmp_encoder_inputs.append(list(reversed(token_ids)) + encoder_pad)
                batch_encoder_inputs = []
                batch_encoder_lens = []
                batch_sentences = []
                batch_flag = 0
                for _ in range(int(len(tmp_encoder_inputs) / FLAGS.batch_size)):
                    encoder_inputs = []
                    for idx in range(_buckets[-1][0]):
                        encoder_inputs.append(
                            np.array([tmp_encoder_inputs[batch_flag+batch_idx][idx]
                                      for batch_idx in range(FLAGS.batch_size)],
                                     dtype = np.int32))
                    batch_encoder_inputs.append(encoder_inputs)
                    batch_encoder_lens.append(tmp_encoder_lens[batch_flag:batch_flag+FLAGS.batch_size])
                    batch_sentences.append(all_f[batch_flag:batch_flag+FLAGS.batch_size])
                    batch_flag += FLAGS.batch_size
                decoder_inputs = [[data_utils.GO_ID for _ in range(FLAGS.batch_size)]]
                print(len(batch_encoder_inputs))

                for batch_idx, encoder_inputs in enumerate(batch_encoder_inputs):
                    outputs, _ = model.dynamic_decode(sess, encoder_inputs, batch_encoder_lens[batch_idx], \
                                                      decoder_inputs)
                    for idx, sentence in enumerate(all_f[batch_idx*FLAGS.batch_size:(batch_idx+1)*FLAGS.batch_size]):
                        #outputs = [int(np.argmax(logit, axis=1)) for logit in outputs]
                        an_output = [output_ids[idx] for output_ids in outputs]
                        if data_utils.EOS_ID in an_output:
                            an_output = an_output[:an_output.index(data_utils.EOS_ID)]
                        ans = " ".join([tf.compat.as_str(rev_vocab[output]) for output in an_output])
                        #if ans in mydata.seq_abbrev(sentence):
                        if grammar_critic.check_ans(sentence.split(), ans):
                            argmax_correct += 1
                        argmax_count += 1

                hist_samples = {}
                coverage = 0.0
                for _ in range(50):
                    for batch_idx, encoder_inputs in enumerate(batch_encoder_inputs):
                        samples, log_prob = model.dynamic_decode(sess, encoder_inputs, batch_encoder_lens[batch_idx], \
                                                                 decoder_inputs, mode='sample')
                        for idx, sentence in enumerate(batch_sentences[batch_idx]):
                            sample = [sample_ids[idx] for sample_ids in samples]
                            if data_utils.EOS_ID in sample:
                                sample = sample[:sample.index(data_utils.EOS_ID)]
                            ans = " ".join([tf.compat.as_str(rev_vocab[output]) for output in sample])
                            if grammar_critic.check_ans(sentence.split(), ans):
                                sample_correct += 1
                                if sentence.strip() in hist_samples:
                                    if ans not in hist_samples[sentence.strip()]:
                                        hist_samples[sentence.strip()].append(ans)
                                        coverage += 1.0 / grammar_critic.possible_ans_num(sentence.split())
                                else:
                                    hist_samples[sentence.strip()] = [ans]
                                    coverage += 1.0 / grammar_critic.possible_ans_num(sentence.split())
                            sample_count += 1
                print('argmax accuracy: {}'.format(float(argmax_correct) / argmax_count))
                print('sample accuracy: {}'.format(float(sample_correct) / sample_count))
                print('sample coverage: {}'.format(coverage / len(tmp_encoder_inputs)))
                    
        elif FLAGS.test_type == 'print_test':
            data_dir, file_name = FLAGS.data_path.rsplit('/',1)
            test_data = os.path.join(data_dir, 'test_' + file_name)
            test_fout = os.path.join(FLAGS.model_dir, 'test_argmax_outs.txt')
            with open(test_data,'r') as f, open(test_fout,'w') as fout:
                all_f = f.readlines()
                all_ref = [all_f[2*i+1] for i in range(int(len(all_f)/2))]
                all_f = [all_f[2*i] for i in range(int(len(all_f)/2))]
                tmp_encoder_inputs = []
                tmp_encoder_lens = []
                for _, sentence in enumerate(all_f):
                    # step
                    token_ids = data_utils.sentence_to_token_ids(
                        tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
                    encoder_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
                    tmp_encoder_lens.append(len(token_ids))
                    tmp_encoder_inputs.append(list(reversed(token_ids)) + encoder_pad)
                batch_encoder_inputs = []
                batch_encoder_lens = []
                encoder_inputs = []
                for _ in range(int(len(tmp_encoder_inputs) / FLAGS.batch_size)):
                    encoder_inputs = []
                    for idx in range(_buckets[-1][0]):
                        encoder_inputs.append(
                            np.array([tmp_encoder_inputs[batch_flag+batch_idx][idx]
                                      for batch_idx in range(FLAGS.batch_size)],
                                     dtype = np.int32))
                    batch_encoder_inputs.append(encoder_inputs)
                    batch_encoder_lens.append(tmp_encoder_lens[batch_flag:batch_flag+FLAGS.batch_size])
                    batch_encoder_inputs.append(encoder_inputs)
                    batch_encoder_lens.append(tmp_encoder_lens[idx])
                decoder_inputs = [[data_utils.GO_ID for _ in range(batch_size)]]
                print(len(batch_encoder_inputs))

                for batch_idx, encoder_inputs in enumerate(batch_encoder_inputs):
                    outputs, _ = model.dynamic_decode(sess, encoder_inputs, batch_encoder_lens[batch_idx], \
                                                      decoder_inputs)
                    #samples = []
                    #for _ in range(10):
                    #   tmp_samples, _ = model.dynamic_decode(sess, encoder_inputs, batch_encoder_lens[batch_idx], \
                    #                                     decoder_inputs, mode='sample')
                    #   samples.append(tmp_samples)
                    for idx, sentence in enumerate(all_f):
                        an_output = [output_ids[idx] for output_ids in outputs]
                        if data_utils.EOS_ID in an_output:
                            an_output = an_output[:an_output.index(data_utils.EOS_ID)]
                        ans = " ".join([tf.compat.as_str(rev_vocab[output]) for output in an_output])
                        fout.write(sentence)
                        fout.write(ans)
                        fout.write('\n')
                        #for idx_sample, sample in enumerate(samples):
                        #    a_sample = [output_ids[idx] for output_ids in sample]
                        #    if data_utils.EOS_ID in a_sample:
                        #        a_sample = a_sample[:a_sample.index(data_utils.EOS_ID)]
                        #    ans = " ".join([tf.compat.as_str(rev_vocab[output]) for output in a_sample])
                        #    fout.write(ans)
                        #    fout.write('\n')
                        #fout.write('\n')
                    
        elif FLAGS.test_type == 'perp':
            with open(FLAGS.test_data,'r') as f:
                all_f = f.readlines()
                batch_size = len(all_f)
                tmp_encoder_inputs = []
                tmp_encoder_lens = []
                for _, sentence in enumerate(all_f):
                    # step
                    token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
                    encoder_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
                    tmp_encoder_lens.append(len(token_ids))
                    # feature in my implementation
                    tmp_encoder_inputs.append(list(reversed(token_ids)) + encoder_pad)
                batch_encoder_inputs = []
                batch_encoder_lens = []
                encoder_inputs = []
                for idx in range(_buckets[-1][0]):
                    encoder_inputs.append(
                        np.array([tmp_encoder_inputs[batch_idx][idx]
                                  for batch_idx in range(batch_size)],
                                 dtype = np.int32))
                batch_encoder_inputs.append(encoder_inputs)
                batch_encoder_lens.append(tmp_encoder_lens)
                decoder_inputs = [[data_utils.GO_ID for _ in range(batch_size)]]

                total_perp = []
                for batch_idx, encoder_inputs in enumerate(batch_encoder_inputs):
                    for _ in range(10):
                        tmp_samples, log_probs = model.dynamic_decode(sess, encoder_inputs, batch_encoder_lens[batch_idx], \
                                                         decoder_inputs, mode='sample')
                        total_perp.extend(log_probs)
                    #outputs, perps = model.dynamic_decode(sess, encoder_inputs, batch_encoder_lens[batch_idx], \
                    #                                  decoder_inputs)
                print(float(sum(total_perp))/len(total_perp))
                

        else:
            sys.stdout.write('> ')
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            while sentence:
                if sentence.strip() == 'exit()':
                    break
                # step
                token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
                encoder_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
                encoder_lens = [len(token_ids)]
                # feature in my implementation
                token_ids = list(reversed(token_ids)) + encoder_pad
                encoder_inputs = []
                for idx in token_ids:
                    encoder_inputs.append([idx])
                decoder_inputs = [[data_utils.GO_ID]]
                
                if FLAGS.test_type == 'realtime_argmax':
                    outs, log_prob = model.dynamic_decode(sess, encoder_inputs, encoder_lens, \
                                                             decoder_inputs)
                    #outs = [int(np.argmax(logit, axis=1)) for logit in outs]
                    
                    outputs = [output_ids[0] for output_ids in outs]
                    if data_utils.EOS_ID in outputs:
                        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                    print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))
                    print(log_prob[0])

                elif FLAGS.test_type == 'realtime_beam_search':
                    outs, log_prob = beam_search(encoder_inputs, encoder_lens)
                    for b in range(len(outs[0])):
                        outputs = [output_ids[b] for output_ids in outs]
                        if data_utils.EOS_ID in outputs:
                            outputs = outputs[1:outputs.index(data_utils.EOS_ID)]
                        print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))
                        print(log_prob[b])

                elif FLAGS.test_type == 'realtime_MMI':
                    outs, log_prob = beam_search(encoder_inputs, encoder_lens)
                    lm_log_prob, lens = MMI(outs)
                    MMI_scores = np.array(log_prob) - 0.5 * np.array(lm_log_prob) + 0.5 * np.array(lens)
                    MMI_idx = np.argsort(MMI_scores)
                    for b in [MMI_idx[-1]]:#range(len(outs[0])):
                        outputs = [output_ids[b] for output_ids in outs]
                        if data_utils.EOS_ID in outputs:
                            outputs = outputs[1:outputs.index(data_utils.EOS_ID)]
                        print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))
                        print(log_prob[b])
                        print(lm_log_prob[b])
                        print(lens[b])

                elif FLAGS.test_type == 'realtime_sample':
                    samples, log_prob = model.dynamic_decode(sess, encoder_inputs, encoder_lens, \
                                                             decoder_inputs, mode='sample')
                    sample = [sample_ids[0] for sample_ids in samples]
                    #if data_utils.EOS_ID in sample:
                    #    sample = sample[:sample.index(data_utils.EOS_ID)]
                    print(sample)
                    print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in sample]))
                    print(log_prob)
                
                sys.stdout.write('> ')
                sys.stdout.flush()
                sentence = sys.stdin.readline()

def test_D():
    with tf.Session() as sess:
        model = Seq2Seq(
            'D_TEST',
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.vocab_size,
            _buckets,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            critic=FLAGS.gan_type,
            critic_size=FLAGS.gan_size,
            critic_num_layers=FLAGS.gan_num_layers,
            use_attn=FLAGS.use_attn,
            output_sample=True,
            input_embed=True,
            feed_prev=True,
            batch_size=FLAGS.batch_size,
            dtype=tf.float32)

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        #ckpt = tf.train.get_checkpoint_state(FLAGS.pre_D_model_dir)
        model.pre_saver.restore(sess, ckpt.model_checkpoint_path)
        print ('read in model from {}'.format(ckpt.model_checkpoint_path))
        #value_path = os.path.join(FLAGS.pre_D_model_dir, '..', 'value')
        V_ckpt = ckpt#tf.train.get_checkpoint_state(value_path)
        model.pre_V_saver.restore(sess, V_ckpt.model_checkpoint_path)
        print ('read in model from {}'.format(V_ckpt.model_checkpoint_path))
        #s2s_path = os.path.join(FLAGS.pre_model_dir, '../MLE')
        s2s_ckpt = ckpt#tf.train.get_checkpoint_state(s2s_path)
        model.pre_s2s_saver.restore(sess, s2s_ckpt.model_checkpoint_path)
        print ('read in model from {}'.format(s2s_ckpt.model_checkpoint_path))
        
        vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        if FLAGS.test_type == 'batch_test_D':
            data_dir, file_name = FLAGS.data_path.rsplit('/',1)
            test_data = os.path.join('test_D_' + file_name)
            out_lists = os.path.join(FLAGS.gan_type + 'test_D_' + '.out')
            all_total_score = []
            all_each_scores = []
            all_uniW = []
            with open(test_data,'r') as f:
                all_f = f.readlines()
                all_ref = [all_f[2*i+1] for i in range(int(len(all_f)/2))]
                all_f = [all_f[2*i] for i in range(int(len(all_f)/2))]
                tmp_Q_inputs = []
                tmp_Q_lens = []
                for _, sentence in enumerate(all_f):
                    token_ids = data_utils.sentence_to_token_ids(
                        tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
                    Q_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
                    tmp_Q_lens.append(len(token_ids))
                    tmp_Q_inputs.append(list(reversed(token_ids)) + Q_pad)
                tmp_A_inputs = []
                tmp_A_lens = []
                for _, sentence in enumerate(all_ref):
                    token_ids = data_utils.sentence_to_token_ids(
                        tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
                    A_pad = [data_utils.PAD_ID] * (_buckets[-1][1] - len(token_ids) - 1)
                    tmp_A_lens.append(len(token_ids))
                    tmp_A_inputs.append(list(token_ids) + [data_utils.EOS_ID] + A_pad)

                batch_Q_inputs = []
                batch_Q_lens = []
                batch_A_inputs = []
                batch_flag = 0
                bs = 1
                for _ in range(int(len(tmp_Q_inputs) / bs)):
                    # Q batch
                    Q_inputs = []
                    for idx in range(_buckets[-1][0]):
                        Q_inputs.append(
                            np.array([tmp_Q_inputs[batch_flag+batch_idx][idx]
                                      for batch_idx in range(bs)],
                                     dtype = np.int32))
                    batch_Q_inputs.append(Q_inputs)
                    batch_Q_lens.append([tmp_Q_lens[batch_flag]])
                    # A batch
                    A_inputs = []
                    for idx in range(_buckets[-1][1]):
                        A_inputs.append(
                            np.array([tmp_A_inputs[batch_flag+batch_idx][idx]
                                      for batch_idx in range(bs)],
                                     dtype = np.int32))
                    batch_A_inputs.append(A_inputs)
                    # batch flag
                    batch_flag += bs

                for Q_inputs, Q_lens, A_inputs in zip(batch_Q_inputs, batch_Q_lens, batch_A_inputs):
                    print(Q_inputs)
                    print(Q_lens)
                    print(A_inputs)
                    total_score, each_scores, uniW = model.test_discriminator(sess, Q_inputs, Q_lens, A_inputs)
                    for idx in range(len(Q_inputs[0])):
                        all_total_score.append(total_score)
                        all_each_scores.append([score[idx] for score in each_scores])
                        all_uniW.append([value[idx] for value in uniW])
                pickle.dump([all_total_score, all_each_scores, all_uniW], open(out_lists,'wb'))

        else:
            sys.stdout.write('> ')
            sys.stdout.flush()
            Q = sys.stdin.readline()
            sys.stdout.write('>> ')
            sys.stdout.flush()
            A = sys.stdin.readline()
            while Q and A:
                if Q.strip() == 'exit()' or A.strip() == 'exit()':
                    break
                token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(Q), vocab, normalize_digits=False)
                Q_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
                Q_lens = [len(token_ids)]
                token_ids = list(reversed(token_ids)) + Q_pad
                Q_inputs = []
                for idx in token_ids:
                    Q_inputs.append([idx])

                token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(A), vocab, normalize_digits=False)
                A_pad = [data_utils.PAD_ID] * (_buckets[-1][1] - len(token_ids)-1)
                token_ids = list(token_ids) + [data_utils.EOS_ID] + A_pad
                A_inputs = []
                for idx in token_ids:
                    A_inputs.append([idx])

                whole_score, each_score, uniW = model.test_discriminator(sess, Q_inputs, Q_lens, A_inputs)
                print(whole_score)
                print(each_score)
                print(uniW)

                sys.stdout.write('> ')
                sys.stdout.flush()
                Q = sys.stdin.readline()
                sys.stdout.write('>> ')
                sys.stdout.flush()
                A = sys.stdin.readline()

def read_data(data_path, maxlen, max_size=None):
    dataset = []
    with tf.gfile.GFile(data_path, mode='r') as data_file:
        source = data_file.readline()
        target = data_file.readline()
        counter = 0
        while source and target and \
                len(source.split()) < maxlen and len(target.split())+1 < maxlen and \
                (not max_size or counter < max_size):
            counter += 1
            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in target.split()]
            target_ids.append(EOS_ID)
            # form dataset
            #if len(target_ids) > 3:
            dataset.append([source_ids, target_ids])
            # next loop
            source = data_file.readline()
            target = data_file.readline()
    return dataset

def read_data_with_buckets(data_path, max_size=None):
    if FLAGS.option == 'MIXER':
        buckets = [_buckets[-1]]
    else:
        buckets = _buckets
    dataset = [[] for _ in buckets]
    with tf.gfile.GFile(data_path, mode='r') as data_file:
        source = data_file.readline()
        target = data_file.readline()
        counter = 0
        while source and target and \
                (not max_size or counter < max_size):
            counter += 1
            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in target.split()]
            target_ids.append(data_utils.EOS_ID)
            # form dataset
            for bucket_id, (source_size, target_size) in enumerate(buckets):
                #TODO: one can also check length of target_id or source_id
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    dataset[bucket_id].append([source_ids, target_ids])
                    break
            # next loop
            source = data_file.readline()
            target = data_file.readline()
    return dataset

def get_batch_with_buckets(data, batch_size, bucket_id, size=None):
    # data should be [whole_data_length x (source, target)] 
    # decoder_input should contain "GO" symbol and target should contain "EOS" symbol
    encoder_size, decoder_size = _buckets[bucket_id]
    encoder_inputs, decoder_inputs, seq_len = [], [], []

    for i in range(batch_size):
        encoder_input, decoder_input = random.choice(data[bucket_id])
        encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
        # feature in my implementation
        encoder_inputs.append(list(reversed(encoder_input)) + encoder_pad)
        seq_len.append(len(encoder_input))
        decoder_pad = [data_utils.PAD_ID] * (decoder_size - len(decoder_input))
        decoder_inputs.append([data_utils.GO_ID] + decoder_input + decoder_pad)

    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
    # make batch for encoder inputs
    for length_idx in range(encoder_size):
        batch_encoder_inputs.append(
            np.array([encoder_inputs[batch_idx][length_idx]
                      for batch_idx in range(batch_size)],
                     dtype = np.int32))
    # make batch for decoder inputs
    for length_idx in range(decoder_size):
        batch_decoder_inputs.append(
            np.array([decoder_inputs[batch_idx][length_idx]
                      for batch_idx in range(batch_size)],
                     dtype = np.int32))
        batch_weight = np.ones(batch_size, dtype = np.float32)
        for batch_idx in range(batch_size):
            if length_idx < decoder_size - 1:
                target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)

    return batch_encoder_inputs, batch_decoder_inputs, batch_weights, seq_len, encoder_inputs

def check_batch_ans(mycritic, inp, inp_lens, ans):
    inp = [[inp[t][i] for t in reversed(range(inp_lens[i]))] for i in range(len(inp[0]))]
    ans = [[ans[t][i] for t in range(len(ans))] for i in range(len(ans[0]))]
    rewards = []
    for per_inp, per_ans in zip(inp, ans):
        per_inp = [ tf.compat.as_str(mycritic.rev_vocab[out]) for out in per_inp ]
        if data_utils.EOS_ID in per_ans:
            per_ans = per_ans[:per_ans.index(data_utils.EOS_ID)]
        per_ans = ' '.join(tf.compat.as_str(mycritic.rev_vocab[out]) for out in per_ans)
        #print(per_inp)
        #print(per_ans)
        if mycritic.check_ans(per_inp, per_ans):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

if __name__ == '__main__':
    if FLAGS.test_type == 'None':
        if FLAGS.option == 'pretrain_D':
            train_critic()
        elif FLAGS.option == 'pretrain_V':
            train_value()
        else:
            if not os.path.exists(FLAGS.model_dir):
                os.makedirs(FLAGS.model_dir)
            with open('{}/model.conf'.format(FLAGS.model_dir),'w') as f:
                for key, value in vars(FLAGS).items():
                    f.write("{}={}\n".format(key, value))
            if FLAGS.gan_type == 'None':
                train_s2s()
            else:
                train_gan()
    elif FLAGS.test_type == 'test_D' or FLAGS.test_type == 'batch_test_D':
        test_D()
    else:
        test()
