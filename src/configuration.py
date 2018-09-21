# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ configuration.py ]
#   Synopsis     [ Training configurations and Hyperparameters for the model. ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import argparse


##################
# CONFIGURATIONS #
##################
def get_config():
	parser = argparse.ArgumentParser(description="daisy configurations")
	#--mode--#
	parser.add_argument('--pre_train', action='store_true', help='activate generator supervise maximum likelihood estimation training mode.')
	parser.add_argument('--gan_train', action='store_true', help='activate generative adversarial Ttraining with reinforcement learning mode.')
	parser.add_argument('--pre_chat', action='store_true', help='activate daisy chat and test mode with the pre trained model.')
	parser.add_argument('--gan_chat', action='store_true', help='activate daisy chat and test mode with the GAN trained model.')
	parser.add_argument('--speak', action='store_true', help='activate daisy speak audio response mode.')
	parser.add_argument('--pre_evaluate', action='store_true', help='evaluate the pre trained models performance.')
	parser.add_argument('--gan_evaluate', action='store_true', help='evaluate the GaAN trained models performance.')
	parser.add_argument('--force_save', action='store_true', help='force model saving at every epoch.')
	parser.add_argument('--read_log', action='store_true', help='read training log file.')
	#--preprocess settings--#
	parser.add_argument('--word_threshold', type=int, default=5, help='threshold to filter out words when building vocabulary set.')
	parser.add_argument('--PAD_ID', type=int, default=0, help='the <pad> token: Special vocabulary symbol ID for vocab dictionary.')
	parser.add_argument('--UNK_ID', type=int, default=1, help='the <unk> token: Special vocabulary symbol ID for vocab dictionary.')
	parser.add_argument('--BOS_ID', type=int, default=2, help='the <bos> token: Special vocabulary symbol ID for vocab dictionary.')
	parser.add_argument('--EOS_ID', type=int, default=3, help='the <eos> token: Special vocabulary symbol ID for vocab dictionary.')
	#--general settings--#
	parser.add_argument('--beam_size', type=int, default=5, help='monte carlo search beam size.')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--pre_train_epochs', type=int, default=400, help='generator supervise maximum likelihood estimation training epochs.')
	parser.add_argument('--gan_train_epochs', type=int, default=500, help='generative adversarial with reinforcement learning training epochs.')
	parser.add_argument('--update_g', type=int, default=1, help='number of time to update generator during GAN training.')
	parser.add_argument('--update_d', type=int, default=3, help='number of time to update discriminator during GAN training.')
	parser.add_argument('--max_grad_norm', type=float, default=5.0, help='max gradient norm for clipping.')
	parser.add_argument('--patient', type=int, default=15, help='number of trials before model checkpoint activates early stopping.')
	#--generator--#
	parser.add_argument('--g_lr', type=float, default=1e-4, help='generator learning rate.')
	parser.add_argument('--g_hidden_dim', type=int, default=256, help='generator hidden state dimension of rnn cell.')
	parser.add_argument('--g_num_layer', type=int, default=3, help='number of generator rnn layer.')
	parser.add_argument('--g_num_sample', type=int, default=1024, help='generator number of samples for sampled softmax.')
	#--discriminator--#
	parser.add_argument('--d_lr', type=float, default=1e-4, help='discriminator learning rate.')
	parser.add_argument('--d_embedding_dim', type=int, default=128, help='discriminator embedding dimension.')
	parser.add_argument('--d_num_layer', type=int, default=3, help='number of discriminator rnn layer.')
	parser.add_argument('--d_num_class', type=int, default=2, help='number of class the discriminator needs to judge.')
	#---path---#
	parser.add_argument('--pre_train_model', type=str, default='pre_train-340', help='specific pre-trained model name.')
	parser.add_argument('--gan_train_model', type=str, default='gan_train-400', help='specific GAN-trained model name.')
	parser.add_argument('--corpus_dir', type=str, default='../cornell_movie_dialog_corpus', help='path to the raw corpus file.')
	parser.add_argument('--data_dir', type=str, default='../data', help='path to the processed data file.')
	parser.add_argument('--model_dir', type=str, default='../model', help='path to model files.')
	#--parse--#
	config = parser.parse_args()
	return config

config = get_config()


###########
# BUCKETS #
###########
"""
	These are some suitable bucket sizes,
	bucket size < 20 is recomended for ChatBot,
	since normal human conversations usualy consist of sentences with length shorter than 20 words
"""
# BUCKETS = [(6, 8), (8, 10), (10, 12), (13, 15), (16, 19), (19, 22), (23, 26), (29, 32), (39, 44)] # >> Distribution: [19530, 17449, 17585, 23444, 22884, 16435, 17085, 18291, 18931]
# BUCKETS = [(8, 10), (12, 14), (16, 19), (23, 26), (39, 43)] # >> Distribution: [37049, 33519, 30223, 33513, 37371]
# BUCKETS = [(8, 10), (12, 14), (16, 19)] # >> Distribution: [43296, 45900, 37681]
# BUCKETS = [(10, 10), (14, 14), (19, 19)] # >> Distribution: [57559, 41828, 38740]
BUCKETS = [(10, 10),] # >> Distribution: [57559]

