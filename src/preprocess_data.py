# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ preprocess_data.py ]
#   Synopsis     [ Pre-processing the raw Cornell Movie Dialogs Corpus dataset ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import pickle
from configuration import config


###############
# GET ID2LINE #
###############
"""
	1. Read from 'movie_lines.tsv'
	2. Create a dictionary with: { key = line_id, value = text }
"""
def get_id2line():
	id2line = {}
	path = os.path.join(config.corpus_dir, 'movie_lines.txt')
	with open(path, 'r', encoding='ISO-8859-1') as f:
		lines = f.readlines()
		for line in lines:
			tokens = line.split(' +++$+++ ')
			if len(tokens) == 5:
				if tokens[4][-1] == '\n': tokens[4] = tokens[4][:-1]
				id2line[tokens[0]] = tokens[4]
	print('>> gathered id2line dictionary.')
	return id2line


####################
# GET COVERSATIONS #
####################
"""
	1. Read from 'movie_conversations.txt'
	2. Create a list of list of line_id's, 2d list with shape: (n_conversations, len_conversation)
"""
def get_conversations():
	conversations = []
	path = os.path.join(config.corpus_dir, 'movie_conversations.txt')
	with open(path, 'r', encoding='ISO-8859-1') as f:
		lines = f.readlines()
		for line in lines:
			tokens = line.split(' +++$+++ ')
			if len(tokens) == 4:
				conversation = []
				for token in tokens[3][1:-2].split(', '):
					conversation.append(token.strip('\''))
				conversations.append(conversation)
	print('>> gathered conversations.')
	return conversations


##########################
# GET CONVERSATION PAIRS #
##########################
"""
	Returns 2 list of all conversations as input output pairs
	1. [encode]
	2. [decode]
"""
def get_conversation_pairs(id2line, conversations):
	encode = []
	decode = []
	for conv in conversations:
		for idx in range(len(conv) - 1):
			encode.append(id2line[conv[idx]])
			decode.append(id2line[conv[idx + 1]])
	assert len(encode) == len(decode)
	print('>> gathered conversation pairs with size: ', len(encode))
	return encode, decode


#############
# TOKENIZER #
#############
"""
	A tokenizer that cleans the text while tokenizing text into tokens.
	Takes a 'line' as input and returns a list of words as 'tokens'
"""
def tokenizer(line, normalize_digits=True):
	# remove the following
	line = line.replace('<u>', '')
	line = line.replace('</u>', '')
	line = line.replace('<i>', '')
	line = line.replace('</i>', '')
	line = line.replace('<b>', '')
	line = line.replace('</b>', '')
	line = line.replace('[', '')
	line = line.replace(']', '')
	line = line.replace('*', '')
	line = line.replace('|', '')
	line = line.replace('(', '')
	line = line.replace(')', '')
	line = line.replace('{', '')
	line = line.replace('}', '')
	line = line.replace('"', '')
	line = line.replace(' - ', '')
	line = line.replace('--', '')
	line = line.replace('---', '')
	line = line.replace('----', '')
	line = line.replace("\x94", '')
	# replace the following
	line = line.replace('good-by.', 'good-bye.')
	line = line.replace('wadn\'t', 'wasn\'t')
	line = line.replace('.', ' . ')
	line = line.replace(',', ' , ')
	line = line.replace('!', ' ! ')
	line = line.replace('?', ' ? ')
	line = line.replace('/', ' . ')
	line = line.replace('>', ' > ')
	line = line.replace('<', ' < ')
	line = line.replace(':', ' : ')
	line = line.replace(';', ' ; ')
	line = line.replace("\x92", "'")
	line = line.replace('$', ' $ ')
	line = line.replace(' .  .  . ', ' ... ')
	tokens = []
	for token in line.strip().lower().split():
		tokens.append(token)
	return tokens


#################
# TOKENIZE TEXT #
#################
"""
	Parse text in to tokens using tokenizer()
	Returns a 2d list with shape: (n_lines, n_tokens_in_line)
"""
def tokenize_text(text):
	tokenized_text = []
	for line in text:
		tokenized_line = tokenizer(line)
		tokenized_text.append(tokenized_line)
	return tokenized_text


###############
# BUILD VOCAB #
###############
"""
	builds vocabulary set, gather all the unique words in 'enocde' and 'decode',
	then filters out words that shows up less than 'config.word_threshold' times.
	Returns a list of words as 'vocab'
"""
def build_vocab(tokenized_encode, tokenized_decode):
	word2count = {}
	tokenized_text = tokenized_encode + tokenized_decode
	for tokenized_line in tokenized_text:
		for token in tokenized_line:
			if not token in word2count:
				word2count[token] = 0
			word2count[token] += 1
	word2count_sort = sorted(word2count, key=word2count.__getitem__, reverse=True)

	vocab = []
	for word in word2count_sort:
		if word2count[word] >= config.word_threshold:
			vocab.append(word)
	print('>> built vocabulary set and filtered words from %d to %d' % (len(word2count)), len(vocab))
	return vocab


#################
# BUILD MAPPING #
#################
"""
	builds mapping from (word to idx) and (idx to word),
	Returns two dictionary mapping: 'word2idx' and 'idx2word'
"""
def build_mapping(vocab):
	idx2word = {}
	idx2word[config.PAD_ID] = '<pad>' # >> 0
	idx2word[config.UNK_ID] = '<unk>' # >> 1
	idx2word[config.BOS_ID] = '<bos>' # >> 2
	idx2word[config.EOS_ID] = '<eos>' # >> 3
	for i, w in enumerate(vocab):
		idx2word[i+4] = w
	word2idx = { w : i for i, w in idx2word.items() }
	print('>> built word to index double-way mapping.')
	return word2idx, idx2word


################
# TOKEN TO IDX #
################
"""
	Convert all the tokens into their corresponding index in the vocabulary.
	Takes 'tokenized_text' as input, uses 'word2idx' as mapping,
	Returns a list of sequence of idx as 'index_text', shape: (n_lines_in_text, n_tokens_in_line)
"""
def token2idx(tokenized_text, word2idx):
	# check each caption's length, truncate if longer than 'n_caption_lstm_steps', pad <eos> at the end
	index_text = []
	unk_count = 0
	for line in tokenized_text:
		index_line = []
		index_line.append(word2idx['<bos>'])
		for word in line:
			if word in word2idx:
				index_line.append(word2idx[word])
			else:
				index_line.append(word2idx['<unk>'])
				unk_count += 1
		index_line.append(word2idx['<eos>'])
		index_text.append(index_line)
	print('>> tokenized text converted into sequence of index, number of unknown words: ', unk_count)
	return index_text


####################
# PREPROCESS DATASET #
####################
"""
	Save the pre-processed data to config.data_dir
	1. processed_corpus.txt : Encoder input for training
	2. word2idx.pkl : word to idx dictionary mapping saved by pickle
	3. idx2word.pkl : idx to word dictionary mapping saved by pickle
	4. train_encode.pkl: model-ready training data: tokenized and index labeled text ready to be fed to the encoder
	5. train_decode.pkl: model-ready training data: tokenized and index labeled text ready to be fed to the decoder
"""
def save_preprocessed_data(tokenized_encode, tokenized_decode, word2idx, idx2word, train_encode, train_decode):	
	# create path to store all the train & test encoder & decoder
	try: os.mkdir(config.data_dir)
	except OSError: pass
	
	with open(os.path.join(config.data_dir, 'processed_corpus.txt'), 'w', encoding='utf-8') as f:
		for i in range(len(tokenized_encode)):
			for token in tokenized_encode[i]:
				f.write(token + ' ')
			f.write('\n')
			for token in tokenized_decode[i]:
				f.write(token + ' ')
			f.write('\n')

	pickle.dump(word2idx, open(os.path.join(config.data_dir, 'map_word2idx.pkl'), 'wb'), True)
	pickle.dump(idx2word, open(os.path.join(config.data_dir, 'map_idx2word.pkl'), 'wb'), True)
	pickle.dump(train_encode, open(os.path.join(config.data_dir, 'train_encode.pkl'), 'wb'), True)
	pickle.dump(train_decode, open(os.path.join(config.data_dir, 'train_decode.pkl'), 'wb'), True)
	print('>> pre-processed data saved to: ', config.data_dir)


#################
# MAIN FUNCTION #
#################
"""
	Process the raw Cornell Movie Dialogs Corpus dataset to be model-ready.
"""
def main():
	print('### Runing data pre-processing on raw Cornell Movie Dialogs Corpus dataset. ###')
	id2line = get_id2line()
	conversations = get_conversations()
	encode, decode = get_conversation_pairs(id2line, conversations)

	print('### Parsing data to be model-ready. ###')
	tokenized_encode = tokenize_text(encode)
	tokenized_decode = tokenize_text(decode)
	vocab = build_vocab(tokenized_encode, tokenized_decode)
	word2idx, idx2word = build_mapping(vocab)
	train_encode = token2idx(tokenized_encode, word2idx)
	train_decode = token2idx(tokenized_decode, word2idx)

	assert len(train_encode) == len(train_decode)

	print('### Pre-processing Complete. ###')
	save_preprocessed_data(tokenized_encode, tokenized_decode, word2idx, idx2word, train_encode, train_decode)


"""***********************"""
if __name__ == "__main__":
	main()
"""***********************"""

