# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ test.py ]
#   Synopsis     [ Test and chat ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import numpy as np
import tensorflow as tf

#--audio importation--#
try:
	import tempfile
	from gtts import gTTS
	from pygame import mixer
except:
	pass

#--BLEU calculation importation--#
import operator
from functools import reduce 

#--daisy importation--#
from generator import Generator
from data_loader import Data_Loader
from preprocess_data import tokenizer
from configuration import config, BUCKETS
from train import _check_restore_parameters


########
# CHAT #
########
"""
	Test the model's performance, no backward path is created in chat mode.
"""
def chat(speak_with_audio=False, model_name=None):
	print('### Model Chat and Test Mode ###')

	#--enable audio mode--#
	if speak_with_audio:
		mixer.init()
		print('### Daisy Speak with Audio Response Mode Enabled ###')

	#--modify batch size--#
	config.batch_size = 1

	#--construct data loader--#
	data_loader = Data_Loader(config, load_bucket=False)
	word2idx, idx2word = data_loader.get_mapping()
	vocab_size = data_loader.get_vocab_size()

	#--construct generator model--#
	GEN = Generator(config, vocab_size, build_train_op=False)

	#--tensorflow saver--#
	saver = tf.train.Saver()

	#--tensorflow session--#
	with tf.Session() as sess:

		#--initialize testing--#
		sess.run(tf.global_variables_initializer())
		GEN = _check_restore_parameters(sess, model=GEN, specific_model=model_name)

		#--output file--#
		with open(os.path.join(config.data_dir, 'chat_dialog.txt'), 'w') as f:
			
			#--greeting message--#
			max_length = BUCKETS[-1][0]
			print('\nHi, my name is Daisy! Nice to meet you, let\'s chat!')
			print('(Input max length: %i, enter empty line to exit. )' % max_length)
			if speak_with_audio: _speak('Hi, my name is Daisy! Nice to meet you, let\'s chat!.')
			
			#--testing loop--#
			while True:

				#--decode from standard input--#
				line = _get_user_input()
				if len(line) > 0 and line[-1] == '\n':
					line = line[:-1]
				if line == '':
					print('Bye!')
					if speak_with_audio: _speak('Bye!')
					break

				#--record human input--#
				f.write('HUMAN +++++++ ' + line + '\n')

				#--get token-ids for the input sentence--#
				token_ids = _sentence2id(word2idx, line)
				if (len(token_ids) > max_length):
					print('Max length I can handle is: %i, tell me something else.' % max_length)
					line = _get_user_input()
					continue

				#--feed input line to the right bucket--#
				bucket_id = _find_right_bucket(len(token_ids))

				#--get a 1-element batch to feed the sentence to the model--#
				(
				chat_encoder_inputs, 
				chat_decoder_inputs, 
				decoder_masks, 
				) = data_loader.generator_get_batch(bucket_id=bucket_id, mode='chat', chat_input=[token_ids,])
				
				#--get output logits for the sentence--#
				_, _, output_logits = GEN.run_step(sess,
												   chat_encoder_inputs, 
												   chat_decoder_inputs, 
												   decoder_masks, 
												   bucket_id,
												   feed_previous=True, # >> in chat mode, we feed forward the previous outputs.
												   add_reward=False,
												   mc_search=False)													  
				#--construct response--#
				response = _construct_response(output_logits, dec_vocab=idx2word, eos_id=config.EOS_ID)
				print('DAISY +++++++ ', response)
				if speak_with_audio:
					_speak(response)

				#--record machine response--#
				f.write('DAISY +++++++ ' + response + '\n')
			f.write('=============================================\n')


############
# EVALUATE #
############
"""
	TODO
"""
def evaluate(model_name):
	print('### Model Chat and Test Mode ###')

	#--construct data loader--#
	data_loader = Data_Loader(config, load_for_test=True)
	word2idx, idx2word = data_loader.get_mapping()
	vocab_size = data_loader.get_vocab_size()

	#--get testing data--#
	test_input, test_output = data_loader.get_test_data()

	#--truncate test data size--#
	n_batch_for_test = 100
	if len(test_input) > config.batch_size*n_batch_for_test:
		test_input = test_input[:config.batch_size*n_batch_for_test] 
		test_output = test_output[:config.batch_size*n_batch_for_test] 

	#--construct generator model--#
	GEN = Generator(config, vocab_size, build_train_op=False)

	#--tensorflow saver--#
	saver = tf.train.Saver()

	#--tensorflow session--#
	with tf.Session() as sess:

		#--initialize testing--#
		eval_output = []
		max_length = BUCKETS[0][0] - 2 # >> the first bucket but without <bos> and <eos> tokens, therefor - 2
		sess.run(tf.global_variables_initializer())
		GEN = _check_restore_parameters(sess, model=GEN, specific_model=model_name)

		#--run over all inputs--#
		n_batch = 0
		while (n_batch * config.batch_size) < len(test_input):

			#--take 'batch_size' lines--#
			batch_inputs = []
			batch_range = np.arange((n_batch*config.batch_size), (n_batch*config.batch_size+config.batch_size))
			for cnt, i in enumerate(batch_range):
				idx = i % len(test_input) # bound idx when i > data_length
				batch_inputs.append(_sentence2id(word2idx, test_input[idx]))
			n_batch += 1

			#--get a 1-element batch to feed the sentence to the model--#
			(
			chat_encoder_inputs, 
			chat_decoder_inputs, 
			decoder_masks, 
			) = data_loader.generator_get_batch(bucket_id=0, mode='chat', chat_input=batch_inputs)
				
			#--get output logits for the sentence--#
			_, _, output_logits = GEN.run_step(sess,
											   chat_encoder_inputs, 
											   chat_decoder_inputs, 
											   decoder_masks, 
											   bucket_id=0,
											   feed_previous=True, # >> in chat mode, we feed forward the previous outputs.
											   add_reward=False,
											   mc_search=False)				
			output_logits = np.asarray(output_logits)

			#--construct response--#
			for i in range(config.batch_size):
				line = np.expand_dims(output_logits[:,i,:], axis=1)
				response = _construct_response(line, dec_vocab=idx2word, eos_id=config.EOS_ID)
				eval_output.append(response)

			print('Evaluation Batch: %i/%i' % (n_batch-1, len(test_input)/config.batch_size), end='\r')
		print('Evaluation Batch: %i/%i' % (n_batch, len(test_input)/config.batch_size))
		assert len(eval_output) == len(test_input)
	
		#--count BLEU score by average--#
		bleu = []
		for i, eval_line in enumerate(eval_output):
			score_per_line = []
			for test_line in test_output:
				score_per_line.append(_BLEU(eval_line, test_line))
			bleu.append(np.mean(score_per_line))
			print('BLEU evaluation: %i/%i' % (i, len(eval_output)), end='\r')
			
		print('BLEU evaluation: %i/%i' % (len(eval_output), len(eval_output)))
		average = np.mean(bleu)
		print("Originally, average bleu score is " + str(average))
		"""
		#--count by the method described in the paper https://aclanthology.info/pdf/P/P02/P02-1040.pdf--#
		bleu = []
		for eval_line, test_line in zip(eval_output, test_output):
			score_per_video = []
			captions = [x.rstrip('.') for x in item['caption']]
			score_per_video.append(BLEU(result[item['id']],captions,True))
			bleu.append(score_per_video[0])
		average = sum(bleu) / len(bleu)
		print("By another method, average bleu score is " + str(average))
		"""		

#****************************************************#	
#***************** HELPER FUNCTIONS *****************#
#****************************************************#


###################
# _GET USER INPUT #
###################
"""
	Called by chat(), this function gets the user's input, which will be transformed into encoder input later.
"""
def _get_user_input():
	user_input = str(input('HUMAN +++++++ '))
	print(end='\r')
	return user_input


##################
# SENTENCE TO ID #
##################
"""
	Called by chat(), this function convert a string sentence input 'line' to a sequence of vocab ID
"""
def _sentence2id(word2idx, line):
	return [word2idx.get(token, word2idx['<unk>']) for token in tokenizer(line)]


######################
# _FIND RIGHT BUCKET #
######################
"""
	Called by chat(), this function finds the proper bucket for an encoder input based on its length.
"""
def _find_right_bucket(length):
	for bucket_id in range(len(BUCKETS)):
		if BUCKETS[bucket_id][0] >= length:
			return bucket_id
	return len(BUCKETS) - 1


#######################
# _CONSTRUCT RESPONSE #
#######################
""" 
	Called by chat(), this function construct a response to the user's encoder input.	
	This is a greedy decoder - outputs are just argmaxes of output_logits.
"""
def _construct_response(output_logits, dec_vocab, eos_id, show_logits=False):
	if show_logits: print(output_logits[0]) # >> the outputs from sequence to sequence wrapper, decoder_size np array, each of dim 1 x DEC_VOCAB
	#--greedy select output--#
	outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
	#--filter out <unk> tokens--#
	outputs = [output for output in outputs if output != config.UNK_ID]
	#--If there is an EOS symbol in outputs, cut them at that point--#
	if eos_id in outputs: outputs = outputs[:outputs.index(eos_id)]
	#--Print out sentence corresponding to outputs--#
	return ' '.join([tf.compat.as_str(dec_vocab[output]) for output in outputs]) + '.'



##########
# _SPEAK #
##########
""" 
	Called by chat(), this function construct and plays an audio response.	
	A temporary file is created and will be deleted once the audio is played.
"""
def _speak(sentence):
	with tempfile.NamedTemporaryFile(delete=True) as fp:
		tts = gTTS(text=sentence, lang='en')
		tts.save(os.path.join(config.data_dir, '{}.mp3'.format(fp.name)))
		mixer.music.load(os.path.join(config.data_dir, '{}.mp3'.format(fp.name)))
		mixer.music.play()


#**************************************************************#	
#***************** BLEU CALCULATION FUNCTIONS *****************#
#**************************************************************#


def _count_ngram(candidate, references, n):
	clipped_count = 0
	count = 0
	r = 0
	c = 0
	for si in range(len(candidate)):
		# Calculate precision for each sentence
		ref_counts = []
		ref_lengths = []
		# Build dictionary of ngram counts
		for reference in references:
			ref_sentence = reference[si]
			ngram_d = {}
			words = ref_sentence.strip().split()
			ref_lengths.append(len(words))
			limits = len(words) - n + 1
			# loop through the sentance consider the ngram length
			for i in range(limits):
				ngram = ' '.join(words[i:i+n]).lower()
				if ngram in ngram_d.keys():
					ngram_d[ngram] += 1
				else:
					ngram_d[ngram] = 1
			ref_counts.append(ngram_d)
		# candidate
		cand_sentence = candidate[si]
		cand_dict = {}
		words = cand_sentence.strip().split()
		limits = len(words) - n + 1
		for i in range(0, limits):
			ngram = ' '.join(words[i:i + n]).lower()
			if ngram in cand_dict:
				cand_dict[ngram] += 1
			else:
				cand_dict[ngram] = 1
		clipped_count += _clip_count(cand_dict, ref_counts)
		count += limits
		r += _best_length_match(ref_lengths, len(words))
		c += len(words)
	if clipped_count == 0:
		pr = 0
	else:
		pr = float(clipped_count) / count
	bp = _brevity_penalty(c, r)
	return pr, bp


def _clip_count(cand_d, ref_ds):
	"""Count the clip count for each ngram considering all references"""
	count = 0
	for m in cand_d.keys():
		m_w = cand_d[m]
		m_max = 0
		for ref in ref_ds:
			if m in ref:
				m_max = max(m_max, ref[m])
		m_w = min(m_w, m_max)
		count += m_w
	return count


def _best_length_match(ref_l, cand_l):
	"""Find the closest length of reference to that of candidate"""
	least_diff = abs(cand_l-ref_l[0])
	best = ref_l[0]
	for ref in ref_l:
		if abs(cand_l-ref) < least_diff:
			least_diff = abs(cand_l-ref)
			best = ref
	return best


def _brevity_penalty(c, r):
	if c > r:
		bp = 1
	else:
		bp = math.exp(1-(float(r)/c))
	return bp


def _geometric_mean(precisions):
	return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))


def _BLEU(s, t, flag=False):

	score = 0.  
	count = 0
	candidate = [s.strip()]
	if flag:
		references = [[t[i].strip()] for i in range(len(t))]
	else:
		references = [[t.strip()]] 
	precisions = []
	pr, bp = _count_ngram(candidate, references, 1)
	precisions.append(pr)
	score = _geometric_mean(precisions) * bp
	return score

	
########
# TEST #
########
"""
	Run testing process
"""
def test():
	chat(speak_with_audio=False, model_name=config.pre_train_model)


if __name__ == '__main__':
	test()

