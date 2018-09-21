# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ data_loader.py ]
#   Synopsis     [ Prepare data to be model-ready ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import pickle
import random
import numpy as np
from configuration import BUCKETS


###############
# RANDOM SEED #
###############
random.seed(19960611)
np.random.seed(19960611)


#####################
# CLASS DATA LOADER #
#####################
class Data_Loader():

	##################
	# INITIALIZATION #
	##################
	"""
		Loads the data processed by 'preprocess_data.py'.
	"""
	def __init__(self, config, load_bucket=True, load_for_test=False):

		#--configuratios--#
		self.data_dir = config.data_dir
		self.batch_size = config.batch_size
		self.buckets = BUCKETS

		#--load model-ready preprocessed data--#
		try:
			self.word2idx = pickle.load(open(os.path.join(self.data_dir, 'map_word2idx.pkl'), 'rb'))
			self.idx2word = pickle.load(open(os.path.join(self.data_dir, 'map_idx2word.pkl'), 'rb'))
			self.encode = pickle.load(open(os.path.join(self.data_dir, 'train_encode.pkl'), 'rb'))
			self.decode = pickle.load(open(os.path.join(self.data_dir, 'train_decode.pkl'), 'rb'))
			print('### Data Loader in operation ###')
		except IOError:
			print('Failed to load preprocessed data, please run \'preprocess_data.py\' first!')

		#--initialize--#
		if load_for_test:
			self.buckets = [self.buckets[0]] # >> load the first bucket only for testing
		elif load_bucket:
			print('>> Total number of vocabulary: ', len(self.word2idx))
			print('>> Total number of training data: ', len(self.encode))
			self.data_buckets_encode, self.data_buckets_decode = self.load_data_to_bucket()


	##################
	# DATA TO BUCKET #
	##################
	"""
		Read each line in 'encode' and 'decode'.
		Place each line into buckets according to each line's length, and pad each line to bucket's max length.
	"""
	def load_data_to_bucket(self):
		#--empty data buckets--#
		data_buckets_encode = [[] for _ in self.buckets] # >> 2D-list, each bucket contains a list [encode_line_ids]
		data_buckets_decode = [[] for _ in self.buckets]
		
		#--load to bucket--#
		for i in range(len(self.encode)):
			print(">> Bucketing conversation number", i, end='\r')
			
			#--read each line--#
			encode_line_ids = self.encode[i]
			decode_line_ids = self.decode[i]

			#--find each line a matching data bucket--#
			for bucket_id, (encode_size, decode_size) in enumerate(self.buckets):
				if len(encode_line_ids) <= encode_size and len(decode_line_ids) <= decode_size:
					#--pad both encoder and decoder, reverse the encoder--#
					data_buckets_encode[bucket_id].append(self._pad_input(list(reversed(encode_line_ids)), encode_size))
					data_buckets_decode[bucket_id].append(self._pad_input(decode_line_ids, decode_size))
					break

		#--finalize and return--#
		self._get_bucket_info(data_buckets_encode)
		for i in range(len(data_buckets_encode)):
			data_buckets_encode[i] = np.asarray(data_buckets_encode[i])
			data_buckets_decode[i] = np.asarray(data_buckets_decode[i])
		assert len(data_buckets_encode) == len(data_buckets_decode)
		return data_buckets_encode, data_buckets_decode


	#####################
	# GET NUMBER BUCKET #
	#####################
	"""
		Get the number of buckets in data_buckets
	"""
	def get_num_buckets(self):
		return len(self.data_buckets_encode)


	#####################
	# GET BUCKET LENGTH #
	#####################
	"""
		Get the length of a specific bucket
	"""
	def get_bucket_len(self, bucket_id):
		return len(self.data_buckets_encode[bucket_id])


	################
	# SHUFFLE DATA #
	################
	"""
		Randomly shuffle data in each bucket
	"""
	def shuffle_data(self):
		for i in range(len(self.data_buckets_encode)):
			indices = np.arange(len(self.data_buckets_encode[i]))
			np.random.shuffle(indices)
			self.data_buckets_encode[i] = self.data_buckets_encode[i][indices]
			self.data_buckets_decode[i] = self.data_buckets_decode[i][indices]


	###################
	# SHUFFLE BUCKETS #
	###################
	"""
		Get random bucket ids.
	"""
	def shuffle_buckets(self):
		bucket_ids = np.arange(len(self.data_buckets_encode))
		np.random.shuffle(bucket_ids)
		return bucket_ids


	#######################
	# GENERATOR GET BATCH #
	#######################
	"""
		Return one batch to feed into the generator model, only pad to the max length of the bucket
	"""
	def generator_get_batch(self, bucket_id, batch_id=None, mode=None, chat_input=None):
		
		encoder_size, decoder_size = self.buckets[bucket_id]
		
		#--pre train / gan train mode--#
		if mode == 'pre_train' or mode == 'gan_train':
			
			#--ready--#
			encoder_inputs = np.zeros((self.batch_size, encoder_size))
			decoder_inputs = np.zeros((self.batch_size, decoder_size))

			#--take 'batch_size' lines--#
			batch_range = np.arange((batch_id*self.batch_size), (batch_id*self.batch_size+self.batch_size))
			for cnt, i in enumerate(batch_range):
				idx = i % self.get_bucket_len(bucket_id) # bound idx when i > data_length
				encoder_inputs[cnt] = self.data_buckets_encode[bucket_id][idx]
				decoder_inputs[cnt] = self.data_buckets_decode[bucket_id][idx]
		
		#--chating mode--#
		elif mode == 'chat' and chat_input is not None:
			encoder_inputs = []
			decoder_inputs = []
			for i in range(len(chat_input)):
				encoder_inputs.append(self._pad_input(list(reversed(chat_input[i])), encoder_size))
				decoder_inputs.append(self._pad_input([], decoder_size))

		else:
			raise ValueError('Invalid mode, legal modes are: \'pre_train\', \'gan_train\', or \'chat\'.')

		#--create batch-major vectors--#
		batch_encoder_inputs = self._reshape_batch(encoder_inputs, encoder_size)
		batch_decoder_inputs = self._reshape_batch(decoder_inputs, decoder_size)
		
		#--get decoder mask--#
		batch_masks = self._get_mask(decoder_inputs, decoder_size)

		#--return--#
		if mode == 'gan_train':
			return batch_encoder_inputs, batch_decoder_inputs, batch_masks, encoder_inputs, decoder_inputs
		else:
			return batch_encoder_inputs, batch_decoder_inputs, batch_masks


	###########################
	# DISCRIMINATOR GET BATCH #
	###########################
	"""
		Return one batch to feed into the generator model, only pad to the max length of the bucket
	"""
	def discriminator_get_batch(max_set, query_set, answer_set, gen_set):
		if self.batch_size % 2 != 0: return ValueError('Discriminator batch size error!')
		train_query = []
		train_answer = []
		train_labels = []
		half_size = self.batch_size / 2
		for _ in range(half_size):
			index = random.randint(0, max_set)
			train_query.append(query_set[index])
			train_answer.append(answer_set[index])
			train_labels.append(1)
			train_query.append(query_set[index])
			train_answer.append(gen_set[index])
			train_labels.append(0)
		return train_query, train_answer, train_labels


	##################
	# GET VOCAB SIZE #
	##################
	"""
		Return 'len(wrod2idx)'.
	"""
	def get_vocab_size(self):
		return len(self.word2idx)


	###############
	# GET MAPPING #
	###############
	"""
		Return 'wrod2idx' and 'idx2word' mappings.
	"""
	def get_mapping(self):
		return self.word2idx, self.idx2word
	

	#################
	# GET TEST DATA #
	#################
	"""
		Return 'test_input' and 'test_output',
		which are encoder inputs and the matching decoder outputs,
		these data are used to evaluate the model's performance.
	"""
	def get_test_data(self):
		try: assert len(self.buckets) == 1
		except: raise ValueError('Please initialize Data Loader with \'load_for_test\' set to \'True\'!')

		#--empty data buckets--#
		test_input = [] # >>  a list [encode_line_ids]
		test_output = []
		
		#--load to bucket--#
		for i in range(len(self.encode)):
			print(">> Bucketing conversation number", i, end='\r')
			
			#--read each line--#
			encode_line_ids = self.encode[i]
			decode_line_ids = self.decode[i]

			#--find each line a matching data bucket--#
			encode_size, decode_size = self.buckets[0]
			if len(encode_line_ids) <= encode_size and len(decode_line_ids) <= decode_size:
				#--reconstruct original text from word ID--#
				test_input.append(' '.join([self.idx2word[ids] for ids in encode_line_ids[1:-1]])) # >> [1:-1] to truncate the <bos> and <eos> token
				test_output.append(' '.join([self.idx2word[ids] for ids in decode_line_ids[1:-1]]))

		#--finalize and return--#
		self._get_bucket_info([test_input])
		assert len(test_input) == len(test_output)
		return test_input, test_output




	#****************************************************#	
	#***************** HELPER FUNCTIONS *****************#
	#****************************************************#


	##############
	# _PAD INPUT #
	##############
	"""
		Called by load_data_to_bucket(), this function pads 'input_' to a given 'size'.
	"""
	def _pad_input(self, input_, size):
		return np.asarray(input_ + [self.word2idx['<pad>']] * (size - len(input_))) # >> idx2word[2] = '<pad>' in 'preprocess_data.py'


	####################
	# _GET BUCKET INFO #
	####################
	"""
		Called by load_data_to_bucket().
		This function calculates relevent information about buckets: each bucket's size, and total size.
	"""
	def _get_bucket_info(self, data_buckets_encode):
		bucket_sizes = [len(data_bucket) for data_bucket in data_buckets_encode]
		total_size = sum(bucket_sizes)
		print('>> Bucketing conversation complete, conversations in buckets: ', total_size)
		print(">> Number of samples in each bucket: ", bucket_sizes)


	##################
	# _GET MASK #
	##################	
	"""
		Called by get_batch(), this function creates decoder masks in a batch-major fashion.
		We set mask to 0 if the corresponding target is a <pad> symbol, the corresponding decoder is decoder_input shifted by 1 forward.
	"""
	def _get_mask(self, decoder_inputs, decoder_size):
		batch_masks = []
		for length_id in range(decoder_size):
			batch_mask = np.ones(self.batch_size, dtype=np.float32)
			for batch_id in range(self.batch_size):
				if length_id < decoder_size - 1:
					target = decoder_inputs[batch_id][length_id + 1]
				if length_id == decoder_size - 1 or target == self.word2idx['<pad>']:
					batch_mask[batch_id] = 0.0
			batch_masks.append(batch_mask)
		return batch_masks


	##################
	# _RESHAPE BATCH #
	##################
	"""
		Called by get_batch(), this function does a matrix transpose on inputs.
		Create batch-major inputs, which are just re-indexed inputs.
		[line_1, line_2, line_3 ... line_n ] -> [timestep_1, timestep_2, timestep_3, ... timestep_size]
		shape: [num_timesteps, batch_size]
	"""		
	def _reshape_batch(self, inputs, bucket_size):
		batch_inputs = []
		for length_idx in range(bucket_size):
			batch_inputs.append(np.array([inputs[idx][length_idx] for idx in range(self.batch_size)], dtype=np.int32))
		return batch_inputs


########
# MAIN #
########
"""
	For testing and debug purpose
"""
if __name__ == '__main__':
	from configuration import config
	data_loader = Data_Loader(config)
	batch_encoder_inputs, batch_decoder_inputs, batch_masks = data_loader.generator_get_batch(bucket_id=0, batch_id=1, mode='pre_train', chat_input=None)
	print(batch_decoder_inputs)
	print(batch_decoder_inputs[0])

