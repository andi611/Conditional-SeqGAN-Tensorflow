# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ generator.py ]
#   Synopsis     [ Generator model ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import math
import numpy as np
import tensorflow as tf
from configuration import config, BUCKETS
import tf_seq2seq_model as seq2seq


###################
# CLASS GENERATOR #
###################
"""
	A Generator model for text generation:
	Sequence to sequence model with attentional decoder, using sampled softmax and bucket mechanism.
	feed_previous: Boolean or scalar Boolean Tensor; if True, only the first of decoder_inputs will be used (the "GO" symbol), 
				   and all other decoder inputs will be taken from previous outputs (as in embedding_rnn_decoder). 
				   If False, decoder_inputs are used as given (the standard decoder case).
"""
class Generator(object):

	##################
	# INITIALIZATION #
	##################
	"""
		Loads Generator parameters and construct model.
	"""
	def __init__(self, config, vocab_size, build_train_op=True):

		#--general settings--#
		self.dtype = tf.float32
		self.buckets = BUCKETS
		self.vocab_size = vocab_size
		self.beam_size = config.beam_size
		self.batch_size = config.batch_size
		self.max_grad_norm = config.max_grad_norm

		#--generator hyper-parameters--#
		self.lr = config.g_lr
		self.hidden_dim = config.g_hidden_dim
		self.num_layer = config.g_num_layer
		self.num_sample = config.g_num_sample

		#--feed placeholders--#
		self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder_input{}'.format(i)) for i in range(self.buckets[-1][0])]
		self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder_input{}'.format(i)) for i in range(self.buckets[-1][1] + 1)]
		self.decoder_masks =  [tf.placeholder(self.dtype, shape=[None], name='mask{}'.format(i)) for i in range(self.buckets[-1][1] + 1)]
		self.reward = 		  [tf.placeholder(self.dtype, shape=None, name='reward{}'.format(i)) for i in range(len(self.buckets))]
		
		#--controler placeholders--#
		self.feed_previous = tf.placeholder(tf.bool, name='feed_previous')
		self.add_reward = tf.placeholder(tf.bool, name='add_reward')
		self.mc_search = tf.placeholder(tf.bool, name='mc_search')

		#--traget--#
		self.targets = self.decoder_inputs[1:] # Our targets are decoder inputs shifted by one (to ignore <s> symbol)

		#--construct model--#
		self.build_model(build_train_op)


	###############
	# BUILD MODEL #
	###############
	"""
		Construct tensorflow model.
	"""
	def build_model(self, build_train_op):
		with tf.variable_scope('generator'):

			#--work arround to avoid tf bug--#
			setattr(tf.nn.rnn_cell.GRUCell, '__deepcopy__', lambda self, _: self)
			setattr(tf.nn.rnn_cell.MultiRNNCell, '__deepcopy__', lambda self, _: self)

			#--output projection for sampled softmax--#
			if self.num_sample > 0 and self.num_sample < self.vocab_size: # Sampled softmax only makes sense if we sample less than vocabulary size.
				self.w = tf.get_variable(name='proj_w',
										 shape=[self.hidden_dim, self.vocab_size], 
										 initializer=tf.truncated_normal_initializer(stddev=0.1),
										 dtype=self.dtype)
				self.b = tf.get_variable(name='proj_b',
										 shape=[self.vocab_size], 
										 initializer=tf.constant_initializer(0.1),
										 dtype=self.dtype)
				self.output_projection = (self.w, self.b)
			
			#--sampled softmax loss--#
			sampled_softmax_loss_function = self._sampled_softmax_loss
			
			#--reward bias--#
			self.reward_bias = tf.get_variable('reward_bias', [1], initializer=tf.constant_initializer(0.1))

			#--rnn cell and layers--#
			single_cell = tf.nn.rnn_cell.GRUCell(num_units=self.hidden_dim)
			self.cell = tf.nn.rnn_cell.MultiRNNCell(cells=[single_cell]*self.num_layer, state_is_tuple=True)

			#--rnn model--#
			print('>> Generator: constructing generator model, this may take a few minutes depending on the number of buckets...\n')
			self.outputs, self.losses, self.encoder_state = seq2seq.model_with_buckets(encoder_inputs=self.encoder_inputs, 
																					   decoder_inputs=self.decoder_inputs, 
																					   targets=self.targets,
																					   weights=self.decoder_masks,
																					   buckets=self.buckets,
																					   vocab_size=self.vocab_size,
																					   batch_size=self.batch_size,
																					   seq2seq=lambda x, y: self._seq2seq_f(x, y, tf.where(self.feed_previous, True, False)),
																					   softmax_loss_function=sampled_softmax_loss_function)
			
			#--If we use output projection, we need to project outputs for decoding--#
			self.policy_outputs = []
			for bucket_id in range(len(self.buckets)):
				self.policy_outputs.append(tf.nn.softmax([tf.nn.xw_plus_b(output, self.output_projection[0], self.output_projection[1]) # >> shape: [time_step, batch_size, vocab_size]
														  for output in self.outputs[bucket_id]]))
				self.outputs[bucket_id] = [ tf.cond(self.feed_previous,
													lambda: tf.nn.xw_plus_b(output, self.output_projection[0], self.output_projection[1]), # >> shape: [time_step, batch_size, vocab_size]
													lambda: output) # >> shape: [time_step, batch_size, embedding_size]
											for output in self.outputs[bucket_id]]

			#--train_op--#
			if build_train_op:
				print('>> Generator: constructing training optimizers, this may take a few minutes depending on the number of buckets...\n')
				self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
				self.params = [param for param in tf.trainable_variables() if 'generator' in param.name]
				self.losses_with_reward = []
				self.gradient_norms = []
				self.train_ops = []

				for bucket_id in range(len(self.buckets)):
					#--calculate reward with bias--#
					R =  tf.subtract(self.reward[bucket_id], self.reward_bias)
					#--cancatenate inputs for reward loss calculation--#
					x = tf.reshape(tf.concat(tf.convert_to_tensor(self.encoder_inputs[:self.buckets[bucket_id][0]]), axis=1), # >> shape: [batch_size, time_step]
																  shape=[self.batch_size, self.buckets[bucket_id][0]])
					#--calculate reward loss--#
					reward_loss = tf.cond(self.add_reward,
										  lambda: #tf.multiply(self.losses[bucket_id], self.reward[bucket_id]),
										  		  tf.reduce_sum(tf.reduce_sum(
												  tf.one_hot(tf.to_int32(tf.reshape(x, [-1])), self.vocab_size, 1.0, 0.0) *
												  tf.log(tf.clip_by_value(tf.reshape(self.policy_outputs[bucket_id], [-1, self.vocab_size]), 1e-20, 1.0)), 1) *
										  		  self.reward[bucket_id]),
										  lambda: self.losses[bucket_id])			
					self.losses_with_reward.append(reward_loss)  
					#--calculate and apply gradients--#
					gradients = tf.gradients(ys=reward_loss, xs=self.params)
					clipped_grads, norm = tf.clip_by_global_norm(t_list=gradients, clip_norm=self.max_grad_norm)
					self.gradient_norms.append(norm)
					self.train_ops.append(self.optimizer.apply_gradients(zip(clipped_grads, self.params)))

		#--model saver--#
		self.variables = [var for var in tf.global_variables() if 'generator' in var.name]
		self.saver = tf.train.Saver(var_list=self.variables, max_to_keep=1)


	############
	# RUN STEP #
	############
	"""
		This function runs one step for the generator.
		If feed_previous is False: generater updates,
		else the generator gives the loss and output for a batch of inputs
		output shape: [time_step, batch_size, vocab_size] >> eg. [10, 64, 20000]
	"""
	def run_step(self, sess, encoder_inputs, decoder_inputs, decoder_masks, bucket_id,
			   reward=1, feed_previous=False, add_reward=False, mc_search=False):

		encoder_size, decoder_size = self.buckets[bucket_id]
		self._assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks)

		#--input feed--#
		feed_dict = { self.feed_previous.name: feed_previous,
					  self.add_reward.name: add_reward,
					  self.mc_search.name: mc_search, }
		for i in range(len(self.buckets)):
			feed_dict[self.reward[i].name] = reward
		for step in range(encoder_size):
			feed_dict[self.encoder_inputs[step].name] = encoder_inputs[step]
		for step in range(decoder_size):
			feed_dict[self.decoder_inputs[step].name] = decoder_inputs[step]
			feed_dict[self.decoder_masks[step].name] = decoder_masks[step]

		#--Since targets are decoder inputs shifted by one, we need one more--#
		last_target = self.decoder_inputs[decoder_size].name
		assert config.PAD_ID == 0
		feed_dict[last_target] = np.zeros([self.batch_size], dtype=np.float32) # >> zero = '<pad>'

		#--normal training--#
		if not feed_previous:
			output_feed = [ self.train_ops[bucket_id],  	  		# update op that does SGD.
							self.gradient_norms[bucket_id],   		# gradient norm.
							self.losses_with_reward[bucket_id] ]  	# loss for this batch.
		
		#--testing or gan training--#
		else:
			output_feed = [ self.encoder_state[bucket_id], 														# encoder state
							self.losses_with_reward[bucket_id] if add_reward else self.losses[bucket_id],		# loss for this batch.
							self.policy_outputs[bucket_id] ]  													# softmax logits	
			for step in range(decoder_size): 					# output logits.
				output_feed.append(self.outputs[bucket_id][step])
		#--run sess--#
		run_outputs = sess.run(output_feed, feed_dict)
		if not feed_previous:
			return run_outputs[2]  # >> loss.
		else:
			return run_outputs[1], run_outputs[2], run_outputs[3:]  # >> loss, softmax, outputs.


	##########
	# SAMPLE #
	##########
	"""
		This function gather (X,Y) and sample (X, ^Y) through ^Y ~ G(*|X)
		If mc_search == True, beam_size of rolls will be preforme,
		else the generator samples 1 time.
	"""
	def sample(self, sess, batch_encoder_inputs, batch_decoder_inputs, decoder_masks, 
			   encoder_inputs, decoder_inputs, bucket_id, mc_search):

		#--ready--#
		answer_len = self.buckets[bucket_id][1]
		train_query = [input_ for input_ in encoder_inputs]
		train_answer = [input_ for input_ in decoder_inputs]
		train_labels = [1 for input_ in encoder_inputs]

		num_roll = self.beam_size if mc_search else 1
		for _ in range(num_roll):
			#--sample from generator--#
			_, _, output_logits = self.run_step(sess,
											 	batch_encoder_inputs, 
											 	batch_decoder_inputs, 
											 	decoder_masks, 
											 	bucket_id,
											 	feed_previous=True,
											 	add_reward=False,
											 	mc_search=mc_search)

			#--distribution to id tokens--#
			output_tokens = []
			for time_step in output_logits:
				batch_token = []
				for a_batch in time_step:
					batch_token.append(int(np.argmax(a_batch, axis=0)))
				output_tokens.append(batch_token) # >> output_tokens shape: (time_step, batch_size)

			#--transpose 'output_tokens' into batch-major vectors--#
			batch_output_tokens = []
			for col in range(len(output_tokens[0])):
				batch_output_tokens.append([output_tokens[row][col] for row in range(len(output_tokens))]) # >> batch_output_tokens shape: (batch_size, time_step)

			responses = []
			for seq in batch_output_tokens:
				if config.EOS_ID in seq:
					responses.append(seq[:seq.index(config.EOS_ID)][:answer_len])
				else:
					responses.append(seq[:answer_len])

			for i, output in enumerate(responses):
				output = output[:answer_len] + [config.PAD_ID] * (answer_len - len(output) if answer_len > len(output) else 0)
				train_query.append(train_query[i])
				train_answer.append(output)
				train_labels.append(0)

		#--transpose to batch-major vectors for discriminator run_step()--#
		train_query = np.transpose(train_query)
		train_answer = np.transpose(train_answer)
		return train_query, train_answer, train_labels
		

	#****************************************************#	
	#***************** HELPER FUNCTIONS *****************#
	#****************************************************#

	
	#########################
	# _SAMPLED SOFTMAX LOSS #
	#########################
	"""
		Called by build_model().
	"""
	def _sampled_softmax_loss(self, labels, inputs):
		labels = tf.reshape(labels, [-1, 1])
		return tf.nn.sampled_softmax_loss(weights=tf.transpose(self.w),
										  biases=self.b,
										  labels=labels, 
										  inputs=inputs, 
										  num_sampled=self.num_sample, 
										  num_classes=self.vocab_size)


	#####################
	# _SEQ2SEQ FUNCTION #
	#####################
	"""
		Called by build_model().
	"""
	def _seq2seq_f(self, encoder_inputs, decoder_inputs, feed_previous):
		return seq2seq.embedding_attention_seq2seq(encoder_inputs=encoder_inputs,
												   decoder_inputs=decoder_inputs,
												   cell=self.cell,
												   num_encoder_symbols=self.vocab_size,
												   num_decoder_symbols=self.vocab_size,
												   embedding_size=self.hidden_dim,
												   output_projection=self.output_projection,
												   feed_previous=feed_previous,
												   mc_search=self.mc_search,
												   dtype=self.dtype)


	###################
	# _ASSERT LENGTHS #
	###################
	"""
		Called by run_step(),
		this function checks that the encoder inputs, decoder inputs, and decoder masks are of the expected lengths.
	"""
	def _assert_lengths(self, encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks):
		if len(encoder_inputs) != encoder_size:
			raise ValueError('Encoder length must be equal to the one in bucket,'
							' %d != %d.' % (len(encoder_inputs), encoder_size))
		if len(decoder_inputs) != decoder_size:
			raise ValueError('Decoder length must be equal to the one in bucket,'
						   ' %d != %d.' % (len(decoder_inputs), decoder_size))
		if len(decoder_masks) != decoder_size:
			raise ValueError('Weights length must be equal to the one in bucket,'
						   ' %d != %d.' % (len(decoder_masks), decoder_size))


########
# MAIN #
########
"""
	For testing and debug purpose
"""
if __name__ == '__main__':
	generator = Generator(config, vocab_size=20000)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		batch_encoder_inputs = np.zeros((BUCKETS[0][0], 64))
		batch_decoder_inputs = np.zeros((BUCKETS[0][1], 64))
		decoder_masks = np.zeros((BUCKETS[0][1], 64))

		_, t1, t2 = generator.run_step(sess,
											 batch_encoder_inputs, 
											 batch_decoder_inputs, 
											 decoder_masks, 
											 0,
											 feed_previous=False,
											 add_reward=False,
											 mc_search=True)

