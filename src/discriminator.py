# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ discriminator.py ]
#   Synopsis     [ Discriminator model ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import math
import tensorflow as tf
from configuration import config, BUCKETS


#######################
# CLASS DISCRIMINATOR #
#######################
"""
	A hierarchical discriminator model for text classification:
	Uses an embedding layer, followed by a two level hierarchical RNN encoder that does a 2 class classification task.
	The dircriminator first enocdes on word level, followed by encoding on sentence level.
"""
class Discriminator(object):

	##################
	# INITIALIZATION #
	##################
	"""
		Loads Discriminator parameters and construct model.
	"""
	def __init__(self, config, vocab_size):

		#--general settings--#
		self.buckets = BUCKETS
		self.vocab_size = vocab_size
		self.batch_size = config.batch_size
		self.max_grad_norm = config.max_grad_norm

		#--discriminator hyper-parameters--#
		self.lr = config.d_lr
		self.embedding_dim = config.d_embedding_dim
		self.num_layer = config.d_num_layer
		self.num_class = config.d_num_class

		#--placeholders--#
		self.query = [tf.placeholder(dtype='int32', shape=[None], name="query{}".format(i)) for i in range(self.buckets[-1][0])]
		self.answer = [tf.placeholder(dtype='int32', shape=[None], name="answer{}".format(i)) for i in range(self.buckets[-1][1])]
		self.target = tf.placeholder(dtype='int32', shape=[None], name="target")

		#--construct model--#
		self.build_model()


	###############
	# BUILD MODEL #
	###############
	"""
		Construct tensorflow model.
	"""
	def build_model(self):
		with tf.variable_scope('discriminator'):

			#--word level cell--#
			word_cell = tf.nn.rnn_cell.GRUCell(num_units=self.embedding_dim)
			multi_word_cell = tf.nn.rnn_cell.MultiRNNCell(cells=[word_cell]*self.num_layer, state_is_tuple=True)
			self.word_lv_cell = tf.contrib.rnn.EmbeddingWrapper(cell=multi_word_cell,
																embedding_classes=self.vocab_size,
																embedding_size=self.embedding_dim,
																initializer=tf.truncated_normal_initializer(stddev=6/math.sqrt(self.embedding_dim)))
			#--sentence level cell--#
			sent_cell = tf.nn.rnn_cell.GRUCell(num_units=self.embedding_dim)
			multi_sent_cell = tf.nn.rnn_cell.MultiRNNCell([sent_cell] * self.num_layer)
			self.sent_lv_cell = multi_sent_cell


			#--record for all buckets--#
			self.b_query_state = []
			self.b_answer_state = []
			self.b_state = []
			self.b_logits = []
			self.b_softmax = []
			self.b_loss = []

			#--build model for all buckets--#
			print('>> Discriminator: constructing generator model, this may take a few minutes depending on the number of buckets...\n')
			for i, bucket in enumerate(self.buckets):
				with tf.variable_scope('word_lv_encoder', reuse=True if i > 0 else None) as scope:
					query_output, query_state = tf.nn.static_rnn(cell=self.word_lv_cell, 		# >> output shape: [bucket_len, batch_size, embedding_dim]
																 inputs=self.query[:bucket[0]], # >> state shape : [num_layer, batch_size, embedding_dim]
																 dtype='float32') 
					scope.reuse_variables()
					answer_output, answer_state = tf.nn.static_rnn(cell=self.word_lv_cell, 
																   inputs=self.answer[:bucket[1]],
																   dtype='float32')
					self.b_query_state.append(query_state)
					self.b_answer_state.append(answer_state)
					sent_lv_input = [query_state[-1], answer_state[-1]]

				with tf.variable_scope('sent_lv_encoder', reuse=True if i > 0 else None):
					output, state = tf.nn.static_rnn(cell=self.sent_lv_cell, # >> output shape: [input_len(2), batch_size, embedding_size]
													 inputs=sent_lv_input,	 # >> state shape : [num_layer, batch_size, embedding_dim]
													 dtype='float32')
					self.b_state.append(state)
					top_state = state[-1]  # top_state shape: [batch_size, embedding_dim]

				#--output_layer--#
				with tf.variable_scope('output_layer', reuse=True if i > 0 else None):
					output_w = tf.get_variable(name='output_w',
												shape=[self.embedding_dim, self.num_class], # >> w shape: [in_units, out_units]
												dtype='float32', 
												initializer=tf.truncated_normal_initializer(stddev=0.1))
					output_b = tf.get_variable(name='output_b',
												shape=[self.num_class], 
												dtype='float32', 
												initializer=tf.truncated_normal_initializer(stddev=0.1))
					logits = tf.nn.xw_plus_b(x=top_state, weights=output_w, biases=output_b) # >> x shape: [batch, in_units]
					softmax = tf.nn.softmax(logits=logits)
					self.b_logits.append(logits) # >> logits shape : [batch, out_units]
					self.b_softmax.append(softmax) # >> softmax shape: same as logits

				#--loss--#
				with tf.name_scope('loss'):
					loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=logits) # >> labels shape: [batch_size] each is an index in [0, num_classes)
					mean_loss = tf.reduce_mean(loss)
					self.b_loss.append(mean_loss)
				
				#--Uncomment and modify this to debug--#
				""" 
				self._vis_tf_tensor_shape(output_=state,
										  input_=self.query,
										  bucket_size=bucket[0],
										  input2_=self.answer,
										  bucket_size_2=bucket[1])
				"""

			#--train_op--#
			with tf.name_scope('train_op'):
				print('>> Discriminator: constructing training optimizers, this may take a few minutes depending on the number of buckets...\n')
				self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
				self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
				self.gradient_norms = []
				self.train_ops = []

				for bucket_id in range(len(self.buckets)):
					gradients = tf.gradients(ys=self.b_loss[bucket_id], xs=self.params)
					grads, norm = tf.clip_by_global_norm(t_list=gradients, clip_norm=self.max_grad_norm)
					self.gradient_norms.append(norm)
					self.train_ops.append(self.optimizer.apply_gradients(zip(grads, self.params)))

			#--model saver--#
			self.variables = [var for var in tf.global_variables() if 'discriminator' in var.name]
			self.saver = tf.train.Saver(var_list=self.variables, max_to_keep=1)

	############
	# RUN STEP #
	############
	"""
		This function runs one step in training.
		Returns the discriminator loss and reward
	"""
	def run_step(self, sess, querys, answers, labels, bucket_id, train=True):
		
		#--feed dict--#
		feed_dict = {}
		for i in range(self.buckets[bucket_id][0]):
			feed_dict[self.query[i].name] = querys[i]
		for i in range(self.buckets[bucket_id][1]):
			feed_dict[self.answer[i].name] = answers[i]
		feed_dict[self.target.name] = labels

		#--output feed--#
		if not train:
			output_feed = [ self.b_logits[bucket_id],
							self.b_softmax[bucket_id],
							self.b_loss[bucket_id] ]
			logits, softmax, loss = sess.run(output_feed, feed_dict)
		else:
			output_feed = [ self.train_ops[bucket_id],
							self.b_logits[bucket_id],
							self.b_softmax[bucket_id],
							self.b_loss[bucket_id] ]
			train_op, logits, softmax, loss = sess.run(output_feed, feed_dict)


		#--calculate reward--#
		reward = 0.0
		#generator_sample_num = 0
		for pred, label in zip(softmax, labels):
			if int(label) == 0: # >> 0 signifies a generated sample
				reward += pred[1]
			elif int(label) == 1:
				reward += -pred[0]
				#generator_sample_num += 1
		#reward = reward / generator_sample_num
		#reward = reward / (1.0 * config.beam_size)

		return reward, loss

				
	#****************************************************#	
	#***************** HELPER FUNCTIONS *****************#
	#****************************************************#


	##########################
	# VISUALIZE TENSOR SHAPE #
	##########################
	"""
		Use for debugging, this function prints and visualizes tensorflow tensor shape.
		May needs to be modify and rewrite manually to see different output shapes.
	"""
	def _vis_tf_tensor_shape(self, output_, input_, bucket_size, input2_=None, bucket_size_2=None):
		import numpy as np
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			feed_dict = {}
			for step in range(bucket_size):
				feed_dict[input_[step].name] = np.zeros((self.batch_size))
			if input2_ is not None and bucket_size_2 is not None:
				for step in range(bucket_size_2):
					feed_dict[input2_[step].name] = np.zeros((self.batch_size))
			print(sess.run(tf.shape(output_), feed_dict))


########
# MAIN #
########
"""
	For testing and debug purpose
"""
if __name__ == '__main__':
	DIS = Discriminator(config, vocab_size=20000)

