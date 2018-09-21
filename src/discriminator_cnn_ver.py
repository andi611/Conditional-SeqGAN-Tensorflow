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
	A CNN Discriminator model for text classification:
	Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
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
		self.vocab_size = vocab_size
		self.max_seq_len = BUCKETS[-1][0] + BUCKETS[-1][1]
		self.batch_size = config.batch_size

		#--discriminator hyper-parameters--#
		self.lr = config.d_lr
		self.embedding_dim = config.d_embedding_dim
		self.num_class = config.d_num_class
		self.l2_reg_lambda = config.d_l2_reg_lambda
		self.dropout_keep_prob = tf.get_variable(name='dropout_keep_prob', shape=[], initializer=tf.constant_initializer(config.d_dropout_keep_prob))
		
		#--discriminator constant-parameters--#
		self.filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
		self.num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

		#--placeholders--#
		self.input_x = tf.placeholder('int32', [None, self.max_seq_len], name='input_x')
		self.input_y = tf.placeholder('float32', [None, self.num_class], name='input_y')

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

			#--embedding layer--#
			with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
				self.W = tf.get_variable(name='W',
										 shape=[self.vocab_size, self.embedding_dim],
										 initializer=tf.truncated_normal_initializer(stddev=6/math.sqrt(self.embedding_dim)))
				self.word_embedding = tf.nn.embedding_lookup(params=self.W, ids=self.input_x)
				self.word_embedding_expanded = tf.expand_dims(input=self.word_embedding, axis=-1)

			#--convolution+maxpool layer for each filter size--#
			pooled_outputs = []
			for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
				with tf.variable_scope('conv_maxpool_%s' % filter_size):

					#--convolution layer--#
					filter_shape = [filter_size, self.embedding_dim, 1, num_filter]
					W = tf.get_variable(name='W', shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
					b = tf.get_variable(name='b', shape=[num_filter], initializer=tf.constant_initializer(0.1))
					conv = tf.nn.conv2d(input=self.word_embedding_expanded,
										filter=W,
										strides=[1, 1, 1, 1],
										padding='VALID',
										name='conv')
					#--add bias and nonlinearity--#
					conv_b = tf.nn.bias_add(value=conv, bias=b, name='conv_b')
					h = tf.nn.relu(conv_b, name='relu')
					#--maxpooling--#
					pooled = tf.nn.max_pool(value=h,
											ksize=[1, self.max_seq_len - filter_size + 1, 1, 1],
											strides=[1, 1, 1, 1],
											padding='VALID',
											name='max_pooling')
					pooled_outputs.append(pooled)
			
			#--combine all the pooled features--#
			total_num_filters = sum(self.num_filters)
			self.h_pool = tf.concat(values=pooled_outputs, axis=3)
			self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_num_filters])

			#--add highway--#
			with tf.name_scope('highway'):
				self.h_highway = self._highway(input_=self.h_pool_flat, size=self.h_pool_flat.get_shape()[1], num_layers=1, bias=0)

			#--add dropout--#
			with tf.name_scope('dropout'):
				self.h_drop = tf.nn.dropout(x=self.h_highway, keep_prob=self.dropout_keep_prob)

			#--l2 regularization loss--#
			l2_loss = tf.constant(0.0)

			#--final (unnormalized) scores and predictions--#
			with tf.name_scope('output'):
				W = tf.get_variable(name='W', shape=[total_num_filters, self.num_class], initializer=tf.truncated_normal_initializer(stddev=0.1))
				b = tf.get_variable(name='b', shape=[self.num_class], initializer=tf.constant_initializer(0.1))
				l2_loss += tf.nn.l2_loss(W)
				l2_loss += tf.nn.l2_loss(b)
				self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
				self.ypred_for_auc = tf.nn.softmax(self.scores)
				self.predictions = tf.argmax(self.scores, 1, name="predictions")

			#--calculat mean cross-entropy loss--#
			with tf.name_scope("loss"):
				losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
				self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

		#--train_op--#
		self.optimizer = tf.train.AdamOptimizer(self.lr)
		self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
		gradients = self.optimizer.compute_gradients(loss=self.loss, var_list=self.params, aggregation_method=2)
		self.train_op = self.optimizer.apply_gradients(gradients)


	#****************************************************#	
	#***************** HELPER FUNCTIONS *****************#
	#****************************************************#


	############
	# _HIGHWAY #
	############
	"""
		Called by build_model(). Highway Network (cf. http://arxiv.org/abs/1505.00387).
		t = sigmoid(Wy + b)
		z = t * g(Wy + b) + (1 - t) * y
		where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
		borrowed from https://github.com/mkroutikov/tf-lstm-char-cnn.
	"""
	def _highway(self, input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
		with tf.variable_scope(scope):
			for idx in range(num_layers):
				g = f(self._linear(input_, size, scope='highway_lin_%d' % idx))
				t = tf.sigmoid(self._linear(input_, size, scope='highway_gate_%d' % idx) + bias)
				output = t * g + (1. - t) * input_
				input_ = output
		return output


	###########
	# _LINEAR #
	###########
	"""	
		Called by _highway().
		An alternative to tf.nn.rnn_cell._linear function, which has been removed in Tensorfow 1.0.1
		Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
		Args:
			input_: a tensor or a list of 2D, batch x n, Tensors.
			output_size: int, second dimension of W[i].
			scope: VariableScope for the created subgraph; defaults to "Linear".
		Returns:
			A 2D Tensor with shape [batch x output_size] equal to
			sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
		Raises:
			ValueError: if some of the arguments has unspecified or wrong shape.
	"""
	def _linear(self, input_, output_size, scope=None):

		shape = input_.get_shape().as_list()
		if len(shape) != 2:
			raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
		if not shape[1]:
			raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
		input_size = shape[1]

		with tf.variable_scope(scope or "SimpleLinear"):
			matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
			bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

		return tf.matmul(input_, tf.transpose(matrix)) + bias_term


########
# MAIN #
########
"""
	For testing and debug purpose
"""
if __name__ == '__main__':
	discriminator = Discriminator(config)

