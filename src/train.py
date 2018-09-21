# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ train.py ]
#   Synopsis     [ MLE pre-train and RL-GAN training procedures ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import pickle
import numpy as np
import tensorflow as tf
from generator import Generator
from discriminator import Discriminator
from data_loader import Data_Loader
from configuration import config, BUCKETS


#############
# PRE TRAIN #
#############
"""
	Generator supervise maximum likelihood estimation training.
	With model checkpoint function supported: 'save model at best loss' and 'early stopping'
"""
def pre_train():
	print('### Generator MLE Pre-Train Mode ###')
	
	#--construct data loader--#
	data_loader = Data_Loader(config)
	vocab_size = data_loader.get_vocab_size()

	#--construct generator model--#
	GEN = Generator(config, vocab_size, build_train_op=True)

	#--tensorflow session--#
	with tf.Session() as sess:

		#--initialize training--#
		sess.run(tf.global_variables_initializer())
		GEN = _check_restore_parameters(sess, model=GEN, specific_model=config.pre_train_model)
		history_loss = []
		best_loss = 777.777
		counter = 0
		
		#--training epochs--#
		for epoch in range(config.pre_train_epochs):
			
			#--initialize epoch--#
			epoch_loss = []
			data_loader.shuffle_data()

			#--run over all buckets--#
			bucket_ids = data_loader.shuffle_buckets()
			for bucket_n, bucket_id in enumerate(bucket_ids):
				
				#--run epoch on each bucket--#
				n_batch = 0
				while (n_batch * config.batch_size) < data_loader.get_bucket_len(bucket_id):
					
					#--get batch--#
					(
					batch_encoder_inputs,
					batch_decoder_inputs,
					decoder_masks
					) = data_loader.generator_get_batch(bucket_id=bucket_id, batch_id=n_batch, mode='pre_train')
					n_batch += 1

					#--update--#
					step_loss = GEN.run_step(sess,
											 batch_encoder_inputs, 
											 batch_decoder_inputs, 
											 decoder_masks, 
											 bucket_id, 
											 feed_previous=False, # >> in train mode, decoder_inputs are used as given, we dont feed the previous output, feed_previous=False
											 add_reward=False,
											 mc_search=False)
					#--print info--#
					epoch_loss.append(step_loss)
					print('Epoch: %i/%i, Bucket: %i/%i, Batch: %i/%i, cur_Loss: %.5f' % (epoch+1,
																						 config.pre_train_epochs,
																						 bucket_n+1,
																						 data_loader.get_num_buckets(),
																						 n_batch-1,
																						 data_loader.get_bucket_len(bucket_id)/config.batch_size,
																						 step_loss), end='\r')

			#--epoch checkpoint--#
			history_loss.append(np.mean(epoch_loss))
			best_loss, to_save = _model_checkpoint_save_best(best_val=best_loss, cur_val=history_loss[-1], mode='min')
			counter, to_stop = _model_checkpoint_earlystopping(counter=counter, reset=to_save, patient=config.patient)

			#--save best and show training info--#
			if config.force_save: to_save = True
			print('Epoch: %i/%i, Bucket: %i/%i, Batch: %i/%i, avg_Loss: %.5f, Saved: %s' % (epoch+1,
																							config.pre_train_epochs,
																							data_loader.get_num_buckets(),
																							data_loader.get_num_buckets(),
																							n_batch-1,
																							n_batch-1,
																							history_loss[-1],
																							'True' if to_save else 'False'))
			#--save model and training log--#
			if to_save == True:
				GEN.saver.save(sess, os.path.join(config.model_dir, 'pre_train'), global_step=(epoch+1))
				pickle.dump(history_loss, open(os.path.join(config.model_dir, 'history_loss.pkl'), 'wb'), True)

			#--earlystopping--#
			if to_stop: break

		#--pre train end--#
		print('Pre-training process complete, model saved to: ', config.model_dir)


#############
# GAN TRAIN #
#############
"""
	Generative Adversarial Training with Reinforcement Learning.
"""
def gan_train():
	print('### Generative Adversarial Training with Reinforcement Learning Mode ###')
	
	#--construct data loader--#
	data_loader = Data_Loader(config)
	vocab_size = data_loader.get_vocab_size()

	#--construct generator and discriminator model--#
	GEN = Generator(config, vocab_size, build_train_op=True)
	DIS = Discriminator(config, vocab_size)

	#--tensorflow session--#
	with tf.Session() as sess:

		#--initialize training--#
		sess.run(tf.global_variables_initializer())
		GEN = _check_restore_parameters(sess, model=GEN, specific_model=config.gan_train_model)
		dis_history_loss = []
		gen_history_loss = []
		tf_history_loss = []
		r_history_loss = []
		best_loss = 777.777
		counter = 0
		
		#--training epochs--#
		for epoch in range(config.gan_train_epochs):
			
			#--initialize epoch--#
			dis_epoch_loss = []
			gen_epoch_loss = []
			tf_epoch_loss = []
			r_epoch_loss = []
			data_loader.shuffle_data()

			#--run over all buckets--#
			bucket_ids = data_loader.shuffle_buckets()
			for bucket_n, bucket_id in enumerate(bucket_ids):
				
				#--run epoch on each bucket--#
				n_batch = 0
				while (n_batch * config.batch_size) < data_loader.get_bucket_len(bucket_id):

					#---Update Discriminator---#
					for _ in range(config.update_d):
						
						#--get batch--#
						(
						batch_encoder_inputs,
						batch_decoder_inputs,
						decoder_masks,
						encoder_inputs, 
						decoder_inputs,
						) = data_loader.generator_get_batch(bucket_id=bucket_id, batch_id=n_batch, mode='gan_train')
						n_batch += 1

						#--gather (X,Y) and sample (X, ^Y) through ^Y ~ G(*|X)--#
						train_query, train_answer, train_labels = GEN.sample(sess, 
																			 batch_encoder_inputs,
																			 batch_decoder_inputs,
																			 decoder_masks, 
																			 encoder_inputs,
																			 decoder_inputs, 
																			 bucket_id, 
																			 mc_search=False)

						#--update D using (X, Y) as positive examples and (X, ^Y) as negative examples--#
						_, dis_step_loss = DIS.run_step(sess,
														train_query, 
														train_answer, 
														train_labels, 
														bucket_id, 
														train=True) # >> This trains the discriminator
						
						#--record--#
						dis_epoch_loss.append(dis_step_loss)

					#---Update Generator---#
					for _ in range(config.update_g):
						
						#--gather (X,Y) and sample (X, ^Y) through ^Y ~ G(*|X) with Monte Carlo search--#
						train_query, train_answer, train_labels = GEN.sample(sess, 
																			 batch_encoder_inputs,
																			 batch_decoder_inputs,
																			 decoder_masks, 
																			 encoder_inputs,
																			 decoder_inputs, 
																			 bucket_id, 
																			 mc_search=True)

						#--compute reward r for (X, ^Y ) using D based on monte carlo search--#
						reward, _ = DIS.run_step(sess,
												 train_query, 
												 train_answer, 
												 train_labels, 
												 bucket_id, 
												 train=False) # >> This does not train the discriminator

						#--update G on (X, ^Y ) using reward r--#
						gen_step_loss = GEN.run_step(sess,
												 	 batch_encoder_inputs, 
													 batch_decoder_inputs, 
													 decoder_masks, 
													 bucket_id,
													 reward=reward, 
													 feed_previous=False, 
													 add_reward=True, # >> add reward
													 mc_search=False)

						#--teacher forcing: update G on (X, Y)--#
						tf_step_loss = GEN.run_step(sess,
													batch_encoder_inputs, 
													batch_decoder_inputs, 
													decoder_masks, 
													bucket_id,
													reward=None, 
													feed_previous=False, 
													add_reward=False,
													mc_search=False)
						
						#--record--#
						gen_epoch_loss.append(gen_step_loss)
						tf_epoch_loss.append(tf_step_loss)
						r_epoch_loss.append(reward)

					#--print info--#
					print('Epoch: %i/%i, Bucket: %i/%i, Batch: %i/%i, Reward: %.5f, dis_Loss: %.5f, gen_Loss: %.5f, tf_Loss: %.5f' % (
																								epoch+1,
																								config.gan_train_epochs,
																								bucket_n+1,
																								data_loader.get_num_buckets(),
																								n_batch-1,
																								data_loader.get_bucket_len(bucket_id)/config.batch_size,
																								reward,
																								dis_step_loss,
																								gen_step_loss,
																								tf_step_loss), end='\r')					
				
			#--epoch checkpoint--#
			dis_history_loss.append(np.mean(dis_epoch_loss))
			gen_history_loss.append(np.mean(gen_epoch_loss))
			tf_history_loss.append(np.mean(tf_epoch_loss))
			r_history_loss.append(np.mean(r_epoch_loss))
			best_loss, to_save = _model_checkpoint_save_best(best_val=best_loss, cur_val=tf_history_loss[-1], mode='min')
			counter, to_stop = _model_checkpoint_earlystopping(counter=counter, reset=to_save, patient=config.patient)

			#--save best and show training info--#
			if config.force_save: to_save = True
			print('Epoch: %i/%i, Bucket: %i/%i, Batch: %i/%i, Reward: %.5f, dis_Loss: %.5f, gen_Loss: %.5f, tf_Loss: %.5f, Saved: %s' % (
																								epoch+1,
																								config.gan_train_epochs,
																								data_loader.get_num_buckets(),
																								data_loader.get_num_buckets(),
																								n_batch-1,
																								n_batch-1,
																								r_history_loss[-1],
																								dis_history_loss[-1],
																								gen_history_loss[-1],
																								tf_history_loss[-1],
																								'True' if to_save else 'False'))
			#--save model and training log--#
			if to_save == True:
				DIS.saver.save(sess, os.path.join(config.model_dir, 'gan_dis_model'), global_step=(epoch+1))
				GEN.saver.save(sess, os.path.join(config.model_dir, 'gan_gen_model'), global_step=(epoch+1))
				pickle.dump(dis_history_loss, open(os.path.join(config.model_dir, 'dis_history_loss.pkl'), 'wb'), True)
				pickle.dump(gen_history_loss, open(os.path.join(config.model_dir, 'gen_history_loss.pkl'), 'wb'), True)
				pickle.dump(tf_history_loss, open(os.path.join(config.model_dir, 'tf_history_loss.pkl'), 'wb'), True)
				pickle.dump(r_history_loss, open(os.path.join(config.model_dir, 'r_history_loss.pkl'), 'wb'), True)
			
			#--earlystopping--#
			if to_stop: break
			if r_history_loss[-1] < 0.00000001 and r_history_loss[-2] < 0.00000001: break

		#--pre train end--#
		print('Generative Adversarial Training process complete, model saved to: ', config.model_dir)


############
# READ LOG #
############
"""
	Read training log file.
"""
def read_log():
	log = pickle.load(open(os.path.join(config.model_dir, 'history_loss.pkl'), 'rb'))
	print('Training log:')
	for epoch, log_line in enumerate(log):
		print('Epoch', epoch+1, ':', log_line)


#****************************************************#	
#***************** HELPER FUNCTIONS *****************#
#****************************************************#


#############################
# _CHECK RESTORE PARAMETERS #
#############################
"""
	Called by pre_train(), this function restore the previously trained parameters if there are any.
	Specify a specific model parameters by passing in 'specific_model'
"""
def _check_restore_parameters(sess, model, specific_model=None):
	#--restore specific model--#
	if specific_model is not None:
		try:
			model.saver.restore(sess, save_path=os.path.join(config.model_dir, specific_model))
			print('>> Loading specified parameters for the generator: ', specific_model)
			return model
		except:
			pass
	#--restore latest checkpoint--#
	ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.model_dir + '/checkpoint'))
	if ckpt is not None:
		model.saver.restore(sess, save_path=tf.train.latest_checkpoint(checkpoint_dir=config.model_dir))
		print('>> Loading existing parameters for the generator...')
		return model
	#--initialize fresh model--#
	else:
		print('>> Initializing fresh parameters for the generator.')
		return model


##############
# _SAVE BEST #
##############
"""
	Called by pre_train(), this function checks and saves the best model.
"""
def _model_checkpoint_save_best(best_val, cur_val, mode):
	if mode == 'min':
		if cur_val < best_val: return cur_val, True
		else: return best_val, False
	elif mode == 'max':
		if cur_val > best_val: return cur_val, True
		else: return best_val, False
	else: raise ValueError('Invalid Mode!')


###################
# _EARLY STOPPING #
###################
"""
	Called by pre_train(), this function checks whether to stop the training process.
"""
def _model_checkpoint_earlystopping(counter, reset, patient=5):
	#--set counter--#
	if reset == True:
		counter = 0
	elif reset == False:
		counter += 1
	#--check patients--#
	if counter > patient:
		return 0, True # >> reset counter and stop training
	else:
		return counter, False # >> return counter and continue training


#########
# TRAIN #
#########
"""
	Run training process.
"""
def train():
	pre_train()
	gan_train()


if __name__ == '__main__':
	train()


