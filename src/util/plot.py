import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

directory = '../../model'

reward_his_path1 = os.path.join(directory, 'history_loss-400.pkl') 
#reward_his_path2 = os.path.join(directory, 'plot_wgan_gp.pkl') 
#reward_his_path3 = os.path.join(directory, 'plot_wgan.pkl')

def plot():
	reward_his1 = pickle.load(open(reward_his_path1, 'rb'))
	#reward_his2 = pickle.load(open(reward_his_path2, 'rb'))
	#reward_his3 = pickle.load(open(reward_his_path3, 'rb'))


	plt.plot(np.arange(len(reward_his1)), reward_his1, 'r')
	#plt.plot(np.arange(len(reward_his2)), reward_his2, 'b')
	#plt.plot(np.arange(len(reward_his3)), reward_his3, 'g')
	plt.ylabel('Loss')
	plt.xlabel('Number of Epochs')
	plt.show()




plot()


