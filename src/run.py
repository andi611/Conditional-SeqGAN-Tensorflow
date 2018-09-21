# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ run.py ]
#   Synopsis     [ Run pre-train, gan-train, or chat mode ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
from configuration import config
from train import pre_train, gan_train, read_log
from test import chat, evaluate


#######
# RUN #
#######
def run():
	if config.read_log:
		read_log()
	if config.pre_train:
		pre_train()
	if config.gan_train:
		gan_train()
	if config.pre_chat:
		chat(speak_with_audio=config.speak, model_name=config.pre_train_model)
	if config.gan_chat:
		chat(speak_with_audio=config.speak, model_name=config.gan_train_model)
	if config.pre_evaluate:
		evaluate(model_name=config.pre_train_model)
	if config.gan_evaluate:
		evaluate(model_name=config.gan_train_model)


if __name__ == '__main__':
	run()