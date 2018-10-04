# Daisy: Dialog Analogous Intellectual System
## Machine Learning: Chatbot
**conditional sequence generative adversarial network trained with policy gradient, Implementation in Tensorflow** 
![](https://github.com/b03901170/machine-learning/blob/master/conditional%20seqGAN/image/framework.png)

## Requirements: 
* **Tensorflow r1.5.1**
* Python 3.6
* Numpy 1.13.3
* tempfile (Optional)
* gtts (Optional)
* pygame (Optional)


## Introduction
Apply Policy Gradient to Generative Adversarial Nets to improve ChatBot dialog generation quality.
- Generator: seq2seq model with attention and embedding
- Discriminator: hierarchical encoder and two class classifier
- Policy Gradient Update: use reward computed by the discriminator to update the generator
- Monte Carlo Rollout: to compute reward for every generation step
- Teacher Forcing: add maximum likelihood estimation training to GAN training
- sampled softmax: to reduce computation complexity from regular softmax
- model checkpoint supported: training early stopping and save model with best loss


## Preperation
1. Download corpus from:
```
1) https://www.cs.cornell.edu/%7Ecristian/Cornell_Movie-Dialogs_Corpus.html (official site)
2) https://drive.google.com/file/d/13TvKXXrKVg9X7IEayu7eYpRbxfZ5GDE0/view?usp=sharing (backup link)
```

2. extract the .zip file, and save them to the path below: 
```
config.corpus_path = '../cornell_movie_dialog_corpus'
```
This default path can be modified by changing the '--corpus_dir' option in 'config.py'.

3. Run Preprocessing on the raw corpus file:
```
python3 preprocess_data.py
```
This saves the pre-processed files to the path below:
```
config.data_dir = '../data'
```
This default path can be modified by changing the '--data_dir' option in 'config.py'.

4. After running pre-processing, 5 files will be created:
```
1. processed_corpus.txt : Encoder input for training
2. word2idx.pkl : word to idx dictionary mapping saved by pickle
3. idx2word.pkl : idx to word dictionary mapping saved by pickle
4. train_encode.pkl: model-ready training data: tokenized and index labeled text ready to be fed to the encoder
5. train_decode.pkl: model-ready training data: tokenized and index labeled text ready to be fed to the decoder
```


## Training
5. To run MLE pre train on the Generator model:
```
python3 run.py --pre_train --force_save
```
(optional: '--force_save' saves model every epoch, otherwise the default model checkpoint is activated and only the model with minimum loss will be saved.)

6. To run generative adversarial training with policy gradient:
```
python3 run.py --gan_train --force_save
```
(optional: '--force_save' saves model every epoch, otherwise the default model checkpoint is activated and only the model with minimum loss will be saved.)


## Inference
7. To chat with the pre-trained model:
```
python3 run.py --pre_chat --speak
```
(optional: '--speak' enables audio response)

8. To chat with the GAN-trained model:
```
python3 run.py --gan_chat --speak
```
(optional: '--speak' enables audio response)

9. To evaluate the BLEU score of the pre-trained model:
```
python3 run.py --pre_evaluate
```

10. To evaluate the BLEU score of the GAN-trained model:
```
python3 run.py --gan_evaluate
```


## Trained Models
11. To use exsiting models, place model files under the 'model' directory.
	Trained Models (pre-trained / gan-trained) can be download from:
```
https://drive.google.com/drive/folders/1C0A53qqQpdGyKIEB7HVb6BqlBD5gvEJy?usp=sharing
```


## Other
12. Source code:
```
1. configuration.py
2. data_loader.py
3. discriminator.py
4. generator.py
5. train.py
6. test.py
7. run.py
8. tf_seq2seq_model.py
```
