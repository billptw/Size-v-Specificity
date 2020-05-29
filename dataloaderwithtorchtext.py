from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe

import numpy as np
import pandas as pd
import logging
import os
import torch
import sys

from copy import deepcopy
from utils import remove_contractions, clean_dataset, remove_stopwords

LOGGER = logging.getLogger('tweets_dataset')

def get_dataset(train, test, fix_length=100, lower=False, vectors=None, glove='6B'):
	
	if vectors is not None:
		lower=True
		
	LOGGER.debug('Preparing CSV files...')
	train = pd.read_csv(train)
	test = pd.read_csv(test)

	prepare_csv(train, test)
	
	TEXT = data.Field(sequential=True, 
#                       tokenize='spacy', 
					  lower=True, 
					  include_lengths=True, 
					  batch_first=True, 
					  fix_length=21)
	LABEL = data.Field(use_vocab=True,
					   sequential=False,
					   dtype=torch.float16)
	ID = data.Field(use_vocab=False,
					sequential=False,
					dtype=torch.float16)
	
	
	LOGGER.debug('Reading train csv files...')
	
	train_temp = data.TabularDataset(path='cache/dataset_train.csv', format='csv',
		skip_header=True,
		fields=[
			('id', ID),
			('target', LABEL),
			('text', TEXT)
		])
	
	LOGGER.debug('Reading test csv file...')
	
	test_temp = data.TabularDataset(
		path='cache/dataset_test.csv', format='csv',
		skip_header=True,
		fields=[
			('id', ID),
			('target', LABEL),
			('text', TEXT)
		]
	)
	
	LOGGER.debug('Building vocabulary...')

	if glove=='27B':
		TEXT.build_vocab(
			train_temp, test_temp,
			max_size=20000,
			min_freq=10,
			vectors=GloVe(name="twitter.27B", dim=200)  # We use it for getting vocabulary of words
		)
	else:
		TEXT.build_vocab(
			train_temp, test_temp,
			max_size=20000,
			min_freq=10,
			vectors=GloVe(name=glove, dim=300)  # We use it for getting vocabulary of words
		)

	LABEL.build_vocab(
		train_temp
	)
	ID.build_vocab(
		train_temp, test_temp
	)
	
	word_embeddings = TEXT.vocab.vectors
	vocab_size = len(TEXT.vocab)
	
	train_iter = get_iterator(train_temp, batch_size=32, 
							  train=True, shuffle=True,
							  repeat=False)
	test_iter = get_iterator(test_temp, batch_size=1, 
							 train=False, shuffle=False,
							 repeat=False)
	
	LOGGER.debug('Done preparing the datasets')
	
	return TEXT, vocab_size, word_embeddings, train_iter, test_iter

def prepare_csv(df_train, df_test, seed=27, val_ratio=0.3):
	idx = np.arange(df_train.shape[0])
	np.random.seed(seed)
	np.random.shuffle(idx)

	# val_size = int(len(idx) * val_ratio)

	if not os.path.exists('cache'):
		os.makedirs('cache')
	
	train_text = df_train['text'] 
	test_text = df_test['text']

	# Remove contractions in both training and testing data
	train_text = train_text.apply(remove_contractions)
	test_text = test_text.apply(remove_contractions)

	# Clean dataset
	train_text = train_text.apply(clean_dataset)
	test_text = test_text.apply(clean_dataset)

	# Removing stopwords
	train_text = remove_stopwords(train_text)
	test_text = remove_stopwords(test_text)

	df_train[['text']] = train_text
	df_test[['text']] = test_text

	df_train[['id', 'target', 'text']].to_csv('cache/dataset_train.csv',
				   index=False)
	
	df_test[['id', 'target', 'text']].to_csv('cache/dataset_test.csv',
				   index=False)

	dataset_text = pd.concat([train_text, test_text])


def get_iterator(dataset, batch_size, train=True,
				 shuffle=True, repeat=False):
	
	device = torch.device('cuda:0' if torch.cuda.is_available()
						  else 'cpu')
	
	dataset_iter = data.Iterator(
		dataset, batch_size=batch_size, device=device,
		train=train, shuffle=shuffle, repeat=repeat,
		sort=False
	)
	
	return dataset_iter


class checkGloveOOV:
	def __init__(self, train_csv, test_csv, glove=''):
		# self.w2v = KeyedVectors.load_word2vec_format('data/crisisNLP_word_vector.bin', binary=True)

		if glove == '27B':
			self.glove = GloVe(name="twitter.27B", dim=200)
		elif glove == '42B':
			self.glove = GloVe(name="42B", dim=300)
		elif glove == '6B':
			self.glove = GloVe(name="6B", dim=300)
		
		print('Glove type:', glove)

		self.twitter = True if glove == '27B' else False

		train_csv = pd.read_csv(train_csv)
		test_csv = pd.read_csv(test_csv)

		train_text = train_csv['text']
		test_text = test_csv['text']

		dataset_text = pd.concat([train_text, test_text])

		# Remove contractions in both training and testing data
		dataset_text = dataset_text.apply(remove_contractions)

		# Clean dataset
		dataset_text = dataset_text.apply(clean_dataset)

		# Removing stopwords
		dataset_text = remove_stopwords(dataset_text)

		self.dataset_text = dataset_text

		self.word2index = {}
		self.n_words = 0  # Count pad

		self.oov = []
		self.num_oov_sentences = 0

	def addSentence(self, sentence):
		has_oov = False
		for word in sentence.split(' '):
			if self.addWord(word): has_oov = True
		if has_oov: self.num_oov_sentences += 1

	def addWord(self, word):
		if word not in self.word2index:
			if word in self.glove.stoi:
				self.word2index[word] = torch.tensor(self.glove.stoi[word])
			else:
				self.word2index[word] = torch.randn(200) if self.twitter else torch.randn(300) 
				self.oov.append(word)
			self.n_words += 1
		return word not in self.glove.stoi


if __name__ == '__main__':
	""" To check for twitter, run dataloaderwithtorchtext.py XXX, where XXX represents any string input
	"""
	print('Inputs: 6B, 27B, 42B')
	if len(sys.argv) > 1:
		lang_obj = checkGloveOOV("./data/train.csv", "./data/test.csv", glove=sys.argv[1])
	# else:
	# 	print("Checking gloVe.twitter.27B.200")
	# 	lang_obj = checkGloveOOV("./data/train.csv", "./data/test.csv", glove='twitter')
	for sentence in lang_obj.dataset_text:
		lang_obj.addSentence(sentence)

	print('Total number of words: {}  OOV words: {}'.format(lang_obj.n_words, len(lang_obj.oov)))
	print('Total number of tweets: {}  OOV tweets: {}'.format(len(lang_obj.dataset_text), lang_obj.num_oov_sentences))
	# print(len(lang_obj.word2index))


