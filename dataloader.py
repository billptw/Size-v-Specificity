import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torchtext.vocab as vocab
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import time, datetime
import sys
import re
import nltk
import logging # for debugging purposes

import itertools
import collections
from collections import Counter

from utils import remove_contractions, clean_dataset, remove_stopwords

class Lang:
    def __init__(self, train_csv, test_csv):
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
        self.word2count = {}
        self.index2word = {0: "pad"}
        self.n_words = 1  # Count pad

    def addSentence(self, sentence):
        # if sentence != '':
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1



class DisasterTweetTrainDataset(Dataset):
    def __init__(self, csv_file, lang_obj):

        if os.path.exists("./debug.log"):
            os.remove("./debug.log") # remove log file and only keep latest
        logging.basicConfig(filename='./debug.log', level=logging.DEBUG)

        #### Loading data using pandas ####
        train = pd.read_csv(csv_file)
        logging.info('Printing top 5 train dataset items')
        logging.debug(train.head())

        train = train[['text', 'target']]
        logging.info('Printing top 5 train dataset items without NULL items')
        logging.debug(train.head())

        # Remove contractions in both training and testing data
        train['text'] = train['text'].apply(remove_contractions)

        # Clean dataset
        train['text'] = train['text'].apply(clean_dataset)

        # Removing stopwords
        train['text'] = remove_stopwords(train['text'])

        train = train[['text', 'target']]
        logging.info('Printing top 5 train dataset cleaned items')
        logging.debug(train.head())

        self.lang_obj = lang_obj
        self.training_data = []
        self.training_label = []

        self.no_words = self.lang_obj.n_words

        for index, sentence in enumerate(train['text']):
            # if sentence != '':
            sentence_list = sentence.split(' ')
            self.training_data.append([self.lang_obj.word2index[word] for word in sentence_list])
            self.training_label.append(train['target'][index])

        self.max_seq = max(len(x) for x in self.training_data) # 21 words

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, index):
        sample = torch.zeros(1, self.max_seq)
        seq_lengths = torch.LongTensor(list(map(len, [self.training_data[index]])))
        sample[0, :len(self.training_data[index])] = torch.tensor(self.training_data[index], dtype=torch.long)
        label = torch.tensor(self.training_label[index])

        return sample, label, seq_lengths

class DisasterTweetTestDataset(Dataset):
    def __init__(self, csv_file, lang_obj):
        
        #### Loading data using pandas ####
        test = pd.read_csv("./data/test.csv")
        logging.info('Printing top 5 test dataset items')
        logging.debug(test.head())

        test = test[['id', 'text', 'target']]
        logging.info('Printing top 5 test dataset items without NULL items')
        logging.debug(test.head())

        # Remove contractions in both training and testing data
        test['text'] = test['text'].apply(remove_contractions)

        # Clean dataset
        test['text'] = test['text'].apply(clean_dataset)

        # Removing stopwords
        test['text'] = remove_stopwords(test['text'])
        pred = test['text']

        test = test[['id', 'text', 'target']]
        logging.info('Printing top 5 test dataset cleaned items')
        logging.debug(test.head())


        self.lang_obj = lang_obj
        self.testing_data = []
        self.testing_label = []
        self.no_words = self.lang_obj.n_words

        for index, sentence in enumerate(test['text']):
            # if sentence != '':
            sentence_list = sentence.split(' ')
            self.testing_data.append([self.lang_obj.word2index[word] for word in sentence_list])
            self.testing_label.append(test['target'][index])

        self.max_seq = 21 # 21 words

    def __len__(self):
        return len(self.testing_data)

    def __getitem__(self, index):
        sample = torch.zeros(1, self.max_seq)
        seq_lengths = torch.LongTensor(list(map(len, [self.testing_data[index]])))
        sample[0, :len(self.testing_data[index])] = torch.tensor(self.testing_data[index], dtype=torch.long)
        label = torch.tensor(self.testing_label[index])

        return sample, label




