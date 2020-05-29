import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import time, datetime
import sys
import re
import nltk
import logging # for debugging purposes

nltk.download('punkt')
from nltk.corpus import stopwords

import itertools
import collections
from collections import Counter
from utils import remove_contractions, clean_dataset, remove_stopwords

from sklearn.metrics import f1_score


import argparse
parser = argparse.ArgumentParser(description='CE7455')
parser.add_argument('--data', type=str, default='data/',
                    help='location of the data corpus')
parser.add_argument('--epochs', type=int, default=4,
                    help='upper epoch limit')
parser.add_argument('--test_epochs', type=int, default=4,
                    help='test every epoch limit')
parser.add_argument('--lr', type=float, default=2e-5,
                    help='initial learning rate')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='adam epsilon')
parser.add_argument('--token_length', type=int, default=40,
                    help='max token length')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--seed', type=int, default=141,
                    help='random seed')
parser.add_argument('--f1', type=str, default='binary',
                    help='f1 average type')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        device = torch.device("cpu")
    else:
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda")
        print("GPU Used:")
        print(torch.cuda.get_device_name())


if os.path.exists("./debug.log"):
	os.remove("./debug.log") # remove log file and only keep latest
logging.basicConfig(filename='./debug.log', level=logging.DEBUG)

#### Loading data using pandas ####
train = pd.read_csv("./data/train.csv")
logging.info('Printing top 5 train dataset items')
logging.debug(train.head())

test = pd.read_csv("./data/test.csv")
logging.info('Printing top 5 test dataset items')
logging.debug(test.head())

train = train[['text', 'target']]
logging.info('Printing top 5 train dataset items without NULL items')
logging.debug(train.head())

test = test[['text', 'target']]
logging.info('Printing top 5 test dataset items without NULL items')
logging.debug(test.head())

# Remove contractions in both training and testing data
train['text'] = train['text'].apply(remove_contractions)
test['text'] = test['text'].apply(remove_contractions)

# Clean dataset
train['text'] = train['text'].apply(clean_dataset)
test['text'] = test['text'].apply(clean_dataset)

# Removing stopwords
train['text'] = remove_stopwords(train['text'])
test['text'] = remove_stopwords(test['text'])
pred = test['text']

train = train[['text', 'target']]
logging.info('Printing top 5 train dataset cleaned items')
logging.debug(train.head())

test = test[['text', 'target']]
logging.info('Printing top 5 test dataset cleaned items')
logging.debug(test.head())

print("Total number of training data: {}".format(len(train['text'])))
print("Total number of testing data: {}".format(len(test['text'])))

all_texts = []
for line in list(train['text']):
    texts = line.split()
    for text in texts:
        all_texts.append(text)

toBeCleanedNew='[%s]' % ' '.join(map(str, all_texts))#remove all the quation marks and commas. 

wordfreq = Counter(all_texts)
print("Total number of words in the vocabulary: {}".format(len(wordfreq)))

# Adapted code from https://mccormickml.com/2019/07/22/BERT-fine-tuning/

from transformers import BertTokenizer

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)

train_size = train['text'].size
test_size = test['text'].size

train_tokens = tokenizer.batch_encode_plus(train['text'], pad_to_max_length=True, max_length=args.token_length, return_tensors = 'pt')
test_tokens = tokenizer.batch_encode_plus(test['text'], pad_to_max_length=True, max_length=args.token_length, return_tensors = 'pt')


input_ids = train_tokens['input_ids']
attention_masks = train_tokens['attention_mask']
labels = torch.tensor(train['target'].append(test['target'], ignore_index=True))

train_input_ids = train_tokens['input_ids']
train_attention_masks = train_tokens['attention_mask']
train_labels = torch.tensor(train['target'])

test_input_ids = test_tokens['input_ids']
test_attention_masks = test_tokens['attention_mask']
test_labels = torch.tensor(test['target'])

logging.info('Printing 3 tensorized inputs')
for i in range(3):
    logging.info('Original: ')
    logging.debug(train['text'][i])
    logging.info('Token IDs:')
    logging.debug(input_ids[i])
    logging.info('Label:')
    logging.debug(labels[i])

from torch.utils.data import TensorDataset, random_split

train_set = TensorDataset(train_input_ids, train_attention_masks, train_labels)
test_set = TensorDataset(test_input_ids, test_attention_masks, test_labels)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = args.batch_size

train_dataloader = DataLoader(
            train_set,  # The training samples.
            sampler = RandomSampler(train_set), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

validation_dataloader = DataLoader(
            test_set, # The validation samples.
            sampler = SequentialSampler(test_set), # Pull out batches sequentially.
            batch_size = 1 # Evaluate with this batch size.
        )

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(test_size))

from transformers import BertForSequenceClassification, AdamW, BertConfig

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()

# Get all of the model's parameters as a list of tuples.
# params = list(model.named_parameters())

params = list(model.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
trainable_params = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])
print('Args:', args)
print('Model total parameters:', total_params)
print('Model trainable parameters:', trainable_params)

optimizer = AdamW(model.parameters(),
                  lr = args.lr,
                  eps = args.eps
                )

from transformers import get_linear_schedule_with_warmup

epochs = args.epochs

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


plot_train_loss = []
plot_eval_loss = []
plot_accuracy_train = []
plot_accuracy_eval = []


def train():
    t0 = time.time()

    total_train_loss = 0
    total_train_accuracy = 0
    total_train_f1 = 0

    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        

        loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)

        total_train_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_train_accuracy += flat_accuracy(logits, label_ids)
        total_train_f1 += f1_score(np.argmax(logits, axis=1).flatten(), label_ids.flatten())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    avg_train_f1 = total_train_f1 / len(train_dataloader)

    return avg_train_accuracy, avg_train_f1, avg_train_loss

def eval():
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0

    predictions = []

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():        
            (loss, logits) = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)            

            logits = logits.detach().cpu().numpy()

            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            total_eval_loss += loss.item()

            logits_ = np.argmax(logits, axis=1).flatten()
            predictions.append(logits_.tolist())

    predictions = np.asarray([item for sublist in predictions for item in sublist]).astype(int)
    df_subm = pd.read_csv("./data/test.csv")
    gt_label = df_subm['target']

    f1 = f1_score(predictions, gt_label, average=args.f1)
    avg_eval_loss = total_eval_loss / len(validation_dataloader)
    plot_eval_loss.append(avg_eval_loss) # per epoch

    avg_eval_accuracy = total_eval_accuracy / len(validation_dataloader)
    plot_accuracy_eval.append(avg_eval_accuracy)

    return avg_eval_accuracy, f1, avg_eval_loss


for i in range(args.epochs):
    avg_train_accuracy, avg_train_f1, avg_train_loss = train()
    avg_eval_accuracy, avg_eval_f1, avg_eval_loss = eval()

    print("Epoch {:2d} / {:2d}  |  Train Acc: {:1.6} |  Train F1: {:1.6f}  |  Train loss: {:1.6f}  |  Test Acc: {:1.6f} |  Test F1: {:1.6f}  |  Test loss: {:1.6f}"
    .format(i + 1, args.epochs, avg_train_accuracy, avg_train_f1, avg_train_loss, avg_eval_accuracy, avg_eval_f1, avg_eval_loss))