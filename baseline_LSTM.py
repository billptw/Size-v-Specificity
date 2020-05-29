import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataloaderwithtorchtext import get_dataset
from dataloader import DisasterTweetTrainDataset, DisasterTweetTestDataset, Lang
from model import LSTMClassifier # LinearModel, LSTM, GRU, 

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

from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description='CE7455')
parser.add_argument('--data', type=str, default='data/',
                    help='location of the data corpus')
parser.add_argument('--train_file', type=str, default='./data/train.csv')
parser.add_argument('--test_file', type=str, default='./data/test.csv')
parser.add_argument('--epochs', type=int, default=8,
                    help='upper epoch limit')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='adam epsilon')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA') # default to false
parser.add_argument('--seed', type=int, default=8888,
                    help='random seed')
parser.add_argument('--pretrained', action='store_true') # default to false
parser.add_argument('--model', type=str, default="LSTMClassifier")
# parser.add_argument("--print_every", type=int, default=500)

args = parser.parse_args()

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Check if GPU is available
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        device = torch.device("cpu")
    else:
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda")
        print("GPU Used:")
        print(torch.cuda.get_device_name())

# Create the vocabulary which consists of all text in both train and test
dictionary = Lang(args.train_file, args.test_file)
for sentence in dictionary.dataset_text:
    dictionary.addSentence(sentence)
print("Dictionary consists of {} total no. of words".format(dictionary.n_words))

train_dataset = DisasterTweetTrainDataset(args.train_file, dictionary)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

test_dataset = DisasterTweetTestDataset(args.test_file, dictionary)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

if args.model == "LSTMClassifier":
    model = LSTMClassifier(dictionary.n_words, output_size=2, embedding_dim=256, hidden_dim=256, weights=None)
    

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

criterion = nn.CrossEntropyLoss()

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

plot_train_loss = []
plot_eval_loss = []
plot_accuracy_train = []
plot_accuracy_eval = []

def train():
    total_train_loss = 0
    total_train_accuracy = 0

    model.train()

    for index, sample in enumerate(train_dataloader):
        data = sample[0].long().to(device).squeeze(1)
        gt_label = sample[1].to(device)
        seq_lengths = sample[2]

        # print(data.size())

        hidden = model.init_hidden(data.size(0))   

        optimizer.zero_grad() # same as model.zero_grad()

        if args.model == "LSTMClassifier":
            output, hidden = model(data, hidden)

        # print(output.size())
        loss = criterion(output, gt_label)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        gt_label_cpu = gt_label.detach().cpu().numpy()
        output_cpu = output.detach().cpu().numpy()

        total_train_accuracy += flat_accuracy(output_cpu, gt_label_cpu)

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader) 
    plot_train_loss.append(avg_train_loss) # per epoch

    avg_train_accuracy = total_train_accuracy / len(train_dataloader)

    plot_accuracy_train.append(avg_train_accuracy)

    return avg_train_accuracy, avg_train_loss


def eval():
    total_eval_accuracy = 0
    total_eval_f1 = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    model.eval()

    predictions = []

    # print(len(test_dataloader))

    # Evaluate data for one epoch
    for index, sample in enumerate(test_dataloader):
        data = sample[0].long().to(device).squeeze(1)
        gt_label = sample[1].to(device)

        hidden = model.init_hidden(data.size(0))

        with torch.no_grad():
            if args.model == "LSTMClassifier":
                output, hidden = model(data, hidden)

            output = output.unsqueeze(0)
            loss = criterion(output, gt_label)

            total_eval_loss += loss.item()

            gt_label_cpu = gt_label.detach().cpu().numpy()
            output_cpu = output.detach().cpu().numpy()

            total_eval_accuracy += flat_accuracy(output_cpu, gt_label_cpu)

            # for submission file
            batch_pred = output.detach().cpu().numpy()
            # # print(batch_pred)
            batch_pred_label = np.argmax(batch_pred).flatten()
            # # print(batch_pred_label)
            predictions.append(batch_pred_label.tolist())

    predictions = np.asarray([item for sublist in predictions for item in sublist]).astype(int)
    df_subm = pd.read_csv("./data/test.csv")
    gt_label = df_subm['target']
    total_eval_f1 = f1_score(predictions, gt_label)

    # Calculate the average loss and accuracy over all of the batches.
    avg_eval_loss = total_eval_loss / len(test_dataloader)
    plot_eval_loss.append(avg_eval_loss) # per epoch

    avg_eval_accuracy = total_eval_accuracy / len(test_dataloader)
    plot_accuracy_eval.append(avg_eval_accuracy)

    return avg_eval_accuracy, total_eval_f1, avg_eval_loss

for i in range(args.epochs):
    avg_train_accuracy, avg_train_loss = train()
    avg_eval_accuracy, f1, avg_eval_loss = eval()

    print("Epoch {:2d} / {:2d}  |  Train Acc: {:1.3f}  |  Train loss: {:1.3f}  |  Test Acc: {:1.3f} |  Test F1: {:1.3f}  |  Test loss: {:1.3f}"
    .format(i + 1, args.epochs, avg_train_accuracy, avg_train_loss, avg_eval_accuracy, f1, avg_eval_loss))
    


plt.figure(1)
plt.plot(range(args.epochs), plot_train_loss, 'g', label="Training Loss")
plt.plot(range(args.epochs), plot_eval_loss, 'b', label="Test Loss")
plt.title('LSTMClassifier')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy-Loss')
plt.legend()

plt.figure(2)
plt.plot(range(args.epochs), plot_accuracy_train, 'g', label="Training F1-Score")
plt.plot(range(args.epochs), plot_accuracy_eval, 'b', label="Test F1-Score")
plt.title('LSTMClassifier')
plt.xlabel('Epochs')
plt.ylabel('F1-Score')
plt.legend()


plt.show()