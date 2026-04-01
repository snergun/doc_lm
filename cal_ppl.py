import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model
import os
from utils import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn',
                    help='location of the data corpus')
parser.add_argument('--save', type=str,default='best.pt',
                    help='model to use the pointer over')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--bptt', type=int, default=5000,
                    help='sequence length')
args = parser.parse_args()

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

eval_batch_size = 1
test_batch_size = 1
train_data = batchify(corpus.train, test_batch_size, args)
val_data = batchify(corpus.valid, test_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
criterion = nn.CrossEntropyLoss()

def one_hot(idx, size, cuda=True):
    a = np.zeros((1, size), np.float32)
    a[0][idx] = 1
    v = Variable(torch.from_numpy(a))
    if cuda: v = v.cuda()
    return v

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    matrix_list = []
    targets_list = []
    prior_total = 0
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        targets = targets.view(-1)
        output, hidden, rnn_outs, _, prior = model(data, hidden, return_h=True)
        loss = nn.functional.nll_loss(output.view(-1, ntokens), targets).data
        total_loss += loss * len(data)
        hidden = repackage_hidden(hidden)
        prior_total += prior.sum(0).data.cpu().numpy()
        output_numpy = output.view(-1, ntokens).data.cpu().numpy()
        matrix_list.append(output_numpy)
        targets_list.append(targets.data.cpu().numpy())
    matrix = np.concatenate(matrix_list)
    targets = np.concatenate(targets_list)
    return total_loss.item() / len(data_source), matrix, targets


# Load the best saved model.
with open(args.save, 'rb') as f:
    if not args.cuda:
        model = torch.load(f, map_location=lambda storage, loc: storage)
    else:
        model = torch.load(f)
print(model)

# Run on val data.
save_dir = os.path.dirname(args.save)
model_name = os.path.basename(args.save).strip('.pt')
results_dir = os.path.join(save_dir, model_name + '_results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

val_loss, val_full_logits, val_targets = evaluate(val_data, test_batch_size)
print(val_full_logits.shape)
np.save(os.path.join(results_dir, 'validation_full_prob.npy'), val_full_logits)
np.save(os.path.join(results_dir, 'validation_targets.npy'), val_targets)

print('=' * 89)
print('| End of pointer | val loss {:5.2f} | val ppl {:8.2f}'.format(
    val_loss, math.exp(val_loss)))
print('=' * 89)

# Run on test data.
test_loss, test_full_logits, test_targets = evaluate(test_data, test_batch_size)
print(test_full_logits.shape)
np.save(os.path.join(results_dir, 'test_full_prob.npy'), test_full_logits)
np.save(os.path.join(results_dir, 'test_targets.npy'), test_targets)
print('=' * 89)
print('| End of pointer | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
