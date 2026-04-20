import argparse
import time
import math
import numpy as np
from numpy.lib.format import open_memmap
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

def evaluate(data_source, batch_size=10, name="validation"):
    # Turn on evaluation mode which disables dropout.
    total_tokens = data_source.size(0) - 1
    # Pre-allocate memmap files on disk (avoids RAM spikes)
    prob_path = os.path.join(results_dir, f'{name}_full_prob.npy')
    target_path = os.path.join(results_dir, f'{name}_targets.npy')
    # 'w+' creates/overwrites the file
    full_probs_mmap = open_memmap(prob_path, mode='w+', dtype='float32', shape=(total_tokens, ntokens))
    targets_mmap = open_memmap(target_path, mode='w+', dtype='int64', shape=(total_tokens,))
    model.eval()

    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)

    prior_total = 0
    pointer = 0
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        targets = targets.view(-1)
        output, hidden, rnn_outs, _, prior = model(data, hidden, return_h=True)
        # Loss calculation
        loss = nn.functional.nll_loss(output.view(-1, ntokens), targets.view(-1)).data
        total_loss += loss * len(data)
        hidden = repackage_hidden(hidden)

        # Write slice to disk
        batch_logits = output.view(-1, ntokens).data.cpu().numpy()
        batch_targets = targets.view(-1).data.cpu().numpy()
        
        num_elements = batch_logits.shape[0]
        full_probs_mmap[pointer : pointer + num_elements] = batch_logits
        targets_mmap[pointer : pointer + num_elements] = batch_targets
        pointer += num_elements
    full_probs_mmap.flush()
    targets_mmap.flush()

    return total_loss.item() / len(data_source), full_probs_mmap, targets_mmap

if args.save.endswith('.pt'):
    model_path = args.save
    model_name = os.path.basename(model_path).strip('.pt')
    os.makedirs(os.path.join(os.path.dirname(model_path), model_name), exist_ok=True)
    results_dir = os.path.join(os.path.dirname(model_path), model_name, "results")
else:
    model_path = os.path.join(args.save, 'model.pt')
    model_name = os.path.basename(args.save)
    results_dir = os.path.join(args.save, "results")


os.makedirs(results_dir, exist_ok=True)

# Load the best saved model.
with open(model_path, 'rb') as f:
    if not args.cuda:
        model = torch.load(f, map_location=lambda storage, loc: storage)
    else:
        model = torch.load(f)
print(model)

if args.cuda:
    print("Using CUDA")
    model.cuda()
else:
    print("Using CPU")
    
# def save_memmap(filename, x):
#     if isinstance(x, torch.Tensor):
#         x = x.cpu().numpy()
#     # Save the array to a .npy file using memory mapping
#     memmap_array = open_memmap(filename, mode='w+', dtype=x.dtype, shape=x.shape)
#     memmap_array[:] = x[:]
#     del memmap_array  # Flush changes to disk
# Run on val data.
val_loss, val_full_logits, val_targets = evaluate(val_data, test_batch_size)
# print(val_full_logits.shape)
# save_memmap(os.path.join(results_dir, 'validation_full_prob.npy'), val_full_logits)
# save_memmap(os.path.join(results_dir, 'validation_targets.npy'), val_targets)
# save_memmap(os.path.join(results_dir, 'validation_prob.npy'), val_full_logits[np.arange(len(val_targets)), val_targets])

print('=' * 89)
print('| End of pointer | val loss {:5.2f} | val ppl {:8.2f}'.format(
    val_loss, math.exp(val_loss)))
print('=' * 89)

# Run on test data.
test_loss, test_full_logits, test_targets = evaluate(test_data, test_batch_size)
print(test_full_logits.shape)
save_memmap(os.path.join(results_dir, 'test_full_prob.npy'), test_full_logits)
save_memmap(os.path.join(results_dir, 'test_targets.npy'), test_targets)
save_memmap(os.path.join(results_dir, 'test_prob.npy'), test_full_logits[np.arange(len(test_targets)), test_targets])
print('=' * 89)
print('| End of pointer | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
