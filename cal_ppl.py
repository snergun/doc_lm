import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import RNNModel
import data
import model

from utils import batchify, get_batch, repackage_hidden

def fix_rnn_buffers(model):
    for m in model.modules():
        if isinstance(m, torch.nn.LSTM):
            print('Fixing LSTM buffers for module:', m)
            # These are the internal buffers that changed between PyTorch 0.2 and 1.x
            stale_attrs = ['_flat_weights', '_flat_weights_names', '_all_weights']
            for attr in stale_attrs:
                if hasattr(m, attr):
                    try:
                        delattr(m, attr)
                    except:
                        pass
            # Force the modern PyTorch engine to rebuild them correctly
            m.flatten_parameters()

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')   
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
    fix_rnn_buffers(model)
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    matrix_list = []
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
    matrix = np.concatenate(matrix_list)
    return total_loss[0] / len(data_source)


# Load the best saved model.
with open(args.save, 'rb') as f:
    if not args.cuda:
        checkpoint = torch.load(f, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(f)

def get_val(obj, attr, default):
    return getattr(obj, attr, default)

print("--- Blueprint Extraction ---")
ntoken = get_val(checkpoint, 'ntoken', 10000)
ninp = checkpoint.encoder.embedding_dim
nhid = checkpoint.rnns[0].module.hidden_size
nhidlast = checkpoint.rnns[-1].module.hidden_size
nlayers = len(checkpoint.rnns)

# Grab dropouts safely from the old object
# If 'lockdrop.dropout' fails, try common alternatives or use defaults
drop_rate = getattr(checkpoint.lockdrop, 'dropout', getattr(checkpoint.lockdrop, 'p', 0.4))
wdrop_rate = getattr(checkpoint.rnns[0], 'dropout', 0.5)
n_experts_true = 20  # Error said checkpoint has 20, replica had 15
num4second_true = 5  # Common default for DOC when weight4second is present

print(f"Arch: {nlayers} layers, {nhid} hidden, {ninp} embed")
print(f"Correcting Experts: {n_experts_true} experts")

model = RNNModel(
    args.model, 10000, 280, 960, 620, 3, 
    dropout=0.4, dropouth=0.225, dropouti=0.4, dropoute=0.1, 
    wdrop=0.5, n_experts=20, num4second=5
)

# 4. Filter the state_dict to remove internal "flat" buffers
# This prevents the old _flat_weights from ever touching your new model
clean_sd = {k: v for k, v in checkpoint.state_dict().items() if '_flat_weights' not in k}

# Run on val data.
val_loss = evaluate(val_data, test_batch_size)
print('=' * 89)
print('| End of pointer | val loss {:5.2f} | val ppl {:8.2f}'.format(
    val_loss, math.exp(val_loss)))
print('=' * 89)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of pointer | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
