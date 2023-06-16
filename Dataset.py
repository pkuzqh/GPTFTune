#written by Qihao Zhu
import sys
import torch
import torch.utils.data as data
import random
import pickle
import os
import numpy as np
import re
from tqdm import tqdm
import json
from copy import deepcopy
from transformers import AutoTokenizer
import math
from tqdm import tqdm
import torch
from torch.utils.data import Sampler
tokenizer = None
args = {}
voc = {}

def pad_seq(seq, maxlen):
    act_len = len(seq)
    if len(seq) < maxlen:
        seq = seq + [args.mask_id] * maxlen
        seq = seq[:maxlen]
    else:
        seq = seq[:maxlen]
        act_len = maxlen
    return seq
def pad_seq2(seq, maxlen):
    act_len = len(seq)
    if len(seq) < maxlen:
        seq = seq + [0] * maxlen
        seq = seq[:maxlen]
    else:
        seq = seq[:maxlen]
        act_len = maxlen
    return seq

def pad_list(seq, maxlen1, maxlen2):
    if len(seq) < maxlen1:
        seq = seq + [[args.mask_id] * maxlen2] * maxlen1
        seq = seq[:maxlen1]
    else:
        seq = seq[:maxlen1]
    return seq

class ChunkedRandomSampler(Sampler):

    def __init__(self, data_source, batch_size):
      self.data_source = data_source
      self.batch_size = batch_size

    def __iter__(self):
      lst = list(range(len(self.data_source)))
      chunked = [lst[i:i+self.batch_size] for i in range(0, len(self.data_source), self.batch_size)]
      random.shuffle(chunked)
      new_lst = [e for piece in chunked for e in piece]
      return iter(new_lst)

    def __len__(self):
      return len(self.data_source)
def pad_and_get_mask(nl, code, max_length):
    global tokenizer
    while (len(code) + len(nl) + 2 > max_length):
        if (len(code) > len(nl)):
            code = code[:-1]
        else:
            nl = nl[:-1]
    inputs = nl + [tokenizer.bos_token_id] + code + [tokenizer.eos_token_id]
    labels = [1] * len(nl) + [2] * (len(code)+1) + [0]
    assert len(inputs) <= max_length
    pad_len = max_length - len(inputs)
    inputs += [tokenizer.pad_token_id] * pad_len
    labels += [0] * pad_len
    assert len(inputs) == len(labels)
    return inputs, labels
def rs_collate_fn1(batch):
    global tokenizer
    rbatch = {}
    binput = []
    btesti = []
    bmask = []

    maxnllen1 = 0
    maxcodelen1 = 0
    for k in (range(len(batch))):
        inputnl = batch[k]['input']
        inputres = batch[k]['output']
        binput.append(inputnl + inputres)
        btesti.append(batch[k]['input'] + [tokenizer.bos_token_id])
        maxnllen1 = max(maxnllen1, len(binput[-1]) + 2)
        maxcodelen1 = max(maxcodelen1, len(btesti[-1]))
    maxnllen1 = min(maxnllen1, args.NlLen)
    maxcodelen1 = min(maxcodelen1, args.NlLen)
    for i in range(len(binput)):
        inputs, labels = pad_and_get_mask(batch[i]['input'], batch[i]['output'], maxnllen1)
        bmask.append(labels)
        binput[i] = inputs
        btesti[i] = pad_seq(btesti[i], maxcodelen1)

    rbatch['input'] = torch.tensor(binput)
    rbatch['mask'] = torch.tensor(bmask)
    rbatch['testi'] = torch.tensor(btesti)
    return rbatch
class SumDataset(data.Dataset):
    def __init__(self, config, code_voc=None, dataName="train", idx=-1, mode='mask', tokenizer1=None):
        global args
        global voc
        global tokenizer
        tokenizer = tokenizer1
        args = config
        tokenizer.pad_token_id = tokenizer.eos_token_id
        config.mask_id = tokenizer.pad_token_id
        config.bertnum = len(tokenizer.get_vocab())
        args.mask_id = tokenizer.pad_token_id
        if dataName == "train":
            self.data = pickle.load(open('fttrain%d.pkl'%idx, "rb"))
            self.data.sort(key=lambda x: len(x['input']) + len(x['output']), reverse=True)
        elif dataName == "valid":
            self.data = pickle.load(open('ftvalid%d.pkl'%idx, 'rb'))
        else:
            self.data = pickle.load(open('fttest%d.pkl'%idx, 'rb'))

    def __getitem__(self, offset):
        return self.data[offset]
    def __len__(self):
        return len(self.data)
