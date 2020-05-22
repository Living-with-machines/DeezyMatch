#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.insert(0,'..')

import argparse
from data_processing import test_tokenize
from evaluation import test_model
import pickle
from utils import read_test_command, read_input_file
from utils import cprint, bc

import torch
# --- set seed for reproducibility
from utils import set_seed_everywhere
set_seed_everywhere(1364)

mode = 'generate_vectors'    # choices: test, generate_vectors

# --- read command args
model_path, dataset_path, train_vocab_path, input_file, test_cutoff, run_type = read_test_command()

if run_type in ["c"]:
    print("Candidates mode")
    # XXX MOVE THIS INTO INPUT FILE!!!!
    path_output_vectors = './embed_candidates/rnn'
    path_save_test_class = "./candidates.df"
elif run_type in ['q']:
    print("Queries mode")
    path_output_vectors = './embed_queries/rnn'
    path_save_test_class = "./queries.df"

if mode.lower() in ['test']:
    output_state_vectors = False
else:
    output_state_vectors = path_output_vectors


# --- read input file
dl_inputs = read_input_file(input_file)

# --- load torch model, send it to the device (CPU/GPU)
model = torch.load(model_path, map_location=dl_inputs['general']['device'])
#print(model.state_dict()['emb.weight'])

# --- create test data class
# read vocabulary
with open(train_vocab_path, 'rb') as handle:
    train_vocab = pickle.load(handle)
# create the actual class here
test_dc = test_tokenize(
    dataset_path, train_vocab,
    preproc_steps=(dl_inputs["preprocessing"]["uni2ascii"],
                   dl_inputs["preprocessing"]["lowercase"],
                   dl_inputs["preprocessing"]["strip"],
                   dl_inputs["preprocessing"]["only_latin_letters"]),
    max_seq_len=dl_inputs['gru_lstm']['max_seq_len'],
    mode=dl_inputs['gru_lstm']['mode'],
    cutoff=test_cutoff, 
    save_test_class=path_save_test_class
    )

# --- output state vectors 
test_acc, test_pre, test_rec, test_f1 = test_model(model, 
                                                   test_dc,
                                                   pooling_mode=dl_inputs['gru_lstm']['pooling_mode'],
                                                   device=dl_inputs['general']['device'],
                                                   batch_size=dl_inputs['gru_lstm']['batch_size'],
                                                   output_state_vectors=output_state_vectors, 
                                                   shuffle=False
                                                   )

cprint('[INFO]', bc.lred,
       'TEST dataset\nacc: {:.3f}; precision: {:.3f}, recall: {:.3f}, f1: {:.3f}'.format(
           test_acc, test_pre, test_rec, test_f1))
