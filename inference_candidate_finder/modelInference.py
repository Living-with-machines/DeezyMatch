#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Add parent path so we can import modules
import sys
sys.path.insert(0,'..')

import argparse
from datetime import datetime
from data_processing import test_tokenize
from evaluation import test_model
import os
import pickle
import shutil
import time

from utils import read_inference_command, read_input_file
from utils import cprint, bc, log_message

import torch

# --- set seed for reproducibility
from utils import set_seed_everywhere
set_seed_everywhere(1364)

# ==== Model inference
start_time = time.time()

# --- read command args
model_path, dataset_path, train_vocab_path, input_file, test_cutoff, inference_mode, query_candidate_mode, scenario = \
    read_inference_command()
if type(test_cutoff) == int:
    test_cutoff = int(test_cutoff)

# --- read input file
dl_inputs = read_input_file(input_file)

if inference_mode in ['test']:
    output_state_vectors = False
    path_save_test_class = False
else:
    scenario_path = ""
    if query_candidate_mode in ["c"]:
        scenario_path = "./candidates/" + scenario + "/"
        if not os.path.isdir(os.path.dirname(scenario_path)):
            os.makedirs(os.path.dirname(scenario_path))
        output_state_vectors = scenario_path + "embed_candidates/rnn"
        path_save_test_class = scenario_path + "candidates.df"
        parent_dir = os.path.abspath(os.path.join(output_state_vectors, os.pardir))
        if os.path.isdir(parent_dir):
            shutil.rmtree(parent_dir)
        if os.path.isfile(path_save_test_class):
            os.remove(path_save_test_class)
    elif query_candidate_mode in ["q"]:
        scenario_path = "./queries/" + scenario + "/"
        if not os.path.isdir(os.path.dirname(scenario_path)):
            os.makedirs(os.path.dirname(scenario_path))
        output_state_vectors = scenario_path + "embed_queries/rnn"
        path_save_test_class = scenario_path + "queries.df"
        parent_dir = os.path.abspath(os.path.join(output_state_vectors, os.pardir))
        if os.path.isdir(parent_dir):
            shutil.rmtree(parent_dir)
        if os.path.isfile(path_save_test_class):
            os.remove(path_save_test_class)
    shutil.copy2(input_file, os.path.dirname(scenario_path))
    msg = datetime.now().strftime("%m/%d/%Y_%H:%M:%S")
    cur_dir = os.path.abspath(os.path.curdir)
    input_command_line = f"python"
    for one_arg in sys.argv:
        input_command_line += f" {one_arg}"
    msg += "\nCurrent directory: " + cur_dir + "\n"
    log_message(msg, mode="w", filename=os.path.join(os.path.dirname(scenario_path), "log.txt"))
    log_message(input_command_line + "\n", mode="a", filename=os.path.join(os.path.dirname(scenario_path), "log.txt"))

# --- load torch model, send it to the device (CPU/GPU)
model = torch.load(model_path, map_location=dl_inputs['general']['device'])
#print(model.state_dict()['emb.weight'])

# --- create test data class
# read vocabulary
with open(train_vocab_path, 'rb') as handle:
    train_vocab = pickle.load(handle)

# create the actual class here
test_dc = test_tokenize(
    dataset_path, train_vocab,dl_inputs["preprocessing"]["missing_char_threshold"],
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
                                                   shuffle=False,
                                                   evaluation=True
                                                   )

cprint('[INFO]', bc.lred,
       'TEST dataset\nacc: {:.3f}; precision: {:.3f}, recall: {:.3f}, f1: {:.3f}'.format(
           test_acc, test_pre, test_rec, test_f1))
print("--- %s seconds ---" % (time.time() - start_time))
