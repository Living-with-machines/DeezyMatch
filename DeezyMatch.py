#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
DeezyMatch main code
"""

import pickle, os

from datetime import datetime
from data_processing import csv_split_tokenize
from rnn_networks import gru_lstm_network, fine_tuning
import shutil
import sys
from utils import read_inputs_command, read_input_file
from utils import cprint, bc, log_message
# --- set seed for reproducibility
from utils import set_seed_everywhere
set_seed_everywhere(1364)

# ===== DeezyMatch
# --- read command args
input_file_path, dataset_path, model_name, pretrained_model_path, pretrained_vocab_path, n_train_examples = \
    read_inputs_command()
# --- read input file
dl_inputs = read_input_file(input_file_path)

# --- log current directory and command line
cur_dir = os.path.abspath(os.path.curdir)
input_command_line = f"python"
for one_arg in sys.argv:
    input_command_line += f" {one_arg}"

# --- Methods for Fuzzy String Matching
if dl_inputs['gru_lstm']['training'] or dl_inputs['gru_lstm']['validation']:
    
    # --- read dataset and split into train/val/test sets
    train_prop = dl_inputs['gru_lstm']['train_proportion']
    val_prop = dl_inputs['gru_lstm']['val_proportion']
    test_prop = dl_inputs['gru_lstm']['test_proportion']
    train_dc, valid_dc, test_dc, dataset_vocab = csv_split_tokenize(
        dataset_path, pretrained_vocab_path, n_train_examples, dl_inputs["preprocessing"]["missing_char_threshold"], train_prop, val_prop, test_prop,
        preproc_steps=(dl_inputs["preprocessing"]["uni2ascii"],
                       dl_inputs["preprocessing"]["lowercase"],
                       dl_inputs["preprocessing"]["strip"],
                       dl_inputs["preprocessing"]["only_latin_letters"]),
        max_seq_len=dl_inputs['gru_lstm']['max_seq_len'],
        mode=dl_inputs['gru_lstm']['mode'],
        read_list_chars=dl_inputs['preprocessing']['read_list_chars'],
        csv_sep=dl_inputs['preprocessing']["csv_sep"]
        )
    
    # --- store vocab
    vocab_path = os.path.join(dl_inputs["general"]["models_dir"], 
                              model_name, model_name + '.vocab')
    vocab_path = os.path.abspath(vocab_path)
    if not os.path.isdir(os.path.dirname(vocab_path)):
        os.makedirs(os.path.dirname(vocab_path))
    with open(vocab_path, 'wb') as handle:
        pickle.dump(dataset_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.copy2(input_file_path, os.path.dirname(vocab_path))

    msg = datetime.now().strftime("%m/%d/%Y_%H:%M:%S")
    msg += "\nCurrent directory: " + cur_dir + "\n"
    log_message(msg, mode="w", filename=os.path.join(os.path.dirname(vocab_path), "log.txt"))
    log_message(input_command_line + "\n", mode="a", filename=os.path.join(os.path.dirname(vocab_path), "log.txt"))
    
    if pretrained_model_path:
        # Fine-tune a pretrained model
        fine_tuning(pretrained_model_path=pretrained_model_path,
                    dl_inputs=dl_inputs, 
                    model_name=model_name, 
                    train_dc=train_dc, valid_dc=valid_dc, test_dc=test_dc)
    else:
        # train a bidirectional_gru from scratch
        gru_lstm_network(dl_inputs=dl_inputs, 
                         model_name=model_name, 
                         train_dc=train_dc, valid_dc=valid_dc, test_dc=test_dc)
