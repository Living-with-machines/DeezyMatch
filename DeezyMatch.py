#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
DeezyMatch main code: select the relevant module (train, finetune, inference, combine_vec, candidate_finder) 
based on the inputs.
"""

from datetime import datetime
import os
import pickle
import shutil
import sys

from candidateFinder import candidate_finder
from candidateFinder import main as candidate_finder_main
from combineVecs import combine_vecs
from combineVecs import main as combine_vecs_main
from data_processing import csv_split_tokenize
from rnn_networks import gru_lstm_network, fine_tuning
from rnn_networks import inference as rnn_inference
from utils import deezy_mode_detector
from utils import read_inputs_command, read_inference_command, read_input_file
from utils import cprint, bc, log_message
# --- set seed for reproducibility
from utils import set_seed_everywhere
set_seed_everywhere(1364)

# ------------------- train --------------------
def train(input_file_path=None, dataset_path=None, model_name=None, 
          n_train_examples=None, run_command_line=False):
    """
    Train a new DeezyMatch model

    Parameters
    ----------
    input_file_path
        path of the input file
    dataset_path
        path of the dataset
    model_name
        name of the model to be saved
    n_train_examples
        number of training examples to be used (optional)
    run_command_line
        False if train is imported as a module
    """

    # --- read input file
    dl_inputs = read_input_file(input_file_path)
    
    # --- log current directory and command line
    cur_dir = os.path.abspath(os.path.curdir)
    if run_command_line:
        input_command_line = f"python"
        for one_arg in sys.argv:
            input_command_line += f" {one_arg}"
    
    # --- read dataset and split into train/val/test sets
    train_prop = dl_inputs['gru_lstm']['train_proportion']
    val_prop = dl_inputs['gru_lstm']['val_proportion']
    test_prop = dl_inputs['gru_lstm']['test_proportion']
    # tokenize and create data classes
    train_dc, valid_dc, test_dc, dataset_vocab = csv_split_tokenize(
        dataset_path=dataset_path, 
        n_train_examples=n_train_examples, 
        train_prop=train_prop, 
        val_prop=val_prop, 
        test_prop=test_prop,
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
    if run_command_line:
        log_message(input_command_line + "\n", mode="a", filename=os.path.join(os.path.dirname(vocab_path), "log.txt"))
    
    # --- train a model from scratch
    gru_lstm_network(dl_inputs=dl_inputs, 
                     model_name=model_name, 
                     train_dc=train_dc, 
                     valid_dc=valid_dc, 
                     test_dc=test_dc)

# ------------------- finetune --------------------
def finetune(input_file_path=None, dataset_path=None, model_name=None, 
             pretrained_model_path=None, pretrained_vocab_path=None, 
             n_train_examples=None, run_command_line=False):
    """
    Fine-tune an already trained model 

    Parameters
    ----------
    input_file_path
        path of the input file
    dataset_path
        path of the dataset
    model_name
        name of the model to be saved
    pretrained_model_path
        Path to the pretrained model
    pretrained_vocab_path
        Path to the pretrained vocabulary
    n_train_examples
        number of training examples to be used (optional)
    run_command_line
        False if train is imported as a module
    """

    # --- read input file
    dl_inputs = read_input_file(input_file_path)
    
    # --- log current directory and command line
    cur_dir = os.path.abspath(os.path.curdir)
    if run_command_line:
        input_command_line = f"python"
        for one_arg in sys.argv:
            input_command_line += f" {one_arg}"
    
    if os.path.isdir(pretrained_model_path):
        pt_model_name = os.path.basename(os.path.abspath(pretrained_model_path))
        pretrained_vocab_path = os.path.join(pretrained_model_path, f"{pt_model_name}.vocab")
        pretrained_model_path = os.path.join(pretrained_model_path, f"{pt_model_name}.model")
    
    # --- read dataset and split into train/val/test sets
    train_prop = dl_inputs['gru_lstm']['train_proportion']
    val_prop = dl_inputs['gru_lstm']['val_proportion']
    test_prop = dl_inputs['gru_lstm']['test_proportion']
    # tokenize and create data classes
    train_dc, valid_dc, test_dc, dataset_vocab = csv_split_tokenize(
        dataset_path=dataset_path, 
        pretrained_vocab_path=pretrained_vocab_path, 
        n_train_examples=n_train_examples, 
        missing_char_threshold=dl_inputs["preprocessing"]["missing_char_threshold"], 
        train_prop=train_prop, 
        val_prop=val_prop, 
        test_prop=test_prop,
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
    if run_command_line:
        log_message(input_command_line + "\n", mode="a", filename=os.path.join(os.path.dirname(vocab_path), "log.txt"))
    
    # --- Fine-tune a pretrained model
    fine_tuning(pretrained_model_path=pretrained_model_path,
                dl_inputs=dl_inputs, 
                model_name=model_name, 
                train_dc=train_dc, 
                valid_dc=valid_dc, 
                test_dc=test_dc)

# ------------------- inference --------------------
def inference(input_file_path=None, dataset_path=None, 
              pretrained_model_path=None, pretrained_vocab_path=None,  
              cutoff=None, inference_mode="test", query_candidate_mode=["q", "c"], scenario=None):
    """
    Use an already trained model for inference/prediction

    Parameters
    ----------
    input_file_path
        path of the input file
    dataset_path
        path of the dataset
    pretrained_model_path
        Path to the pretrained model
    pretrained_vocab_path
        Path to the pretrained vocabulary
    cutoff
        number of examples to be used (optional)
    inference_mode
        Three options: test (inference), vect (generate vector representations)
    query_candidate_mode
        Two modes: q (generate query vectors), c (generate candidate vectors)
    scenario
        Name of the experiment/top-directory
    """

    # --- read input file
    dl_inputs = read_input_file(input_file_path)

    if not type(query_candidate_mode) in [list, tuple]:
        query_candidate_mode = [query_candidate_mode]

    if inference_mode in ["test"]:
        # query_candidate_mode is not used:
        query_candidate_mode = ["q"]

    for qc_mode in query_candidate_mode:
        # --- Inference / generate vector representations
        rnn_inference(model_path=pretrained_model_path, 
                      dataset_path=dataset_path, 
                      train_vocab_path=pretrained_vocab_path, 
                      input_file_path=input_file_path, 
                      test_cutoff=cutoff, 
                      inference_mode=inference_mode, 
                      query_candidate_mode=qc_mode, 
                      scenario=scenario, 
                      dl_inputs=dl_inputs)

# ------------------- main --------------------
def main():
    """
    Initiate main when DeezyMatch is called from the command line
    """
    # detect DeezyMatch mode
    dm_mode = deezy_mode_detector()
    
    if dm_mode in ["train", "finetune"]:
        # --- read command args
        input_file_path, dataset_path, model_name, pretrained_model_path, pretrained_vocab_path, n_train_examples = \
            read_inputs_command()
        
        if pretrained_model_path:
            finetune(input_file_path=input_file_path, 
                     dataset_path=dataset_path, 
                     model_name=model_name, 
                     pretrained_model_path=pretrained_model_path, 
                     pretrained_vocab_path=pretrained_vocab_path, 
                     n_train_examples=n_train_examples,
                     run_command_line=True)
        else:
            train(input_file_path=input_file_path, 
                  dataset_path=dataset_path, 
                  model_name=model_name, 
                  n_train_examples=n_train_examples, 
                  run_command_line=True)

    elif dm_mode in ["inference"]:
        # --- read command args for inference
        model_path, dataset_path, train_vocab_path, input_file_path, test_cutoff, \
        inference_mode, query_candidate_mode, scenario = \
            read_inference_command()

        inference(input_file_path=input_file_path, 
                  dataset_path=dataset_path, 
                  pretrained_model_path=model_path, 
                  pretrained_vocab_path=train_vocab_path, 
                  cutoff=test_cutoff, 
                  inference_mode=inference_mode, 
                  query_candidate_mode=query_candidate_mode, 
                  scenario=scenario)
    
    elif dm_mode in ["combine_vecs"]:
        combine_vecs_main()

    elif dm_mode in ["candidate_finder"]:
        candidate_finder_main()

if __name__ == '__main__':
    main()
