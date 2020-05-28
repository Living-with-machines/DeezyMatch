#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from utils import read_inputs_command, read_input_file
from utils import cprint, bc

from rnn_networks import gru_lstm_network
from data_processing import csv_split_tokenize

import pickle,os,gensim,torch
# --- set seed for reproducibility
from utils import set_seed_everywhere
set_seed_everywhere(1364)

# only if you run on this on a mac
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ===== DeezyMatch

# --- read command args
input_file_path, dataset_path, model_name, pretrained_model_path,pretrained_model_vocab_path,n_train_examples,pretrained_embs = read_inputs_command()

# --- read input file
dl_inputs = read_input_file(input_file_path)

# we are not using it yet
#if pretrained_embs:
#    model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_embs)
#    pretrained_embs_weights = torch.FloatTensor(model.vectors)
#    pretrained_embs_vocab = model.vocab
#else:
#    pretrained_embs_weights = None
#    pretrained_embs_vocab = None


# --- various!!! methods for Fuzzy String Matching
if dl_inputs['gru_lstm']['training'] or dl_inputs['gru_lstm']['evaluation']:
        
    # --- read dataset and split into train/val/test sets
    train_prop = dl_inputs['gru_lstm']['train_proportion']
    val_prop = dl_inputs['gru_lstm']['val_proportion']
    test_prop = dl_inputs['gru_lstm']['test_proportion']
    train_dc, valid_dc, test_dc, dataset_vocab = csv_split_tokenize(
        dataset_path, n_train_examples,train_prop, val_prop, test_prop,
        preproc_steps=(dl_inputs["preprocessing"]["uni2ascii"],
                       dl_inputs["preprocessing"]["lowercase"],
                       dl_inputs["preprocessing"]["strip"],
                       dl_inputs["preprocessing"]["only_latin_letters"]),
        max_seq_len=dl_inputs['gru_lstm']['max_seq_len'],
        mode=dl_inputs['gru_lstm']['mode'])
    
    # --- store vocab
    vocab_path = os.path.join('vocabs', model_name + '.pickle')
    if not os.path.isdir("vocabs"):
        os.makedirs("vocabs")
    with open(vocab_path, 'wb') as handle:
        pickle.dump(dataset_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # if we have a pretrained model to fine-tune
    if pretrained_model_path:
        fine_tuning(pretrained_model_path=pretrained_model_path,dl_inputs=dl_inputs, model_name=model_name, train_dc=train_dc, valid_dc=valid_dc, test_dc=test_dc)
    else:
    # we train the bidirectional_gru from scratch
        gru_lstm_network(dl_inputs=dl_inputs, model_name=model_name, train_dc=train_dc, valid_dc=valid_dc, test_dc=test_dc)
