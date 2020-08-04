#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import pandas as pd
import sys
from torch.utils.data import DataLoader
import uuid

from .data_processing import test_tokenize
from .rnn_networks import test_model
from .combineVecs import combine_vecs 

def query_vector_gen(query, model, train_vocab, dl_inputs):
    """Generate query vectors on the fly

    Args:
        query: a string or list of strings (queries)
        model: a DeezyMatch model
        train_vocab: a DeezyMatch vovabulary 
        dl_inputs: inputs, normally read from a yaml file 

    Returns:
        path to the generated temporary dir 
    """

    if isinstance(query, str):
        query = [query]

    tmp_dirname = "tmp_" + str(uuid.uuid4())

    query_input_pd = pd.DataFrame(query, columns=['s1'])
    query_input_pd['s2'] = query
    query_input_pd['label'] = "False"

    # create test class 
    test_dc = test_tokenize(
        query_input_pd, 
        train_vocab,
        preproc_steps=(dl_inputs["preprocessing"]["uni2ascii"],
                       dl_inputs["preprocessing"]["lowercase"],
                       dl_inputs["preprocessing"]["strip"],
                       dl_inputs["preprocessing"]["only_latin_letters"],
                       dl_inputs["preprocessing"]["prefix_suffix"],
                       ),
        max_seq_len=dl_inputs['gru_lstm']['max_seq_len'],
        mode=dl_inputs['gru_lstm']['mode'],
        save_test_class=os.path.join(tmp_dirname, "query", "dataframe.df"),
        dataframe_input=True,
        verbose=False
        )
    
    test_dl = DataLoader(dataset=test_dc, 
                         batch_size=dl_inputs['gru_lstm']['batch_size'], 
                         shuffle=False)
    num_batch_test = len(test_dl)

    # inference
    test_model(model, 
               test_dl,
               eval_mode='test',
               pooling_mode=dl_inputs['gru_lstm']['pooling_mode'],
               device=dl_inputs['general']['device'],
               evaluation=True,
               output_state_vectors=os.path.join(tmp_dirname, "query", f"embeddings", "rnn"), 
               output_preds=False,
               output_preds_file=False,
               csv_sep=dl_inputs['preprocessing']['csv_sep'],
               print_epoch=False
               )
     
    # combine vectors 
    combine_vecs(rnn_passes=['fwd', 'bwd'],  
                 input_scenario=os.path.join(tmp_dirname, 'query'),  
                 output_scenario=os.path.join(tmp_dirname, "combined", "query_on_fly"),  
                 print_every=10,
                 sel_device=dl_inputs["general"]["device"],
                 save_df=False) 
    
    return tmp_dirname

def candidate_conf_calc(query_candidate_pd, model, train_vocab, dl_inputs, cutoffs):
    # create test class 
    test_dc = test_tokenize(
        query_candidate_pd, 
        train_vocab,
        preproc_steps=(dl_inputs["preprocessing"]["uni2ascii"],
                       dl_inputs["preprocessing"]["lowercase"],
                       dl_inputs["preprocessing"]["strip"],
                       dl_inputs["preprocessing"]["only_latin_letters"],
                       dl_inputs["preprocessing"]["prefix_suffix"],
                       ),
        max_seq_len=dl_inputs['gru_lstm']['max_seq_len'],
        mode=dl_inputs['gru_lstm']['mode'],
        cutoff=cutoffs,
        save_test_class=False,
        dataframe_input=True,
        verbose=False
        )
    
    test_dl = DataLoader(dataset=test_dc, 
                        batch_size=dl_inputs['gru_lstm']['batch_size'], 
                        shuffle=False)
    num_batch_test = len(test_dl)
    
    # inference
    all_preds = test_model(model, 
                           test_dl,
                           eval_mode='test',
                           pooling_mode=dl_inputs['gru_lstm']['pooling_mode'],
                           device=dl_inputs['general']['device'],
                           evaluation=True,
                           output_state_vectors=False, 
                           output_preds=True,
                           output_preds_file=False,
                           csv_sep=dl_inputs['preprocessing']['csv_sep'],
                           print_epoch=False
                           )

    if len(all_preds) != len(query_candidate_pd):
        print(f"[ERROR] lengths of all predictions ({len(all_preds)}) and processed data ({len(query_candidate_pd)}) are not the same!")
        sys.exit("[ERROR] This should not happen! Contact developers.")

    return all_preds
