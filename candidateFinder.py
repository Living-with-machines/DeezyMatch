#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Add parent path so we can import modules
import sys
sys.path.insert(0,'..')

from argparse import ArgumentParser
from collections import OrderedDict
import faiss
import glob
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import time

import torch
from torch.utils.data import DataLoader

from data_processing import test_tokenize
from rnn_networks import test_model
from utils import read_input_file
from utils import read_command_candidate_finder
# --- set seed for reproducibility
from utils import set_seed_everywhere
set_seed_everywhere(1364)

# skip future warnings for now XXX
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ===== candidateFinder main code
start_time = time.time()

output_filename, selection_threshold, ranking_metric, search_size, num_candidates, \
    par_dir, input_file_path, number_test_rows, model_path, vocab_path = \
    read_command_candidate_finder()

if input_file_path in ["default"]:
    detect_input_files = glob.iglob(os.path.join(par_dir, "*.yaml"))
    for detected_inp in detect_input_files:
        if os.path.isfile(detected_inp):
            input_file_path = detected_inp
            break

# read input file
dl_inputs = read_input_file(input_file_path)

# ----- COMBINE VECTORS
# ----- CANDIDATES
path1_combined = os.path.join(par_dir, "candidates_fwd.pt")
path2_combined = os.path.join(par_dir, "candidates_bwd.pt")
path_id_combined = os.path.join(par_dir, "candidates_fwd_id.pt")
path_items_combined = os.path.join(par_dir, "candidates_fwd_items.npy")

vecs_ids_candidates = torch.load(path_id_combined, map_location=dl_inputs['general']['device'])
vecs_items_candidates = np.load(path_items_combined, allow_pickle=True)
vecs1_candidates = torch.load(path1_combined, map_location=dl_inputs['general']['device'])
vecs2_candidates = torch.load(path2_combined, map_location=dl_inputs['general']['device'])
vecs_candidates = torch.cat([vecs1_candidates, vecs2_candidates], dim=1)

# ----- QUERIES
path1_combined = os.path.join(par_dir, "queries_fwd.pt")
path2_combined = os.path.join(par_dir, "queries_bwd.pt")
path_id_combined = os.path.join(par_dir, "queries_fwd_id.pt")
path_items_combined = os.path.join(par_dir, "queries_fwd_items.npy")

vecs_ids_query = torch.load(path_id_combined, map_location=dl_inputs['general']['device'])
vecs_items_query = np.load(path_items_combined, allow_pickle=True)
vecs1_query = torch.load(path1_combined, map_location=dl_inputs['general']['device'])
vecs2_query = torch.load(path2_combined, map_location=dl_inputs['general']['device'])
vecs_query = torch.cat([vecs1_query, vecs2_query], dim=1)
# ----- END COMBINED VECTORS

# --- start FAISS
faiss_id_candis = faiss.IndexFlatL2(vecs_candidates.size()[1])   # build the index
print("Is faiss_id_candis already trained? %s" % faiss_id_candis.is_trained)
faiss_id_candis.add(vecs_candidates.detach().cpu().numpy())

if number_test_rows > 0:
    len_vecs_query = number_test_rows
else:
    len_vecs_query = len(vecs_query)

if not model_path in [False, None]:
    # --- load torch model, send it to the device (CPU/GPU)
    model = torch.load(model_path, map_location=dl_inputs['general']['device'])
    # --- create test data class
    # read vocabulary
    with open(vocab_path, 'rb') as handle:
        train_vocab = pickle.load(handle)

# Empty dataframe to collect data
output_pd = pd.DataFrame()
for iq in range(len_vecs_query):
    print("=========== Start the search for %s" % iq, vecs_items_query[iq])
    collect_neigh_pd = pd.DataFrame()
    num_found_candidates = 0
    # start with 0:seach_size
    # If the number of selected candidates < num_candidates
    # Increase the search size
    id_0_neigh = 0
    id_1_neigh = search_size
    while (num_found_candidates < num_candidates):
        if id_1_neigh > len(vecs_candidates):
            id_1_neigh = len(vecs_candidates)
        if id_0_neigh == id_1_neigh:
            break

        found_neighbours = faiss_id_candis.search(vecs_query[iq:(iq+1)].detach().cpu().numpy(), id_1_neigh)
    
        # Candidates
        orig_id_candis = found_neighbours[1][0, id_0_neigh:id_1_neigh]
        all_candidates = vecs_items_candidates[orig_id_candis]
    
        # Queries
        orig_id_queries = vecs_ids_query[iq].item()
        all_queries = [vecs_items_query[orig_id_queries]]*(id_1_neigh - id_0_neigh)

        query_candidate_pd = pd.DataFrame(all_queries, columns=['s1'])
        query_candidate_pd['s2'] = all_candidates
        query_candidate_pd['label'] = "False"

        # Compute cosine similarity
        cosine_sim = cosine_similarity(vecs_query[iq:(iq+1)].detach().cpu().numpy(), 
                                       vecs_candidates.detach().cpu().numpy()[orig_id_candis])

        if not model_path in [False, None]:
            # create test class 
            test_dc = test_tokenize(
                query_candidate_pd, 
                train_vocab,
                preproc_steps=(dl_inputs["preprocessing"]["uni2ascii"],
                               dl_inputs["preprocessing"]["lowercase"],
                               dl_inputs["preprocessing"]["strip"],
                               dl_inputs["preprocessing"]["only_latin_letters"]),
                max_seq_len=dl_inputs['gru_lstm']['max_seq_len'],
                mode=dl_inputs['gru_lstm']['mode'],
                cutoff=(id_1_neigh - id_0_neigh),
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
                                   output_preds_file="./tmp.txt",
                                   csv_sep=dl_inputs['preprocessing']['csv_sep']
                                   )
            if len(all_queries) != len(query_candidate_pd):
                print(f"[ERROR] lengths of all queries ({len(all_queries)}) and processed data ({len(query_candidate_pd)}) are not the same!")
                sys.exit("[ERROR] This should not happen! Contact developers.")

            all_preds = torch.exp(all_preds)
            query_candidate_pd['dl_match'] = all_preds.detach().cpu().numpy()

        else:
            query_candidate_pd['dl_match'] = [None]*len(query_candidate_pd)


        query_candidate_pd['faiss_dist'] = found_neighbours[0][0, id_0_neigh:id_1_neigh]
        query_candidate_pd['cosine_sim'] = cosine_sim[0] 
        query_candidate_pd['s1_orig_ids'] = orig_id_queries 
        query_candidate_pd['s2_orig_ids'] = orig_id_candis 

        if ranking_metric.lower() in ["faiss"]:
            query_candidate_filtered_pd = query_candidate_pd[query_candidate_pd["faiss_dist"] <= selection_threshold]
        elif ranking_metric.lower() in ["cosine"]:
            query_candidate_filtered_pd = query_candidate_pd[query_candidate_pd["cosine_sim"] >= selection_threshold]
        elif ranking_metric.lower() in ["conf"]:
            if not model_path in [False, None]:
                query_candidate_filtered_pd = query_candidate_pd[query_candidate_pd["dl_match"] >= selection_threshold]
            else:
                sys.exit(f"ranking_metric: {ranking_metric} is selected, but --model_path is not specified.")
        else:
            sys.exit(f"[ERROR] ranking_metric: {ranking_metric} is not implemented. See the documentation.")

        num_found_candidates += len(query_candidate_filtered_pd)
        print("ID: %s/%s -- Number of found candidates so far: %s, search span: 0, %s" % (iq, len(vecs_query), num_found_candidates, id_1_neigh))

        if num_found_candidates > 0:
            collect_neigh_pd = collect_neigh_pd.append(query_candidate_filtered_pd)

        # Go to the next zone    
        if (num_found_candidates < num_candidates):
            id_0_neigh, id_1_neigh = id_1_neigh, id_1_neigh + search_size

    
    # write results to output_pd
    mydict_dl_match = OrderedDict({})
    mydict_faiss_dist = OrderedDict({})
    mydict_candid_id = OrderedDict({})
    mydict_cosine_sim = OrderedDict({})
    if ranking_metric.lower() in ["faiss"]:
        collect_neigh_pd = collect_neigh_pd.sort_values(by="faiss_dist")[:num_candidates]
    elif ranking_metric.lower() in ["cosine"]:
        collect_neigh_pd = collect_neigh_pd.sort_values(by="cosine_sim", ascending=False)[:num_candidates]
    elif ranking_metric.lower() in ["conf"]:
        collect_neigh_pd = collect_neigh_pd.sort_values(by="dl_match", ascending=False)[:num_candidates]

    for i_row, row in collect_neigh_pd.iterrows():
        if not model_path in [False, None]:
            mydict_dl_match[row["s2"]] = round(row["dl_match"], 4)
        else:
            mydict_dl_match[row["s2"]] = row["dl_match"]
        mydict_faiss_dist[row["s2"]] = round(row["faiss_dist"], 4)
        mydict_cosine_sim[row["s2"]] = round(row["cosine_sim"], 4)
        mydict_candid_id[row["s2"]] = row["s2_orig_ids"]
    one_row = {
        "id": orig_id_queries, 
        "toponym": all_queries[0], 
        "pred_score": [mydict_dl_match], 
        "faiss_distance": [mydict_faiss_dist], 
        "cosine_sim": [mydict_cosine_sim],
        "candidate_original_ids": [mydict_candid_id], 
        "query_original_id": orig_id_queries,
        "num_all_searches": id_1_neigh 
        }
    output_pd = output_pd.append(pd.DataFrame.from_dict(one_row))
       
output_pd = output_pd.set_index("id")
output_pd.to_pickle(os.path.join(par_dir, f"{output_filename}.pkl"))
elapsed = time.time() - start_time
print("TOTAL TIME: %s" % elapsed)
