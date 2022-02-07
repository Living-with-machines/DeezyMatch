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
import shutil
import time

import torch
from torch.utils.data import DataLoader

from .data_processing import test_tokenize
from .rnn_networks import test_model
from .utils import read_input_file
from .utils import read_command_candidate_ranker
from .utils_candidate_ranker import query_vector_gen
from .utils_candidate_ranker import candidate_conf_calc
# --- set seed for reproducibility
from .utils import set_seed_everywhere
set_seed_everywhere(1364)

# skip future warnings for now XXX
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------- candidateRanker --------------------
class candidate_ranker_init:
    """
    Wrapper for candidate_ranker
    """
    def __init__(self, input_file_path="default", query_scenario=None, candidate_scenario=None,
                 ranking_metric="faiss", selection_threshold=0.8, query=None, num_candidates=10,
                 search_size=4, length_diff=None, use_predict=True, output_path="ranker_output",
                 pretrained_model_path=None, pretrained_vocab_path=None, number_test_rows=-1):
        
        self.input_file_path = input_file_path 
        self.query_scenario = query_scenario 
        self.candidate_scenario = candidate_scenario 
        self.ranking_metric = ranking_metric 
        self.selection_threshold = selection_threshold 
        self.query = query 
        self.num_candidates = num_candidates 
        self.search_size = search_size 
        self.length_diff = length_diff
        self.use_predict = use_predict
        self.output_path = output_path 
        self.pretrained_model_path = pretrained_model_path 
        self.pretrained_vocab_path = pretrained_vocab_path 
        self.number_test_rows = number_test_rows
        self.detected_input_file_path = None

    def rank(self):
        self.output = \
            candidate_ranker(input_file_path=self.input_file_path,
                             query_scenario=self.query_scenario,
                             candidate_scenario=self.candidate_scenario,
                             ranking_metric=self.ranking_metric,
                             selection_threshold=self.selection_threshold,
                             query=self.query,
                             num_candidates=self.num_candidates,
                             search_size=self.search_size,
                             length_diff=self.length_diff,
                             use_predict=self.use_predict,
                             output_path=self.output_path,
                             pretrained_model_path=self.pretrained_model_path,
                             pretrained_vocab_path=self.pretrained_vocab_path,
                             number_test_rows=self.number_test_rows
                            )
    
    def set_query(self, query=None, query_scenario=None, ranking_metric=None, 
                  selection_threshold=None, num_candidates=None, search_size=None,
                  length_diff=None, use_predict=True, number_test_rows=None, output_path=None):
        if query: self.query=query
        if query_scenario: self.query_scenario=query_scenario
        if ranking_metric: self.ranking_metric=ranking_metric
        if selection_threshold: self.selection_threshold=selection_threshold
        if num_candidates: self.num_candidates=num_candidates
        if search_size: self.search_size=search_size
        if length_diff: self.length_diff=length_diff
        if use_predict: self.use_predict=use_predict
        if number_test_rows: self.number_test_rows=number_test_rows
        if output_path: self.output_path=output_path
    
    def __str__(self):
        msg = "-------------------------\n"
        msg += "* Candidate ranker params\n"
        msg += "-------------------------\n\n"
        if self.query:
            msg += "Queries are based on the following list:\n"
            msg += f"{self.query}\n\n"
        else:
            msg += "Queries are based on the following file:\n"
            msg += f"{self.query_scenario}\n\n"
        
        if (self.input_file_path in ["default"]) and (not self.detected_input_file_path):
            detect_input_files = glob.iglob(os.path.join(self.candidate_scenario, "*.yaml"))
            for detected_inp in detect_input_files:
                if os.path.isfile(detected_inp):
                    self.detected_input_file_path = detected_inp
                    break

        msg += f"candidate_scenario:\t{self.candidate_scenario}\n"
        msg += f"---Searching params---\n"
        msg += f"num_candidates:\t\t{self.num_candidates}\n"
        msg += f"ranking_metric:\t\t{self.ranking_metric}\n"
        msg += f"selection_threshold:\t{self.selection_threshold}\n"
        msg += f"search_size:\t\t{self.search_size}\n"
        msg += f"length_diff:\t\t{self.length_diff}\n"
        msg += f"use_predict:\t\t{self.use_predict}\n"
        msg += f"number_test_rows:\t{self.number_test_rows}\n"
        msg += f"---I/O---\n"
        if self.input_file_path in ["default"]:
            msg += f"input_file_path:\t{self.input_file_path} (path: {self.detected_input_file_path})\n"
        else:
            msg += f"input_file_path:\t{self.input_file_path}\n"
        msg += f"output_path:\t\t{self.output_path}\n"
        msg += f"pretrained_model_path:\t{self.pretrained_model_path}\n"
        msg += f"pretrained_vocab_path:\t{self.pretrained_vocab_path}\n"
        return msg
    

# ------------------- candidate_ranker --------------------
def candidate_ranker(input_file_path="default", query_scenario=None, candidate_scenario=None,
                     ranking_metric="faiss", selection_threshold=0.8, query=None, num_candidates=10,
                     search_size=4, length_diff=None, use_predict=True, output_path="ranker_output",
                     pretrained_model_path=None, pretrained_vocab_path=None, number_test_rows=-1):
    """
    find and rank a set of candidates (from a dataset) for given queries in the same or another dataset

    Parameters
    ----------
    input_file_path
        path to the input file. "default": read input file in `candidate_scenario`
    query_scenario
        directory that contains all the assembled query vectors
    candidate_scenario
        directory that contains all the assembled candidate vectors
    ranking_metric
        choices are `faiss` (used here, L2-norm distance), 
                    `cosine` (cosine distance), 
                    `conf` (confidence as measured by DeezyMatch prediction outputs)
    selection_threshold 
        changes according to the `ranking_metric`:
          A candidate will be selected if:
              faiss-distance <= threshold
              cosine-distance <= threshold
              prediction-confidence >= threshold
    query
        one string or a list of strings to be used in candidate ranking on-the-fly
    num_candidates
        number of desired candidates
    search_size
        number of candidates to be tested at each iteration
    length_diff
        max length difference allowed between query and candidate strings
    use_predict
        boolean on whether to use prediction in ranking or not
    output_path
        path to the output file
    pretrained_model_path
        path to the pretrained model
    pretrained_vocab_path
        path to the pretrained vocabulary
    number_test_rows 
        number of examples to be used (optional, normally for testing)
    """

    start_time = time.time()
    
    if input_file_path in ["default"]:
        found_input = False
        detect_input_files = glob.iglob(os.path.join(candidate_scenario, "*.yaml"))
        for detected_inp in detect_input_files:
            if os.path.isfile(detected_inp):
                input_file_path = detected_inp
                found_input = True
                break
        if not found_input:
            sys.exit(f"[ERROR] no input file (*.yaml file) could be found in the dir: {candidate_scenario}")
    
    # read input file
    dl_inputs = read_input_file(input_file_path)

    if (ranking_metric.lower() in ["faiss"]) and (selection_threshold < 0):
        sys.exit(f"[ERROR] Threshold for the selected metric: '{ranking_metric}' should be >= 0.")
    if (ranking_metric.lower() in ["cosine", "conf"]) and not (0 <= selection_threshold <= 1):
        sys.exit(f"[ERROR] Threshold for the selected metric: '{ranking_metric}' should be between 0 and 1.")
    
    if not ranking_metric.lower() in ["faiss", "cosine", "conf"]:
        sys.exit(f"[ERROR] ranking_metric of {ranking_metric.lower()} is not supported. "\
                  "Current ranking methods are: 'faiss', 'cosine', 'conf'")
    
    if num_candidates == 0:
        sys.exit(f"[ERROR] num_candidates must be larger than 0.")
    
    if search_size == 0:
        sys.exit(f"[ERROR] search_size must be larger than 0.")
    
    # ----- CANDIDATES
    path1_combined = os.path.join(candidate_scenario, "fwd.pt")
    path2_combined = os.path.join(candidate_scenario, "bwd.pt")
    path_id_combined = os.path.join(candidate_scenario, "fwd_id.pt")
    path_items_combined = os.path.join(candidate_scenario, "fwd_items.npy")
    
    vecs_ids_candidates = torch.load(path_id_combined, map_location=dl_inputs['general']['device'])
    vecs_items_candidates = np.load(path_items_combined, allow_pickle=True)
    vecs1_candidates = torch.load(path1_combined, map_location=dl_inputs['general']['device'])
    vecs2_candidates = torch.load(path2_combined, map_location=dl_inputs['general']['device'])
    vecs_candidates = torch.cat([vecs1_candidates, vecs2_candidates], dim=1)
           
    if (not pretrained_model_path in [False, None]) or query:
        # --- load torch model, send it to the device (CPU/GPU)
        model = torch.load(pretrained_model_path, map_location=dl_inputs['general']['device'])
        # --- create test data class
        # read vocabulary
        with open(pretrained_vocab_path, 'rb') as handle:
            train_vocab = pickle.load(handle)

    # ----- QUERIES
    if query:
        tmp_dirname = query_vector_gen(query, model, train_vocab, dl_inputs)
        query_scenario = os.path.join(tmp_dirname, "combined", "query_on_fly")
        mydf = pd.read_pickle(os.path.join(tmp_dirname, "query", "dataframe.df"))
        vecs_items = mydf[['s1_unicode', "s1"]].to_numpy()
        np.save(os.path.join(tmp_dirname, "fwd_items.npy"), vecs_items)
        path_items_combined = os.path.join(tmp_dirname, "fwd_items.npy")
    else:
        path_items_combined = os.path.join(query_scenario, "fwd_items.npy")

    path1_combined = os.path.join(query_scenario, f"fwd.pt")
    path2_combined = os.path.join(query_scenario, f"bwd.pt")
    path_id_combined = os.path.join(query_scenario, f"fwd_id.pt")
    
    vecs_ids_query = torch.load(path_id_combined, map_location=dl_inputs['general']['device'])
    vecs_items_query = np.load(path_items_combined, allow_pickle=True)
    vecs1_query = torch.load(path1_combined, map_location=dl_inputs['general']['device'])
    vecs2_query = torch.load(path2_combined, map_location=dl_inputs['general']['device'])
    vecs_query = torch.cat([vecs1_query, vecs2_query], dim=1)

    if query:
        shutil.rmtree(tmp_dirname)

    if (number_test_rows > 0) and (number_test_rows < len(vecs_query)):
        len_vecs_query = number_test_rows
    else:
        len_vecs_query = len(vecs_query)

    # --- start FAISS
    faiss_id_candis = faiss.IndexFlatL2(vecs_candidates.size()[1])   # build the index
    print("Is faiss_id_candis already trained? %s" % faiss_id_candis.is_trained)
    faiss_id_candis.add(vecs_candidates.detach().cpu().numpy())

    # Empty dataframe to collect data
    output_pd = pd.DataFrame()
    
    for iq in range(len_vecs_query):
        print("=========== Start the search for %s" % iq, vecs_items_query[iq][1])
        collect_neigh_pd = pd.DataFrame()
        num_found_candidates = 0
        
        # start with 0:search_size
        # If the number of selected candidates < num_candidates
        # Increase the search size
        id_0_neigh = 0
        id_1_neigh = search_size

        # If use_predict is false, the search strategy is skipped
        if use_predict == False:
            id_1_neigh = search_size

        while (num_found_candidates < num_candidates):
            if id_1_neigh > len(vecs_candidates):
                id_1_neigh = len(vecs_candidates)
            if id_0_neigh == id_1_neigh:
                break
    
            found_neighbours = faiss_id_candis.search(vecs_query[iq:(iq+1)].detach().cpu().numpy(), id_1_neigh)
        
            # Candidates
            orig_id_candis = found_neighbours[1][0, id_0_neigh:id_1_neigh]
            all_candidates = vecs_items_candidates[orig_id_candis][:, 0]
            all_candidates_orig = vecs_items_candidates[orig_id_candis][:, 1]
        
            # Queries
            orig_id_queries = vecs_ids_query[iq].item()
            all_queries = [vecs_items_query[orig_id_queries][0]]*(id_1_neigh - id_0_neigh)
            all_queries_no_preproc = [vecs_items_query[orig_id_queries][1]]*(id_1_neigh - id_0_neigh)
    
            query_candidate_pd = pd.DataFrame(all_queries, columns=['s1'])
            query_candidate_pd['s2'] = all_candidates
            query_candidate_pd['s2_orig'] = all_candidates_orig

            query_candidate_pd['label'] = "False"
    
            # Compute cosine similarity
            cosine_sim = cosine_similarity(vecs_query[iq:(iq+1)].detach().cpu().numpy(), 
                                           vecs_candidates.detach().cpu().numpy()[orig_id_candis])
            cosine_dist = 1. - cosine_sim
    
            if use_predict == True:
                if not pretrained_model_path in [False, None]:
                    all_preds = candidate_conf_calc(query_candidate_pd, 
                                                    model, 
                                                    train_vocab, 
                                                    dl_inputs, 
                                                    cutoffs=(id_1_neigh - id_0_neigh))
                    query_candidate_pd['dl_match'] = all_preds.detach().cpu().numpy()
        
                else:
                    query_candidate_pd['dl_match'] = [None]*len(query_candidate_pd)
            else:
                query_candidate_pd['dl_match'] = [None]*len(query_candidate_pd)
    
            query_candidate_pd['faiss_dist'] = found_neighbours[0][0, id_0_neigh:id_1_neigh]
            query_candidate_pd['cosine_dist'] = cosine_dist[0] 
            query_candidate_pd['s1_orig_ids'] = orig_id_queries 
            query_candidate_pd['s2_orig_ids'] = orig_id_candis 

            # Filter out candidates that have a larger string length difference than the one allowed:
            if isinstance(length_diff, int):
                query_candidate_pd = query_candidate_pd[abs(query_candidate_pd["s1"].str.len() - query_candidate_pd["s2"].str.len()) <= length_diff]
    
            if ranking_metric.lower() in ["faiss"]:
                query_candidate_filtered_pd = query_candidate_pd[query_candidate_pd["faiss_dist"] <= selection_threshold]
            elif ranking_metric.lower() in ["cosine"]:
                query_candidate_filtered_pd = query_candidate_pd[query_candidate_pd["cosine_dist"] <= selection_threshold]
            elif ranking_metric.lower() in ["conf"]:
                if use_predict == False:
                    sys.exit(f"ranking_metric: {ranking_metric} is selected, but use_predict is set to {use_predict}")
                elif not pretrained_model_path in [False, None]:
                    query_candidate_filtered_pd = query_candidate_pd[query_candidate_pd["dl_match"] >= selection_threshold]
                else:
                    sys.exit(f"ranking_metric: {ranking_metric} is selected, but --model_path is not specified.")
            else:
                sys.exit(f"[ERROR] ranking_metric: {ranking_metric} is not implemented. See the documentation.")
    
            # remove duplicates
            query_candidate_filtered_pd = query_candidate_filtered_pd[~query_candidate_filtered_pd.duplicated(["s2_orig"])]

            if len(query_candidate_filtered_pd) > 0:
                collect_neigh_pd = collect_neigh_pd.append(query_candidate_filtered_pd)
                collect_neigh_pd = collect_neigh_pd[~collect_neigh_pd.duplicated(["s2_orig"])]

            num_found_candidates = len(collect_neigh_pd)
            print("ID: %s/%s -- Number of found candidates so far: %s, searched: %s" % (iq+1, len(vecs_query), num_found_candidates, id_1_neigh))
            
            if ranking_metric.lower() in ["faiss"]:
                if query_candidate_pd["faiss_dist"].max() > selection_threshold:
                    break
            elif ranking_metric.lower() in ["cosine"]:
                if query_candidate_pd["cosine_dist"].max() > selection_threshold:
                    break 
    
            # Go to the next zone    
            if (num_found_candidates < num_candidates):
                id_0_neigh, id_1_neigh = id_1_neigh, id_1_neigh + search_size
    
        # write results to output_pd
        mydict_dl_match = OrderedDict({})
        mydict_dl_1_minus_match = OrderedDict({})
        mydict_faiss_dist = OrderedDict({})
        mydict_candid_id = OrderedDict({})
        mydict_cosine_dist = OrderedDict({})
        if len(collect_neigh_pd) == 0:
            one_row = {
                "id": orig_id_queries, 
                "query": all_queries_no_preproc[0], 
                "pred_score": [mydict_dl_match], 
                "1-pred_score": [mydict_dl_1_minus_match],
                "faiss_distance": [mydict_faiss_dist], 
                "cosine_dist": [mydict_cosine_dist],
                "candidate_original_ids": [mydict_candid_id], 
                "query_original_id": orig_id_queries,
                "num_all_searches": id_1_neigh 
                }
            output_pd = output_pd.append(pd.DataFrame.from_dict(one_row))
            continue
        if ranking_metric.lower() in ["faiss"]:
            collect_neigh_pd = collect_neigh_pd.sort_values(by="faiss_dist")[:num_candidates]
        elif ranking_metric.lower() in ["cosine"]:
            collect_neigh_pd = collect_neigh_pd.sort_values(by="cosine_dist")[:num_candidates]
        elif ranking_metric.lower() in ["conf"]:
            collect_neigh_pd = collect_neigh_pd.sort_values(by="dl_match", ascending=False)[:num_candidates]
    
        for i_row, row in collect_neigh_pd.iterrows():
            if use_predict == True:
                if not pretrained_model_path in [False, None]:
                    mydict_dl_match[row["s2_orig"]] = round(row["dl_match"], 4)
                    mydict_dl_1_minus_match[row["s2_orig"]] = 1. - round(row["dl_match"], 4)
                else:
                    mydict_dl_match[row["s2_orig"]] = row["dl_match"]
                    mydict_dl_1_minus_match[row["s2_orig"]] = 1. - row["dl_match"]
            mydict_faiss_dist[row["s2_orig"]] = round(row["faiss_dist"], 4)
            mydict_cosine_dist[row["s2_orig"]] = round(row["cosine_dist"], 4)
            mydict_candid_id[row["s2_orig"]] = row["s2_orig_ids"]
        one_row = {
            "id": orig_id_queries, 
            "query": all_queries_no_preproc[0], 
            "pred_score": [mydict_dl_match], 
            "1-pred_score": [mydict_dl_1_minus_match], 
            "faiss_distance": [mydict_faiss_dist], 
            "cosine_dist": [mydict_cosine_dist],
            "candidate_original_ids": [mydict_candid_id], 
            "query_original_id": orig_id_queries,
            "num_all_searches": id_1_neigh 
            }
        output_pd = output_pd.append(pd.DataFrame.from_dict(one_row))
           
    if len(output_pd) == 0:
        return None
    output_pd = output_pd.set_index("id")
    output_path = os.path.abspath(output_path)
    if not os.path.isdir(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    output_pd.to_pickle(os.path.join(f"{output_path}.pkl"))
    elapsed = time.time() - start_time
    print("TOTAL TIME: %s" % elapsed)
    return output_pd

def main():
    # --- read args from the command line
    input_file_path, query_scenario, candidate_scenario, ranking_metric, selection_threshold, query, num_candidates,\
        search_size, length_diff, use_predict, output_path, pretrained_model_path, pretrained_vocab_path, number_test_rows = \
        read_command_candidate_ranker()
    
    # --- 
    candidate_ranker(input_file_path=input_file_path, 
                     query_scenario=query_scenario, 
                     candidate_scenario=candidate_scenario,
                     ranking_metric=ranking_metric, 
                     selection_threshold=selection_threshold, 
                     query=query,
                     num_candidates=num_candidates, 
                     search_size=search_size,
                     length_diff=length_diff,
                     use_predict=use_predict,
                     output_path=output_path,
                     pretrained_model_path=pretrained_model_path, 
                     pretrained_vocab_path=pretrained_vocab_path, 
                     number_test_rows=number_test_rows)

if __name__ == '__main__':
    main()
