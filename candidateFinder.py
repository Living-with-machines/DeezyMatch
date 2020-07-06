#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Add parent path so we can import modules
import sys
sys.path.insert(0,'..')

from argparse import ArgumentParser
import faiss
import glob
import numpy as np
import os
import pandas as pd
import pickle
import time

import torch

from utils import read_input_file
# --- set seed for reproducibility
from utils import set_seed_everywhere
set_seed_everywhere(1364)

def read_command():
    parser = ArgumentParser()

    parser.add_argument("-fd", "--max_faiss_distance",
                        help="max FAISS distance", default=0.8)

    parser.add_argument("-n", "--num_candidates",
                        help="Number of candidates", default=10)

    parser.add_argument("-o", "--output_filename",
                        help="output filename")

    parser.add_argument("-sz", "--search_size",
                        help="search size", default=4)

    parser.add_argument("-comb", "--combined_path",
                        help="path of the combined folder")
    
    parser.add_argument("-i", "--input_file_path",
                        help="Path of the input file, if 'default', search for files with .yaml extension in -sc", 
                        default="default")
    
    parser.add_argument("-tn", "--number_test_rows",
                        help="Only for testing", 
                        default=-1)

    args = parser.parse_args()
    num_candidates = int(args.num_candidates)
    output_filename = args.output_filename
    max_faiss_distance = float(args.max_faiss_distance)
    search_size = int(args.search_size)
    comb_path = args.combined_path
    input_file_path = args.input_file_path
    number_test_rows = int(args.number_test_rows)
    return output_filename, max_faiss_distance, search_size, num_candidates, comb_path, input_file_path, number_test_rows

start_time = time.time()
output_filename, max_faiss_distance, search_size, num_candidates, comb_path, input_file_path, number_test_rows = read_command()

if input_file_path in ["default"]:
    detect_input_files = glob.iglob(os.path.join(comb_path, "*.yaml"))
    for detected_inp in detect_input_files:
        if os.path.isfile(detected_inp):
            input_file_path = detected_inp
            break

dl_inputs = read_input_file(input_file_path)

# ----- COMBINE VECTORS, USER
par_dir = comb_path
path1_combined = os.path.join(par_dir, "candidates_fwd.pt")
path2_combined = os.path.join(par_dir, "candidates_bwd.pt")
path_id_combined = os.path.join(par_dir, "candidates_fwd_id.pt")
path_items_combined = os.path.join(par_dir, "candidates_fwd_items.npy")

vecs_ids_candidates = torch.load(path_id_combined, map_location=dl_inputs['general']['device'])
vecs_items_candidates = np.load(path_items_combined, allow_pickle=True)
vecs1_candidates = torch.load(path1_combined, map_location=dl_inputs['general']['device'])
vecs2_candidates = torch.load(path2_combined, map_location=dl_inputs['general']['device'])
vecs_candidates = torch.cat([vecs1_candidates, vecs2_candidates], dim=1)

par_dir = comb_path
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

output_pd = pd.DataFrame()
for iq in range(len_vecs_query):
    print("=========== Start the search for %s" % iq, vecs_items_query[iq])
    collect_neigh_pd = pd.DataFrame()
    num_found_candidates = 0
    # start with 0:seach_size, increase later
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
        query_candidate_pd['faiss_dist'] = found_neighbours[0][0, id_0_neigh:id_1_neigh]
        query_candidate_pd['s1_orig_ids'] = orig_id_queries 
        query_candidate_pd['s2_orig_ids'] = orig_id_candis 

        query_candidate_filtered_pd = query_candidate_pd[query_candidate_pd["faiss_dist"] <= max_faiss_distance]
        num_found_candidates += len(query_candidate_filtered_pd)
        print("ID: %s/%s -- Number of found candidates so far: %s, search span: 0, %s" % (iq, len(vecs_query), num_found_candidates, id_1_neigh))

        if num_found_candidates > 0:
            collect_neigh_pd = collect_neigh_pd.append(query_candidate_filtered_pd)

        # Go to the next zone    
        if (num_found_candidates < num_candidates):
            id_0_neigh, id_1_neigh = id_1_neigh, id_1_neigh + search_size

    
    # write results to output_pd
    mydict_dl_match = {}
    mydict_faiss_dist = {}
    mydict_candid_id = {}
    for i_row, row in collect_neigh_pd.iterrows():
        #mydict_dl_match[row["s2"]] = round(row["dl_match"], 4)
        mydict_faiss_dist[row["s2"]] = round(row["faiss_dist"], 4)
        mydict_candid_id[row["s2"]] = row["s2_orig_ids"]
    one_row = {
        "id": orig_id_queries, 
        "toponym": all_queries[0], 
        #"DeezyMatch_score": [mydict_dl_match], 
        "faiss_distance": [mydict_faiss_dist], 
        "candidate_original_ids": [mydict_candid_id], 
        "query_original_id": orig_id_queries,
        "num_all_searches": id_1_neigh 
        }
    output_pd = output_pd.append(pd.DataFrame.from_dict(one_row))
       
output_pd = output_pd.set_index("id")
output_pd.to_pickle(par_dir + "/" + output_filename + ".pkl")
elapsed = time.time() - start_time
print("TOTAL TIME: %s" % elapsed)
#import ipdb; ipdb.set_trace()
