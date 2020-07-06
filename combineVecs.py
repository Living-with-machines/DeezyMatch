#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Add parent path so we can import modules
import sys
sys.path.insert(0,'..')

from argparse import ArgumentParser
import glob
import numpy as np
import os
import pandas as pd
import shutil
import sys
import time

import torch

start_time = time.time()

from utils import read_input_file
# --- set seed for reproducibility
from utils import set_seed_everywhere
set_seed_everywhere(1364)

import re
key_pat = re.compile(r"^(\D+)(\d+)$")
def sort_key(item):
    m = key_pat.match(item)
    return m.group(1), int(m.group(2))

def read_command():
    parser = ArgumentParser()

    parser.add_argument("-qc", "--candidate_or_query",
                        help="select mode: candidate (c) or query (q)")

    parser.add_argument("-sc", "--candidate_query_scenario", 
                        help="name of the candidate or query scenario")

    parser.add_argument("-p", "--rnn_pass", 
                        help="rnn pass: bwd (backward) or fwd (forward)")

    parser.add_argument("-combs", "--combined_scenario",
                        help="name of the combined scenario")

    parser.add_argument("-i", "--input_file_path",
                        help="Path of the input file, if 'default', search for files with .yaml extension in -sc", 
                        default="default")

    args = parser.parse_args()
    qc_mode = args.candidate_or_query
    cq_sc = args.candidate_query_scenario
    rnn_pass = args.rnn_pass
    combined_sc = args.combined_scenario
    input_file_path = args.input_file_path
    return qc_mode, cq_sc, rnn_pass, combined_sc, input_file_path

qc_modes, cq_sc, rnn_passes, combined_sc, input_file_path = read_command()

rnn_passes_list = rnn_passes.split(",")
rnn_passes_list = [x.strip() for x in rnn_passes_list]

qc_modes_list = qc_modes.split(",")
qc_modes_list = [x.strip() for x in qc_modes_list]

for qc_mode in qc_modes_list:
    for rnn_pass in rnn_passes_list:

        # paths to create tensors/arrays
        outputpath = "./combined/" + combined_sc + "/"
        
        path_vec_combined = outputpath
        path_id_combined = outputpath
        path_items_combined = outputpath
        
        if not os.path.isdir(outputpath):
            os.makedirs(outputpath)

        path2vecs = ""
        path2ids = ""
        pathdf = ""
        if qc_mode == "q":
            path2vecs = "./queries/" + cq_sc + "/embed_queries/rnn_" + rnn_pass + "*"
            path2ids = "./queries/" + cq_sc + "/embed_queries/rnn_indxs*"
            pathdf = "./queries/" + cq_sc + "/queries.df"
            path_vec_combined += "queries_" + rnn_pass + ".pt"
            path_id_combined += "queries_" + rnn_pass + "_id.pt"
            path_items_combined += "queries_" + rnn_pass + "_items.npy"
            inp_par_dir = "./queries/" + cq_sc
        elif qc_mode == "c":
            path2vecs = "./candidates/" + cq_sc + "/embed_candidates/rnn_" + rnn_pass + "*"
            path2ids = "./candidates/" + cq_sc + "/embed_candidates/rnn_indxs*"
            pathdf = "./candidates/" + cq_sc + "/candidates.df"
            path_vec_combined += "candidates_" + rnn_pass + ".pt"
            path_id_combined += "candidates_" + rnn_pass + "_id.pt"
            path_items_combined += "candidates_" + rnn_pass + "_items.npy"
            inp_par_dir = "./candidates/" + cq_sc
        
        if input_file_path in ["default"]:
            detect_input_files = glob.iglob(os.path.join(inp_par_dir, "*.yaml"))
            for detected_inp in detect_input_files:
                if os.path.isfile(detected_inp):
                    shutil.copy2(detected_inp, outputpath)
                    input_file_path = detected_inp
        else:
            shutil.copy2(input_file_path, outputpath)
        
        dl_inputs = read_input_file(input_file_path)
        
        print(f"\n\nReading vectors from {path2vecs}")
        list_files = glob.glob(os.path.join(path2vecs))
        list_files.sort(key=sort_key)
        vecs = []
        print("Combine vectors")
        for lfile in list_files:
            print(lfile, end="; ")
            if len(vecs) == 0:
                vecs = torch.load(f"{lfile}", map_location=dl_inputs['general']['device'])
            else:
                vecs = torch.cat((vecs, torch.load(f"{lfile}", map_location=dl_inputs['general']['device'])))
        print()
        torch.save(vecs, path_vec_combined)
        
        list_files = glob.glob(os.path.join(path2ids))
        list_files.sort(key=sort_key)
        vecs_ids = []
        print("\nCombine IDs")
        for lfile in list_files: 
            print(lfile, end="; ")
            if len(vecs_ids) == 0:
                vecs_ids = torch.load(f"{lfile}", map_location=dl_inputs['general']['device'])
            else:
                vecs_ids = torch.cat((vecs_ids, torch.load(f"{lfile}", dl_inputs['general']['device'])))
        print()
        torch.save(vecs_ids, path_id_combined)
        
        mydf = pd.read_pickle(pathdf)
        vecs_items = mydf['s1_unicode'].to_numpy()
        np.save(path_items_combined, vecs_items)

print("--- %s seconds ---" % (time.time() - start_time))

