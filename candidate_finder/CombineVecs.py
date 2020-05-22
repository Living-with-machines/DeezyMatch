#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import glob
import numpy as np
import os
import pandas as pd
import sys

import torch
# --- set seed for reproducibility
from utils import set_seed_everywhere
set_seed_everywhere(1364)

# XXX move this to an input file or a better design
run_type = sys.argv[1]

if not run_type in ["c", "q"]:
    sys.exit("python CombineVecs.py <q or c, q: query mode, c: candidate mode>")

# --- path to fwd and bwd saved vectors and other stuff
if run_type in ["c"]:
    print("Candidates mode")
    path1 = "./embed_candidates/rnn_fwd"
    path2 = "./embed_candidates/rnn_bwd"
    path_id = "./embed_candidates/rnn_indxs"
    path_df = "./candidates.df"
    path2save_combined = "./combined_candidates"
elif run_type in ['q']:
    print("Queries mode")
    path1 = "./embed_queries/rnn_fwd"
    path2 = "./embed_queries/rnn_bwd"
    path_id = "./embed_queries/rnn_indxs"
    path_df = "./queries.df"
    path2save_combined = "./combined_queries"

# paths to create tensors/arrays
path1_combined = os.path.join(path2save_combined, "vecs1_combined.pt")
path2_combined = os.path.join(path2save_combined, "vecs2_combined.pt")
path_id_combined = os.path.join(path2save_combined, "vecs_ids_combined.pt")
path_items_combined = os.path.join(path2save_combined, "vecs_items_combined.npy")

output_par_dir = os.path.abspath(os.path.join(path1_combined, os.pardir))
if not os.path.isdir(output_par_dir):
    os.mkdir(output_par_dir)

if path1:
    print(f"\n\nReading vectors from {path1}")
    list_files_1 = glob.glob(path1 + "*")
    vecs1 = []
    for lfile in range(0, len(list_files_1)):
        print(lfile, end=' ', flush=True)
        if len(vecs1) == 0:
            vecs1 = torch.load(f"{path1}_{lfile}")
        else:
            vecs1 = torch.cat((vecs1, torch.load(f"{path1}_{lfile}")))
    torch.save(vecs1, path1_combined)

if path2:
    print(f"\n\nReading vectors from {path2}")
    vecs2 = []
    for lfile in range(0, len(list_files_1)):
        print(lfile, end=' ', flush=True)
        if len(vecs2) == 0:
            vecs2 = torch.load(f"{path2}_{lfile}")
        else:
            vecs2 = torch.cat((vecs2, torch.load(f"{path2}_{lfile}")))
    torch.save(vecs2, path2_combined)

if path_id:
    print(f"\n\nReading vectors from {path_id}")
    vecs_ids = []
    for lfile in range(0, len(list_files_1)):
        print(lfile, end=' ', flush=True)
        if len(vecs_ids) == 0:
            vecs_ids = torch.load(f"{path_id}_{lfile}")
        else:
            vecs_ids = torch.cat((vecs_ids, torch.load(f"{path_id}_{lfile}")))
    torch.save(vecs_ids, path_id_combined)

if path_df:
    mydf = pd.read_pickle(path_df)
    vecs_items = mydf['s1_unicode'].to_numpy()
    np.save(path_items_combined, vecs_items)