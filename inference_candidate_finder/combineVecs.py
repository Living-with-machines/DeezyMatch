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
import sys

import torch
# --- set seed for reproducibility
from utils import set_seed_everywhere
set_seed_everywhere(1364)

def read_command():
    parser = ArgumentParser()

    parser.add_argument("-p", "--path_vectors", 
                        help="path to parent directory where vectors are stored")

    parser.add_argument("-p_id", "--path_ids", 
                        help="Path to IDs")

    parser.add_argument("-df", "--path_df", 
                        help="path to dataframe")

    parser.add_argument("-n", "--name_combined", 
                        help="name of combined vector")

    parser.add_argument("-o", "--output_dir", 
                        help="path to parent directory where combined vectors will be stored")

    args = parser.parse_args()
    path2vecs = args.path_vectors
    path2ids = args.path_ids
    outputpath = args.output_dir
    name_combined = args.name_combined
    path_df = args.path_df
    return path2vecs, path2ids, outputpath, name_combined, path_df

path2vecs, path2ids, outputpath, name_combined, path_df = read_command()

# paths to create tensors/arrays
path_vec_combined = os.path.join(outputpath, f"{name_combined}.pt")
path_id_combined = os.path.join(outputpath, f"{name_combined}_id.pt")
path_items_combined = os.path.join(outputpath, f"{name_combined}_items.npy")

if not os.path.isdir(outputpath):
    os.makedirs(outputpath)

print(f"\n\nReading vectors from {path2vecs}")
list_files = glob.glob(os.path.join(path2vecs))
vecs = []
for lfile in list_files:
    print(lfile)
    if len(vecs) == 0:
        vecs = torch.load(f"{lfile}")
    else:
        vecs = torch.cat((vecs, torch.load(f"{lfile}")))
torch.save(vecs, path_vec_combined)

vecs_ids = []
list_files = glob.glob(os.path.join(path2ids))
for lfile in list_files: 
    print(lfile)
    if len(vecs_ids) == 0:
        vecs_ids = torch.load(f"{lfile}")
    else:
        vecs_ids = torch.cat((vecs_ids, torch.load(f"{lfile}")))
torch.save(vecs_ids, path_id_combined)

mydf = pd.read_pickle(path_df)
vecs_items = mydf['s1_unicode'].to_numpy()
np.save(path_items_combined, vecs_items)
