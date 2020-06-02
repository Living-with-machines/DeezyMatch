"""A simple code to compare DeezyMatch embeddings"""

import pandas as pd
import torch

# --- inputs
path2dataset = "./candidates/tests.csv"
path2embedding = "./embed_queries/rnn_bwd_0"
ref_id = 0
compare_mode = "l2"
# ---

embed = torch.load(path2embedding)
dataset = pd.read_csv(path2dataset, header=None, sep="\t")

print(f"{'string to compare':>40s}, {'reference string':>40s}, {'l2_norm'}")
for i in range(len(embed)):
    if compare_mode == "l2":
        l2_norm = torch.sum(torch.abs(embed[i] - embed[ref_id])**2)
        print(f"{dataset.iloc[i][0]:>40s}, {dataset.iloc[ref_id][0]:>40s}, {l2_norm}")
