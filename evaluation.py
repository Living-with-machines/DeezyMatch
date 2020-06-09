#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import numpy as np
from tqdm import tqdm, tnrange

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import torch_summarize
# --- set seed for reproducibility
from utils import set_seed_everywhere
set_seed_everywhere(1364)

# ------------------- test_model --------------------
def test_model(model, test_dc, pooling_mode='attention', device='cpu', batch_size=256, 
               output_state_vectors=False, output_preds=False, shuffle=False, evaluation=True,
               output_preds_file="./pred_results.txt"):

    model.eval()

    # print info about the model only in the first epoch
    #torch_summarize(model)

    y_true_test = list()
    y_pred_test = list()
    total_loss_test = 0

    test_dl = DataLoader(dataset=test_dc, batch_size=batch_size, shuffle=shuffle)
    num_batch_test = len(test_dl)

    # XXX HARD CODED! Also in rnn_networks
    loss_fn=F.nll_loss
    # In first dump of the results, we add a header to the output file
    first_dump = True

    wtest_counter = 0
    t_test = tqdm(iter(test_dl), leave=False, total=num_batch_test)
    t_test.set_description('Test')
    for x1, len1, x2, len2, y, indxs in t_test:
        if output_state_vectors:
            output_par_dir = os.path.abspath(os.path.join(output_state_vectors, os.pardir))
            if not os.path.isdir(output_par_dir):
                os.mkdir(output_par_dir)
            torch.save(indxs, f'{output_state_vectors}_indxs_{wtest_counter}')
        wtest_counter += 1

        # transpose x1 and x2
        x1 = x1.transpose(0, 1)
        x2 = x2.transpose(0, 1)

        x1 = Variable(x1.to(device))
        x2 = Variable(x2.to(device))
        y = Variable(y.to(device))
        len1 = len1.numpy()
        len2 = len2.numpy()

        with torch.no_grad():
            pred = model(x1, len1, x2, len2, pooling_mode=pooling_mode, device=device, output_state_vectors=output_state_vectors, evaluation=evaluation)
            if output_state_vectors:
                continue

            loss = loss_fn(pred, y)
            pred_idx = torch.max(pred, dim=1)[1]

            if wtest_counter == 1:
                # Confidence for label 1
                all_preds = pred[:, 1]
            else:
                all_preds = torch.cat([all_preds, pred[:, 1]])

            y_true_test += list(y.cpu().data.numpy())
            y_pred_test += list(pred_idx.cpu().data.numpy())

            pred_results = np.vstack([test_dl.dataset.df.loc[indxs]["s1_unicode"].to_numpy(), 
                                      test_dl.dataset.df.loc[indxs]["s2_unicode"].to_numpy(), 
                                      pred_idx.cpu().data.numpy().T, 
                                      torch.exp(pred).T.cpu().data.numpy(), 
                                      y.cpu().data.numpy().T])
            with open(output_preds_file, "a+") as pred_f:
                if first_dump:
                    np.savetxt(pred_f, pred_results.T, 
                               fmt=('%s', '%s', '%d', '%.4f', '%.4f', '%d'), delimiter='\t', 
                               header="s1_unicode\ts2_unicode\tprediction\tp0\tp1\tlabel")
                    first_time = False
                else:
                    np.savetxt(pred_f, pred_results.T, 
                               fmt=('%s', '%s', '%d', '%.4f', '%.4f', '%d'), delimiter='\t')

            total_loss_test += loss.data

    if output_preds:
        return all_preds, 0, 0, 0
    elif output_state_vectors:
        return 0, 0, 0, 0
    else:
        test_acc = accuracy_score(y_true_test, y_pred_test)
        test_pre = precision_score(y_true_test, y_pred_test)
        test_rec = recall_score(y_true_test, y_pred_test)
        test_f1 = f1_score(y_true_test, y_pred_test, average='weighted')

        return (test_acc, test_pre, test_rec, test_f1)
