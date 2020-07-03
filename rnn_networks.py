#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
References:
- Main: https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130
- https://github.com/hpanwar08/sentence-classification-pytorch/blob/master/Sentiment%20analysis%20pytorch.ipynb
New version of the above implementation (not used here)
- https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8
- https://github.com/hpanwar08/sentence-classification-pytorch/blob/master/Sentiment%20analysis%20pytorch%201.0.ipynb
Others:
- https://blog.floydhub.com/gru-with-pytorch/
- https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
- https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_recurrent_neuralnetwork/
- https://www.kaggle.com/francescapaulin/character-level-lstm-in-pytorch
- https://medium.com/intel-student-ambassadors/implementing-attention-models-in-pytorch-f947034b3e66
"""

import time, os
from tqdm import tqdm, tnrange

from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import glob
import numpy as np

from utils import cprint, bc, log_message
from utils import print_stats
from utils import torch_summarize
from utils import create_parent_dir
# --- set seed for reproducibility
from utils import set_seed_everywhere
set_seed_everywhere(1364)

# ------------------- gru_lstm_network --------------------
def gru_lstm_network(dl_inputs, model_name,train_dc, valid_dc=False, test_dc=False):
    """
    Main function for training and evaluation of GRU/LSTM network for matching
    """
    start_time = time.time()

    print("\n\n")
    cprint('[INFO]', bc.magenta,
           '******************************'.format(dl_inputs['gru_lstm']['main_architecture'].upper()))
    cprint('[INFO]', bc.magenta,
           '**** (Bi-directional) {} ****'.format(dl_inputs['gru_lstm']['main_architecture'].upper()))
    cprint('[INFO]', bc.magenta,
           '******************************'.format(dl_inputs['gru_lstm']['main_architecture'].upper()))

    # --- read inputs
    cprint('[INFO]', bc.dgreen, 'read inputs')
    vocab_size = len(train_dc.vocab)
    embedding_dim = dl_inputs['gru_lstm']['embedding_dim']
    rnn_hidden_dim = dl_inputs['gru_lstm']['rnn_hidden_dim']
    output_dim = dl_inputs['gru_lstm']['output_dim']
    batch_size = dl_inputs['gru_lstm']['batch_size']
    epochs = dl_inputs['gru_lstm']['epochs']
    learning_rate = dl_inputs['gru_lstm']['learning_rate']
    rnn_n_layers = dl_inputs['gru_lstm']['num_layers']
    bidirectional = dl_inputs['gru_lstm']['bidirectional']
    rnn_drop_prob = dl_inputs['gru_lstm']['gru_dropout']
    rnn_bias = dl_inputs['gru_lstm']['bias']
    fc_dropout = dl_inputs['gru_lstm']['fc_dropout']
    att_dropout = dl_inputs['gru_lstm']['att_dropout']
    fc1_out_features = dl_inputs['gru_lstm']['fc1_out_dim']
    pooling_mode = dl_inputs['gru_lstm']['pooling_mode']
    dl_shuffle = dl_inputs['gru_lstm']['dl_shuffle']

    # --- create the model
    cprint('[INFO]', bc.dgreen, 'create a two_parallel_rnns model')
    model_gru = two_parallel_rnns(vocab_size, embedding_dim, rnn_hidden_dim, output_dim,
                                  rnn_n_layers, bidirectional, pooling_mode, rnn_drop_prob, rnn_bias,
                                  fc1_out_features, fc_dropout, att_dropout)
    model_gru.to(dl_inputs['general']['device'])

    # --- optimisation
    if dl_inputs['gru_lstm']['optimizer'].lower() in ['adam']:
        opt = optim.Adam(model_gru.parameters(), learning_rate)

    cprint('[INFO]', bc.lgreen, 'start fitting parameters')
    train_dl = DataLoader(dataset=train_dc, batch_size=batch_size, shuffle=dl_shuffle)
    valid_dl = DataLoader(dataset=valid_dc, batch_size=batch_size, shuffle=dl_shuffle)

    if dl_inputs['gru_lstm']['create_tensor_board']:
        tboard_path = os.path.join(dl_inputs["general"]["models_dir"], model_name, dl_inputs['gru_lstm']['create_tensor_board'])
    else:
        tboard_path = None

    fit(model=model_gru,
        train_dl=train_dl, 
        valid_dl=valid_dl,
        loss_fn=F.nll_loss,  # The negative log likelihood loss
        opt=opt,
        epochs=epochs,
        pooling_mode=pooling_mode,
        device=dl_inputs['general']['device'], 
        tboard_path=tboard_path,
        model_path=os.path.join(dl_inputs["general"]["models_dir"], model_name)
        )

    # --- save the model
    cprint('[INFO]', bc.lgreen, 'saving the model')
    model_path = os.path.join(dl_inputs["general"]["models_dir"], 
                              model_name,
                              model_name + '.model')
    if not os.path.isdir(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    torch.save(model_gru, model_path)

    """
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    """

    # --- print some simple stats on the run
    print_stats(start_time)
    
# ------------------- fine_tuning --------------------
def fine_tuning(pretrained_model_path, dl_inputs, model_name, 
                train_dc, valid_dc=False, test_dc=False):
    """
    Fine tuning function for further training a model on new data
    """
    batch_size = dl_inputs['gru_lstm']['batch_size']
    dl_shuffle = dl_inputs['gru_lstm']['dl_shuffle']
    device=dl_inputs['general']['device']
    learning_rate = dl_inputs['gru_lstm']['learning_rate']
    epochs = dl_inputs['gru_lstm']['epochs']
    pooling_mode = dl_inputs['gru_lstm']['pooling_mode']
    
    pretrained_model = torch.load(pretrained_model_path, map_location=torch.device(device))
    
    layers_to_freeze = dl_inputs['gru_lstm']['layers_to_freeze']
    for one_layer in layers_to_freeze:
        for name, param in pretrained_model.named_parameters():
            if one_layer in name:
                param.requires_grad = False

    print("\n")
    print(20*"===")
    print(f"List all parameters in the model")
    print(20*"===")
    for name, param in pretrained_model.named_parameters():
        n = name.split(".")[0].split("_")[0]
        print(name, param.requires_grad)
    print(20*"===")
    
    if dl_inputs['gru_lstm']['optimizer'].lower() in ['adam']:
        opt = optim.Adam(filter(lambda p: p.requires_grad, pretrained_model.parameters()), learning_rate)

    start_time = time.time()

    print("\n\n")
    cprint('[INFO]', bc.magenta,
           '******************************'.format(dl_inputs['gru_lstm']['main_architecture'].upper()))
    cprint('[INFO]', bc.magenta,
           '**** (Bi-directional) {} ****'.format(dl_inputs['gru_lstm']['main_architecture'].upper()))
    cprint('[INFO]', bc.magenta,
           '******************************'.format(dl_inputs['gru_lstm']['main_architecture'].upper()))
    
    train_dl = DataLoader(dataset=train_dc, batch_size=batch_size, shuffle=dl_shuffle)
    valid_dl = DataLoader(dataset=valid_dc, batch_size=batch_size, shuffle=dl_shuffle)
    fit(model=pretrained_model,
        train_dl=train_dl, 
        valid_dl=valid_dl,
        loss_fn=F.nll_loss,  # The negative log likelihood loss
        opt=opt,
        epochs=epochs,
        pooling_mode=pooling_mode,
        device=dl_inputs['general']['device'], 
        tboard_path=dl_inputs['gru_lstm']['create_tensor_board'],
        model_path=os.path.join(dl_inputs["general"]["models_dir"], model_name)
        )

    # --- save the model
    cprint('[INFO]', bc.lgreen, 'saving the model')
    model_path = os.path.join(dl_inputs["general"]["models_dir"], 
                              model_name,
                              model_name + '.model')
    if not os.path.isdir(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))

    torch.save(pretrained_model, model_path)

    """
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    """

    # --- print some simple stats on the run
    print_stats(start_time)
    
# ------------------- fit  --------------------
def fit(model, train_dl, valid_dl, loss_fn, opt, epochs=3, pooling_mode='attention', device='cpu', tboard_path=False, model_path=False):

    num_batch_train = len(train_dl)
    num_batch_valid = len(valid_dl)
    cprint('[INFO]', bc.dgreen, 'Number of batches: {}'.format(num_batch_train))
    cprint('[INFO]', bc.dgreen, 'Number of epochs: {}'.format(epochs))
    
    if tboard_path:
        try:
            from torch.utils.tensorboard import SummaryWriter       
            tboard_writer = SummaryWriter(tboard_path) 
        except ImportError:
            cprint('[WARNING]', bc.dred, 'SummaryWriter could not be imported! Continue without creating a tensorboard.')
            tboard_writer = False
    else:
        tboard_writer = False

    print_summary = True
    wtrain_counter = 0
    wvalid_counter = 0
    for epoch in tnrange(epochs):
        if train_dl:
            model.train()
            y_true_train = list()
            y_pred_train = list()
            total_loss_train = 0

            t_train = tqdm(iter(train_dl), leave=False, total=num_batch_train)
            t_train.set_description('Epoch {}/{}'.format(epoch+1, epochs))
            for x1, len1, x2, len2, y, train_indxs in t_train:
                # transpose x1 and x2
                x1 = x1.transpose(0, 1)
                x2 = x2.transpose(0, 1)

                x1 = Variable(x1.to(device))
                x2 = Variable(x2.to(device))
                y = Variable(y.to(device))
                len1 = len1.numpy()
                len2 = len2.numpy()

                # step 1. zero the gradients
                opt.zero_grad()
                # step 2. compute the output
                pred = model(x1, len1, x2, len2, pooling_mode=pooling_mode, device=device)
                if print_summary:
                    # print info about the model only in the first epoch
                    torch_summarize(model)
                    print_summary = False
                # step 3. compute the loss
                loss = loss_fn(pred, y)
                # step 4. use loss to produce gradients
                loss.backward()
                # step 5. use optimizer to take gradient step
                opt.step()

                t_train.set_postfix(loss=loss.data)
                pred_idx = torch.max(pred, dim=1)[1]

                y_true_train += list(y.cpu().data.numpy())
                y_pred_train += list(pred_idx.cpu().data.numpy())
                total_loss_train += loss.data

                # if tboard_writer:    
                #     # XXX not working at this point, but the results can be plotted here: https://projector.tensorflow.org/
                #     # XXX TODO: change the metadata to the string name, plot embeddings derived for evaluation or test dataset
                #     s1s2_strings = train_dl.dataset.df[train_dl.dataset.df["index"].isin(train_indxs.tolist())]["s1"].to_list()
                #     s1s2_strings.extend(train_dl.dataset.df[train_dl.dataset.df["index"].isin(train_indxs.tolist())]["s2"].to_list())
                #     x1x2_tensors = torch.cat((x1.T, x2.T))
                #     try:
                #         tboard_writer.add_embedding(x1x2_tensors,
                #                                     global_step=wtrain_counter, 
                #                                     metadata=s1s2_strings,
                #                                     tag="Embedding")
                #         tboard_writer.flush()
                #     except:
                #         continue

                wtrain_counter += 1

            train_acc = accuracy_score(y_true_train, y_pred_train)
            train_pre = precision_score(y_true_train, y_pred_train)
            train_rec = recall_score(y_true_train, y_pred_train)
            train_f1 = f1_score(y_true_train, y_pred_train, average='weighted')
            train_loss = total_loss_train / len(train_dl)
            epoch_log = '{} -- Epoch: {}/{}; Train; loss: {:.3f}; acc: {:.3f}; precision: {:.3f}, recall: {:.3f}, f1: {:.3f}'.format(
                    datetime.now().strftime("%m/%d/%Y_%H:%M:%S"), epoch+1, epochs, train_loss, train_acc, train_pre, train_rec, train_f1)
            cprint('[INFO]', bc.orange, epoch_log)
            if model_path:
                log_message(epoch_log + "\n", mode="a+", filename=os.path.join(model_path, "log.txt"))
            else:
                log_message(epoch_log + "\n", mode="a+")

            if tboard_writer:    
                # Record loss
                tboard_writer.add_scalar('Train/Loss', loss.item(), epoch)
                # Record accuracy
                tboard_writer.add_scalar('Train/Accuracy', train_acc, epoch)
                tboard_writer.flush()

        if valid_dl:
            valid_desc = 'Epoch: {}/{}; Valid'.format(epoch+1, epochs)
            test_model(model, valid_dl, 
                       eval_mode="valid", valid_desc=valid_desc,
                       pooling_mode=pooling_mode, 
                       device=device,
                       model_path=model_path, 
                       tboard_writer=tboard_writer)

        if model_path:
            # --- save the model
            cprint('[INFO]', bc.lgreen, 'saving the model')
            checkpoint_path = os.path.join(model_path, f'checkpoint{epoch:05d}.model')
            if not os.path.isdir(os.path.dirname(checkpoint_path)):
                os.makedirs(os.path.dirname(checkpoint_path))
            torch.save(model, checkpoint_path)

# ------------------- test_model --------------------
def test_model(model, test_dl, eval_mode='test', valid_desc=None,
               pooling_mode='attention', device='cpu', evaluation=True,
               output_state_vectors=False, output_preds=False, 
               output_preds_file=False, model_path=False, tboard_writer=False):

    model.eval()

    # print info about the model only in the first epoch
    #torch_summarize(model)

    y_true_test = list()
    y_pred_test = list()
    total_loss_test = 0

    # XXX HARD CODED! Also in rnn_networks
    loss_fn=F.nll_loss
    # In first dump of the results, we add a header to the output file
    first_dump = True

    wtest_counter = 0
    t_test = tqdm(iter(test_dl), leave=False, total=len(test_dl))
    if eval_mode == 'valid':
        eval_desc = valid_desc
    elif eval_mode == 'test':
        eval_desc = "test"
    
    t_test.set_description(eval_mode)

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

            if eval_mode == 'valid':
                t_test.set_postfix(loss=loss.data)
                
            pred_idx = torch.max(pred, dim=1)[1]

            if wtest_counter == 1:
                # Confidence for label 1
                all_preds = pred[:, 1]
            else:
                all_preds = torch.cat([all_preds, pred[:, 1]])

            y_true_test += list(y.cpu().data.numpy())
            y_pred_test += list(pred_idx.cpu().data.numpy())

            if output_preds_file:
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
                        first_dump = False
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
        test_loss = total_loss_test / len(test_dl)
        epoch_log = '{} -- {}; loss: {:.3f}; acc: {:.3f}; precision: {:.3f}, recall: {:.3f}, f1: {:.3f}'.format(
               datetime.now().strftime("%m/%d/%Y_%H:%M:%S"), eval_desc, test_loss, test_acc, test_pre, test_rec, test_f1)
        cprint('[INFO]', bc.lred, epoch_log)
        if model_path:
            log_message(epoch_log + "\n", mode="a+", filename=os.path.join(model_path, "log.txt"))
        else:
            log_message(epoch_log + "\n", mode="a+")

        if tboard_writer:
            # Record loss
            tboard_writer.add_scalar('Test/Loss', loss.item(), epoch)
            # Record Accuracy, precision, recall, F1 on validation set 
            tboard_writer.add_scalar('Test/Accuracy', test_acc, epoch)
            tboard_writer.add_scalar('Test/Precision', test_pre, epoch)
            tboard_writer.add_scalar('Test/Recall', test_rec, epoch)
            tboard_writer.add_scalar('Test/F1', test_f1, epoch)
            tboard_writer.flush()
            
        return (test_acc, test_pre, test_rec, test_f1)

# ------------------- two_parallel_rnns  --------------------
class two_parallel_rnns(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_hidden_dim, output_dim,
                 rnn_n_layers, bidirectional, pooling_mode, rnn_drop_prob, rnn_bias,
                 fc1_out_features, fc_dropout=[0.5, 0.5], att_dropout=[0.5, 0.5], 
                 maxpool_kernel_size=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.output_dim = output_dim
        self.gru_output_dim = embedding_dim
        self.rnn_n_layers = rnn_n_layers
        self.bidirectional = bidirectional
        self.pooling_mode = pooling_mode
        self.rnn_drop_prob = rnn_drop_prob
        self.rnn_bias = rnn_bias
        self.fc1_out_features = fc1_out_features
        self.fc1_dropout = fc_dropout[0]
        self.fc2_dropout = fc_dropout[1]
        self.att1_dropout = att_dropout[0]
        self.att2_dropout = att_dropout[1]

        self.maxpool_kernel_size = maxpool_kernel_size

        if self.pooling_mode in ['attention', 'average', 'max', 'maximum', 'context']:
            fc1_multiplier = 4
        elif self.pooling_mode in ["context_layers"]:
            fc1_multiplier = 8
        elif self.pooling_mode in ["context_layers_simple"]:
            fc1_multiplier = 4
        else:
            fc1_multiplier = 4

        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # --- methods
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.gru_1 = nn.GRU(self.embedding_dim, self.rnn_hidden_dim, self.rnn_n_layers,
                            bias=self.rnn_bias, dropout=self.rnn_drop_prob,
                            bidirectional=self.bidirectional)

        #self.gru_2 = nn.GRU(self.embedding_dim, self.rnn_hidden_dim, self.rnn_n_layers,
        #                    bias=self.rnn_bias, dropout=self.rnn_drop_prob,
        #                    bidirectional=self.bidirectional)

        self.attn_step1 = nn.Linear(self.rnn_hidden_dim * self.num_directions, self.embedding_dim)
        self.attn_step2 = nn.Linear(self.embedding_dim, 1)

        self.fc1 = nn.Linear(self.rnn_hidden_dim*fc1_multiplier*self.num_directions, self.fc1_out_features)
        self.fc2 = nn.Linear(self.fc1_out_features, self.output_dim)

    # ------------------- forward 
    def forward(self, x1_seq, len1, x2_seq, len2, pooling_mode='context', device="cpu", output_state_vectors=False, evaluation=False):

        if evaluation:
            # XXX Set dropouts to zero manually
            self.att1_dropout = 0
            self.att2_dropout = 0
            self.fc1_dropout = 0
            self.fc2_dropout = 0

        if output_state_vectors:
            create_parent_dir(output_state_vectors)

        self.h1 = self.init_hidden(x1_seq.size(1), device)
        x1_embs_not_packed = self.emb(x1_seq)
        x1_embs = pack_padded_sequence(x1_embs_not_packed, len1, enforce_sorted=False)
        gru_out_1, self.h1 = self.gru_1(x1_embs, self.h1)
        gru_out_1, len1 = pad_packed_sequence(gru_out_1)

        if output_state_vectors:
            # the layers can be separated using h_n.view(num_layers, num_directions, batch, hidden_size).
            h1_reshape = self.h1.view(self.rnn_n_layers, self.num_directions, gru_out_1.shape[1], self.rnn_hidden_dim)

            output_h_layer = self.rnn_n_layers - 1
            file_id = len(glob.glob(output_state_vectors + "_fwd_*"))

            torch.save(h1_reshape[output_h_layer, 0], f'{output_state_vectors}_fwd_{file_id}')
            if self.bidirectional:
                torch.save(h1_reshape[output_h_layer, 1], f'{output_state_vectors}_bwd_{file_id}')
                return (h1_reshape[output_h_layer, 0], h1_reshape[output_h_layer, 1])
            else:
                return (h1_reshape[output_h_layer, 0], False)

        if pooling_mode in ['attention']:
            attn_weight_flag = False
            attn_weight_array = False
            for i_nhs in range(np.shape(gru_out_1)[0]):
                attn_weight = F.relu(self.attn_step1(F.dropout(gru_out_1[i_nhs], self.att1_dropout)))
                attn_weight = self.attn_step2(F.dropout(attn_weight, self.att2_dropout))
                if not attn_weight_flag:
                    attn_weight_array = attn_weight
                    attn_weight_flag = True
                else:
                    attn_weight_array = torch.cat((attn_weight_array, attn_weight), dim=1)
            attn_weight_array = F.softmax(attn_weight_array, dim=1)
            attn_vect_1 = torch.squeeze(torch.bmm(gru_out_1.permute(1, 2, 0), torch.unsqueeze(attn_weight_array, 2)))
        elif pooling_mode in ['average']:
            pool_1 = F.adaptive_avg_pool1d(gru_out_1.permute(1, 2, 0), 1).view(x1_seq.size(1), -1)
        elif pooling_mode in ['max', 'maximum']:
            pool_1 = F.adaptive_max_pool1d(gru_out_1.permute(1, 2, 0), 1).view(x1_seq.size(1), -1)
        elif pooling_mode in ['context']:
            context_1_fwd_bwd = self.h1.view(self.rnn_n_layers, self.num_directions, gru_out_1.shape[1], self.rnn_hidden_dim)
            context_1 = context_1_fwd_bwd[self.rnn_n_layers - 1, 0]
            if self.bidirectional:
                context_1 = torch.cat((context_1, context_1_fwd_bwd[self.rnn_n_layers - 1, 1]), dim=1)
        elif pooling_mode in ['context_layers', 'context_layers_simple']:
            context_1_fwd_bwd = self.h1.view(self.rnn_n_layers, self.num_directions, gru_out_1.shape[1], self.rnn_hidden_dim)
            context_1 = context_1_fwd_bwd[0, 0]
            for rlayer in range(1, self.rnn_n_layers):
                context_1 = torch.cat((context_1, context_1_fwd_bwd[rlayer, 0]), dim=1)
            if self.bidirectional:
                context_1_bwd = context_1_fwd_bwd[0, 1]
                for rlayer in range(1, self.rnn_n_layers):
                    context_1_bwd = torch.cat((context_1_bwd, context_1_fwd_bwd[rlayer, 1]), dim=1)
                context_1 = torch.cat((context_1, context_1_bwd), dim=1)

        self.h2 = self.init_hidden(x2_seq.size(1), device)
        x2_embs_not_packed = self.emb(x2_seq)
        x2_embs = pack_padded_sequence(x2_embs_not_packed, len2, enforce_sorted=False)
        # Share parameters between two GRUs
        # Previously, we had gru_out_2, self.h2 = self.gru_2(x2_embs, self.h2)
        gru_out_2, self.h2 = self.gru_1(x2_embs, self.h2)
        gru_out_2, len2 = pad_packed_sequence(gru_out_2)

        if pooling_mode in ['attention']:
            attn_weight_flag = False
            attn_weight_array = False
            for i_nhs in range(np.shape(gru_out_2)[0]):
                attn_weight = F.relu(self.attn_step1(F.dropout(gru_out_2[i_nhs], self.att1_dropout)))
                attn_weight = self.attn_step2(F.dropout(attn_weight, self.att2_dropout))
                if not attn_weight_flag:
                    attn_weight_array = attn_weight
                    attn_weight_flag = True
                else:
                    attn_weight_array = torch.cat((attn_weight_array, attn_weight), dim=1)
            attn_weight_array = F.softmax(attn_weight_array, dim=1)
            attn_vect_2 = torch.squeeze(torch.bmm(gru_out_2.permute(1, 2, 0), torch.unsqueeze(attn_weight_array, 2)))
        elif pooling_mode in ['average']:
            pool_2 = F.adaptive_avg_pool1d(gru_out_2.permute(1, 2, 0), 1).view(x2_seq.size(1), -1)
        elif pooling_mode in ['max', 'maximum']:
            pool_2 = F.adaptive_max_pool1d(gru_out_2.permute(1, 2, 0), 1).view(x2_seq.size(1), -1)
        elif pooling_mode in ['context']:
            context_2_fwd_bwd = self.h2.view(self.rnn_n_layers, self.num_directions, gru_out_2.shape[1], self.rnn_hidden_dim)
            context_2 = context_2_fwd_bwd[self.rnn_n_layers - 1, 0]
            if self.bidirectional:
                context_2 = torch.cat((context_2, context_2_fwd_bwd[self.rnn_n_layers - 1, 1]), dim=1) 
        elif pooling_mode in ['context_layers', 'context_layers_simple']:
            context_2_fwd_bwd = self.h2.view(self.rnn_n_layers, self.num_directions, gru_out_2.shape[1], self.rnn_hidden_dim)
            context_2 = context_2_fwd_bwd[0, 0]
            for rlayer in range(1, self.rnn_n_layers):
                context_2 = torch.cat((context_2, context_2_fwd_bwd[rlayer, 0]), dim=1)
            if self.bidirectional:
                context_2_bwd = context_2_fwd_bwd[0, 1]
                for rlayer in range(1, self.rnn_n_layers):
                    context_2_bwd = torch.cat((context_2_bwd, context_2_fwd_bwd[rlayer, 1]), dim=1)
                context_2 = torch.cat((context_2, context_2_bwd), dim=1)

        # Combine outputs from GRU1 and GRU2
        if pooling_mode in ['attention']:
            attn_vec_cat = torch.cat((attn_vect_1, attn_vect_2), dim=1)
            attn_vec_mul = attn_vect_1 * attn_vect_2
            attn_vec_dif = attn_vect_1 - attn_vect_2
            output_combined = torch.cat((attn_vec_cat,
                                         attn_vec_mul,
                                         attn_vec_dif), dim=1)
        elif pooling_mode in ['average', 'max', 'maximum']:
            pool_rnn_cat = torch.cat((pool_1, pool_2), dim=1)
            pool_rnn_mul = pool_1 * pool_2
            pool_rnn_dif = pool_1 - pool_2
            output_combined = torch.cat((pool_rnn_cat,
                                         pool_rnn_mul,
                                         pool_rnn_dif), dim=1)
        elif pooling_mode in ['context', 'context_layers']:
            context_rnn_cat = torch.cat((context_1, context_2), dim=1)
            context_rnn_mul = context_1 * context_2
            context_rnn_dif = context_1 - context_2
            output_combined = torch.cat((context_rnn_cat,
                                         context_rnn_mul,
                                         context_rnn_dif), dim=1)
        elif pooling_mode in ['context_layers_simple']:
            output_combined = torch.cat((context_1, context_2), dim=1)

        y_out = F.relu(self.fc1(F.dropout(output_combined, self.fc1_dropout)))
        y_out = self.fc2(F.dropout(y_out, self.fc2_dropout))
        return F.log_softmax(y_out, dim=-1)
        #return F.log_softmax(output_combined, dim=-1)

    def init_hidden(self, batch_size, device):
        first_dim = self.rnn_n_layers
        if self.bidirectional:
            first_dim *= 2
        return Variable(torch.zeros((first_dim, batch_size, self.rnn_hidden_dim)).to(device))
