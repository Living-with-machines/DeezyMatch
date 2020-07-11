#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import re
import socket
import time
import unicodedata
import yaml
from argparse import ArgumentParser
from sklearn.metrics import average_precision_score
import torch
from torch.nn.modules.module import _addindent


# ------------------- normalizeString --------------------
def normalizeString(s, uni2ascii=False, lowercase=False, strip=False, only_latin_letters=False):
    if uni2ascii:
        s = unicodedata.normalize('NFKD', str(s))
    if lowercase:
        s = s.lower()
    if strip:
        s = s.strip()
    if only_latin_letters:
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    #return "|" + s + "|"
    return s


# ------------------- sort_key --------------------
def sort_key(item):
    key_pat = re.compile(r"^(\D+)(\d+)$")
    item = os.path.abspath(item)
    item2match = os.path.basename(item)
    m = key_pat.match(item2match)
    return m.group(1), int(m.group(2))


# ------------------- set_seed_everywhere --------------------
def set_seed_everywhere(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------- computing_map --------------------
# from: https://github.com/iesl/stance/blob/master/src/main/eval/EvalMap.py

#NOTE! this expects labels as 1 and 0
def eval_map(list_of_list_of_labels,list_of_list_of_scores,randomize=True):
    """Compute Mean Average Precision
    Given a two lists with one element per test example compute the
    mean average precision score.
    The i^th element of each list is an array of scores or labels corresponding
    to the i^th training example.
    :param list_of_list_of_labels: Binary relevance labels. One list per example.
    :param list_of_list_of_scores: Predicted relevance scores. One list per example.
    :return: the mean average precision
    """

    set_seed_everywhere(1364)

    assert len(list_of_list_of_labels) == len(list_of_list_of_scores)
    aps = []
    for i in range(len(list_of_list_of_labels)):
        if randomize == True:
            perm = np.random.permutation(len(list_of_list_of_labels[i]))
            list_of_list_of_labels[i] = np.asarray(list_of_list_of_labels[i])[perm]
            list_of_list_of_scores[i] = np.asarray(list_of_list_of_scores[i])[perm]

        # NOTE! In case there are no positive labels, the entry will be skipped
        if sum(list_of_list_of_labels[i]) > 0:
            aps.append(average_precision_score(list_of_list_of_labels[i],
                                               list_of_list_of_scores[i]))
    return sum(aps) / len(aps)


# ------------------- string_split --------------------
def string_split(x, ngram=1):
    if ngram == 1:
        return [sub_x for sub_x in x]
    else:
        if len(x) <= 1:
            return [sub_x for sub_x in x]
        else:
            return [x[i:i+ngram] for i in range(len(x)-ngram+1)]

# ------------------- deezy_mode_detector --------------------
def deezy_mode_detector():

    parser = ArgumentParser()
    parser.add_argument("--deezy_mode", 
                        help="DeezyMatch mode (options: train, finetune, inference, combine_vecs, candidate_finder)",
                        default="train",
                        )
    dm_mode, unknown = parser.parse_known_args()
    dm_mode = dm_mode.deezy_mode.lower()
    if dm_mode not in ["train", "finetune", "inference", "combine_vecs", "candidate_finder"]:
        parser.exit(f"ERROR: implemeted modes are: train, finetune, inference, combine_vecs, candidate_finder (input: {dm_mode})")     
        
    return dm_mode

# ------------------- read_inputs_command --------------------
def read_inputs_command():
    """
    read inputs from the command line
    :return:
    """    
    parser = ArgumentParser()
    
    parser.add_argument("--deezy_mode",
                    help="DeezyMatch mode",
                    default=None
                    )

    parser.add_argument("-i", "--input_file_path",
                    help="Path of the input file",
                    default=None
                    )
    
    parser.add_argument("-d", "--dataset_path",
                    help="Path of the dataset",
                    default=None
                    )
    
    parser.add_argument("-m", "--model_name",
                    help="Name of the model to be saved",
                    default=None
                    )
    
    parser.add_argument("-f", "--fine_tuning",
                    help="Path to the folder of the model to be fine-tuned (note: if you use -v, then you should provide here the path to the .model file)",
                    default=None
                    )

    parser.add_argument("-v", "--vocabulary",
                    help="Path to the vocabulary to be used when fine-tuning (note: in this case -f should point to the .model file)",
                    default=None
                    )

    parser.add_argument("-n", "--number_training_examples",
                    help="the number of training examples to be used (optional)", 
                    default=None)

    parser.add_argument("-lp", "--log_plot",
                    help="Plot a log file and exit. In this case, you need to specify -ld flag as well.", 
                    default=None)

    parser.add_argument("-ld", "--log_dataset",
                    help="Name of the dataset for which the log will be plotted. This name is used in the figures. See -lp flag.", 
                    default=None)

    parser.add_argument("-pm", "--print_model_layers",
                    help="Print all the layers in a saved model.", 
                    default=None)
    
    args = parser.parse_args()

    if args.log_plot:
        if not args.log_dataset:
            parser.exit("ERROR: -ld is not defined.")
        log_plotter(args.log_plot, args.log_dataset)
        sys.exit("Exit normally")

    if args.print_model_layers:
        model_explorer(args.print_model_layers)
        sys.exit("Exit normally")

    input_file_path = args.input_file_path
    dataset_path = args.dataset_path
    model_name = args.model_name
    fine_tuning_model = args.fine_tuning
    vocab_path = args.vocabulary
    n_train_examples = args.number_training_examples
    
    if input_file_path is None or dataset_path is None or model_name is None:
        parser.print_help()
        parser.exit("ERROR: Missing input arguments.")
        
    if os.path.exists(input_file_path) and os.path.exists(dataset_path):
        fine_tuning_model_path = None
        if fine_tuning_model:
            
            if vocab_path:
                if fine_tuning_model.endswith(".model") is False:
                    parser.exit(f"ERROR: when using -v you need to provide with -f the path to the .model file")     
                
                if vocab_path.endswith(".vocab") is False:
                    parser.exit(f"ERROR: when using -v you need to provide the path to the .vocab file")     
                
                fine_tuning_model_path = fine_tuning_model 
                if os.path.exists(fine_tuning_model_path) is False:
                    parser.exit(f"ERROR: model {fine_tuning_model_path} not found!") 
                
                if os.path.exists(vocab_path) is False:
                    parser.exit(f"ERROR: vocab {vocab_path} not found!")
            
            else:
                fine_tuning_model_name = os.path.split(fine_tuning_model)[-1]
                if fine_tuning_model_name.endswith(".model"):
                        parser.exit(f"ERROR: with -f you need to provide the path to the model folder, not the .model file")     

                if os.path.exists(fine_tuning_model) is False:
                        parser.exit(f"ERROR: model folder {fine_tuning_model} not found!") 
                
                fine_tuning_model_path = os.path.join(fine_tuning_model,fine_tuning_model_name + '.model')   
                vocab_path = os.path.join(fine_tuning_model,fine_tuning_model_name + '.vocab')   
                    
                if os.path.exists(fine_tuning_model_path) is False:
                    parser.exit(f"ERROR: model {fine_tuning_model_path} not found!") 

                if os.path.exists(vocab_path) is False:
                    parser.exit(f"ERROR: vocab {vocab_path} not found!") 

        return input_file_path, dataset_path, model_name, fine_tuning_model_path, vocab_path, n_train_examples
    else:
        parser.exit("ERROR: Input file or dataset not found.")

# ------------------- read_inference_command --------------------
def read_inference_command():
    """
    read inputs from the command line
    :return:
    """
    
    cprint('[INFO]', bc.dgreen, 'read inputs from the command')
    try:
        parser = ArgumentParser()
        parser.add_argument("--deezy_mode",
                        help="DeezyMatch mode",
                        default=None
                        )
        parser.add_argument("-m", "--model_path")
        parser.add_argument("-d", "--dataset_path")
        parser.add_argument("-v", "--vocabulary_path")
        parser.add_argument("-i", "--input_file_path")
        parser.add_argument("-n", "--number_examples")
        parser.add_argument("-mode", "--inference_mode", default="test")
        parser.add_argument("-qc", "--query_candidate_mode", default="q")
        parser.add_argument("-sc", "--scenario")
        args = parser.parse_args()
        
        model_path = args.model_path
        dataset_path = args.dataset_path
        train_vocab_path = args.vocabulary_path
        input_file = args.input_file_path
        test_cutoff = args.number_examples
        inference_mode = args.inference_mode
        query_candidate_mode = args.query_candidate_mode
        scenario = args.scenario

    except IndexError as error:
        cprint('[syntax error]', bc.red, 'syntax: python <modelInference.py> /path/to/model /path/to/dataset /path/to/train/vocab /path/to/input/file n_examples_cutoff')
        sys.exit("[ERROR] {}".format(error))
    
    return model_path, dataset_path, train_vocab_path, input_file, test_cutoff, inference_mode, query_candidate_mode, scenario

# ------------------- read_command_combinevecs --------------------
def read_command_combinevecs():
    parser = ArgumentParser()

    parser.add_argument("--deezy_mode",
                    help="DeezyMatch mode",
                    default="combine_vecs"
                    )

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

# ------------------- read_command_candidate_finder --------------------
def read_command_candidate_finder():
    parser = ArgumentParser()

    parser.add_argument("--deezy_mode",
                    help="DeezyMatch mode",
                    default="candidate_finder"
                    )

    parser.add_argument("-t", "--threshold",
                        help="Selection criterion. NOTE: changes according to the ranking metric specified by -rm. " \
                             "A candidate will be selected if:" \
                             "faiss-distance <= threshold, " \
                             "cosine-similarity >= threshold, " \
                             "prediction-confidence >= threshold", 
                        default=0.8)

    parser.add_argument("-rm", "--ranking_metric",
                        help="Choices between faiss, cosine, conf", 
                        default="faiss")

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
    
    parser.add_argument("-mp", "--model_path",
                        help="Path to a DeezyMatch model, normally /path/to/file.model", 
                        default=False)
    
    parser.add_argument("-v", "--vocab_path",
                        help="Path to a vocabulary file, normally /path/to/file.vocab", 
                        default=False)

    args = parser.parse_args()
    num_candidates = int(args.num_candidates)
    output_filename = args.output_filename
    selection_threshold = float(args.threshold)
    ranking_metric = args.ranking_metric
    search_size = int(args.search_size)
    comb_path = args.combined_path
    input_file_path = args.input_file_path
    number_test_rows = int(args.number_test_rows)
    model_path = args.model_path
    vocab_path = args.vocab_path
    return output_filename, selection_threshold, ranking_metric, search_size, num_candidates, \
           comb_path, input_file_path, number_test_rows, model_path, vocab_path

# ------------------- read_input_file --------------------
def read_input_file(input_file_path):
    """
    read inputs from input_file_path
    :param input_file_path:
    :return:
    """
    cprint('[INFO]', bc.dgreen, "read input file: {}".format(input_file_path))
    with open(input_file_path, 'r') as input_file_read:
        dl_inputs = yaml.load(input_file_read, Loader=yaml.FullLoader)
        dl_inputs['gru_lstm']['learning_rate'] = float(dl_inputs['gru_lstm']['learning_rate'])

        # initialize before checking if GPU actually exists
        device = torch.device("cpu")
        dl_inputs['general']['is_cuda'] = False
        if dl_inputs['general']['use_gpu']:
            # --- check cpu/gpu availability
            # returns a Boolean True if a GPU is available, else it'll return False
            is_cuda = torch.cuda.is_available()
            if is_cuda:
                device = torch.device(dl_inputs["general"]["gpu_device"])
                dl_inputs['general']['is_cuda'] = True
            else:
                cprint('[INFO]', bc.lred, 'GPU was requested but not available.')

        dl_inputs['general']['device'] = device
        cprint('[INFO]', bc.lgreen, 'pytorch will use: {}'.format(dl_inputs['general']['device']))

        # XXX separation in the input CSV file
        # Hardcoded, see issue #38
        dl_inputs['preprocessing']['csv_sep'] = "\t"

    return dl_inputs

# ------------------- model_explorer --------------------
def model_explorer(model_path):
    """Output all the layers in a model"""
    pretrained_model = torch.load(model_path)

    print("\n")
    print(20*"===")
    print(f"List all parameters in {model_path}")
    print(20*"===")
    for name, param in pretrained_model.named_parameters():
        n = name.split(".")[0].split("_")[0]
        print(name, param.requires_grad)
    print(20*"===")
    print("Any of the above parameters can be freezed for fine-tuning.")
    print("You can also input, e.g., 'gru_1' and in this case, all weights/biases related to that layer will be freezed.")
    print("See input file.")
    print(20*"===")

# ------------------- log_message --------------------
def log_message(msg2push, filename="./log.txt", mode="w"):
    """log messages into a file"""
    log_fio = open(filename, mode)
    log_fio.writelines(msg2push)
    log_fio.close()

# ------------------- bc --------------------
class bc:
    lgrey = '\033[1;90m'
    grey = '\033[90m'           # broing information
    yellow = '\033[93m'         # FYI
    orange = '\033[0;33m'       # Warning

    lred = '\033[1;31m'         # there is smoke
    red = '\033[91m'            # fire!
    dred = '\033[2;31m'         # Everything is on fire

    lblue = '\033[1;34m'
    blue = '\033[94m'
    dblue = '\033[2;34m'

    lgreen = '\033[1;32m'       # all is normal
    green = '\033[92m'          # something else
    dgreen = '\033[2;32m'       # even more interesting

    lmagenta = '\033[1;35m'
    magenta = '\033[95m'        # for title
    dmagenta = '\033[2;35m'

    cyan = '\033[96m'           # system time
    white = '\033[97m'          # final time

    black = '\033[0;30m'

    end = '\033[0m'
    bold = '\033[1m'
    under = '\033[4m'


# ------------------- get_time --------------------
def get_time():
    time = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    return time


# ------------------- cprint --------------------
def cprint(type_info, bc_color, text):
    """
    simple print function used for colored logging
    """
    ho_nam = socket.gethostname().split('.')[0]

    print(bc.green          + get_time() + bc.end,
          bc.magenta        + ho_nam     + bc.end,
          bc.bold + bc.grey + type_info  + bc.end,
          bc_color          + text       + bc.end)


# ------------------- print_stats --------------------
def print_stats(t1):
    print("\n\n")
    print(20*"=")
    print("User time: {:.4f}".format(time.time() - t1))
    print(20*"=")


# ------------------- create_3d_input_arrays_chars --------------------
def create_3d_input_arrays_chars(mylist, char_labels, max_seq_len, len_chars, tmp_file_suffix, mycounter):
    """
    Create 3-D arrays as inputs for bidirectional_gru
    XXXXX
    """
    aux_arr = np.memmap("tmp-{}-{}".format(mycounter, tmp_file_suffix),
                        mode="w+",
                        shape=(len(mylist), max_seq_len, len_chars),
                        dtype=np.bool)

    for i, one_example in enumerate(mylist):
        for t, char in enumerate(one_example):
            if t < max_seq_len:
                aux_arr[i, t, char_labels[char]] = 1
            else:
                break
    return aux_arr


# ------------------- torch_summarize --------------------
def torch_summarize(model, show_weights=True, show_parameters=True):
    """
    SOURCE: https://stackoverflow.com/a/45528544
    Summarizes torch model by showing trainable parameters and weights.
    """
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'

    print("\n\n\n" + 20*"=")
    print("Total number of params: {}\n".format(sum([param.nelement() for param in model.parameters()])))
    print(tmpstr)
    print(20*"=" + "\n\n")

# ------------------- create_parent_dir --------------------
def create_parent_dir(file_path):
    output_par_dir = os.path.abspath(os.path.join(file_path, os.pardir))
    if not os.path.isdir(output_par_dir):
        os.mkdir(output_par_dir)

# ------------------- log_plotter --------------------
def log_plotter(path2log, dataset="DEFAULT"):
    """Plot the generated log file for each model"""

    # set path for the output
    path2log = os.path.abspath(path2log)
    path2fig_dir = os.path.dirname(path2log)
    path2fig_dirname = os.path.basename(path2fig_dir)

    if dataset in [None, "DEFAULT"]:
        dataset = path2fig_dirname 

    log_fio = open(path2log, "r")
    log = log_fio.readlines()

    # collect info of train and valid sets
    train_arr = []
    valid_arr = []
    time_arr = []
    for one_line in log[3:]:
        line_split = one_line.split()
        datetime_str = line_split[0]
        epoch = int(line_split[3].split("/")[0])
        loss = float(line_split[6][:-1])
        acc = float(line_split[8][:-1])
        prec = float(line_split[10][:-1])
        recall = float(line_split[12][:-1])
        macrof1 = float(line_split[14][:-1])
        weightedf1 = float(line_split[16][:-1])
    
        if line_split[4].lower() in ["train;", "train"]:
            train_arr.append([epoch, loss, acc, prec, recall, macrof1,weightedf1])
            time_arr.append(datetime.strptime(datetime_str, '%d/%m/%Y_%H:%M:%S'))
        elif line_split[4].lower() in ["valid;", "valid"]:
            #to be added
            #map_score = float(line_split[18])
            valid_arr.append([epoch, loss, acc, prec, recall, macrof1,weightedf1])
    
    diff_time = []
    for i in range(len(time_arr)-1):
        diff_time.append((time_arr[i+1] - time_arr[i]).seconds)
    total_time = (time_arr[-1] - time_arr[0]).seconds
    
    print(f"Dataset: {dataset}\nTime: {total_time}s")
    print(f"Dataset: {dataset}\nTime / epoch: {total_time/(len(time_arr)-1):.3f}s")
    
    train_arr = np.array(train_arr)
    valid_arr = np.array(valid_arr)
    if len(valid_arr > 0):
        min_valid_arg = np.argmin(valid_arr[:, 1])
        plot_valid = True
    else:
        plot_valid = False
    
    plt.figure(figsize=(15, 12))
    plt.subplot(3, 2, 1)
    plt.plot(train_arr[:, 0], train_arr[:, 1], label="train loss", c="k", lw=2)
    if plot_valid:
        plt.plot(valid_arr[:, 0], valid_arr[:, 1], label="valid loss", c='r', lw=2)
        plt.axvline(valid_arr[min_valid_arg, 0], 0, 1, ls="--", c="k")
        plt.text(valid_arr[min_valid_arg, 0]*1.05, min(min(valid_arr[:, 1]), min(train_arr[:, 1])), 
                 f"Epoch: {min_valid_arg}, Loss: {valid_arr[min_valid_arg, 1]}", fontsize=12, color="r")
    plt.xlabel("Epoch", size=18)
    plt.ylabel("Loss", size=18)
    plt.legend(fontsize=14, loc=7)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid()
    
    plt.subplot(3, 2, 2)
    plt.plot(train_arr[:, 0], train_arr[:, 5], label="train macro F1", c="k", lw=2)
    if plot_valid:  
        plt.plot(valid_arr[:, 0], valid_arr[:, 5], label="valid macro F1", c='r', lw=2)
        plt.axvline(valid_arr[min_valid_arg, 0], 0, 1, ls="--", c="k")
        plt.text(valid_arr[min_valid_arg, 0]*1.05, min(min(valid_arr[:, 5]), min(train_arr[:, 5])), 
             f"Epoch: {min_valid_arg}, macro F1: {valid_arr[min_valid_arg, 5]}", fontsize=12, color="r")
    plt.xlabel("Epoch", size=18)
    plt.ylabel("macro F1", size=18)
    plt.legend(fontsize=14, loc=4)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid()
    
    plt.subplot(3, 2, 3)
    plt.plot(train_arr[:, 0], train_arr[:, 2], label="train acc", c="k", lw=2)
    if plot_valid:
        plt.plot(valid_arr[:, 0], valid_arr[:, 2], label="valid acc", c='r', lw=2)
        plt.axvline(valid_arr[min_valid_arg, 0], 0, 1, ls="--", c="k")
        plt.text(valid_arr[min_valid_arg, 0]*1.05, min(min(valid_arr[:, 2]), min(train_arr[:, 2])), 
                 f"Epoch: {min_valid_arg}, Acc: {valid_arr[min_valid_arg, 2]}", fontsize=12, color="r")
    plt.xlabel("Epoch", size=18)
    plt.ylabel("Accuracy", size=18)
    plt.legend(fontsize=14, loc=4)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid()
    
    plt.subplot(3, 2, 4)
    plt.plot(train_arr[:, 0], train_arr[:, 3], label="train prec", c="k", ls="-", lw=2)
    plt.plot(train_arr[:, 0], train_arr[:, 4], label="train recall", c="k", ls="--", lw=2)
    if plot_valid:
        plt.plot(valid_arr[:, 0], valid_arr[:, 3], label="valid prec", c='r', ls="-", lw=2)
        plt.plot(valid_arr[:, 0], valid_arr[:, 4], label="valid recall", c='r', ls="--", lw=2)
        plt.axvline(valid_arr[min_valid_arg, 0], 0, 1, ls="--", c="k")
        plt.text(valid_arr[min_valid_arg, 0]*1.05, min(min(valid_arr[:, 3]), min(valid_arr[:, 4]), min(train_arr[:, 3]), min(train_arr[:, 4])), 
                 f"Epoch: {min_valid_arg}, Prec/Recall: {valid_arr[min_valid_arg, 3]}/{valid_arr[min_valid_arg, 4]}", fontsize=12, color="r")
    plt.xlabel("Epoch", size=18)
    plt.ylabel("Precision/Recall", size=18)
    plt.legend(fontsize=14, loc=4)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid()
    
    plt.subplot(3, 2, 5)
    plt.title(f"Dataset: {dataset}\nTotal time: {total_time}s, Ave. Time / epoch: {total_time/(len(time_arr)-1):.3f}s", size=16)
    plt.plot(train_arr[1:, 0], diff_time, c="k", lw=2)
    if plot_valid:
        plt.axvline(valid_arr[min_valid_arg, 0], 0, 1, ls="--", c="k")
        plt.text(valid_arr[min_valid_arg, 0]*1.05, min(diff_time)*0.98, 
                 f"Epoch: {min_valid_arg}, Time to solution: {np.cumsum(diff_time[:min_valid_arg])[-1]}s", fontsize=12, color="r")
    plt.xlabel("Epoch", size=18)
    plt.ylabel("Time (each epoch) / sec", size=18)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.ylim(min(diff_time)*0.97)
    plt.grid()
    
    plt.tight_layout()
    path2fig = os.path.join(path2fig_dir, f"log_{dataset}.png")
    plt.savefig(path2fig, dpi=300)
