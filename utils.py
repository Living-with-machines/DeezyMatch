#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from datetime import datetime
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
    return s


# ------------------- set_seed_everywhere --------------------
def set_seed_everywhere(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------- string_split --------------------
def string_split(x):
    return [sub_x for sub_x in x]


# ------------------- read_inputs_command --------------------
def read_inputs_command():
    """
    read inputs from the command line
    :return:
    """    
    parser = ArgumentParser()
    
    parser.add_argument("-i", "--input-file-path",
                    help="add the path of the input file")
    
    parser.add_argument("-d", "--dataset-path",
                    help="add the path of the dataset")
    
    parser.add_argument("-m", "--model-name",
                    help="add the name of the model")
    
    parser.add_argument("-f", "--fine-tuning",
                    help="add the name of the model to be fine-tuned (model and vocab should be located in models and vocabs folders), a new version of the model will be saved")

    parser.add_argument("-n", "--number-training-examples",
                    help="the number of training example to be used (optional)")
    
    parser.add_argument("-p", "--pretrained-embeddings",
                    help="the path to the pretrained char embeddings (optional)")
    args = parser.parse_args()
        
    input_file_path = args.input_file_path
    dataset_path = args.dataset_path
    model_name = args.model_name
    fine_tuning_model = args.fine_tuning
    train_examples = args.number_training_examples
    pretrained_embs = args.pretrained_embeddings
    
    if input_file_path is None or dataset_path is None or model_name is None:
        parser.print_help()
        parser.exit("ERROR: Missing input arguments.")
        
    if os.path.exists(input_file_path) and os.path.exists(dataset_path):
        if fine_tuning_model:
            fine_tuning_model_vocab_path = os.path.join('vocabs', fine_tuning_model + '.pickle')
            fine_tuning_model_path = os.path.join('models', fine_tuning_model + '.model')
            if os.path.exists(fine_tuning_model_path) and os.path.exists(fine_tuning_model_vocab_path):
                return input_file_path,dataset_path,model_name,fine_tuning_model_path,fine_tuning_model_vocab_path,train_examples,None
            else:
                parser.exit("ERROR: model or vocab file not found: they should be inside models and vocabs folders.")               
        else:
            if pretrained_embs:
                return input_file_path,dataset_path,model_name,None,None,None,pretrained_embs
                            
            else:
                return input_file_path,dataset_path,model_name,None,None,None,None
    else:
        parser.exit("ERROR: Input file or dataset not found.")


# ------------------- read_test_command --------------------
def read_test_command():
    """
    read inputs from the command line
    :return:
    """
    
    cprint('[INFO]', bc.dgreen, 'read inputs from the command')
    try:
        model_path = sys.argv[1]
        dataset_path = sys.argv[2]
        train_vocab_path = sys.argv[3]
        input_file = sys.argv[4]
        test_cutoff = int(sys.argv[5])
        query_candid_mode = sys.argv[6]
    except IndexError as error:
        cprint('[syntax error]', bc.red, 'syntax: python <TestModel.py> /path/to/model /path/to/dataset /path/to/train/vocab /path/to/input/file n_examples_cutoff <q or c, q: query mode, c: candidate mode>')
        sys.exit("[ERROR] {}".format(error))
    
    return model_path, dataset_path, train_vocab_path, input_file, test_cutoff, query_candid_mode


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
                device = torch.device("cuda")
                dl_inputs['general']['is_cuda'] = True
            else:
                cprint('[INFO]', bc.lred, 'GPU was requested but not available.')

        dl_inputs['general']['device'] = device
        cprint('[INFO]', bc.lgreen, 'pytorch will use: {}'.format(dl_inputs['general']['device']))
    return dl_inputs


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
