# DeezyMatch (Deep Fuzzy Match)

## A flexible Deep Neural Network framework for fuzzy string matching

DeezyMatch can be applied for performing the following tasks:

- candidate selection for entity linking systems
- record linkage

Credits:

- This project extensively uses the ideas/neural-network-architecture published in https://github.com/ruipds/Toponym-Matching. 

### Run DeezyMatch

After [installing DeezyMatch on your machine](#installation), DeezyMatch can be run in two ways:

1. As a Python library
2. Command line


Examples on how to run DeezyMatch can be found [here](./examples).

### DeezyMatch as a python library

```python
from DeezyMatch import train as dm_train
from DeezyMatch import finetune as dm_finetune
from DeezyMatch import inference as dm_inference
from DeezyMatch import combine_vecs
from DeezyMatch import candidate_finder

# train a new model
dm_train(input_file_path="input_dfm.yaml", 
         dataset_path="dataset/dataset-string-similarity_test.txt", 
         model_name="test001")

# fine-tune a pretrained model
dm_finetune(input_file_path="input_dfm.yaml", 
            dataset_path="dataset/dataset-string-similarity_test.txt", 
            model_name="finetuned_test001",
            pretrained_model_path="./models/test001/test001.model", 
            pretrained_vocab_path="./models/test001/test001.vocab")

# model inference
dm_inference(input_file_path="./input_dfm.yaml",
             dataset_path="dataset/dataset-string-similarity_test.txt", 
             pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
             pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab")

# generate vectors for queries and candidates
dm_inference(input_file_path="./input_dfm.yaml",
             dataset_path="dataset/dataset-string-similarity_test.txt", 
             pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
             pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab",
             inference_mode="vect",
             query_candidate_mode=["q", "c"],
             scenario="test")

# combine vectors
combine_vecs(qc_modes=['q', 'c'], 
             rnn_passes=['fwd', 'bwd'], 
             input_scenario='test', 
             output_scenario='test', 
             print_every=1)

# Find candidates
candidates_pd = \
    candidate_finder(scenario="./combined/test/", 
                     ranking_metric="conf", 
                     selection_threshold=0.51, 
                     num_candidates=1, 
                     search_size=4, 
                     output_filename="test_candidates_deezymatch", 
                     pretrained_model_path="./models/test001/test001.model", 
                     pretrained_vocab_path="./models/test001/test001.vocab", 
                     number_test_rows=20) 
```

In fine-tuning a pretrained model, it is possible to specify a directory name in the argument `pretrained_model_path`, e.g., 

```python
from DeezyMatch import finetune as dm_finetune
# fine-tune a pretrained model
dm_finetune(input_file_path="input_dfm.yaml", 
            dataset_path="dataset/dataset-string-similarity_test.txt", 
            model_name="finetuned_test001",
            pretrained_model_path="./models/test001")
```

In this case, DeezyMatch will create the `pretrained_model_path` and `pretrained_vocab_path` using the input directory name.

### Command line
Train a new model: a new classifier can be trained by:

```bash
python DeezyMatch.py -i input_dfm.yaml -d dataset/dataset-string-similarity_test.txt -m test001
```

NOTE: 
* Currently, the third column (label column) should be one of: ["true", "false", "1", "0"]
* Delimiter is fixed to \t for now.

DeezyMatch keeps some information about the metrics (e.g., loss/accuracy/precision/recall/F1) for each epoch. It is possible to plot the log-file by:

```bash
python DeezyMatch.py -lp ./models/test001/log.txt -ld test001
```

In this command, `-lp`: runs the log plotter and `-ld` is a name assigned to the log which will be used in the figure. This command generates a figure `log_test001.png`.

---

There are three steps needed to fine-tune a previously trained DeezyMatch model:

1. Print all parameters in the model

```bash
python DeezyMatch.py -pm ./models/test001/test001.model
```

which generates:

```bash
============================================================
List all parameters in ./models/test001/test001.model
============================================================
emb.weight True
gru_1.weight_ih_l0 True
gru_1.weight_hh_l0 True
gru_1.bias_ih_l0 True
gru_1.bias_hh_l0 True
gru_1.weight_ih_l0_reverse True
gru_1.weight_hh_l0_reverse True
gru_1.bias_ih_l0_reverse True
gru_1.bias_hh_l0_reverse True
gru_1.weight_ih_l1 True
gru_1.weight_hh_l1 True
gru_1.bias_ih_l1 True
gru_1.bias_hh_l1 True
gru_1.weight_ih_l1_reverse True
gru_1.weight_hh_l1_reverse True
gru_1.bias_ih_l1_reverse True
gru_1.bias_hh_l1_reverse True
attn_step1.weight True
attn_step1.bias True
attn_step2.weight True
attn_step2.bias True
fc1.weight True
fc1.bias True
fc2.weight True
fc2.bias True
============================================================
Any of the above parameters can be freezed for fine-tuning.
You can also input, e.g., 'gru_1' and in this case, all weights/biases related to that layer will be freezed.
See input file.
============================================================
Exit normally
```

2. Modify the input file:

```bash
layers_to_freeze: ["emb", "gru_1", "attn"]
```

3. Fine-tune on a dataset (in this example, we fine-tune on the same dataset, but the argument of `-d` can point to other datasets):

```bash
python DeezyMatch.py -i input_dfm.yaml -d dataset/dataset-string-similarity_test.txt -f ./models/test001 -m finetuned_test001
```

Note that it is also possible to add the argument `-n 100` to only use the first 100 rows for fine-tuning. In the above command, we use all the rows.

The above command outputs:

```bash
============================================================
List all parameters in the model
============================================================
emb.weight False
gru_1.weight_ih_l0 False
gru_1.weight_hh_l0 False
gru_1.bias_ih_l0 False
gru_1.bias_hh_l0 False
gru_1.weight_ih_l0_reverse False
gru_1.weight_hh_l0_reverse False
gru_1.bias_ih_l0_reverse False
gru_1.bias_hh_l0_reverse False
gru_1.weight_ih_l1 False
gru_1.weight_hh_l1 False
gru_1.bias_ih_l1 False
gru_1.bias_hh_l1 False
gru_1.weight_ih_l1_reverse False
gru_1.weight_hh_l1_reverse False
gru_1.bias_ih_l1_reverse False
gru_1.bias_hh_l1_reverse False
attn_step1.weight False
attn_step1.bias False
attn_step2.weight False
attn_step2.bias False
fc1.weight True
fc1.bias True
fc2.weight True
fc2.bias True
============================================================
```

The first column lists the learnable parameters, and the second column specifies if those parameters will be used in the optimization or not. In our example, we set `["emb", "gru_1", "attn"]` and all the parameters except for `fc1` and `fc2` will not be changed during the training.

DeezyMatch stores models, vocabularies, input file, log file and checkpoints (for each epoch) in the following directory structure:

```bash
models
├── finetuned_test001
│   ├── checkpoint00000.model
│   ├── checkpoint00001.model
│   ├── checkpoint00002.model
│   ├── checkpoint00003.model
│   ├── checkpoint00004.model
│   ├── finetuned_test001.model
│   ├── finetuned_test001.vocab
│   ├── input_dfm.yaml
│   └── log.txt
└── test001
    ├── checkpoint00000.model
    ├── checkpoint00001.model
    ├── checkpoint00002.model
    ├── checkpoint00003.model
    ├── checkpoint00004.model
    ├── input_dfm.yaml
    ├── log.txt
    ├── test001.model
    └── test001.vocab
```

After training/fine-tuning a model, DeezyMatch model can be used for inference or for candidate selection. 

### Model inference

To use an already trained model for inference/prediction:

```bash
python DeezyMatch.py --deezy_mode inference -m ./models/finetuned_test001/finetuned_test001.model -d dataset/dataset-string-similarity_test.txt -v ./models/finetuned_test001/finetuned_test001.vocab -i ./input_dfm.yaml -mode test
```

This command creates a file: `models/finetuned_test001/pred_results_dataset-string-similarity_test.txt` in which:

```bash
# s1_unicode    s2_unicode      prediction      p0      p1      label
la dom nxy      ลําโดมน้อย        1       0.1482  0.8518  1
krutoy  крутой  1       0.0605  0.9395  1
sharunyata      shartjugskij    0       0.9568  0.0432  0
sutangcun       羊山村  1       0.1534  0.8466  0
jowkār-e shafī‘ جوکار شفیع      1       0.0226  0.9774  1
rongreiyn ban hwy h wk cxmthxng rongreiyn ban hnxng xu  0       0.8948  0.1052  0
同心村  tong xin cun    1       0.0572  0.9428  1
engeskjæran     abrahamskjeret  0       0.9289  0.0711  0
izumo-zaki      tsumo-zaki      1       0.4662  0.5338  1
```

`p0` and `p1` are probabilities assigned to labels 0 and 1, respectively. For example, in the first row, the actual label is 1 (last column), the predicted label is 1 (third column), and the model confidence on the predicted label is 0.8518.

In the above example, we used a fine-tuned model. The model inference can be done with any trained models, e.g., 

```bash
python DeezyMatch.py --deezy_mode inference -m ./models/test001/test001.model -d dataset/dataset-string-similarity_test.txt -v ./models/test001/test001.vocab -i ./input_dfm.yaml -mode test
```

### Candidate selection

Candidate selection consists of the following steps:
1. Generate vectors for both queries and candidates
2. Combine vectors
3. For each query, find a list of candidates

----

1. In the first step, we create vectors for both query and candidate tokens:

```bash
# queries
python DeezyMatch.py --deezy_mode inference -m ./models/finetuned_test001/finetuned_test001.model -d dataset/dataset-string-similarity_test.txt -v ./models/finetuned_test001/finetuned_test001.vocab -i ./input_dfm.yaml -mode vect --scenario test -qc q

# candidates
python DeezyMatch.py --deezy_mode inference -m ./models/finetuned_test001/finetuned_test001.model -d dataset/dataset-string-similarity_test.txt -v ./models/finetuned_test001/finetuned_test001.vocab -i ./input_dfm.yaml -mode vect --scenario test -qc c
```

2. Combine vectors. This step is required if candidates or queries are distributed on several files. At this step, we combined those vectors.

```bash
python DeezyMatch.py --deezy_mode combine_vecs -qc q,c -sc test -p fwd,bwd -combs test
```

Alternatively:

```bash
python combineVecs.py -qc q,c -sc test -p fwd,bwd -combs test
```

3. CandidateFinder. Various options are available to find a set of candidates (from a dataset) for a given query in the same or another dataset.

* Select candidates based on L2-norm distance (aka faiss distance):

```bash
python DeezyMatch.py --deezy_mode candidate_finder -t 0.5 -rm faiss -n 1 -o test_candidates_deezymatch -sz 4 -comb combined/test -mp ./models/test001/test001.model -v ./models/test001/test001.vocab -tn 20
```

Alternatively:

```bash
python candidateFinder.py -t 0.5 -rm faiss -n 1 -o test_candidates_deezymatch -sz 4 -comb combined/test -mp ./models/test001/test001.model -v ./models/test001/test001.vocab -tn 20
```

in which `-t` is threshold:

```text
Selection criterion. NOTE: changes according to the ranking metric specified by -rm.
A candidate will be selected if:
    faiss-distance <= threshold, 
    cosine-similarity >= threshold,
    prediction-confidence >= threshold
```

`-rm` specifies the ranking metric, choices are `faiss` (used here, L2-norm distance), `cosine` (cosine similarity), `conf` (confidence as measured by DeezyMatch prediction outputs).

`-sz` is the search size. At each iteration, the selected metric between a query and `-sz` candidates are computed, and if the number of desired candidates is not reached (specified by `-n`), a new batch of candidates with the size of `-sz` is examined.

`-tn` should be used only for testing. The argument of `-tn` specifies the number of queries to be used for testing.

This command creates a pandas dataframe stored in `combined/test/test_candidates_deezymatch.pkl` and the first few rows are:

```bash
                            toponym                                   DeezyMatch_score                            faiss_distance                                cosine_sim                  candidate_original_ids  query_original_id  num_all_searches
id                                                                                                                                                                                                                                                     
0                        la dom nxy                 {'la dom nxy': 0.7220154404640198}                       {'la dom nxy': 0.0}                       {'la dom nxy': 1.0}                       {'la dom nxy': 0}                  0                 4
1                            krutoy                       {'krutoy': 0.79023677110672}                           {'krutoy': 0.0}                           {'krutoy': 1.0}                           {'krutoy': 1}                  1                 4
2                        sharunyata                 {'sharunyata': 0.8818759322166443}                       {'sharunyata': 0.0}                       {'sharunyata': 1.0}                       {'sharunyata': 2}                  2                 4
3                         sutangcun                  {'sutangcun': 0.7015400528907776}                        {'sutangcun': 0.0}                        {'sutangcun': 1.0}                        {'sutangcun': 3}                  3                 4
```

* Select candidates based on DeezyMatch predictions and their confidence:

```bash
python candidateFinder.py -t 0.51 -rm conf -n 1 -o test_candidates_deezymatch -sz 4 -comb combined/test -mp ./models/test001/test001.model -v ./models/test001/test001.vocab -tn 20
```

* Select candidates based on cosine similarity:

```bash
python candidateFinder.py -t 0.51 -rm cosine -n 1 -o test_candidates_deezymatch -sz 4 -comb combined/test -mp ./models/test001/test001.model -v ./models/test001/test001.vocab -tn 20
```


**Note on vocabulary:** `characters_v001.vocab` contains all characters in the wikigaz, OCR, gb1900, and santos training and test datasets (7,540 characters from multiple alphabets, containing special characters). 

### Installation

We strongly recommend installation via Anaconda:

1. Refer to [Anaconda website and follow the instructions](https://docs.anaconda.com/anaconda/install/).

2. Create a new environment for DeezyMatch

```bash
conda create -n py37deezy python=3.7
```

3. Activate the environment:

```bash
conda activate py37deezy
```

3. Install DeezyMatch dependencies:

```
pip install -r requirements.txt
```

:warning: If you get `ModuleNotFoundError: No module named '_swigfaiss'` error when running `candidateFinder.py`, one way to solve this issue is by:

```bash
pip install faiss-cpu --no-cache
```

Refer to [this page](https://github.com/facebookresearch/faiss/issues/821).