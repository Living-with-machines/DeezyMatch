<div align="center">
    <br>
    <h1>DeezyMatch</h1>
    <h2>A Flexible Deep Neural Network Approach to Fuzzy String Matching</h2>
</div>
 
<p align="center">
    <a href="https://pypi.org/project/DeezyMatch/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/DeezyMatch">
    </a>
    <a href="https://github.com/Living-with-machines/DeezyMatch/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
    <br/>
</p>

DeezyMatch can be applied for performing the following tasks:

- Toponym matching
- Candidate selection for entity linking systems
- Record linkage

Table of contents
-----------------

- [Run DeezyMatch as a Python module or via command line](#run-deezymatch-as-a-python-module-or-via-command-line)
    * [Data and directory structure in tutorials](#data-and-directory-structure-in-tutorials)
    * [Train a new model](#train-a-new-model)
    * [Finetune a pretrained model](#finetune-a-pretrained-model)
    * [Model inference](#model-inference)
    * [Generate query and candidate vectors](#generate-query-and-candidate-vectors)
    * [Candidate ranker and assembling vector representations](#candidate-ranker-and-assembling-vector-representations)
    * [Candidate ranking on-the-fly](#candidate-ranking-on-the-fly)
    * [Tips / Suggestions on DeezyMatch functionalities](#tips--suggestions-on-deezymatch-functionalities)
- [Examples on how to run DeezyMatch](./examples)
- [Installation and setup](#installation)
- [Credits](#credits)

## Data and directory structure in tutorials

In the tutorials, we assume the following directory structure:

```bash
.
├── dataset
│   ├── characters_v001.vocab
│   └── dataset-string-similarity_test.txt
└── inputs
    └── input_dfm.yaml
```

For this, we first create a test directory (<ins>**note that we strongly recommend creating this directory outside of the DeezyMarch directory cloned and installed following the installation section. After installation, DeezyMatch command lines and modules are accessible from anywhere on your local machine.**</ins>):

```bash
mkdir ./test_deezy
cd ./test_deezy
mkdir dataset inputs
# Now, copy characters_v001.vocab, dataset-string-similarity_test.txt and input_dfm.yaml from DeezyMatch repo
# Arrange the files according to the above directory structure
```

These three files can be downloaded directly from `inputs` and `dataset` directories on [DeezyMatch repo](https://github.com/Living-with-machines/DeezyMatch).

**Note on vocabulary:** `characters_v001.vocab` combines all characters from the different datasets we have used in our experiments (Santos et al, 2018 and other datasets which will be described in a forthcoming publication). It consists of 7,540 characters from multiple alphabets, containing special characters.

`dataset-string-similarity_test.txt` contains 9995 example rows. The original dataset can be found here: https://github.com/ruipds/Toponym-Matching.


## Run DeezyMatch as a Python module or via command line

Refer to [installation section](#installation) to set-up DeezyMatch on your local machine. 

:warning: In the following tutorials, we assume a directory structure specified in the [this section](#data-and-directory-structure-in-tutorials).
 
 Written in the Python programming language, DeezyMatch can be used as a stand-alone command-line tool or can be integrated as a module with other Python codes. In what follows, we describe DeezyMatch's functionalities in different examples and by providing both command lines and python modules syntaxes.

### Train a new model

DeezyMatch `train` module can be used to train a new model:

```python
from DeezyMatch import train as dm_train

# train a new model
dm_train(input_file_path="./inputs/input_dfm.yaml", 
         dataset_path="dataset/dataset-string-similarity_test.txt", 
         model_name="test001")
```

The same model can be trained via command line:

```bash
DeezyMatch -i ./inputs/input_dfm.yaml -d dataset/dataset-string-similarity_test.txt -m test001
```

A new model directory called `test001` will be created in `models` directory (as specified in the `models_dir` in the input file).

:warning: Dataset (e.g., `dataset/dataset-string-similarity_test.txt` in the above command)
* Currently, the third column (label column) should be one of (case-insensitive): ["true", "false", "1", "0"]
* Delimiter is fixed to `\t` for now.

DeezyMatch keeps some information about the metrics (e.g., loss/accuracy/precision/recall/F1) for each epoch. It is possible to plot the log-file by:

```bash
DeezyMatch -lp ./models/test001/log.txt -ld test001
```

In this command, 
* `-lp`: runs the log plotter
* `-ld` is a name assigned to the log which will be used in the figure. 

This command generates a figure `log_test001.png` and stores it in `models/test001` directory.

DeezyMatch stores models, vocabularies, input file, log file and checkpoints (for each epoch) in the following directory structure:

```bash
models/test001
├── checkpoint00000.model
├── checkpoint00001.model
├── checkpoint00002.model
├── checkpoint00003.model
├── checkpoint00004.model
├── input_dfm.yaml
├── log.txt
├── log_test001.png
├── test001.model
└── test001.vocab
```

### Finetune a pretrained model

`finetune` module can be used to fine-tune a pretrained model:

```python
from DeezyMatch import finetune as dm_finetune

# fine-tune a pretrained model
dm_finetune(input_file_path="./inputs/input_dfm.yaml", 
            dataset_path="dataset/dataset-string-similarity_test.txt", 
            model_name="finetuned_test001",
            pretrained_model_path="./models/test001/test001.model", 
            pretrained_vocab_path="./models/test001/test001.vocab")
```

`dataset_path` specifies the dataset to be used for finetuning. For this example, we use the same dataset as in training however, other datasets are normally used to finetune an already trained model. The paths to model and vocabulary of the pretrained model are specified in `pretrained_model_path` and `pretrained_vocab_path`, respectively.

The same can be done via command line:

```bash
DeezyMatch -i ./inputs/input_dfm.yaml -d dataset/dataset-string-similarity_test.txt -m finetuned_test001 -f ./models/test001/test001.model -v ./models/test001/test001.vocab 
```

:warning: Note that it is also possible to add the argument `-n 100` to the above command to only use 100 rows for fine-tuning. In this example, we use all the rows. If `-n` flag is not specified, the train/valid/test proportions are read from the input file.

A new fine-tuned model called `finetuned_test001` will be stored in `models` directory. To fine-tune the pretrained model, two components in the neural network architecture were frozen, that is, not changed during fine-tuning (see `layers_to_freeze` in the input file). When running the above command, DeezyMatch lists the parameters in the model and whether or not they will be used in finetuning:

```
============================================================
List all parameters in the model
============================================================
emb.weight False
rnn_1.weight_ih_l0 False
rnn_1.weight_hh_l0 False
rnn_1.bias_ih_l0 False
rnn_1.bias_hh_l0 False
rnn_1.weight_ih_l0_reverse False
rnn_1.weight_hh_l0_reverse False
rnn_1.bias_ih_l0_reverse False
rnn_1.bias_hh_l0_reverse False
rnn_1.weight_ih_l1 False
rnn_1.weight_hh_l1 False
rnn_1.bias_ih_l1 False
rnn_1.bias_hh_l1 False
rnn_1.weight_ih_l1_reverse False
rnn_1.weight_hh_l1_reverse False
rnn_1.bias_ih_l1_reverse False
rnn_1.bias_hh_l1_reverse False
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

The first column lists the parameters in the model, and the second column specifies if those parameters will be used in the optimization or not. In our example, we set `["emb", "rnn_1", "attn"]` and all the parameters except for `fc1` and `fc2` will not be changed during the training.

In fact, it is possible to print all parameters in a model by:

```bash
DeezyMatch -pm models/finetuned_test001/finetuned_test001.model
```

which generates similar output as above.

In fine-tuning, it is also possible to specify a directory name for the argument `pretrained_model_path`. For example: 

```python
from DeezyMatch import finetune as dm_finetune

# fine-tune a pretrained model
dm_finetune(input_file_path="./inputs/input_dfm.yaml", 
            dataset_path="dataset/dataset-string-similarity_test.txt", 
            model_name="finetuned_test001",
            pretrained_model_path="./models/test001")
```

In this case, DeezyMatch will create the `pretrained_model_path` and `pretrained_vocab_path` using the input directory name, namely, `./models/test001/test001.model` and `./models/test001/test001.vocab`.

### Model inference

When a model is trained/fine-tuned, `inference` module can be used for predictions/model-inference. Again, we use the same dataset (`dataset/dataset-string-similarity_test.txt`) as before in this example. The paths to model and vocabulary of the pretrained model are specified in `pretrained_model_path` and `pretrained_vocab_path`, respectively. 

```python
from DeezyMatch import inference as dm_inference

# model inference
dm_inference(input_file_path="./inputs/input_dfm.yaml",
             dataset_path="dataset/dataset-string-similarity_test.txt", 
             pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
             pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab")
```

Similarly via command line:

```bash
DeezyMatch --deezy_mode inference -i ./inputs/input_dfm.yaml -d dataset/dataset-string-similarity_test.txt -m ./models/finetuned_test001/finetuned_test001.model  -v ./models/finetuned_test001/finetuned_test001.vocab  -mode test
```

The inference component creates a file: `models/finetuned_test001/pred_results_dataset-string-similarity_test.txt` in which:

```bash
# s1_unicode    s2_unicode      prediction      p0      p1      label
la dom nxy      ลําโดมน้อย        1       0.1635  0.8365  1
krutoy  крутой  1       0.0632  0.9368  1
sharunyata      shartjugskij    0       0.8696  0.1304  0
sutangcun       羊山村  1       0.4821  0.5179  0
rongreiyn ban hwy h wk cxmthxng rongreiyn ban hnxng xu  0       0.9156  0.0844  0
同心村  tong xin cun    1       0.1178  0.8822  1
engeskjæran     abrahamskjeret  0       0.8976  0.1024  0
izumo-zaki      tsumo-zaki      1       0.3394  0.6606  1
```

`p0` and `p1` are probabilities assigned to labels 0 and 1, respectively. For example, in the first row, the actual label is 1 (last column), the predicted label is 1 (third column), and the model confidence on the predicted label is `0.8365`. In these examples, DeezyMatch correctly predicts the label in all rows except for `sutangcun       羊山村`. By looking at the confidence scores, it is clear that DeezyMatch is not confident which label to assign (`p0=0.4821` and `p1=0.5179`). It should be noted, in this example and for showcasing DeezyMatch's functionalities, the model was trained and used for model inference on one dataset. In practice, we train a model on a dataset and use it for prediction on another dataset(s). Also, the dataset used to train the above model has around ~10K rows. Again, in practice, we use larger datasets for training and fine-tuning.

### Generate query and candidate vectors

`inference` module can also be used to generate vector representations for a set of strings in a dataset. This is **a required step for alias selection** (which we will [talk about later](#candidate-ranker-and-assembling-vector-representations). We first create vector representations for **query** mentions (we assume the query mentions are stored in `dataset/dataset-string-similarity_test.txt`):

```python
from DeezyMatch import inference as dm_inference

# generate vectors for queries and candidates
dm_inference(input_file_path="./inputs/input_dfm.yaml",
             dataset_path="dataset/dataset-string-similarity_test.txt", 
             pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
             pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab",
             inference_mode="vect",
             query_candidate_mode="q",
             scenario="test")
```

Compared to the previous section, here we have three additional arguments: 
* `inference_mode="vect"`: generate vector representations for the first column in `dataset_path`.
* `query_candidate_mode`: can be `"q"` or `"c"` for `queries` and `candidates`, respectively.
* `scenario`: directory (inside `queries` or `candidates` directories) where all the vector representations are stored.

Alternatively, the same can be done via command line:

```bash
DeezyMatch --deezy_mode inference -i ./inputs/input_dfm.yaml -d dataset/dataset-string-similarity_test.txt -m ./models/finetuned_test001/finetuned_test001.model -v ./models/finetuned_test001/finetuned_test001.vocab -mode vect -qc q --scenario test
```

The resulting directory structure is:

```
queries
└── test
    ├── embed_queries
    ├── input_dfm.yaml
    ├── log.txt
    └── queries.df
```
(`embed_queries` contains all the vector representations).

We repeat this step for `candidates` (again, we use the same dataset):

```python
from DeezyMatch import inference as dm_inference

# generate vectors for queries and candidates
# Note the only difference compared to the previous command is query_candidate_mode="c"
dm_inference(input_file_path="./inputs/input_dfm.yaml",
             dataset_path="dataset/dataset-string-similarity_test.txt", 
             pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
             pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab",
             inference_mode="vect",
             query_candidate_mode="c",
             scenario="test")
```

or via command line:

```bash
DeezyMatch --deezy_mode inference -i ./inputs/input_dfm.yaml -d dataset/dataset-string-similarity_test.txt -m ./models/finetuned_test001/finetuned_test001.model -v ./models/finetuned_test001/finetuned_test001.vocab -mode vect -qc c --scenario test
```

The resulting directory structure is:

```
candidates
└── test
    ├── candidates.df
    ├── embed_candidates
    ├── input_dfm.yaml
    └── log.txt
```

### Changing query/candidate directory names

In the above examples, DeezyMatch creates `queries` and `candidates` directories, by default, and store the `scenario` (in this example, it is set to `test`) inside these directories. This behaviour can be changed by (see `query_candidate_dirname="my_query_dir"`):

```python
from DeezyMatch import inference as dm_inference

# generate vectors for queries and candidates
# note the new argument: query_candidate_dirname 
dm_inference(input_file_path="./inputs/input_dfm.yaml",
             dataset_path="dataset/dataset-string-similarity_test.txt", 
             pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
             pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab",
             inference_mode="vect",
             query_candidate_mode="q",
             query_candidate_dirname="my_query_dir",
             scenario="test")
```

In this case, the directory structure is:

```
my_query_dir/
└── test
    ├── embed_my_query_dir
    ├── input_dfm.yaml
    ├── log.txt
    └── my_query_dir.df
```

(this can also be done for candidates.)

### Candidate ranker and assembling vector representations

Before using the `candidate_ranker` module of DeezyMatch, we need to:

1. Generate vector representations for both query and candidate mentions
2. Combine vector representations

----

Step 1 is already discussed in detail in the previous [section: Generate query and candidate vectors](#generate-query-and-candidate-vectors).

:warning: `queries` and `candidates` directories need to be in the same directory, for example:

```bash
├── candidates
│   └── test
├── dataset
│   ├── characters_v001.vocab
│   └── dataset-string-similarity_test.txt
├── inputs
│   └── input_dfm.yaml
├── models
│   ├── finetuned_test001
│   └── test001
└── queries
    └── test
```

#### Combine vector representations 

This step is required if query or candidate vectors are stored on several files (normally the case!). `combine_vecs` module can assemble those vector representations and store the results in `combined/output_scenario` directory (`output_scenario` is an argument in `combine_vecs` function): 

```python
from DeezyMatch import combine_vecs

# combine vectors
combine_vecs(qc_modes=['q', 'c'], 
             rnn_passes=['fwd', 'bwd'], 
             input_scenario='test', 
             output_scenario='test', 
             print_every=10)
```

Here, `qc_modes` specifies that `combine_vecs` should assemble both query and candidate embeddings stored in `input_scenario` directory (`input_scenario` is a directory inside `queries` or `candidates` directories). `rnn_passes` tells `combine_vecs` to assemble all vectors generated in both forward and backward RNN/GRU/LSTM passes (we have a backward pass only if `bidirectional` is set to True in the input file).

Similarly, this can be done via command line:

```bash
DeezyMatch --deezy_mode combine_vecs -qc q,c -p fwd,bwd -sc test -combs test
```

In this command, compared to `combine_vecs` module explained above:
* `-qc`: `qc_modes`
* `-p`: `rnn_passes`
* `-sc`: `input_scenario`
* `-combs`: `output_scenario`

The results are sored in the `output_scenario` (in python module mode) or `-combs` (in command-line mode) as follows:

```bash
combined
└── test
    ├── candidates_bwd.pt
    ├── candidates_bwd_id.pt
    ├── candidates_bwd_items.npy
    ├── candidates_fwd.pt
    ├── candidates_fwd_id.pt
    ├── candidates_fwd_items.npy
    ├── input_dfm.yaml
    ├── queries_bwd.pt
    ├── queries_bwd_id.pt
    ├── queries_bwd_items.npy
    ├── queries_fwd.pt
    ├── queries_fwd_id.pt
    ├── queries_fwd_items.npy
    └── test_candidates_deezymatch.pkl
```

In case `query_candidate_dirname` was set in [changing query/candidate directory names](#changing-querycandidate-directory-names), the vector representations can be combined by:

```python
from DeezyMatch import combine_vecs

# combine vectors
combine_vecs(qc_modes='q', 
             rnn_passes=['fwd', 'bwd'], 
             input_scenario='test', 
             query_candidate_dirname='my_query_dir',
             output_scenario='test', 
             print_every=10)
```

(and similarly for candidate vectors).

Moreover, it is possible to change the default dirname where combined vector representations are stored (by default, it is `combined`, see the above directory structure):

```python
from DeezyMatch import combine_vecs

# combine vectors
combine_vecs(qc_modes='q', 
             rnn_passes=['fwd', 'bwd'], 
             input_scenario='test', 
             output_scenario='test', 
             query_candidate_dirname='my_query_dir',
             output_par_dir="my_combined_dir",
             print_every=10)
```

:warning: in this case, `output_par_dir="my_combined_dir"` should be set for `qc_modes='c'` as well, that is:

```python
from DeezyMatch import combine_vecs

# combine vectors
combine_vecs(qc_modes='c', 
             rnn_passes=['fwd', 'bwd'], 
             input_scenario='test', 
             output_scenario='test', 
             query_candidate_dirname='my_candidate_dir',
             output_par_dir="my_combined_dir",
             print_every=10)
```

which results in the following directory structure:

```
my_combined_dir/
└── test
    ├── candidates_bwd_id.pt
    ├── candidates_bwd_items.npy
    ├── candidates_bwd.pt
    ├── candidates_fwd_id.pt
    ├── candidates_fwd_items.npy
    ├── candidates_fwd.pt
    ├── input_dfm.yaml
    ├── queries_bwd_id.pt
    ├── queries_bwd_items.npy
    ├── queries_bwd.pt
    ├── queries_fwd_id.pt
    ├── queries_fwd_items.npy
    └── queries_fwd.pt
```

#### CandidateRanker

Various options are available to find a set of candidates (from a dataset) for a given query in the same or another dataset.

* Select candidates based on L2-norm distance (aka faiss distance):

```python
from DeezyMatch import candidate_ranker

# Find candidates
candidates_pd = \
    candidate_ranker(scenario="./combined/test/", 
                     ranking_metric="faiss", 
                     selection_threshold=5., 
                     num_candidates=1, 
                     search_size=4, 
                     output_filename="test_candidates_deezymatch", 
                     pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
                     pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab", 
                     number_test_rows=20)
```

Similarly, if `output_par_dir="my_combined_dir"` was set in [combine vector representations](#combine-vector-representations), we need to set `scenario="./my_combined_dir/test"`: 

```python
from DeezyMatch import candidate_ranker

# Find candidates
candidates_pd = \
    candidate_ranker(scenario="./my_combined_dir/test", 
                     ranking_metric="faiss", 
                     selection_threshold=5., 
                     num_candidates=1, 
                     search_size=4, 
                     output_filename="test_candidates_deezymatch", 
                     pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
                     pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab", 
                     number_test_rows=20)
```

`scenario` is the directory that contains all the assembled vectors [(see)](#combine-vector-representations). 

`ranking_metric`: choices are `faiss` (used here, L2-norm distance), `cosine` (cosine similarity), `conf` (confidence as measured by DeezyMatch prediction outputs). 

:warning: In our experiments, `conf` was not a good metric to rank candidates. Consider using `faiss` or `cosine`.

`selection_threshold`: changes according to the `ranking_metric`:

```text
A candidate will be selected if:
    faiss-distance <= threshold
    cosine-similarity >= threshold
    prediction-confidence >= threshold
```

:warning: Note that `cosine` and `conf` scores are between [0, 1] while `faiss` distance can take any values from [0, +&#8734;). 

`search_size`: At each iteration, the selected ranking metric between a query and candidates (with the size of `search_size`) are computed, and if the number of desired candidates (specified by `num_candidates`) is not reached, a new batch of candidates with the size of `search_size` is tested. This continues until candidates with the size of `num_candidates` are found or all the candidates are tested.

The DeezyMatch model and its vocabulary are specified by `pretrained_model_path` and `pretrained_vocab_path`, respectively. 

`number_test_rows`: **only for testing**. It specifies the number of queries to be used for testing.

The results can be accessed directly from `candidates_pd` variable (see the above command). Also, they are saved in `combined/test/test_candidates_deezymatch.pkl` (specified by `output_filename`) which is in a pandas dataframe fromat. The first few rows are:

```bash
                query                     pred_score              faiss_distance                  cosine_sim    candidate_original_ids  query_original_id  num_all_searches
id                                                                                                                                                                         
0          la dom nxy         {'la dom nxy': 0.7165}         {'la dom nxy': 0.0}         {'la dom nxy': 1.0}         {'la dom nxy': 0}                  0                 4
1              krutoy             {'krutoy': 0.7733}             {'krutoy': 0.0}             {'krutoy': 1.0}             {'krutoy': 1}                  1                 4
2          sharunyata         {'sharunyata': 0.7062}         {'sharunyata': 0.0}         {'sharunyata': 1.0}         {'sharunyata': 2}                  2                 4
3           sutangcun          {'sutangcun': 0.6194}          {'sutangcun': 0.0}          {'sutangcun': 1.0}          {'sutangcun': 3}                  3                 4
```

As expected, candidate mentions (in `pred_score`, `faiss_distance`, `cosine_sim` and `candidate_original_ids`) are the same as the queries (second column), as we used one dataset for both queries and candidates.

Similarly, the above results can be generated via command line:

```bash
DeezyMatch --deezy_mode candidate_ranker -comb ./combined/test -rm faiss -t 5 -n 1 -sz 4 -o test_candidates_deezymatch -mp ./models/finetuned_test001/finetuned_test001.model -v ./models/finetuned_test001/finetuned_test001.vocab -tn 20
```

In this command, compared to `candidate_ranker` module explained above:
* `-comb`: `scenario`
* `-rm`: `ranking_metric`
* `-t`: `selection_threshold`
* `-n`: `num_candidates`
* `-sz`: `search_size`
* `-o`: `output_filename`
* `-mp`: `pretrained_model_path`
* `-v`: `pretrained_vocab_path`
* `-tn`: `number_test_rows`

**Other methods**

* Select candidates based on cosine similarity:

```python
from DeezyMatch import candidate_ranker

# Find candidates
candidates_pd = \
    candidate_ranker(scenario="./combined/test/", 
                     ranking_metric="cosine", 
                     selection_threshold=0.51, 
                     num_candidates=1, 
                     search_size=4, 
                     output_filename="test_candidates_deezymatch", 
                     pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
                     pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab", 
                     number_test_rows=20)
```


Note that the only difference compared to the previous command is `ranking_metric="cosine"`, or via command line:

```bash
DeezyMatch --deezy_mode candidate_ranker -comb ./combined/test -rm cosine -t 0.51 -n 1 -sz 4 -o test_candidates_deezymatch -mp ./models/finetuned_test001/finetuned_test001.model -v ./models/finetuned_test001/finetuned_test001.vocab -tn 20
```


## Tips / Suggestions on DeezyMatch functionalities

### Candidate ranker

* As already mentioned, based on our experiments, `conf` is not a good metric for ranking candidates. Consider using `faiss` or `cosine`.

* Adding prefix/suffix to input strings (see `prefix_suffix` option in the input file) can greatly enhance the ranking results.

* In `candidate_ranker`, the user specifies a `ranking_metric` based on which the candidates are selected. However, DeezyMatch also reports the values of other metrics for those candidates. For example, if the user selects `ranking_metric="faiss"`, the candidates are selected based on the `faiss`-distance metric. At the same time, the values of `cosine` and `conf` metrics for **those candidates** are also reported.

* In most use cases, `search_size` should be set `>= num_candidates`. However, if `num_candidates` is very large, it is better to set the `search_size` to lower values. 


## Installation

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

4. Clone DeezyMatch repositories on your local machine.

5. Install DeezyMatch dependencies:

```
cd /path/to/my/DeezyMatch
pip install -r requirements.txt
```

We have provided some [Jupyter Notebooks to show how different components in DeezyMatch can be run]((./examples)). To allow the newly created `py37deezy` environment to show up in the notebooks:

```bash
python -m ipykernel install --user --name py37deezy --display-name "Python (py37deezy)"
```

6. Install DeezyMatch:

```
cd /path/to/my/DeezyMatch
python setup.py install
```

Alternatively:

```
cd /path/to/my/DeezyMatch
pip install -v -e .
```

7. Continue with the [Tutorial](#run-deezymatch-as-a-python-module-or-via-command-line)!

---

:warning: If you get `ModuleNotFoundError: No module named '_swigfaiss'` error when running `candidateRanker.py`, one way to solve this issue is by:

```bash
pip install faiss-cpu --no-cache
```

Refer to [this page](https://github.com/facebookresearch/faiss/issues/821).

## Credits

This project extensively uses the ideas/neural-network-architecture published in https://github.com/ruipds/Toponym-Matching. 
