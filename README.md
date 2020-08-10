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

- Record linkage
- Candidate selection for entity linking systems
- Toponym matching

Table of contents
-----------------

- [Installation and setup](#installation)
- [Data and directory structure in tutorials](#data-and-directory-structure-in-tutorials)
- [Run DeezyMatch as a Python module or via command line](#run-deezymatch-as-a-python-module-or-via-command-line)
    * [Quick tour](#quick-tour)
    * [Train a new model](#train-a-new-model)
    * [Finetune a pretrained model](#finetune-a-pretrained-model)
    * [Model inference](#model-inference)
    * [Generate query and candidate vectors](#generate-query-and-candidate-vectors)
    * [Candidate ranker and assembling vector representations](#candidate-ranker-and-assembling-vector-representations)
    * [Candidate ranking on-the-fly](#candidate-ranking-on-the-fly)
    * [Tips / Suggestions on DeezyMatch functionalities](#tips--suggestions-on-deezymatch-functionalities)
- [Examples on how to run DeezyMatch](./examples)
- [Credits](#credits)

## Installation

We strongly recommend installation via Anaconda:

* Refer to [Anaconda website and follow the instructions](https://docs.anaconda.com/anaconda/install/).

* Create a new environment for DeezyMatch

```bash
conda create -n py37deezy python=3.7
```

* Activate the environment:

```bash
conda activate py37deezy
```

DeezyMatch can be installed in different ways:

1. **install DeezyMatch via [PyPi](https://pypi.org/project/DeezyMatch/)**: which tends to be the most user-friendly option:

    ```bash
    pip install DeezyMatch
    ```

    * We have provided some [Jupyter Notebooks to show how different components in DeezyMatch can be run]((./examples)). To allow the newly created `py37deezy` environment to show up in the notebooks:

    ```bash
    python -m ipykernel install --user --name py37deezy --display-name "Python (py37deezy)"
    ```

    * Continue with the [Tutorial](#run-deezymatch-as-a-python-module-or-via-command-line)!

2. **install DeezyMatch from the source code**:

    * Clone DeezyMatch source code:

    ```bash
    git clone https://github.com/Living-with-machines/DeezyMatch.git
    ```

    * Install DeezyMatch dependencies:

    ```
    cd /path/to/my/DeezyMatch
    pip install -r requirements.txt
    ```

    * Install DeezyMatch:

    ```
    cd /path/to/my/DeezyMatch
    python setup.py install
    ```

    Alternatively:

    ```
    cd /path/to/my/DeezyMatch
    pip install -v -e .
    ```

    * We have provided some [Jupyter Notebooks to show how different components in DeezyMatch can be run]((./examples)). To allow the newly created `py37deezy` environment to show up in the notebooks:

    ```bash
    python -m ipykernel install --user --name py37deezy --display-name "Python (py37deezy)"
    ```

    * Continue with the [Tutorial](#run-deezymatch-as-a-python-module-or-via-command-line)!

---

:warning: If you get `ModuleNotFoundError: No module named '_swigfaiss'` error when running `candidateRanker.py`, one way to solve this issue is by:

```bash
pip install faiss-cpu --no-cache
```

Refer to [this page](https://github.com/facebookresearch/faiss/issues/821).

## Data and directory structure in tutorials

In the tutorials, we assume the following directory structure:

```bash
test_deezy/
├── dataset
│   ├── characters_v001.vocab
│   └── dataset-string-similarity_test.txt
└── inputs
    └── input_dfm.yaml
```

For this, we first create a test directory (:warning: note that this directory can be created outside of the DeezyMarch source code. After installation, DeezyMatch command lines and modules are accessible from anywhere on your local machine.):

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

:warning: In the following tutorials, we assume a directory structure specified in [this section](#data-and-directory-structure-in-tutorials).
 
 Written in the Python programming language, DeezyMatch can be used as a stand-alone command-line tool or can be integrated as a module with other Python codes. In what follows, we describe DeezyMatch's functionalities in different examples and by providing both command lines and python modules syntaxes.

### Quick tour 

In this "quick tour", we go through all the DeezyMatch main functionalities with minimal explanations. If you want to know more about each module, refer to the relevant part of README (also referenced in this section):

* [Train a new model](#train-a-new-model):

```python
from DeezyMatch import train as dm_train

# train a new model
dm_train(input_file_path="./inputs/input_dfm.yaml", 
         dataset_path="dataset/dataset-string-similarity_test.txt", 
         model_name="test001")
```

* Plot the log file (stored at `./models/test001/log.txt` and contains loss/accuracy/recall/F1-scores as a function of epoch):

```python
from DeezyMatch import plot_log

# plot log file
plot_log(path2log="./models/test001/log.txt", 
         dataset="t001")
```

* [Finetune a pretrained model](#finetune-a-pretrained-model):

```python
from DeezyMatch import finetune as dm_finetune

# fine-tune a pretrained model stored at pretrained_model_path and pretrained_vocab_path 
dm_finetune(input_file_path="./inputs/input_dfm.yaml", 
            dataset_path="dataset/dataset-string-similarity_test.txt", 
            model_name="finetuned_test001",
            pretrained_model_path="./models/test001/test001.model", 
            pretrained_vocab_path="./models/test001/test001.vocab")
```

* [Model inference](#model-inference):

```python
from DeezyMatch import inference as dm_inference

# model inference using a model stored at pretrained_model_path and pretrained_vocab_path 
dm_inference(input_file_path="./inputs/input_dfm.yaml",
             dataset_path="dataset/dataset-string-similarity_test.txt", 
             pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
             pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab")
```

 * [Generate query vectors](#generate-query-and-candidate-vectors):
 
 ```python
 from DeezyMatch import inference as dm_inference

# generate vectors for queries (specified in dataset_path) 
# using a model stored at pretrained_model_path and pretrained_vocab_path 
dm_inference(input_file_path="./inputs/input_dfm.yaml",
             dataset_path="dataset/dataset-string-similarity_test.txt", 
             pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
             pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab",
             inference_mode="vect",
             scenario="queries/test")
 ```

 * [Generate candidate vectors](#generate-query-and-candidate-vectors):

```python
from DeezyMatch import inference as dm_inference

# generate vectors for candidates (specified in dataset_path) 
# using a model stored at pretrained_model_path and pretrained_vocab_path 
dm_inference(input_file_path="./inputs/input_dfm.yaml",
             dataset_path="dataset/dataset-string-similarity_test.txt", 
             pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
             pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab",
             inference_mode="vect",
             scenario="candidates/test")
```

* [Assembling queries vector representations](#candidate-ranker-and-assembling-vector-representations):

```python
from DeezyMatch import combine_vecs

# combine vectors stored in queries/test and save them in combined/queries_test
combine_vecs(rnn_passes=['fwd', 'bwd'], 
             input_scenario='queries/test', 
             output_scenario='combined/queries_test', 
             print_every=10)
```

* [Assembling candidates vector representations](#candidate-ranker-and-assembling-vector-representations):

```python
from DeezyMatch import combine_vecs

# combine vectors stored in candidates/test and save them in combined/candidates_test
combine_vecs(rnn_passes=['fwd', 'bwd'], 
             input_scenario='candidates/test', 
             output_scenario='combined/candidates_test', 
             print_every=10)
```

* [Candidate ranker](#candidate-ranker-and-assembling-vector-representations):

```python

from DeezyMatch import candidate_ranker

# Select candidates based on L2-norm distance (aka faiss distance):
# find candidates from candidate_scenario 
# for queries specified in query_scenario
candidates_pd = \
    candidate_ranker(query_scenario="./combined/queries_test",
                     candidate_scenario="./combined/candidates_test", 
                     ranking_metric="faiss", 
                     selection_threshold=5., 
                     num_candidates=2, 
                     search_size=2, 
                     output_path="ranker_results/test_candidates_deezymatch", 
                     pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
                     pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab", 
                     number_test_rows=20)
```

* [Candidate ranking on-the-fly](#candidate-ranking-on-the-fly):

```python
from DeezyMatch import candidate_ranker

# Ranking on-the-fly
# find candidates from candidate_scenario 
# for queries specified by the `query` argument
candidates_pd = \
    candidate_ranker(candidate_scenario="./combined/candidates_test",
                     query=["DeezyMatch", "kasra", "fede", "mariona"],
                     ranking_metric="faiss", 
                     selection_threshold=5., 
                     num_candidates=1, 
                     search_size=100, 
                     output_path="ranker_results/test_candidates_deezymatch_on_the_fly", 
                     pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
                     pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab", 
                     number_test_rows=20)
```

The candidate ranker can be initialised, to be used multiple times, by running:

```python
from DeezyMatch import candidate_ranker_init

# initializing candidate_ranker via candidate_ranker_init
myranker = candidate_ranker_init(candidate_scenario="./combined/candidates_test",
                                 query=["DeezyMatch", "kasra", "fede", "mariona"],
                                 ranking_metric="faiss", 
                                 selection_threshold=5., 
                                 num_candidates=1, 
                                 search_size=100, 
                                 output_path="ranker_results/test_candidates_deezymatch_on_the_fly", 
                                 pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
                                 pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab", 
                                 number_test_rows=20)

# print the content of myranker by:
print(myranker)

# To rank the queries:
myranker.rank()

#The results are stored in:
myranker.output
```

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

A new model directory called `test001` will be created in `models` directory (as specified in the input file, see `models_dir` in `./inputs/input_dfm.yaml`).

:warning: Dataset (e.g., `dataset/dataset-string-similarity_test.txt` in the above command)
* Currently, the third column (label column) should be one of (case-insensitive): ["true", "false", "1", "0"]
* Delimiter is fixed to `\t` for now.

DeezyMatch keeps some information about the metrics (e.g., loss/accuracy/precision/recall/F1) for each epoch. It is possible to plot the log-file by:

```python
from DeezyMatch import plot_log

# plot log file
plot_log(path2log="./models/test001/log.txt", 
         dataset="t001")
```

or:

```bash
DeezyMatch -lp ./models/test001/log.txt -ld t001
```

In this command, 
* `-lp`: runs the log plotter on `./models/test001/log.txt` file. 
* `-ld` is a name assigned to the log which will be used in the figure. 

This command generates a figure `log_test001.png` and stores it in `models/test001` directory.

![Example output of plot_log module](./figs/log_t001.png)

DeezyMatch stores models, vocabularies, input file, log file and checkpoints (for each epoch) in the following directory structure:

```bash
models/
└── test001
    ├── checkpoint00001.model
    ├── checkpoint00002.model
    ├── checkpoint00003.model
    ├── checkpoint00004.model
    ├── checkpoint00005.model
    ├── input_dfm.yaml
    ├── log_t001.png
    ├── log.txt
    ├── test001.model
    └── test001.vocab
```

### Finetune a pretrained model

`finetune` module can be used to fine-tune a pretrained model:

```python
from DeezyMatch import finetune as dm_finetune

# fine-tune a pretrained model stored at pretrained_model_path and pretrained_vocab_path 
dm_finetune(input_file_path="./inputs/input_dfm.yaml", 
            dataset_path="dataset/dataset-string-similarity_test.txt", 
            model_name="finetuned_test001",
            pretrained_model_path="./models/test001/test001.model", 
            pretrained_vocab_path="./models/test001/test001.vocab")
```

`dataset_path` specifies the dataset to be used for finetuning. For this example, we use the same dataset as in training; normally, other datasets are used to finetune a model. The paths to model and vocabulary of the pretrained model are specified in `pretrained_model_path` and `pretrained_vocab_path`, respectively. 

It is also possible to fine-tune a model on a specified number of examples/rows from `dataset_path` (see the `n_train_examples` argument):

```python
from DeezyMatch import finetune as dm_finetune

# fine-tune a pretrained model stored at pretrained_model_path and pretrained_vocab_path 
dm_finetune(input_file_path="./inputs/input_dfm.yaml", 
            dataset_path="dataset/dataset-string-similarity_test.txt", 
            model_name="finetuned_test001",
            pretrained_model_path="./models/test001/test001.model", 
            pretrained_vocab_path="./models/test001/test001.vocab",
            n_train_examples=100)
```

The same can be done via command line:

```bash
DeezyMatch -i ./inputs/input_dfm.yaml -d dataset/dataset-string-similarity_test.txt -m finetuned_test001 -f ./models/test001/test001.model -v ./models/test001/test001.vocab -n 100 
```

:warning: If `-n` flag (or `n_train_examples` argument) is not specified, the train/valid/test proportions are read from the input file.

A new fine-tuned model called `finetuned_test001` will be stored in `models` directory. In this example, two components in the neural network architecture were frozen, that is, not changed during fine-tuning (see `layers_to_freeze` in the input file). When running the above command, DeezyMatch lists the parameters in the model and whether or not they will be used in finetuning:

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

It is possible to print all parameters in a model by:

```bash
DeezyMatch -pm models/finetuned_test001/finetuned_test001.model
```

which generates similar output as above.

### Model inference

When a model is trained/fine-tuned, `inference` module can be used for predictions/model-inference. Again, we use the same dataset (`dataset/dataset-string-similarity_test.txt`) as before in this example. The paths to model and vocabulary of the pretrained model are specified in `pretrained_model_path` and `pretrained_vocab_path`, respectively. 

```python
from DeezyMatch import inference as dm_inference

# model inference using a model stored at pretrained_model_path and pretrained_vocab_path 
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
# s1	s2	prediction	p0	p1	label
Krutoy	Крутой	1	0.1432	0.8568	1
Sharunyata	Shartjugskij	0	0.9553	0.0447	0
Sutangcun	羊山村	1	0.3031	0.6969	0
同心村	tong xin cun	1	0.1666	0.8334	1
Engeskjæran	Abrahamskjeret	0	0.7942	0.2058	0
Izumo-zaki	Tsumo-zaki	1	0.3216	0.6784	1
Qermez Khalifeh-ye `Olya	Qermez Khalīfeh-ye ‘Olyā	1	0.1267	0.8733	1
კირენია	Κυρά	0	0.8817	0.1183	0
Ozero Pogoreloe	Ozero Pogoreloye	1	0.2111	0.7889	1
Anfijld	ਮਾਕਰੋਨ ਸਟੇਡੀਅਮ	0	0.6214	0.3786	0
Qanât el-Manzala	El-Manzala Canal	1	0.2361	0.7639	1
Yŏhangmyŏn-samuso	yeohangmyeonsamuso	1	0.0820	0.9180	1
Mājra	Lahāri Tibba	0	0.5295	0.4705	0
```

`p0` and `p1` are probabilities assigned to labels 0 and 1, respectively. For example, in the first row, the actual label is 1 (last column), the predicted label is 1 (third column), and the model confidence on the predicted label is `0.8568`. In these examples, DeezyMatch correctly predicts the label in all rows except for `Sutangcun       羊山村` (with `p0=0.3031` and `p1=0.6969`). It should be noted, in this example and for showcasing DeezyMatch's functionalities, the model was trained and used for inference on one dataset. <ins>In practice, we train a model on a dataset and use it for prediction/inference on other datasets</ins>. Also, the dataset used to train the above model has around ~10K rows. Again, in practice, we use <ins>larger datasets</ins> for training.

### Generate query and candidate vectors

`inference` module can also be used to generate vector representations for a set of strings in a dataset. This is **a required step for alias selection and candidate ranking** (which we will [talk about later](#candidate-ranker-and-assembling-vector-representations)). We first create vector representations for **query** mentions (we assume the query mentions are stored in `dataset/dataset-string-similarity_test.txt`):

```python
from DeezyMatch import inference as dm_inference

# generate vectors for queries (specified in dataset_path) 
# using a model stored at pretrained_model_path and pretrained_vocab_path 
dm_inference(input_file_path="./inputs/input_dfm.yaml",
             dataset_path="dataset/dataset-string-similarity_test.txt", 
             pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
             pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab",
             inference_mode="vect",
             scenario="queries/test")
```

Compared to the previous section, here we have three additional arguments: 
* `inference_mode="vect"`: generate vector representations for the first column in `dataset_path`.
* `scenario`: directory to store the vectors.

The same can be done via command line:

```bash
DeezyMatch --deezy_mode inference -i ./inputs/input_dfm.yaml -d dataset/dataset-string-similarity_test.txt -m ./models/finetuned_test001/finetuned_test001.model -v ./models/finetuned_test001/finetuned_test001.vocab -mode vect --scenario queries/test
```

The resulting directory structure is:

```
queries/
└── test
    ├── dataframe.df
    ├── embeddings
    │   ├── rnn_fwd_0
    │   ├── rnn_fwd_1
    │   ├── rnn_fwd_2
    │   ├── rnn_fwd_3
    │   ├── rnn_fwd_4
    │   ├── rnn_fwd_5
    │   ├── rnn_fwd_6
    │   ├── rnn_fwd_7
    │   ├── rnn_fwd_8
    │   ├── rnn_fwd_9
    │   └── ...
    ├── input_dfm.yaml
    └── log.txt
```

(`embeddings` dir contains all the vector representations).

We repeat this step for `candidates` (again, we use the same dataset):

```python
from DeezyMatch import inference as dm_inference

# generate vectors for candidates (specified in dataset_path) 
# using a model stored at pretrained_model_path and pretrained_vocab_path 
dm_inference(input_file_path="./inputs/input_dfm.yaml",
             dataset_path="dataset/dataset-string-similarity_test.txt", 
             pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
             pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab",
             inference_mode="vect",
             scenario="candidates/test")
```

or via command line:

```bash
DeezyMatch --deezy_mode inference -i ./inputs/input_dfm.yaml -d dataset/dataset-string-similarity_test.txt -m ./models/finetuned_test001/finetuned_test001.model -v ./models/finetuned_test001/finetuned_test001.vocab -mode vect --scenario candidates/test
```

The resulting directory structure is:

```
candidates/
└── test
    ├── dataframe.df
    ├── embeddings
    │   ├── rnn_fwd_0
    │   ├── rnn_fwd_1
    │   ├── rnn_fwd_2
    │   ├── rnn_fwd_3
    │   ├── rnn_fwd_4
    │   ├── rnn_fwd_5
    │   ├── rnn_fwd_6
    │   ├── rnn_fwd_7
    │   ├── rnn_fwd_8
    │   ├── rnn_fwd_9
    │   └── ...
    ├── input_dfm.yaml
    └── log.txt
```

### Candidate ranker and assembling vector representations

Before using the `candidate_ranker` module of DeezyMatch, we need to:

1. Generate vector representations for both query and candidate mentions
2. Combine vector representations

----

Step 1 is already discussed in detail in the previous [section: Generate query and candidate vectors](#generate-query-and-candidate-vectors).

#### Combine vector representations 

This step is required if query or candidate vector representations are stored on several files (<ins>normally the case!</ins>). `combine_vecs` module assembles those vectors and store the results in `output_scenario` (see function below). For query vectors: 

```python
from DeezyMatch import combine_vecs

# combine vectors stored in queries/test and save them in combined/queries_test
combine_vecs(rnn_passes=['fwd', 'bwd'], 
             input_scenario='queries/test', 
             output_scenario='combined/queries_test', 
             print_every=10)
```

Similarly, for candidate vectors:

```python
from DeezyMatch import combine_vecs

# combine vectors stored in candidates/test and save them in combined/candidates_test
combine_vecs(rnn_passes=['fwd', 'bwd'], 
             input_scenario='candidates/test', 
             output_scenario='combined/candidates_test', 
             print_every=10)
```

Here, `rnn_passes` specifies that `combine_vecs` should assemble all vectors generated in the forward and backward RNN/GRU/LSTM passes (and stored in the `input_scenario` directory). NOTE: we have a backward pass only if `bidirectional` is set to `True` in the input file.

The results (for both query and candidate vectors) are sored in the `output_scenario` as follows:

```bash
combined/
├── candidates_test
│   ├── bwd_id.pt
│   ├── bwd_items.npy
│   ├── bwd.pt
│   ├── fwd_id.pt
│   ├── fwd_items.npy
│   ├── fwd.pt
│   └── input_dfm.yaml
└── queries_test
    ├── bwd_id.pt
    ├── bwd_items.npy
    ├── bwd.pt
    ├── fwd_id.pt
    ├── fwd_items.npy
    ├── fwd.pt
    └── input_dfm.yaml
```

The above steps can be done via command line, for query vectors:

```bash
DeezyMatch --deezy_mode combine_vecs -p fwd,bwd -sc queries/test -combs combined/queries_test
```

For candidate vectors:

```bash
DeezyMatch --deezy_mode combine_vecs -p fwd,bwd -sc candidates/test -combs combined/candidates_test
```

In this command, compared to `combine_vecs` module explained above:
* `-p`: `rnn_passes`
* `-sc`: `input_scenario`
* `-combs`: `output_scenario`

#### CandidateRanker

Candidate ranker uses the vector representations, generated and assembled in the previous sections, to find a set of candidates (from a dataset) for given queries in the same or another dataset. In the following example, for queries stored in `query_scenario`, we want to find 2 candidates (specified by `num_candidates`) from a dataset stored in `candidate_scenario`.

:warning: It is also possible to do [candidate ranking on-the-fly](#candidate-ranking-on-the-fly) in which query vectors are generated on-the-fly (and not stored in a dataset).

```python
from DeezyMatch import candidate_ranker

# Select candidates based on L2-norm distance (aka faiss distance):
# find candidates from candidate_scenario 
# for queries specified in query_scenario
candidates_pd = \
    candidate_ranker(query_scenario="./combined/queries_test",
                     candidate_scenario="./combined/candidates_test", 
                     ranking_metric="faiss", 
                     selection_threshold=5., 
                     num_candidates=2, 
                     search_size=2, 
                     output_path="ranker_results/test_candidates_deezymatch", 
                     pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
                     pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab", 
                     number_test_rows=20)
```

`query_scenario` is the directory that contains all the assembled query vectors [(see)](#combine-vector-representations) while `candidate_scenario` contains the assembled candidate vectors.

`ranking_metric`: choices are `faiss` (used here, L2-norm distance), `cosine` (cosine similarity), `conf` (confidence as measured by DeezyMatch prediction outputs). 

:warning: In our experiments, `conf` was not a good metric to rank candidates. Consider using `faiss` or `cosine` instead.

`selection_threshold`: changes according to the `ranking_metric`:

```text
A candidate will be selected if:
    faiss-distance <= threshold
    cosine-similarity >= threshold
    prediction-confidence >= threshold
```

:warning: Note that `cosine` and `conf` scores are between [0, 1] while `faiss` distance can take any values from [0, +&#8734;). 

`search_size`: for a given query, DeezyMatch searches for candidates iteratively. At each iteration, the selected ranking metric between a query and candidates (with the size of `search_size`) are computed, and if the number of desired candidates (specified by `num_candidates`) is not reached, a new batch of candidates with the size of `search_size` is tested in the next iteration. This continues until candidates with the size of `num_candidates` are found or all the candidates are tested. 

The DeezyMatch model and its vocabulary are specified by `pretrained_model_path` and `pretrained_vocab_path`, respectively. 

`number_test_rows`: **only for testing**. It specifies the number of queries to be used for testing.

The results can be accessed directly from `candidates_pd` variable (see the above command). Also, they are saved in `output_path` which is in a pandas dataframe fromat. The first few rows are:

```bash
                              query                                         pred_score                                     faiss_distance                                         cosine_sim                             candidate_original_ids  query_original_id  num_all_searches
id                                                                                                                                                                                                                                                                                  
0                        la dom nxy      {'la dom nxy': 0.7976, 'Laohuzhuang': 0.7717}         {'la dom nxy': 0.0, 'Laohuzhuang': 2.6753}         {'la dom nxy': 1.0, 'Laohuzhuang': 0.8917}             {'la dom nxy': 0, 'Laohuzhuang': 3743}                  0                 2
1                            Krutoy             {'Krutoy': 0.6356, 'Krugloye': 0.6229}                {'Krutoy': 0.0, 'Krugloye': 1.9234}                {'Krutoy': 1.0, 'Krugloye': 0.9526}                    {'Krutoy': 1, 'Krugloye': 4549}                  1                 2
2                        Sharunyata  {'Sharunyata': 0.7585, 'Shēlah-ye Nasar-e Jari...  {'Sharunyata': 0.0, 'Shēlah-ye Nasar-e Jaritā'...  {'Sharunyata': 1.0, 'Shēlah-ye Nasar-e Jaritā'...  {'Sharunyata': 2, 'Shēlah-ye Nasar-e Jaritā': ...                  2                 2
3                         Sutangcun       {'Sutangcun': 0.7508, 'Senge Pa`in': 0.7564}          {'Sutangcun': 0.0, 'Senge Pa`in': 2.2971}          {'Sutangcun': 1.0, 'Senge Pa`in': 0.9071}              {'Sutangcun': 3, 'Senge Pa`in': 8304}                  3                 2
```

As expected, candidate mentions (in `pred_score`, `faiss_distance`, `cosine_sim` and `candidate_original_ids`) are the same as the queries (second column), as we used one dataset for both queries and candidates.

Similarly, the above results can be generated via command line:

```bash
eezyMatch --deezy_mode candidate_ranker -qs ./combined/queries_test -cs ./combined/candidates_test -rm faiss -t 5 -n 2 -sz 2 -o ranker_results/test_candidates_deezymatch -mp ./models/finetuned_test001/finetuned_test001.model -v ./models/finetuned_test001/finetuned_test001.vocab -tn 20
```

In this command, compared to `candidate_ranker` module explained above:
* `-qs`: `query_scenario`
* `-cs`: `candidate_scenario`
* `-rm`: `ranking_metric`
* `-t`: `selection_threshold`
* `-n`: `num_candidates`
* `-sz`: `search_size`
* `-o`: `output_path`
* `-mp`: `pretrained_model_path`
* `-v`: `pretrained_vocab_path`
* `-tn`: `number_test_rows`

**Other methods**

* Select candidates based on cosine similarity:

```python
from DeezyMatch import candidate_ranker

# Select candidates based on cosine similarity:
# find candidates from candidate_scenario 
# for queries specified in query_scenario
candidates_pd = \
    candidate_ranker(query_scenario="./combined/queries_test",
                     candidate_scenario="./combined/candidates_test", 
                     ranking_metric="cosine", 
                     selection_threshold=0.51, 
                     num_candidates=2, 
                     search_size=2, 
                     output_path="ranker_results/test_candidates_deezymatch", 
                     pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
                     pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab", 
                     number_test_rows=20)
```

Note that the only differences compared to the previous command are `ranking_metric="cosine"` and `selection_threshold=0.51`.

## Candidate ranking on-the-fly

For a list of input strings (specified in `query` argument), DeezyMatch can rank candidates (stored in `candidate_scenario`) on-the-fly. Here, DeezyMatch generates and assembles the vector representations of strings in `query` on-the-fly.

```python
from DeezyMatch import candidate_ranker

# Ranking on-the-fly
# find candidates from candidate_scenario 
# for queries specified by the `query` argument
candidates_pd = \
    candidate_ranker(candidate_scenario="./combined/candidates_test",
                     query=["DeezyMatch", "kasra", "fede", "mariona"],
                     ranking_metric="faiss", 
                     selection_threshold=5., 
                     num_candidates=1, 
                     search_size=100, 
                     output_path="ranker_results/test_candidates_deezymatch_on_the_fly", 
                     pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
                     pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab", 
                     number_test_rows=20)
```

The candidate ranker can be initialised, to be used multiple times, by running:

```python
from DeezyMatch import candidate_ranker_init

# initializing candidate_ranker via candidate_ranker_init
myranker = candidate_ranker_init(candidate_scenario="./combined/candidates_test",
                                 query=["DeezyMatch", "kasra", "fede", "mariona"],
                                 ranking_metric="faiss", 
                                 selection_threshold=5., 
                                 num_candidates=1, 
                                 search_size=100, 
                                 output_path="ranker_results/test_candidates_deezymatch_on_the_fly", 
                                 pretrained_model_path="./models/finetuned_test001/finetuned_test001.model", 
                                 pretrained_vocab_path="./models/finetuned_test001/finetuned_test001.vocab", 
                                 number_test_rows=20)
```

The content of `myranker` can be printed by:

```python
print(myranker)
```

which results in:

```bash
-------------------------
* Candidate ranker params
-------------------------

Queries are based on the following list:
["DeezyMatch", "kasra", "fede", "mariona"]

candidate_scenario:     ./combined/candidates_test
---Searching params---
num_candidates:         1
ranking_metric:         faiss
selection_threshold:    5.0
search_size:            100
number_test_rows:       20
---I/O---
input_file_path:        default
output_path:            ranker_results/test_candidates_deezymatch_on_the_fly
pretrained_model_path:  ./models/finetuned_test001/finetuned_test001.model
pretrained_vocab_path:  ./models/finetuned_test001/finetuned_test001.vocab
```

To rank the queries:

```python
myranker.rank()
```

The results are stored in:

```python
myranker.output
```

All the query-related parameters can be changed via `set_query` method, for example:

```python
myranker.set_query(query=["another_example"])
```

other parameters include: 
```bash
query
query_scenario
ranking_metric
selection_threshold
num_candidates
search_size
number_test_rows
output_path
```

Again, we can rank the candidates for the new query by:

```python
myranker.rank()
# to access output:
myranker.output
```

## Tips / Suggestions on DeezyMatch functionalities

### Candidate ranker

* As already mentioned, based on our experiments, `conf` is not a good metric for ranking candidates. Consider using `faiss` or `cosine`.

* Adding prefix/suffix to input strings (see `prefix_suffix` option in the input file) can greatly enhance the ranking results. However, we recommend one-character-long prefix/suffix; otherwise, this may affect the computation time.

* In `candidate_ranker`, the user specifies a `ranking_metric` based on which the candidates are selected. However, DeezyMatch also reports the values of other metrics for those candidates. For example, if the user selects `ranking_metric="faiss"`, the candidates are selected based on the `faiss`-distance metric. At the same time, the values of `cosine` and `conf` metrics for **those candidates** (ranked according to the selected metric, in this case faiss) are also reported.

* In most use cases, `search_size` should be set `>= num_candidates`. However, if `num_candidates` is very large, it is better to set the `search_size` to lower values. Let's clarify this in an example. First, assume `num_candidates=4` (number of desired candidates is 4 for each query). If we set the `search_size` to values less than 4, let's say, 2. DeezyMatch needs to do at least two iterations. In the first iteration, it looks at the closest 2 candidate vectors (as `search_size` is 2). In the second iteration, candidate vectors 3 and 4 will be examined. So two iterations. Another choice is `search_size=4`. Here, DeezyMatch looks at 4 candidates in the first iteration, if they pass the threshold, it is done. If not, it will seach candidates 5-8 in the next iteration. Now, let's assume `num_candidates=1001` (i.e., number of desired candidates is 1001 for each query). If we set the `search_size=1000`, DeezyMatch has to search at least 2000 candidates (2 x 1000 `search_size`). If we set `search_size=100`, this time, DeezyMatch has to search at least 1100 candidates (11 x 100 `search_size`). So 900 vectors less. In the end, it is a trade-off between iterations and `search_size`.

## Credits

This project extensively uses the ideas/neural-network-architecture published in https://github.com/ruipds/Toponym-Matching. 
