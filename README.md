# DeezyMatch (Deep Fuzzy Match)

## A flexible Deep Neural Network framework for fuzzy string matching

DeezyMatch has been applied to the following problems:

- toponym matching

Credits:

- This project extensively uses the ideas/neural-network-architecture published in https://github.com/ruipds/Toponym-Matching. 

### Run DeezyMatch

After installing DeezyMatch on your machine, a new classifier can be trained by:

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

To perform inference on a dataset:

```bash
python DeezyMatch.py --deezy_mode inference -m ./models/finetuned_test001/finetuned_test001.model -d dataset/dataset-string-similarity_test.txt -v ./models/finetuned_test001/finetuned_test001.vocab -i ./input_dfm.yaml -mode test
```

In candidate selection, we first need to create vectors for both query and candidates. This can be done by:

```bash
# queries
python DeezyMatch.py --deezy_mode inference -m ./models/finetuned_test001/finetuned_test001.model -d dataset/dataset-string-similarity_test.txt -v ./models/finetuned_test001/finetuned_test001.vocab -i ./input_dfm.yaml -mode vect --scenario test -qc q

# candidates
python DeezyMatch.py --deezy_mode inference -m ./models/finetuned_test001/finetuned_test001.model -d dataset/dataset-string-similarity_test.txt -v ./models/finetuned_test001/finetuned_test001.vocab -i ./input_dfm.yaml -mode vect --scenario test -qc c
```

Refer to `inference_candidate_finder` directory for more information.

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

---

Run `deepneuralnetwork.py`:

```
python deepneuralnetwork.py dataset/dataset-string-similarity_test.txt
```

For testing, you can change `nb_epoch=20` to `nb_epoch=2` in deepneuralnetwork.py.

---

**Changes**

(from the original github repo https://github.com/ruipds/Toponym-Matching):


* In `deepneuralnetwork.py`, we had to change the `call` function: (changes are based on https://github.com/richliao/textClassifier/issues/13)

```
    def call(self, x, mask=None):
        eij = K.tanh(K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1))
        ai = K.exp(eij)
        weights = ai/K.expand_dims(K.sum(ai, axis=1),1)
        weighted_input = x*K.expand_dims(weights,2)
        return K.sum(weighted_input, axis=1)

        """
        eij = K.tanh(K.dot(x, self.W))
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        weighted_input = x*weights.dimshuffle(0,1,'x')
        self.attention = weights
        return weighted_input.sum(axis=1)
        """
```
