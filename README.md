# DeezyMatch (Deep Fuzzy Match)

## A flexible Deep Neural Network framework for fuzzy string matching

DeezyMatch has been applied to the following problems:

- toponym matching

Credits:

- This project extensively uses the codes published in https://github.com/ruipds/Toponym-Matching. 

### Run DeezyMatch

After installing DeezyMatch on your machine, a new classifier can be trained by:

```bash
python DeezyMatch.py -i input_dfm.yaml -d dataset/dataset-string-similarity_test.txt -m test001
```

---

Fine-Tune a previously trained DeezyMatch model (using, for instance, only 100 training instances):

```bash
python DeezyMatch.py -i input_dfm.yaml -d dataset/dataset-string-similarity_test.txt -f test001 -n 100 -m finetuned_model
```

DeezyMatch stores both the model and vocabulary used in the following directory structure:

```bash
models
└── test001
    ├── test001.model
    └── test001.vocab
```

To fine-tune an existing model:

```bash
python DeezyMatch.py -i input_dfm.yaml -d dataset/dataset-string-similarity_test.txt -f ./models/test001/test001.model -m finetuned_test001
```

It is also possible to fine-tune on a limited number of rows (note `-n 100`)

```bash
python DeezyMatch.py -i input_dfm.yaml -d dataset/dataset-string-similarity_test.txt -f ./models/test001/test001.model -n 100 -m finetuned_test001
```

After fine-tuning, there are two models in the `models` directory:

```bash
models
├── finetuned_test001
│   ├── finetuned_test001.model
│   └── finetuned_test001.vocab
└── test001
    ├── test001.model
    └── test001.vocab
```

After training/fine-tuning a model, DeezyMatch model can be used for inference or for candidate selection. Refer to `inference_candidate_finder` directory for more information.

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
