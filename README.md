# DeezyMatch (Deep Fuzzy Match)

## A flexible Deep Neural Network framework for fuzzy string matching

DeezyMatch has been applied to the following problems:

- toponym matching

Credits:

- This project extensively uses the codes published in https://github.com/ruipds/Toponym-Matching. 

---

Run DeezyMatch:

```bash
python DeezyMatch.py input_dfm.yaml dataset/dataset-string-similarity_test.txt model_name
```

---

Installation:

```
conda env create -f environment.yaml
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
