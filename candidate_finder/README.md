## Generate vectors

### Test the embeddings

To test the embeddings learned by DeezyMatch:

1. We create a simple dataset and call it `tests.csv` (already located in `./candidates/tests.csv`)

Note that column 2 and 3 are omitted in the process. This will be changed in the next version of DeezyMatch (XXX).

2. Generate vectors for all queries: (Note that `-qc q` which sets the `query_candidate_mode` to `query`)

```bash
python modelInference.py -m ../models/test_model.model -d ./candidates/tests.csv -v ../vocabs/test_model.pickle -i ../input_dfm.yaml -n 10000 -mode generate_vectors -qc q
``` 

3. compare the embeddings (the inputs can be changed in the header):

```bash
python compare_embeddings.py
```

# --- continue here (candidates)

2. Generate vectors for all candidates: (Note that `-qc c` which sets the `query_candidate_mode` to `candidate`)

```bash
python modelInference.py -m ../models/test_model.model -d ./candidates/tests.csv -v ../vocabs/test_model.pickle -i ../input_dfm.yaml -n 10000 -mode generate_vectors -qc c
``` 

--

# OLD README FILE


### Generate vectors for all candidates (note "c" at the very end of the command! which will be changed in the next version):
```bash
python TestModel.py Models/testme.model candidate_selection/new_set/uniqueAltnamesGeonames.csv Vocabs/testme.pickle input_testme.yaml 1000 c
```

### Generate vectors for queries (note the ugly "q" at the very end):
```bash
python TestModel.py Models/testme.model candidate_selection/new_set/uniqueAltnamesGeonames.csv Vocabs/testme.pickle input_testme.yaml 32 q
```

## Combine all vectors

### Candidates
```bash
python CombineVecs.py c
```

### Queries
```bash
python CombineVecs.py q
```

## Candidate finder:
First change the inputs at the top of the file, e.g.,

```python
# --- inputs
output_filename = "test"
num_desired_candidates = 10
min_threshold_deezy = 0.8
input_file = "./input_testme.yaml"
train_vocab_path = "Vocabs/testme.pickle"
model_path = "Models/testme.model"
# set number of neighbours to use in search 
search_size = 25
```

Run:

```bash
python CandidateFinder.py
```
