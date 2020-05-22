## Generate vectors

1. Set the `query_candidate_mode` to `c` (i.e., candidate mode) in the input file.
2. Generate vectprs for all candidates:

```bash
python modelInference.py ../models/test_001.model.model ./candidates/uniqueAltnamesGeonames.csv ../vocabs/test_001.model.pickle ../input_dfm.yaml 100
```

3. repeat step 2 with `query_candidate_mode: "q"`
4. combine vectors


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
