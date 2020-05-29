### Test embeddings

To test the embeddings learned by DeezyMatch:

1. We create a simple dataset and call it `tests.csv` (already located in `./candidates/tests.csv`). Note that column 2 and 3 are omitted in the process. This will be changed in the next version of DeezyMatch (XXX).

2. Generate vectors for all queries: (Note that `-qc q` which sets the `query_candidate_mode` to `query`).

```bash
python modelInference.py -m ../models/test_model.model -d ./candidates/tests.csv -v ../vocabs/test_model.pickle -i ../input_dfm.yaml -n 10000 -mode generate_vectors -qc q
``` 

3. compare the embeddings (the inputs can be changed in the header):

```bash
python compare_embeddings.py
```

Depending on your model, the results may significantly differ from what we report here:

```bash
                       string to compare,                         reference string, l2_norm
                                  London,                                   London, 0.0
                                  Londan,                                   London, 0.046379465609788895
                                  nodnol,                                   London, 0.8457154035568237
                                  Lndoon,                                   London, 0.25316423177719116
                            Ban Pho Tinh,                                   London, 1.9595017433166504
                              Storey Bay,                                   London, 2.2820239067077637
                          Desa Sukmajaya,                                   London, 2.0710322856903076
                  Hotel Pine House Hotel,                                   London, 1.3341679573059082
           Jonesville Lock Landing Strip,                                   London, 1.1363288164138794
                      Letnik Kara-Kochku,                                   London, 1.0316883325576782
                     Leonards Beach Pond,                                   London, 0.49187901616096497
```

# Candidate selection

1. Generate vectors for all queries: (Note that `-qc q` which sets the `query_candidate_mode` to `query`).

```bash
python modelInference.py -m ../models/test_model.model -d ./candidates/tests.csv -v ../vocabs/test_model.pickle -i ../input_dfm.yaml -n 10000 -mode generate_vectors -qc q
``` 

2. Generate vectors for all candidates: (Note that `-qc c` which sets the `query_candidate_mode` to `candidate`)

```bash
python modelInference.py -m ../models/test_model.model -d ./candidates/tests.csv -v ../vocabs/test_model.pickle -i ../input_dfm.yaml -n 10000 -mode generate_vectors -qc c
``` 

3. Combine vectors. This step is required if candidates or queries are distributed on several files (e.g., candidates are in 5 separate files. For each file, steps 1-2 should be repeated. This results in 5 fwd and bwd embeddings). At this step, we combined those vectors.

```bash
python combineVecs.py -p embed_candidates/rnn_fwd* -p_id embed_candidates/rnn_indxs_0 -df df/candidates.df -n candidates_fwd -o combined
python combineVecs.py -p embed_candidates/rnn_bwd* -p_id embed_candidates/rnn_indxs_0 -df df/candidates.df -n candidates_bwd -o combined
python combineVecs.py -p embed_queries/rnn_fwd* -p_id embed_queries/rnn_indxs_0 -df df/queries.df -n queries_fwd -o combined
python combineVecs.py -p embed_queries/rnn_bwd* -p_id embed_queries/rnn_indxs_0 -df df/queries.df -n queries_bwd -o combined
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
