#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pytest


def test_pipeline_ngram():
    
    from DeezyMatch import train as dm_train
    # train a new model
    dm_train(input_file_path="./inputs/input_dfm_pytest_002.yaml",
             dataset_path="./dataset/dataset-string-matching_train.txt",
             model_name="test002")
    
    from DeezyMatch import finetune as dm_finetune
    # fine-tune a pretrained model stored at pretrained_model_path and pretrained_vocab_path
    dm_finetune(input_file_path="./inputs/input_dfm_pytest_002.yaml",
                dataset_path="./dataset/dataset-string-matching_train.txt",
                model_name="finetuned_test002",
                pretrained_model_path="./models/test002/test002.model",
                pretrained_vocab_path="./models/test002/test002.vocab")
    
    from DeezyMatch import inference as dm_inference
    
    # model inference using a model stored at pretrained_model_path and pretrained_vocab_path
    dm_inference(input_file_path="./inputs/input_dfm_pytest_002.yaml",
                 dataset_path="./dataset/dataset-string-matching_train.txt",
                 pretrained_model_path="./models/finetuned_test002/finetuned_test002.model",
                 pretrained_vocab_path="./models/finetuned_test002/finetuned_test002.vocab")
    
    
    from DeezyMatch import inference as dm_inference
    
    # generate vectors for queries (specified in dataset_path)
    # using a model stored at pretrained_model_path and pretrained_vocab_path
    dm_inference(input_file_path="./inputs/input_dfm_pytest_002.yaml",
                 dataset_path="./dataset/dataset-string-matching_train.txt",
                 pretrained_model_path="./models/finetuned_test002/finetuned_test002.model",
                 pretrained_vocab_path="./models/finetuned_test002/finetuned_test002.vocab",
                 inference_mode="vect",
                 scenario="queries_002/test")
    
    from DeezyMatch import inference as dm_inference
    
    # generate vectors for candidates (specified in dataset_path)
    # using a model stored at pretrained_model_path and pretrained_vocab_path
    dm_inference(input_file_path="./inputs/input_dfm_pytest_002.yaml",
                 dataset_path="./dataset/dataset-string-matching_train.txt",
                 pretrained_model_path="./models/finetuned_test002/finetuned_test002.model",
                 pretrained_vocab_path="./models/finetuned_test002/finetuned_test002.vocab",
                 inference_mode="vect",
                 scenario="candidates_002/test")
    
    
    from DeezyMatch import combine_vecs
    
    # combine vectors stored in queries/test and save them in combined/queries_test
    combine_vecs(rnn_passes=['fwd', 'bwd'], 
                 input_scenario='queries_002/test', 
                 output_scenario='combined_002/queries_test', 
                 print_every=10)
    
    from DeezyMatch import combine_vecs
    
    # combine vectors stored in candidates/test and save them in combined/candidates_test
    combine_vecs(rnn_passes=['fwd', 'bwd'],
                 input_scenario='candidates_002/test',
                 output_scenario='combined_002/candidates_test',
                 print_every=10)
    
    from DeezyMatch import candidate_ranker
    
    # Select candidates based on L2-norm distance (aka faiss distance):
    # find candidates from candidate_scenario
    # for queries specified in query_scenario
    candidates_pd = \
        candidate_ranker(query_scenario="./combined_002/queries_test",
                         candidate_scenario="./combined_002/candidates_test",
                         ranking_metric="faiss",
                         selection_threshold=5.,
                         num_candidates=2,
                         search_size=10,
                         output_path="ranker_results_002/test_candidates_deezymatch",
                         pretrained_model_path="./models/finetuned_test002/finetuned_test002.model",
                         pretrained_vocab_path="./models/finetuned_test002/finetuned_test002.vocab",
                         number_test_rows=5)
    
    for s in candidates_pd["query"].to_list():
        assert candidates_pd.loc[candidates_pd["query"] == s]["faiss_distance"].iloc[0][s] == pytest.approx(0.0)
    